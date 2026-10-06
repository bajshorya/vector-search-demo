// Step 11 — Hardening the LLM call.
//
// ask.ts used to call Ollama with a bare `fetch` and no safety net: one hiccup
// (server still loading the model, a dropped connection, a request that hangs
// forever) took the whole pipeline down with an ugly stack trace. This module
// wraps that call in the three things a production-minded LLM client needs:
//
//   1. A TIMEOUT            — never let a hung request block forever.
//   2. ERROR CLASSIFICATION — is this failure worth retrying, or is it our bug?
//   3. RETRY WITH BACKOFF   — retry the *retryable* failures, backing off each time.
//
// The call runs against a LOCAL model (Ollama / llama3.2), so there is no real
// rate limit or cloud outage here. But the failure *modes* are identical to a
// hosted API — timeouts, 5xx while the server is busy, a 400 from a malformed
// request — so the handling generalises directly to Claude, OpenAI, etc. Swap
// the URL and body shape and everything below is unchanged.
//
// Everything is returned as a Result<string, LLMError> (see result.ts) rather
// than thrown, so callers must deal with failure explicitly.

import { type Result, ok, err } from "./result.js";

const OLLAMA_URL = "http://localhost:11434/api/chat";
const MODEL = "llama3.2";

/** Default ceiling on a single LLM call. Past this we abort and report a timeout. */
export const DEFAULT_TIMEOUT_MS = 30_000;

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  /** Present on an assistant turn that decided to call tools (Ollama's shape). */
  tool_calls?: ToolCall[];
  /** Set on a role:"tool" message so the model knows which call this answers. */
  tool_name?: string;
}

// --- Tool-calling types (Ollama /api/chat `tools`) ---------------------------
//
// Ollama follows the OpenAI-style function-tool schema: each tool is a function
// with a name, a human-readable description (the model reads this to decide when
// to call it), and a JSON-Schema `parameters` object describing its arguments.

export interface Tool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, unknown>;
      required?: string[];
    };
  };
}

/** One tool call the model asked for. `arguments` is already parsed JSON. */
export interface ToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

export interface LLMOptions {
  /** Abort (and fail) the call if it takes longer than this. */
  timeoutMs?: number;
  /** 0 = deterministic. The re-ranker wants this low; generation can be higher. */
  temperature?: number;
  /** Ask Ollama to constrain output to valid JSON (used by the re-ranker). */
  format?: "json";
}

// --- The operational-vs-programmer-error distinction (Day 5, applied) --------
//
// OPERATIONAL: something outside our control went transiently wrong — the server
//   was busy (5xx), we got rate-limited (429), the network blipped, the request
//   timed out. These are worth RETRYING: the same request may well succeed a
//   moment later.
//
// PROGRAMMER: *we* sent something wrong — a malformed body, an unknown model, a
//   bad field (4xx that isn't 429). Retrying is pointless and actively harmful:
//   the same broken request will fail identically every time, so we just burn
//   time and hammer the server. Surface it loudly instead so the bug gets fixed.
export type LLMErrorKind = "operational" | "programmer";

export interface LLMError {
  kind: LLMErrorKind;
  message: string;
  /** HTTP status, when the failure came back as a response. */
  status?: number;
}

const operational = (message: string, status?: number): LLMError => ({
  kind: "operational",
  message,
  status,
});

const programmer = (message: string, status?: number): LLMError => ({
  kind: "programmer",
  message,
  status,
});

/** Only operational errors are worth trying again. */
export const isRetryable = (e: LLMError): boolean => e.kind === "operational";

/**
 * ONE attempt at an Ollama chat completion, with a hard timeout. Returns the
 * full answer text on success, or a classified LLMError on failure — never
 * throws. Non-streaming on purpose: a retry that had already streamed half an
 * answer to the screen would be a mess, so we take the whole reply atomically.
 */
export async function chatOnce(
  messages: ChatMessage[],
  options: LLMOptions = {},
): Promise<Result<string, LLMError>> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  // AbortController is how you put a deadline on fetch: the timer fires abort(),
  // fetch rejects, and we turn that into a clean operational timeout below.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(OLLAMA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: MODEL,
        stream: false,
        format: options.format,
        options: { temperature: options.temperature ?? 0 },
        messages,
      }),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      // 4xx (except 429) means we sent a bad request — our bug, don't retry.
      // 429 (rate limit) and 5xx (server busy/unavailable) are transient.
      if (
        response.status >= 400 &&
        response.status < 500 &&
        response.status !== 429
      ) {
        return err(
          programmer(
            `Ollama rejected the request: ${body || response.statusText}`,
            response.status,
          ),
        );
      }
      return err(
        operational(
          `Ollama returned ${response.status}: ${body || response.statusText}`,
          response.status,
        ),
      );
    }

    const data = (await response.json()) as { message?: { content?: string } };
    const content = data.message?.content;
    if (!content) {
      // A 200 with no content is not something a retry fixes — treat as our bug.
      return err(programmer("Ollama returned a 200 with no message content."));
    }
    return ok(content);
  } catch (e) {
    // fetch itself failed (never reached the point of an HTTP status).
    if (e instanceof DOMException && e.name === "AbortError") {
      return err(operational(`LLM call timed out after ${timeoutMs}ms.`));
    }
    // Connection refused / DNS / dropped socket — transient network trouble.
    const cause = String((e as { cause?: unknown })?.cause ?? "");
    if (e instanceof TypeError) {
      const detail = cause.includes("ECONNREFUSED")
        ? "can't reach Ollama at localhost:11434 (is `ollama serve` running?)"
        : cause || e.message;
      return err(operational(`Network error calling Ollama: ${detail}`));
    }
    // Anything else is unexpected — surface it as a programmer error.
    return err(
      programmer(`Unexpected error calling Ollama: ${(e as Error).message}`),
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * A small, reusable retry-with-backoff wrapper — generic over any Result-returning
 * async function, not just LLM calls. Retries only the failures `isRetryable`
 * approves (operational ones), sleeping `baseDelayMs` and DOUBLING it each time
 * (exponential backoff: 500ms → 1s → 2s). A programmer error, or a run out of
 * attempts, returns immediately.
 */
export async function withRetry<T, E>(
  fn: () => Promise<Result<T, E>>,
  opts: {
    retries?: number;
    baseDelayMs?: number;
    isRetryable: (e: E) => boolean;
    onRetry?: (e: E, attempt: number, delayMs: number) => void;
  },
): Promise<Result<T, E>> {
  const retries = opts.retries ?? 3;
  const baseDelayMs = opts.baseDelayMs ?? 500;

  let attempt = 0;
  // Loop is `retries` retries AFTER the first try => retries + 1 total attempts.
  while (true) {
    const result = await fn();
    if (result.ok) return result;

    const outOfTries = attempt >= retries;
    if (outOfTries || !opts.isRetryable(result.error)) return result;

    const delayMs = baseDelayMs * 2 ** attempt; // 500, 1000, 2000, ...
    opts.onRetry?.(result.error, attempt + 1, delayMs);
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    attempt++;
  }
}

/**
 * The everyday entry point: a chat completion that is timed out AND retried.
 * This is what ask.ts and rerank.ts call. On failure the returned LLMError is
 * already classified, so callers can print a friendly message and move on.
 */
export function chat(
  messages: ChatMessage[],
  options: LLMOptions = {},
): Promise<Result<string, LLMError>> {
  return withRetry(() => chatOnce(messages, options), {
    isRetryable,
    onRetry: (e, attempt, delayMs) =>
      console.error(
        `   ⚠️  LLM call failed (${e.message}) — retry ${attempt} in ${delayMs}ms`,
      ),
  });
}

// --- Tool-calling variant ----------------------------------------------------
//
// Same transport, timeout, error classification and retry as chatOnce/chat, but
// it (a) sends a `tools` array so the model may call them, and (b) returns the
// FULL assistant message (content + any tool_calls) instead of just text — the
// caller needs the tool_calls to drive the agent loop. Non-tool-calling code is
// untouched and keeps using chat() above.

async function chatWithToolsOnce(
  messages: ChatMessage[],
  tools: Tool[],
  options: LLMOptions = {},
): Promise<Result<ChatMessage, LLMError>> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(OLLAMA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: MODEL,
        stream: false,
        options: { temperature: options.temperature ?? 0 },
        tools,
        messages,
      }),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      if (
        response.status >= 400 &&
        response.status < 500 &&
        response.status !== 429
      ) {
        return err(
          programmer(
            `Ollama rejected the request: ${body || response.statusText}`,
            response.status,
          ),
        );
      }
      return err(
        operational(
          `Ollama returned ${response.status}: ${body || response.statusText}`,
          response.status,
        ),
      );
    }

    const data = (await response.json()) as { message?: ChatMessage };
    const message = data.message;
    if (!message) {
      return err(programmer("Ollama returned a 200 with no message."));
    }
    return ok(message);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      return err(operational(`LLM call timed out after ${timeoutMs}ms.`));
    }
    const cause = String((e as { cause?: unknown })?.cause ?? "");
    if (e instanceof TypeError) {
      const detail = cause.includes("ECONNREFUSED")
        ? "can't reach Ollama at localhost:11434 (is `ollama serve` running?)"
        : cause || e.message;
      return err(operational(`Network error calling Ollama: ${detail}`));
    }
    return err(
      programmer(`Unexpected error calling Ollama: ${(e as Error).message}`),
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Tool-aware chat: timed out AND retried like chat(), but returns the whole
 * assistant message so the caller can inspect `tool_calls`. This is what the
 * agent loop in tool-ask.ts calls.
 */
export function chatWithTools(
  messages: ChatMessage[],
  tools: Tool[],
  options: LLMOptions = {},
): Promise<Result<ChatMessage, LLMError>> {
  return withRetry(() => chatWithToolsOnce(messages, tools, options), {
    isRetryable,
    onRetry: (e, attempt, delayMs) =>
      console.error(
        `   ⚠️  LLM call failed (${e.message}) — retry ${attempt} in ${delayMs}ms`,
      ),
  });
}

// what does this code do summary:
// This TypeScript code provides a robust interface for interacting with a local LLM (Large Language Model) server, specifically Ollama running the llama3.2 model. It defines types and functions to handle chat messages, manage timeouts, classify errors, and implement retry logic with exponential backoff.
//function wise :
// 1. `chatOnce`: Makes a single request to the LLM server, handling timeouts and classifying errors into operational or programmer errors. It returns a Result type indicating success or failure.
// 2. `withRetry`: A generic function that retries a given asynchronous operation based on the classification of errors. It implements exponential backoff for retryable errors.
// 3. `chat`: The main entry point for making chat requests to the LLM server, combining the functionality of `chatOnce` and `withRetry` to provide a robust and fault-tolerant interface for users.
