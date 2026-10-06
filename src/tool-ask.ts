// Tool calling — the standard agent loop, added alongside the RAG pipeline.
//
// RAG (ask.ts) stuffs retrieved text into the prompt and asks for one answer.
// Tool calling is different: we hand the model some TOOLS it can choose to call,
// let it ask for calls, run them in OUR code, feed the results back, and repeat
// until it returns plain text. The loop:
//
//   user question -> LLM -> (tool calls?) -> we run the tools -> tool results
//                 -> LLM -> ... -> final text answer
//
// Three tools, all backed by the mock store in bookmarks.ts:
//   get_bookmarks(tag)              — read
//   add_bookmark(url, title, tags)  — write
//   count_bookmarks()               — read, no args
//
// Every tool call is VALIDATED with Zod before we run it, and every tool error is
// caught and handed back to the model as a readable tool_result instead of
// crashing. Nothing in the RAG pipeline is touched; this reuses only the hardened
// LLM transport from llm.ts (timeout + retry + error classification).
//
// Run with:  npm run tools "find my typescript bookmarks and tell me how many I have"

import { z } from "zod";
import { pathToFileURL } from "node:url";
import { chatWithTools, type ChatMessage, type Tool, type ToolCall } from "./llm.js";
import { getBookmarks, addBookmark, countBookmarks } from "./bookmarks.js";

// --- 1. Tool registry: schema + description + implementation, in one place ---
//
// Each tool pairs a Zod schema (the single source of truth for its arguments —
// used BOTH to generate the JSON schema the model sees AND to validate what the
// model sends back) with the actual code to run. `run` returns any JSON-
// serialisable value; the loop stringifies it for the tool_result.
interface ToolSpec {
  description: string;
  schema: z.ZodObject<z.ZodRawShape>;
  run: (args: Record<string, unknown>) => unknown;
}

const REGISTRY: Record<string, ToolSpec> = {
  get_bookmarks: {
    description:
      "Look up the user's saved bookmarks filed under a given tag. Use this " +
      "when the user asks about their bookmarks, saved links, or reading list " +
      "for a topic. Returns a list of {title, url, tags}.",
    schema: z.object({
      tag: z
        .string()
        .min(1)
        .describe("The single tag to filter by, e.g. 'typescript'. Lowercase, no '#'."),
    }),
    run: (args) => getBookmarks(args.tag as string),
  },

  add_bookmark: {
    description:
      "Save a new bookmark for the user. Use this when the user asks to save, " +
      "add, or bookmark a link. Returns the saved bookmark.",
    schema: z.object({
      url: z.url().describe("The full URL to bookmark, including https://."),
      title: z.string().min(1).describe("A short human-readable title for the link."),
      tags: z
        .array(z.string())
        .default([])
        .describe("Zero or more topic tags, e.g. ['rag','postgres']."),
    }),
    run: (args) =>
      addBookmark(args.url as string, args.title as string, args.tags as string[]),
  },

  count_bookmarks: {
    description:
      "Count how many bookmarks the user has saved in total. Takes no arguments.",
    schema: z.object({}),
    run: () => ({ count: countBookmarks() }),
  },
};

// Build the tool definitions the model sees, deriving each `parameters` JSON
// schema straight from the Zod schema so the two can never drift apart.
const TOOLS: Tool[] = Object.entries(REGISTRY).map(([name, spec]) => ({
  type: "function",
  function: {
    name,
    description: spec.description,
    parameters: z.toJSONSchema(spec.schema) as Tool["function"]["parameters"],
  },
}));

// --- 2. Executing one tool call: validate, run, never throw ------------------

export interface ToolOutcome {
  /** false for an unknown tool, invalid arguments, or a thrown error. */
  ok: boolean;
  /** JSON string to hand back to the model as the tool_result content. */
  content: string;
}

/**
 * Run a single tool call safely. Returns a ToolOutcome whose `content` is always
 * a JSON string suitable for a tool_result — it NEVER throws, so a bad call can't
 * take the loop down. Handles three failure modes distinctly, each as a readable
 * error the model can recover from:
 *   - unknown / hallucinated tool name
 *   - arguments that fail Zod validation
 *   - the tool implementation itself throwing
 */
export function executeToolCall(call: ToolCall): ToolOutcome {
  const name = call.function.name;
  const spec = REGISTRY[name];

  // (a) Unknown / hallucinated tool.
  if (!spec) {
    const available = Object.keys(REGISTRY).join(", ");
    return {
      ok: false,
      content: JSON.stringify({
        error: `Unknown tool "${name}". Available tools: ${available}.`,
      }),
    };
  }

  // (b) Validate arguments with Zod before running anything.
  const parsed = spec.schema.safeParse(call.function.arguments ?? {});
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`)
      .join("; ");
    return {
      ok: false,
      content: JSON.stringify({ error: `Invalid arguments for ${name}: ${issues}` }),
    };
  }

  // (c) Run it, catching any error the implementation throws.
  try {
    const result = spec.run(parsed.data as Record<string, unknown>);
    return { ok: true, content: JSON.stringify(result ?? null) };
  } catch (e) {
    return {
      ok: false,
      content: JSON.stringify({ error: `${name} failed: ${(e as Error).message}` }),
    };
  }
}

/**
 * Run every tool call in a batch and return all results, in order. This is how
 * PARALLEL tool calls are handled: the model may ask for several at once, and we
 * execute them all and return one tool_result per call. (The tools here are
 * synchronous, so "parallel" just means "all of them this turn".)
 */
export function executeToolCalls(calls: ToolCall[]): ToolOutcome[] {
  return calls.map(executeToolCall);
}

// --- 3. The agent loop -------------------------------------------------------

export const MAX_TOOL_ITERATIONS = 5; // hard cap so a confused model can't loop forever

const SYSTEM_PROMPT =
  "You are a helpful assistant that manages the user's bookmarks using the " +
  "provided tools: get_bookmarks(tag), add_bookmark(url, title, tags), and " +
  "count_bookmarks(). Call tools when they help answer the request — you may " +
  "call more than one. After the tools return, answer in plain language, " +
  "including any titles and URLs you found. If get_bookmarks returns an empty " +
  "list, say nothing is bookmarked under that tag.";

/**
 * Drive the full loop for one question and return the model's final text answer.
 * Throws on an LLM transport failure or if the loop exceeds MAX_TOOL_ITERATIONS;
 * the CLI wrapper at the bottom turns those into a clean exit. Returning (rather
 * than process.exit-ing) is what lets the tests drive this directly.
 */
export async function run(question: string): Promise<string> {
  console.log(`\n👤 User message: "${question}"`);

  const messages: ChatMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: question },
  ];

  for (let iteration = 1; iteration <= MAX_TOOL_ITERATIONS; iteration++) {
    const result = await chatWithTools(messages, TOOLS);
    if (!result.ok) {
      const { kind, message } = result.error;
      throw new Error(`LLM call failed (${kind} error): ${message}`);
    }

    const reply = result.value;
    messages.push(reply); // keep the assistant turn in the transcript

    const toolCalls = reply.tool_calls ?? [];

    // Zero tool calls => the model is done; this is the final text answer.
    if (toolCalls.length === 0) {
      const answer = reply.content || "(empty response)";
      console.log(`\n✅ Final answer (iteration ${iteration}):\n`);
      process.stdout.write(answer + "\n");
      return answer;
    }

    // One or more (possibly parallel) tool calls: run them ALL and feed every
    // result back before the model's next turn.
    console.log(
      `\n🔧 ${toolCalls.length} tool call(s) requested (iteration ${iteration}):`,
    );
    const outcomes = executeToolCalls(toolCalls);

    toolCalls.forEach((call, i) => {
      const outcome = outcomes[i];
      const mark = outcome.ok ? "✓" : "✗ FAILED";
      console.log(`   ${mark} ${call.function.name}(${JSON.stringify(call.function.arguments)})`);
      console.log(`      result: ${outcome.content}`);

      messages.push({
        role: "tool",
        tool_name: call.function.name,
        content: outcome.content,
      });
    });
    // Loop: the model now sees the tool results and gets another turn.
  }

  throw new Error(
    `Gave up after ${MAX_TOOL_ITERATIONS} tool-loop iterations without a final answer.`,
  );
}

// --- 4. CLI entry (only when run directly, not when imported by tests) -------

const isMain =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) {
  const question =
    process.argv.slice(2).join(" ") ||
    "What typescript bookmarks do I have, and how many bookmarks total?";

  run(question).catch((e: unknown) => {
    console.error(`\n❌ ${(e as Error).message}`);
    process.exit(1);
  });
}
