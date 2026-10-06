// Tests for the tool-calling layer (src/tool-ask.ts).
//
// Two tiers:
//   1. DETERMINISTIC unit tests on executeToolCall / executeToolCalls — these
//      cover argument validation, error handling, parallel execution and unknown
//      tools without touching the LLM, so they're fast and never flaky.
//   2. LIVE integration tests that drive the real loop through Ollama. They are
//      skipped automatically when Ollama isn't running, so the suite still passes
//      offline.
//
// Run with:  npm test
//
// The five scenarios the task asks for are labelled [SCENARIO n] below.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  executeToolCall,
  executeToolCalls,
  run,
  type ToolOutcome,
} from "./tool-ask.js";
import type { ToolCall } from "./llm.js";
import { countBookmarks } from "./bookmarks.js";

// Small helper to build a ToolCall the way the model would send one.
const call = (name: string, args: Record<string, unknown> = {}): ToolCall => ({
  function: { name, arguments: args },
});

const parse = (o: ToolOutcome) => JSON.parse(o.content);

// --- Deterministic unit tests ------------------------------------------------

test("[SCENARIO 2] one-tool request: get_bookmarks returns matching bookmarks", () => {
  const outcome = executeToolCall(call("get_bookmarks", { tag: "typescript" }));
  assert.equal(outcome.ok, true);
  const bookmarks = parse(outcome);
  assert.ok(Array.isArray(bookmarks));
  assert.ok(bookmarks.length >= 1);
  assert.ok(bookmarks.every((b: { tags: string[] }) => b.tags.includes("typescript")));
});

test("get_bookmarks for an unknown tag returns an empty list (not an error)", () => {
  const outcome = executeToolCall(call("get_bookmarks", { tag: "no-such-tag" }));
  assert.equal(outcome.ok, true);
  assert.deepEqual(parse(outcome), []);
});

test("add_bookmark then count_bookmarks reflects the new entry", () => {
  const before = countBookmarks();

  const added = executeToolCall(
    call("add_bookmark", {
      url: "https://example.com/zod",
      title: "Zod docs",
      tags: ["Zod", "typescript"],
    }),
  );
  assert.equal(added.ok, true);
  const bookmark = parse(added);
  assert.equal(bookmark.url, "https://example.com/zod");
  // tags are normalised to lowercase + de-duped by the store
  assert.deepEqual(bookmark.tags, ["zod", "typescript"]);

  const counted = executeToolCall(call("count_bookmarks"));
  assert.equal(counted.ok, true);
  assert.equal(parse(counted).count, before + 1);
});

test("add_bookmark rejects a duplicate url as a readable tool error (no crash)", () => {
  // Adding the same url twice: first succeeds, second is caught and reported.
  executeToolCall(
    call("add_bookmark", { url: "https://example.com/dup", title: "Dup", tags: [] }),
  );
  const second = executeToolCall(
    call("add_bookmark", { url: "https://example.com/dup", title: "Dup", tags: [] }),
  );
  assert.equal(second.ok, false);
  assert.match(parse(second).error, /already exists/i);
});

test("[SCENARIO 3] multiple / parallel tool calls all run and all return results", () => {
  const outcomes = executeToolCalls([
    call("get_bookmarks", { tag: "rag" }),
    call("count_bookmarks"),
    call("get_bookmarks", { tag: "postgres" }),
  ]);
  assert.equal(outcomes.length, 3);
  assert.ok(outcomes.every((o) => o.ok));
  assert.ok(Array.isArray(parse(outcomes[0])));
  assert.equal(typeof parse(outcomes[1]).count, "number");
  assert.ok(Array.isArray(parse(outcomes[2])));
});

test("[SCENARIO 4] invalid arguments are rejected by Zod with a readable error", () => {
  // Missing required `tag`.
  const missing = executeToolCall(call("get_bookmarks", {}));
  assert.equal(missing.ok, false);
  assert.match(parse(missing).error, /Invalid arguments/i);

  // Empty `tag` violates .min(1).
  const empty = executeToolCall(call("get_bookmarks", { tag: "" }));
  assert.equal(empty.ok, false);

  // Malformed url for add_bookmark.
  const badUrl = executeToolCall(
    call("add_bookmark", { url: "not-a-url", title: "x", tags: [] }),
  );
  assert.equal(badUrl.ok, false);
  assert.match(parse(badUrl).error, /Invalid arguments/i);
});

test("[SCENARIO 5] unknown / hallucinated tool is handled gracefully", () => {
  const outcome = executeToolCall(call("delete_all_the_things", { confirm: true }));
  assert.equal(outcome.ok, false);
  assert.match(parse(outcome).error, /Unknown tool/i);
  // and it lists the real tools so the model can recover
  assert.match(parse(outcome).error, /get_bookmarks/);
});

test("[SCENARIO 1 prep] a turn with zero tool calls executes nothing", () => {
  assert.deepEqual(executeToolCalls([]), []);
});

// --- Live integration tests (skipped when Ollama is offline) -----------------

async function ollamaUp(): Promise<boolean> {
  try {
    const res = await fetch("http://localhost:11434/api/tags", {
      signal: AbortSignal.timeout(2000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

const live = await ollamaUp();
const skip = live ? false : "Ollama not running on localhost:11434";

test("[SCENARIO 1] a normal question needing no tool returns plain text", { skip }, async () => {
  const answer = await run("In one sentence, what is a hash map?");
  assert.equal(typeof answer, "string");
  assert.ok(answer.trim().length > 0);
});

test("[SCENARIO 2 live] a one-tool request produces a grounded answer", { skip }, async () => {
  const answer = await run("What typescript bookmarks do I have saved?");
  assert.equal(typeof answer, "string");
  assert.ok(answer.trim().length > 0);
});
