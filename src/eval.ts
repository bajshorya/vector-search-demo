// Step 13 — A lightweight eval: "does the answer actually use the context?"
//
// This is NOT a full evaluation framework — it's the first honest signal, the
// seed of the logging/eval work that comes later. It runs the real pipeline
// (retrieve → re-rank → generate) against a handful of questions whose answers
// we already know, and checks two things that matter most in RAG:
//
//   1. When the answer IS in the document, does it answer correctly?
//   2. When the answer is NOT in the document, does it correctly DECLINE
//      ("I don't know...") instead of hallucinating?
//
// (2) is the one people forget. A RAG system that always sounds confident is
// worse than useless — the failure is invisible. So half the cases below are
// deliberately unanswerable from the README.
//
// The pass/fail check here is a crude heuristic (keyword presence + whether it
// declined), meant to flag regressions at a glance, not to be a ground truth.
// The point is to log question / chunks / answer side by side so you can eyeball
// them. Read the answers; don't just trust the ✅.
//
// Run with:  npm run eval

import "dotenv/config";
import { retrieveAndRerank } from "./rerank.js";
import { chat } from "./llm.js";
import { chunkLabel } from "./chunk.js";

const SYSTEM_PROMPT = `You are a helpful assistant answering questions about a codebase, using ONLY the context chunks provided in the user's message.

Rules:
- Base every claim on the context. Do not use outside knowledge to fill gaps.
- If the context does not contain the answer, say "I don't know based on the indexed documents" — do not guess.
- When you use a chunk, cite it inline like [1] or [2] matching the chunk ids.
- Be concise: a few sentences or a short list is usually enough.`;

interface TestCase {
  question: string;
  /** True if the README can answer it; false if it deliberately can't. */
  shouldAnswer: boolean;
  /** For answerable cases: at least one of these should appear in a good answer. */
  expectKeywords?: string[];
  why: string;
}

// 6 cases: 4 answerable from README.md, 2 deliberately outside it.
const CASES: TestCase[] = [
  {
    question: "What is the default chunk size in characters?",
    shouldAnswer: true,
    expectKeywords: ["900"],
    why: "A specific fact stated in the README (CHUNK_SIZE = 900).",
  },
  {
    question: "How much overlap is used between chunks?",
    shouldAnswer: true,
    expectKeywords: ["150"],
    why: "Another specific fact (CHUNK_OVERLAP = 150 chars).",
  },
  {
    question: "What embedding model does this project use?",
    shouldAnswer: true,
    expectKeywords: ["minilm", "all-minilm", "384", "transformers"],
    why: "Stated: all-MiniLM-L6-v2, 384 dims, via Transformers.js.",
  },
  {
    question: "Why do we use overlap between chunks at all?",
    shouldAnswer: true,
    expectKeywords: ["boundary", "cracks", "sentence", "straddle", "split", "context"],
    why: "The README explains overlap stops ideas falling through the cracks.",
  },
  {
    question: "What is the capital of France?",
    shouldAnswer: false,
    why: "Pure outside knowledge — a well-grounded RAG must decline this.",
  },
  {
    question: "How do I deploy this project to Kubernetes with autoscaling?",
    shouldAnswer: false,
    why: "On-topic sounding but not covered anywhere in the README.",
  },
];

const DECLINE_MARKERS = ["i don't know", "i do not know", "don't know based", "not contain", "no information"];

function declined(answer: string): boolean {
  const a = answer.toLowerCase();
  return DECLINE_MARKERS.some((m) => a.includes(m));
}

function judge(tc: TestCase, answer: string): { pass: boolean; note: string } {
  const didDecline = declined(answer);
  if (tc.shouldAnswer) {
    if (didDecline) return { pass: false, note: "declined a question it should have answered" };
    const hit = (tc.expectKeywords ?? []).some((k) => answer.toLowerCase().includes(k.toLowerCase()));
    return hit
      ? { pass: true, note: "answered and hit an expected keyword" }
      : { pass: false, note: `answered but missed expected keywords: ${(tc.expectKeywords ?? []).join(", ")}` };
  }
  // shouldAnswer === false: passing means correctly declining.
  return didDecline
    ? { pass: true, note: "correctly declined (no hallucination)" }
    : { pass: false, note: "HALLUCINATED — answered a question not covered by the document" };
}

async function runCase(tc: TestCase) {
  const retrieved = await retrieveAndRerank(tc.question);
  const hits = retrieved.ok ? retrieved.value : [];

  const answerResult = await chat([
    { role: "system", content: SYSTEM_PROMPT },
    {
      role: "user",
      content:
        `<context>\n${hits.map((h, i) => `<chunk id="${i + 1}">\n${h.content}\n</chunk>`).join("\n\n")}\n</context>\n\nQuestion: ${tc.question}`,
    },
  ]);

  const answer = answerResult.ok ? answerResult.value : `⚠️ LLM error: ${answerResult.error.message}`;
  const verdict = answerResult.ok ? judge(tc, answer) : { pass: false, note: "LLM call failed" };

  return { tc, hits, answer, verdict };
}

async function main() {
  console.log("\n=== RAG eval — does the answer actually use the context? ===\n");
  console.log("Heuristic pass/fail — read the answers, don't just trust the marks.\n");

  let passed = 0;
  for (const tc of CASES) {
    const { hits, answer, verdict } = await runCase(tc);
    const mark = verdict.pass ? "✅ PASS" : "❌ FAIL";
    if (verdict.pass) passed++;

    console.log("─".repeat(78));
    console.log(`${mark}  Q: ${tc.question}`);
    console.log(`        expectation: ${tc.shouldAnswer ? "should ANSWER" : "should DECLINE"} — ${tc.why}`);
    console.log(`        chunks:      ${hits.length ? hits.map((h) => chunkLabel(h.content)).join(" · ") : "(none)"}`);
    console.log(`        answer:      ${answer.replace(/\s+/g, " ").trim().slice(0, 240)}`);
    console.log(`        judgment:    ${verdict.note}`);
  }

  console.log("─".repeat(78));
  console.log(`\nScore: ${passed}/${CASES.length} passed.\n`);
}

main().catch((err) => {
  console.error("Eval error:", err);
  process.exit(1);
});
