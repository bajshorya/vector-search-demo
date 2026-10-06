// Step 10 — Generation: the second half of RAG.  (Hardened in steps 11–12.)
//
// search.ts stops at "here are the most similar chunks". This file finishes
// the pipeline:
//
//     question  ->  retrieveAndRerank()  ->  best chunks     (Retrieve + Re-rank)
//               ->  buildPrompt()                            (Prompt + Context)
//               ->  chat()  (Ollama llama3.2, local)         (LLM — timed out + retried)
//               ->  grounded answer + cited sources          (Answer)
//
// What changed when we hardened it:
//   - Retrieval now pulls a wider set and RE-RANKS it (rerank.ts) before
//     building the prompt, so the model sees fewer, more relevant chunks.
//   - The LLM call goes through llm.ts: it has a timeout, it retries transient
//     failures with backoff, and it returns a Result instead of throwing — so a
//     hung or failing model degrades to a clear message, not a stack trace.
//
// The grounding rule is unchanged and is what makes this RAG and not just a
// chatbot: the model answers ONLY from the retrieved chunks and says it doesn't
// know when they don't contain the answer.
//
// Run with:  npm run ask "how do I choose chunk size and overlap?"

import "dotenv/config"; // loads DATABASE_URL from .env
import { retrieveAndRerank, type RankedHit } from "./rerank.js";
import { chat } from "./llm.js";
import { chunkLabel } from "./chunk.js";

const MODEL = "llama3.2";

// The system prompt carries the grounding rule. It never changes per question.
const SYSTEM_PROMPT = `You are a helpful assistant answering questions about a codebase, using ONLY the context chunks provided in the user's message.

Rules:
- Base every claim on the context. Do not use outside knowledge to fill gaps.
- If the context does not contain the answer, say "I don't know based on the indexed documents" — do not guess.
- When you use a chunk, cite it inline like [1] or [2] matching the chunk ids.
- Be concise: a few sentences or a short list is usually enough.`;

/**
 * Prompt + Context: paste the retrieved chunks into the user message, each
 * wrapped in a tagged block so the model can tell chunk boundaries apart and
 * cite them by id. The question goes last.
 */
function buildPrompt(question: string, hits: RankedHit[]): string {
  const context = hits
    .map((hit, i) => `<chunk id="${i + 1}">\n${hit.content}\n</chunk>`)
    .join("\n\n");

  return `<context>\n${context}\n</context>\n\nQuestion: ${question}`;
}

async function ask(question: string) {
  // ---- Retrieve + Re-rank (steps 8 & 12) ----
  const retrieved = await retrieveAndRerank(question);
  if (!retrieved.ok) {
    // Re-rank degrades gracefully on its own, so this is only reached on a hard
    // failure. Report it and stop rather than guessing.
    console.error(`\n❌ Retrieval failed: ${retrieved.error.message}`);
    process.exit(1);
  }
  const hits = retrieved.value;

  console.log(`\n🔎 Question: "${question}"`);
  console.log(`📚 Using ${hits.length} chunks (after re-ranking):`);
  hits.forEach((hit, i) => {
    const score = hit.rerankScore != null ? `score ${hit.rerankScore}/10, ` : "";
    console.log(`   [${i + 1}] (${score}distance ${Number(hit.distance).toFixed(4)}) ${chunkLabel(hit.content)}`);
  });
  console.log(`\n🤖 ${MODEL} (local via Ollama):\n`);

  // ---- Prompt + Context -> LLM -> Answer (timed out + retried) ----
  const answer = await chat([
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: buildPrompt(question, hits) },
  ]);

  if (!answer.ok) {
    // The error is already classified (operational vs programmer) by llm.ts.
    const { kind, message } = answer.error;
    console.error(`❌ Could not generate an answer (${kind} error): ${message}`);
    if (kind === "operational") {
      console.error("   This looks transient — check that `ollama serve` is running and try again.");
    } else {
      console.error("   This looks like a bug in the request — retrying won't help; check the code.");
    }
    process.exit(1);
  }

  process.stdout.write(answer.value + "\n");

  // Show which chunks fed the answer, so the grounding is inspectable.
  console.log(`\n— sources: ${hits.map((h, i) => `[${i + 1}] ${chunkLabel(h.content)}`).join("  ·  ")}`);
}

const question =
  process.argv.slice(2).join(" ") || "How do I choose the chunk size and overlap?";

ask(question).catch((err) => {
  console.error("Ask error:", err);
  process.exit(1);
});
