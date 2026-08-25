// Step 10 — Generation: the second half of RAG.
//
// search.ts stops at "here are the 3 most similar chunks". This file finishes
// the pipeline:
//
//     question  ->  search()   ->  top chunks          (Retrieve — already built)
//               ->  buildPrompt()                      (Prompt + Context)
//               ->  Ollama (llama3.2, local)           (LLM)
//               ->  grounded answer + cited sources    (Answer)
//
// Like the embeddings, the LLM runs entirely on this machine: Ollama serves
// llama3.2 at http://localhost:11434. No API key, no cost, works offline.
//
// The grounding rule is what makes this RAG and not just a chatbot: the model
// is instructed to answer ONLY from the retrieved chunks, and to say it
// doesn't know when they don't contain the answer. The knowledge lives in
// Postgres; the model just reads and phrases it.
//
// Run with:  npm run ask "how do I choose chunk size and overlap?"

import "dotenv/config"; // loads DATABASE_URL from .env
import { search, TOP_K, type SearchHit } from "./search.js"; // reuse retrieval — same embedder, same table
import { chunkLabel } from "./chunk.js";

const OLLAMA_URL = "http://localhost:11434/api/chat";
const MODEL = "llama3.2"; // 3B model, ~2GB, already pulled via `ollama pull llama3.2`

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
function buildPrompt(question: string, hits: SearchHit[]): string {
  const context = hits
    .map((hit, i) => `<chunk id="${i + 1}">\n${hit.content}\n</chunk>`)
    .join("\n\n");

  return `<context>\n${context}\n</context>\n\nQuestion: ${question}`;
}

/**
 * Call the local Ollama server and stream the answer to stdout as it's
 * generated. Ollama's /api/chat returns NDJSON: one JSON object per line,
 * each carrying the next fragment of the reply in .message.content.
 */
async function generate(question: string, hits: SearchHit[]): Promise<void> {
  const response = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      stream: true,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: buildPrompt(question, hits) },
      ],
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Ollama returned ${response.status}: ${await response.text()}`);
  }

  // Read the NDJSON stream line by line, printing each text fragment.
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk as Uint8Array, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? ""; // keep any incomplete trailing line
    for (const line of lines) {
      if (!line.trim()) continue;
      const part = JSON.parse(line);
      if (part.message?.content) process.stdout.write(part.message.content);
    }
  }
  process.stdout.write("\n");
}

async function ask(question: string) {
  // ---- Retrieve (built in step 8) ----
  const hits = await search(question, TOP_K);

  console.log(`\n🔎 Question: "${question}"`);
  console.log(`📚 Retrieved ${hits.length} chunks:`);
  hits.forEach((hit, i) => {
    console.log(`   [${i + 1}] (distance ${Number(hit.distance).toFixed(4)}) ${chunkLabel(hit.content)}`);
  });
  console.log(`\n🤖 ${MODEL} (local via Ollama):\n`);

  // ---- Prompt + Context -> LLM -> Answer ----
  await generate(question, hits);

  // Show which chunks fed the answer, so the grounding is inspectable.
  console.log(`\n— sources: ${hits.map((h, i) => `[${i + 1}] ${chunkLabel(h.content)}`).join("  ·  ")}`);
}

const question =
  process.argv.slice(2).join(" ") || "How do I choose the chunk size and overlap?";

ask(question).catch((err) => {
  if (err instanceof TypeError && String(err.cause ?? "").includes("ECONNREFUSED")) {
    console.error(
      "\n❌ Can't reach Ollama at localhost:11434.\n" +
        "   Start it with:  ollama serve\n" +
        "   (and make sure the model is pulled:  ollama pull llama3.2)",
    );
  } else {
    console.error("Ask error:", err);
  }
  process.exit(1);
});
