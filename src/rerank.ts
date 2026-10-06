// Step 12 — Re-ranking: closing the gap between "closest" and "most useful".
//
// Plain retrieval (search.ts) returns the chunks whose EMBEDDINGS are nearest
// the question's embedding. But "nearest by vector distance" and "actually
// answers the question" are not the same thing:
//
//   - A chunk can sit close in vector space because it repeats the question's
//     words ("How do I choose chunk size?" matching a heading "Choosing size")
//     without containing the answer.
//   - The single most useful chunk can rank 6th by raw distance, behind five
//     that are merely on-topic.
//
// The fix is a two-stage retrieval, the standard pattern in production RAG:
//
//     search(top 10 by distance)   →   LLM scores each 1–10   →   keep top 4
//     -------- cheap, fuzzy ------      ------ smarter, reads meaning ------
//
// Stage 1 casts a wider net cheaply. Stage 2 spends one small LLM call to judge
// relevance the way a reader would, then keeps only the best. The final prompt
// is built from those, so the generator sees fewer, better chunks.

import { search, type SearchHit } from "./search.js";
import { chat, type LLMError } from "./llm.js";
import { type Result, ok } from "./result.js";
import { chunkLabel } from "./chunk.js";

/** Cast a wide net first... */
export const INITIAL_K = 10;
/** ...then keep only this many after the LLM has judged relevance. */
export const FINAL_K = 4;

export interface RankedHit extends SearchHit {
  /** 1–10 relevance score from the re-ranker (absent if we fell back). */
  rerankScore?: number;
}

const SCORER_SYSTEM = `You are a search relevance judge. You are given a question and several numbered context chunks.
Score how well EACH chunk helps answer the question, from 1 (irrelevant) to 10 (directly answers it).
Judge usefulness for answering, not mere keyword overlap: a chunk that only repeats the question's words without answering scores low.
Respond with ONLY a JSON object of the form {"scores": [{"id": 1, "score": 8}, ...]} covering every chunk id. No prose.`;

function buildScoringPrompt(question: string, hits: SearchHit[]): string {
  const chunks = hits
    .map((hit, i) => `<chunk id="${i + 1}">\n${hit.content}\n</chunk>`)
    .join("\n\n");
  return `Question: ${question}\n\n${chunks}`;
} // explanation of this code : This function builds a prompt for the LLM to score the relevance of each search hit. It takes a question and an array of search hits, formats each hit with a unique ID, and returns a string that includes the question followed by the formatted chunks. The LLM will use this prompt to evaluate how well each chunk answers the question.
//what is search hit : A search hit is an individual result returned from a search query. In this context, it represents a piece of content (chunk) that has been retrieved based on its relevance to the user's question. Each search hit includes the content itself and a distance metric indicating how closely it matches the query in vector space.
// example: If a user asks "What is the capital of France?", a search hit might be a chunk of text that says "Paris is the capital city of France." The search hit would include this content and a distance score indicating how closely it relates to the question.

/**
 * Retrieve INITIAL_K by vector distance, then re-rank down to FINAL_K with one
 * LLM scoring call. Returns the kept hits (best first).
 *
 * Graceful degradation is deliberate: re-ranking is a QUALITY improvement, not a
 * correctness requirement, so if the scoring call fails or returns unparseable
 * JSON we fall back to the plain top-FINAL_K by distance rather than erroring
 * out. The pipeline still answers; it just skips the extra polish.
 */
export async function retrieveAndRerank(
  question: string,
  initialK: number = INITIAL_K,
  finalK: number = FINAL_K,
): Promise<Result<RankedHit[], LLMError>> {
  const candidates = await search(question, initialK);

  // Nothing to re-rank if we already have finalK or fewer.
  if (candidates.length <= finalK) return ok(candidates);

  const scored = await chat(
    [
      { role: "system", content: SCORER_SYSTEM },
      { role: "user", content: buildScoringPrompt(question, candidates) },
    ],
    // explanation of this chat call:
    // This chat call sends a request to the LLM to score the relevance of each candidate chunk. It includes a system message that sets the context for the LLM, instructing it to act as a search relevance judge, and a user message that contains the question and the formatted chunks. The options specify that the response should be deterministic (temperature: 0), formatted as JSON, and have a timeout of 60 seconds to allow for potentially longer processing time. The result of this call will be used to re-rank the candidates based on their relevance scores.

    // Scoring 10 chunks as JSON is a bigger job than a normal answer (and the
    // first call may include model load time), so allow more headroom by :-
    // 1) giving the model more time to respond, and 2) asking for deterministic output.
    { temperature: 0, format: "json", timeoutMs: 60_000 },
  );

  if (!scored.ok) {
    console.error(
      `   ⚠️  Re-ranker unavailable (${scored.error.message}) — falling back to vector order.`,
    );
    return ok(candidates.slice(0, finalK));
  }

  const scores = parseScores(scored.value, candidates.length);
  if (!scores) {
    console.error(
      "   ⚠️  Re-ranker returned unparseable JSON — falling back to vector order.",
    );
    return ok(candidates.slice(0, finalK));
  }

  // Attach scores, sort by them (desc), keep the best finalK.
  const ranked: RankedHit[] = candidates
    .map((hit, i) => ({ ...hit, rerankScore: scores[i] ?? 0 }))
    .sort((a, b) => (b.rerankScore ?? 0) - (a.rerankScore ?? 0))
    .slice(0, finalK);

  return ok(ranked);
}
// line by line explanation of the above code :
/**
 * This function retrieves a set of search hits based on vector distance and then re-ranks them using an LLM scoring call. It takes a question, an optional initialK (number of hits to retrieve), and an optional finalK (number of hits to keep after re-ranking). The function returns a Result type containing either the ranked hits or an LLMError.
 *
 * 1. It first retrieves candidates using the search function based on the question and initialK.
 * 2. If the number of candidates is less than or equal to finalK, it returns them as they are.
 * 3. It then constructs a scoring prompt and makes a chat call to the LLM to score the relevance of each candidate.
 * 4. If the scoring call fails or returns unparseable JSON, it logs an error and falls back to returning the top finalK candidates by vector distance.
 * 5. If scoring is successful, it parses the scores, attaches them to the candidates, sorts them by score in descending order, and keeps only the top finalK hits.
 * 6. Finally, it returns the ranked hits wrapped in an ok Result type.
 *
 * example: If a user asks "What is the capital of France?" and the initial search retrieves 10 chunks, the function will score each chunk for relevance. If the scoring is successful, it will return the top 4 chunks that are most relevant to the question, based on the LLM's scoring. If scoring fails, it will simply return the top 4 chunks based on vector distance.
 */

/**
 * Parse the scorer's JSON into an array indexed by (id - 1). Returns null if the
 * shape is wrong, so the caller can fall back. Missing ids default to 0 later.
 *
 * example output : If the LLM returns a JSON response like `{"scores": [{"id": 1, "score": 8}, {"id": 2, "score": 5}]}`, this function will parse it and return an array like `[8, 5, 0, 0, ...]` where the index corresponds to the chunk ID minus one. If the JSON is malformed or does not contain the expected structure, it will return null to indicate that parsing failed.
 */
function parseScores(raw: string, count: number): number[] | null {
  try {
    const parsed = JSON.parse(raw) as {
      scores?: Array<{ id?: number; score?: number }>;
    };
    if (!Array.isArray(parsed.scores)) return null;

    const byIndex: number[] = new Array(count).fill(0);
    for (const entry of parsed.scores) {
      const idx = Number(entry.id) - 1;
      if (idx >= 0 && idx < count && Number.isFinite(entry.score)) {
        byIndex[idx] = Number(entry.score);
      }
    }
    return byIndex;
  } catch {
    return null;
  }
}

// CLI: `npm run rerank "your question"` — shows the before/after so you can SEE
// re-ranking move chunks around (vector rank vs. final rank + score).
import { fileURLToPath } from "node:url";
const isDirectRun = process.argv[1] === fileURLToPath(import.meta.url);
// explanation of the above code : This section of the code checks if the script is being run directly from the command line. It uses the `fileURLToPath` function to convert the module URL to a file path and compares it with the second argument in `process.argv`, which represents the script being executed. If they match, it indicates that the script is being run directly, and the subsequent code block will execute to perform a search and re-ranking based on a provided question or a default question.
if (isDirectRun) {
  const question =
    process.argv.slice(2).join(" ") ||
    "How do I choose the chunk size and overlap?";

  (async () => {
    const initial = await search(question, INITIAL_K);
    console.log(`\n🔎 "${question}"\n`);
    console.log(`Stage 1 — top ${initial.length} by vector distance:`);
    initial.forEach((h, i) =>
      console.log(
        `   #${i + 1}  d=${Number(h.distance).toFixed(4)}  ${chunkLabel(h.content)}`,
      ),
    );

    const result = await retrieveAndRerank(question);
    if (!result.ok) {
      console.error("Re-rank error:", result.error);
      process.exit(1);
    }

    console.log(
      `\nStage 2 — kept top ${result.value.length} after LLM re-ranking:`,
    );
    result.value.forEach((h, i) =>
      console.log(
        `   #${i + 1}  score=${h.rerankScore ?? "—"}/10  d=${Number(h.distance).toFixed(4)}  ${chunkLabel(h.content)}`,
      ),
    );
    console.log();
  })().catch((err) => {
    console.error("Re-rank error:", err);
    process.exit(1);
  });
}
//explanation of the CLI code : This section of the code allows the script to be run directly from the command line. It checks if the script is being executed directly and, if so, retrieves a question from the command line arguments or uses a default question. It then performs an initial search to retrieve candidates based on vector distance and logs the results. After that, it calls the `retrieveAndRerank` function to re-rank the candidates using the LLM scoring call. The final ranked results are logged, showing both the score and distance for each chunk. If any errors occur during this process, they are logged, and the script exits with an error code.