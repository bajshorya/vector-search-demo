// Step 8 — Searching (Retrieve).
//
// A user query is NOT matched as text. Instead:
//     query text  ->  embed()  ->  query vector
// ...then Postgres compares that vector against every stored document vector
// and returns the closest ones (smallest distance = most similar meaning).
//
// Run with:  npm run search "how can I build a caching server?"
//
// This file also exports `search()` so ask.ts can reuse retrieval as the
// first half of the full RAG loop (retrieve -> prompt -> LLM -> answer).

import "dotenv/config"; // loads DATABASE_URL from .env
import { Client } from "pg";
import { embed } from "./embed.js"; // the SAME embedder used to index the documents
import { chunkLabel } from "./chunk.js"; // just for pretty console labels

export const TOP_K = 3; // how many similar chunks to return

export interface SearchHit {
  content: string;
  distance: number;
}

export async function search(query: string, topK: number = TOP_K): Promise<SearchHit[]> {
  const db = new Client({ connectionString: process.env.DATABASE_URL });
  await db.connect();

  // 1) Turn the query into a vector — same model/space as the stored documents.
  const queryVector = await embed(query);
  const queryLiteral = JSON.stringify(queryVector); // pgvector wants '[...]'

  // 2) Let Postgres do the comparison.
  //    `<=>` is pgvector's COSINE DISTANCE operator. It computes the distance
  //    between the query vector and each row's embedding. ORDER BY distance +
  //    LIMIT gives us the nearest neighbours — the top matches by meaning.
  const result = await db.query(
    `SELECT content,
            embedding <=> $1::vector AS distance
     FROM documents
     ORDER BY distance
     LIMIT $2`,
    [queryLiteral, topK]
  );

  await db.end();
  return result.rows as SearchHit[];
}

// Only run the CLI below when this file is executed directly (`npm run search`).
// When ask.ts imports `search`, this block is skipped — same pattern as embed.ts.
import { fileURLToPath } from "node:url";
const isDirectRun = process.argv[1] === fileURLToPath(import.meta.url);

if (isDirectRun) {
  // Take the query from the command line, or use a default.
  const query =
    process.argv.slice(2).join(" ") || "How can I build a caching server?";

  search(query)
    .then((rows) => {
      console.log(`\n🔎 Query: "${query}"\n`);
      console.log(`Top ${rows.length} most similar chunks:\n`);
      rows.forEach((row, i) => {
        const distance = Number(row.distance).toFixed(4);
        console.log(`${i + 1}. [distance ${distance}]  ${chunkLabel(row.content)}`);
        // Chunks are multi-line markdown, so indent the whole block.
        console.log(
          row.content
            .split("\n")
            .map((line) => `   ${line}`)
            .join("\n") + "\n",
        );
      });
    })
    .catch((err) => {
      console.error("Search error:", err);
      process.exit(1);
    });
}
