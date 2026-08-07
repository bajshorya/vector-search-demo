// Step 6 — Generate Embeddings.
//
// For every chunk:
//     chunk text  ->  Embedding API  ->  384 numbers (a vector)
// ...and we store that vector alongside the text in Postgres.
//
// The corpus is no longer hand-written (src/snippets.ts). It is now produced by
// slicing a *real* document:
//
//     README.md  ->  chunkText()  ->  chunks  ->  embed()  ->  pgvector
//
// Only the first arrow is new. Embedding and storing are unchanged.
//
// Run with:  npm run embed

import "dotenv/config"; // loads DATABASE_URL from .env
import { readFile } from "node:fs/promises";
import { pipeline } from "@xenova/transformers"; // runs the embedding model locally — no API key, no cost
import { Client } from "pg"; // npm install pg, pg will be used to connect to the Postgres database
import { chunkText, chunkLabel, CHUNK_SIZE, CHUNK_OVERLAP } from "./chunk.js";

// The document we want to make searchable. Swap this path for any text/markdown
// file — nothing else in the pipeline needs to change.
const SOURCE_DOCUMENT = new URL("../README.md", import.meta.url);

// The embedding model. all-MiniLM-L6-v2 runs entirely on your machine and
// outputs 384 numbers per input, which matches the vector(384) column in the
// documents table. First run downloads the model (~90MB) and caches it.
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

// What is 1536 ? => The number 1536 refers to the dimensionality of the embedding vector produced by the "text-embedding-3-small" model. When you input a piece of text into this model, it generates a vector representation of that text in a 1536-dimensional space. Each of the 1536 numbers in the vector captures different semantic features of the input text, allowing for effective comparison and retrieval of similar texts based on their embeddings.

// Load the model once (lazily). `extractor` is a function we call to turn
// text into a vector. Loading is async, so we keep the promise and await it.
const extractorPromise = pipeline("feature-extraction", EMBEDDING_MODEL);

/**
 * Turn one piece of text into its embedding: an array of 1536 numbers.
 * We don't care what the numbers mean — we just store them.
 */

/** example embedding :- 
 *  [
  -0.0023456789,
  0.123456789,
  -0.987654321,
  ...
  0.4567890123
] of the text "Mini Redis. An in-memory key-value database supporting TTL, persistence, and pub/sub. Built in TypeScript to explore how real caches handle expiry and durability.",
  it is not possible to interpret the meaning of each individual number in the embedding vector. The embedding is a high-dimensional representation of the input text, capturing its semantic meaning and relationships to other texts in a way that allows for similarity comparisons and retrieval tasks. The specific values in the embedding vector are learned by the model during training and do not have an inherent human-readable interpretation.
 */

//this function takes a string of text as input and returns a Promise that resolves to an array of numbers representing the embedding of that text. The embedding is generated using the OpenAI Embeddings API, which transforms the input text into a high-dimensional vector representation. This vector captures the semantic meaning of the text, allowing for tasks such as similarity comparisons and information retrieval. The function uses the specified embedding model (text-embedding-3-small) to produce a 1536-dimensional embedding for each input text.

export async function embed(text: string): Promise<number[]> {
  const extractor = await extractorPromise;
  // pooling: "mean" averages the per-word vectors into one sentence vector.
  // normalize: true scales it to unit length, which makes cosine similarity clean.
  const output = await extractor(text, { pooling: "mean", normalize: true });
  // output.data is a typed array (Float32Array); convert to a plain number[].
  return Array.from(output.data as Float32Array);
}
/**
 * pgvector expects a vector literal formatted like a JSON array: '[0.1,0.2,...]'.
 * JSON.stringify on a number[] produces exactly that string.
 */

/** toVectorLiteral function will convert an array of numbers into a JSON string formatted as a vector literal.
 * example :-
 * const embedding = [0.1, 0.2, 0.3, 0.4];
 * const vectorLiteral = toVectorLiteral(embedding);
 * console.log(vectorLiteral); // Output: "[0.1,0.2,0.3,0.4]"
 *
 * This function is useful for preparing the embedding data to be stored in a database that expects vector literals in JSON format.
 *
 */
function toVectorLiteral(embedding: number[]): string {
  return JSON.stringify(embedding);
}
async function main() {
  const db = new Client({ connectionString: process.env.DATABASE_URL });
  await db.connect();

  // Start clean so re-running doesn't create duplicates.
  await db.query("TRUNCATE documents RESTART IDENTITY");

  // --- THE NEW STEP: one document in, many chunks out. ---
  const document = await readFile(SOURCE_DOCUMENT, "utf8");
  const chunks = chunkText(document);

  console.log(
    `Read ${document.length} characters from README.md`,
  );
  console.log(
    `Split into ${chunks.length} chunks (target ${CHUNK_SIZE} chars, ${CHUNK_OVERLAP} overlap)\n`,
  );
  console.log(`Embedding ${chunks.length} chunks with ${EMBEDDING_MODEL}...\n`);

  // --- UNCHANGED FROM YESTERDAY: a chunk is just a string, like a snippet. ---
  for (const content of chunks) {
    const embedding = await embed(content); // text -> 384 numbers

    await db.query(
      "INSERT INTO documents (content, embedding) VALUES ($1, $2)",
      [content, toVectorLiteral(embedding)],
    );

    // Show the first few numbers so you can SEE the vector, like in the diagram.
    const preview = embedding
      .slice(0, 3)
      .map((n) => n.toFixed(3))
      .join(", ");
    console.log(
      `✅ ${chunkLabel(content).padEnd(42)} ${String(content.length).padStart(4)} chars -> [${preview}, ...] (${embedding.length} dims)`,
    );
  }

  await db.end();
  console.log(
    "\nDone. Every chunk of README.md now has a vector in the documents table.",
  );
}

// Only run the indexing (embed + insert all snippets) when this file is
// executed directly — e.g. `npm run embed`. When search.ts imports `embed`
// from here, this block is skipped so we don't re-index on every search.
import { fileURLToPath } from "node:url";
const isDirectRun = process.argv[1] === fileURLToPath(import.meta.url);

if (isDirectRun) {
  main().catch((err) => {
    console.error("Error while embedding:", err);
    process.exit(1);
  });
}
