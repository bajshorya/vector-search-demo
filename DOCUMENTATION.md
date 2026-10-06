# Vector Search Demo — Complete Technical Documentation

A full, file-by-file explanation of this project: a **Retrieval-Augmented Generation (RAG)** pipeline built from scratch in TypeScript, running **entirely locally** — local embeddings, local database, local LLM. No API keys, no cost, works offline.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [The complete pipeline](#2-the-complete-pipeline)
3. [Repository layout](#3-repository-layout)
4. [Core concepts](#4-core-concepts)
5. [The database](#5-the-database)
6. [File-by-file walkthrough](#6-file-by-file-walkthrough)
   - [6.1 `src/chunk.ts` — cutting a document into pieces](#61-srcchunkts--cutting-a-document-into-pieces)
   - [6.2 `src/embed.ts` — text → vectors → Postgres](#62-srcembedts--text--vectors--postgres)
   - [6.3 `src/search.ts` — similarity search (Retrieve)](#63-srcsearchts--similarity-search-retrieve)
   - [6.4 `src/result.ts` — errors as values](#64-srcresultts--errors-as-values)
   - [6.5 `src/llm.ts` — the hardened LLM client](#65-srcllmts--the-hardened-llm-client)
   - [6.6 `src/rerank.ts` — two-stage retrieval (Re-rank)](#66-srcrerankts--two-stage-retrieval-re-rank)
   - [6.7 `src/ask.ts` — generation (Prompt + Context → LLM → Answer)](#67-srcaskts--generation-prompt--context--llm--answer)
   - [6.8 `src/bookmarks.ts` — the mock bookmarks store](#68-srcbookmarksts--the-mock-bookmarks-store)
   - [6.9 `src/tool-ask.ts` — tool calling (the agent loop)](#69-srctool-askts--tool-calling-the-agent-loop)
   - [6.10 `src/tool-ask.test.ts` — testing the tool layer](#610-srctool-asktestts--testing-the-tool-layer)
   - [6.11 `src/eval.ts` — a lightweight eval](#611-srcevalts--a-lightweight-eval)
   - [6.12 `src/snippets.ts` — the legacy corpus](#612-srcsnippetsts--the-legacy-corpus)
   - [6.13 Configuration files](#613-configuration-files)
7. [End-to-end trace of one question](#7-end-to-end-trace-of-one-question)
8. [Design decisions and invariants](#8-design-decisions-and-invariants)
9. [Running it](#9-running-it)
10. [Troubleshooting](#10-troubleshooting)
11. [Extending the project](#11-extending-the-project)

---

## 1. What this project is

This project answers natural-language questions about a document (the project's own `README.md`) by implementing every stage of a RAG pipeline by hand:

```
Document → Chunk → Embedding → pgvector → Retrieve → Re-rank → Prompt + Context → LLM → Answer
```

Ask it *"how do I choose the chunk size?"* and it will:

1. Convert your question into a 384-number vector,
2. Find the 10 chunks of the README whose vectors are closest in meaning,
3. **Re-rank** them with one LLM scoring call and keep the 4 most useful,
4. Hand those chunks to a local LLM with strict instructions to answer **only** from them,
5. Return a grounded, cited answer — behind a timeout and automatic retries.

The point is educational: every stage that frameworks like LangChain hide behind abstractions is written out here in plain TypeScript, small enough to read in one sitting (~900 lines across seven core files, much of it teaching comments). The pipeline has also been *hardened* — a two-stage retrieval (re-rank), a resilient LLM client (timeout, retry, typed errors), and a lightweight eval — so it reads like production-minded code, not just a demo.

**Everything runs on your machine:**

| Stage | Runs where | Tool |
|---|---|---|
| Embeddings | Locally (CPU) | `all-MiniLM-L6-v2` via Transformers.js |
| Vector storage & search | Locally | PostgreSQL + pgvector |
| Answer generation | Locally | `llama3.2` (3B) via Ollama |

---

## 2. The complete pipeline

Two phases. **Indexing** runs once (or whenever the document changes). **Querying** runs on every question.

```
INDEXING  (offline — `npm run embed`)
─────────────────────────────────────
  README.md                     one ~30,000-character markdown file
      │
      ▼
  chunkText()                   src/chunk.ts — cut at paragraph boundaries,
      │                         never inside code fences, ~900 chars each,
      │                         150-char overlap between neighbours
      ▼
  ~40 chunks                    each chunk is just a string
      │
      ▼
  embed()                       src/embed.ts — all-MiniLM-L6-v2, runs locally
      │
      ▼
  40 × vector(384)              one 384-number vector per chunk
      │
      ▼
  INSERT INTO documents         PostgreSQL + pgvector = the knowledge base


QUERYING  (online — `npm run ask "..."`)
────────────────────────────────────────
  question (plain English)
      │
      ▼
  embed()                       the SAME function, model, and vector space
      │
      ▼
  query vector(384)
      │
      ▼
  SELECT ... ORDER BY           src/search.ts — pgvector's <=> operator
  embedding <=> $1 LIMIT 10     computes cosine distance to every stored row
      │
      ▼
  top 10 candidates             a wide, cheap net (INITIAL_K = 10)
      │
      ▼
  retrieveAndRerank()           src/rerank.ts — one LLM scoring call rates each
      │                         chunk 1–10 for usefulness; keep the best 4
      ▼                         (FINAL_K = 4). Falls back to vector order on failure.
  top 4 chunks                  the "Retrieve + Re-rank" step ends here
      │
      ▼
  buildPrompt()                 src/ask.ts — chunks wrapped in <chunk id="n">
      │                         tags + grounding rules + the question
      ▼
  chat()                        src/llm.ts — Ollama /api/chat (llama3.2) behind a
      │                         timeout + retry-with-backoff; returns a Result
      ▼
  grounded answer               cited [1] [2] [3], or "I don't know" if the
                                chunks lack the answer
```

The critical property of this diagram: **the two `embed()` boxes are the same function.** Query vectors and document vectors must live in the same 384-dimensional space, produced by the same model, or comparing them is meaningless. This is why `search.ts` imports `embed` from `embed.ts` instead of having its own copy.

---

## 3. Repository layout

```
vector-search-demo/
├── src/
│   ├── chunk.ts       # Pure string logic: document → overlapping chunks. No dependencies.
│   ├── embed.ts       # INDEXING entry point (npm run embed). Exports embed().
│   ├── search.ts      # RETRIEVAL entry point (npm run search). Exports search().
│   ├── result.ts      # Generic Result<T, E> type — errors as values, not exceptions.
│   ├── llm.ts         # Hardened Ollama client: timeout, retry+backoff, typed errors. Exports chat() + chatWithTools().
│   ├── rerank.ts      # RE-RANK entry point (npm run rerank). Exports retrieveAndRerank().
│   ├── ask.ts         # FULL RAG entry point (npm run ask). Retrieve + re-rank + generate.
│   ├── bookmarks.ts   # Mock bookmarks store — the data source behind the tools. Exports get/add/countBookmarks().
│   ├── tool-ask.ts    # TOOL-CALLING entry point (npm run tools). The Zod-validated agent loop.
│   ├── tool-ask.test.ts # Tests for the tool layer (npm test, node:test).
│   ├── eval.ts        # EVAL entry point (npm run eval). 6-question pass/fail harness.
│   └── snippets.ts    # Legacy hand-written corpus. Unused; kept for comparison.
├── .env               # DATABASE_URL. Git-ignored.
├── .gitignore         # node_modules/, .env, .DS_Store
├── package.json       # Dependencies + the npm scripts (embed/search/rerank/ask/tools/eval/test)
├── tsconfig.json      # Strict TypeScript, ESM, no emit (tsx runs TS directly)
├── README.md          # Project README — and also the document being indexed
├── HARDENING.md       # Why/how the pipeline was hardened (re-rank, llm, eval)
└── DOCUMENTATION.md   # This file
```

Dependency graph between source files (arrows mean "imports from"):

```
ask.ts ──► rerank.ts ──► search.ts ──► embed.ts ──► chunk.ts
   │           │              │                        ▲
   │           └──► llm.ts ──► result.ts               │
   └───────────────────────────────────────────────────┘  (chunkLabel, console output)

eval.ts ──► rerank.ts, llm.ts     (runs the real pipeline over known questions)

tool-ask.ts ──► llm.ts (chatWithTools), bookmarks.ts, zod   (tool-calling loop; independent of RAG)
tool-ask.test.ts ──► tool-ask.ts, bookmarks.ts              (node:test)
```

`chunk.ts` and `result.ts` sit at the bottom with zero project imports; `ask.ts` sits at the top and touches everything. `llm.ts` depends only on `result.ts`, so the hardened LLM client is reusable in isolation. The tool-calling feature (`tool-ask.ts`) hangs off `llm.ts` and `bookmarks.ts` only — it shares the hardened transport with RAG but is otherwise a separate branch, so neither can break the other.

---

## 4. Core concepts

### 4.1 Embeddings: meaning as coordinates

An **embedding model** maps a piece of text to a fixed-length list of numbers — here, 384 of them. The model is trained so that *texts with similar meaning land near each other* in that 384-dimensional space:

```
"How do I build a cache?"        →  [0.021, -0.043, 0.011, ...]   ┐ close
"An in-memory key-value store"   →  [0.019, -0.040, 0.014, ...]   ┘ together

"The capital of France"          →  [-0.310, 0.220, -0.150, ...]  far away
```

No individual number means anything by itself. Only *distances between vectors* carry information. That single property converts "search by meaning" into geometry: find the stored vectors nearest to the query vector.

### 4.2 Cosine distance

pgvector's `<=>` operator computes **cosine distance** = `1 − cosine similarity`, i.e. how far apart two vectors *point*, ignoring their lengths:

- `0.0` — identical direction (identical meaning)
- `~0.3–0.6` — related meaning (typical for a good match here)
- `~0.8+` — unrelated
- `2.0` — opposite direction (theoretical maximum)

Because `embed()` normalizes every vector to unit length (`normalize: true`), all vectors lie on the unit sphere and cosine distance behaves cleanly and consistently.

### 4.3 Why chunking is necessary

Feeding a whole 30,000-character document to the embedder fails three ways:

1. **Input limit.** `all-MiniLM-L6-v2` reads only the first ~256 word-pieces and silently discards the rest — most of the document would never be embedded at all.
2. **Blurry averages.** Even if it fit, one vector for a whole document is an average of every topic in it. A question about installing pgvector would match the entire README weakly instead of the install section strongly.
3. **Retrieval granularity.** Search returns whole rows. Small rows give the LLM precise, relevant context; huge rows drown the answer in noise.

So the document is cut into ~900-character pieces, each embedded independently. The chunk becomes the atomic unit of the whole system.

### 4.4 What makes it RAG, not just a chatbot

`llama3.2` already "knows" many things from its training. RAG deliberately doesn't use that knowledge. The system prompt in `ask.ts` orders the model to:

- answer **only** from the supplied chunks,
- cite which chunk each claim came from,
- say **"I don't know based on the indexed documents"** when the chunks lack the answer.

This is called **grounding**. It's verifiable: asked *"what is the capital of France?"* — something the model certainly knows — the pipeline answers *"I don't know based on the indexed documents"*, because the README never mentions it. The knowledge lives in Postgres; the LLM only reads and phrases it.

---

## 5. The database

One table, living in a local PostgreSQL database called `vector_demo`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: adds the vector type + operators

CREATE TABLE documents (
  id        SERIAL PRIMARY KEY,          -- auto-incrementing row id
  content   TEXT NOT NULL,               -- the chunk's raw text
  embedding VECTOR(384)                  -- the chunk's 384-number embedding
);
```

Why each piece is what it is:

- **`VECTOR(384)`** — a pgvector column type: a fixed-length float array with distance operators. The `384` must equal the embedding model's output size. Switch models → change this number → re-run `npm run embed`.
- **`content` stored alongside `embedding`** — search happens on vectors, but the *result* you need is the text. Storing both in one row means one query returns everything.
- **No vector index (HNSW/IVFFlat)** — with 40 rows, a brute-force scan comparing the query against every row is instant. Indexes matter at tens of thousands of rows; adding one here would only obscure the concept.
- **Operators pgvector provides:** `<->` L2/Euclidean distance, `<=>` cosine distance (used here), `<#>` negative inner product.

---

## 6. File-by-file walkthrough

### 6.1 `src/chunk.ts` — cutting a document into pieces

**Zero dependencies — pure string manipulation.** ~165 lines. Exports two constants, one interface, and two functions.

#### The constants

```ts
export const CHUNK_SIZE = 900;     // target chunk length, in characters
export const CHUNK_OVERLAP = 150;  // how much of each chunk's tail repeats at the next chunk's head
```

- **900 characters ≈ 200–230 tokens**, safely inside the embedder's ~256-token reading window with headroom for overlap. The guiding instinct: *a chunk should be the smallest piece of text that still answers a question on its own.*
- **150-character overlap (~17%)** exists because a sentence that straddles a chunk boundary would otherwise be cut in half, leaving *neither* chunk carrying its full meaning. Repeating a little context across the seam is cheap insurance.

#### `chunkText(text, options?)` — the public entry point

```ts
export function chunkText(text: string, options: ChunkOptions = {}): string[] {
  const size = options.size ?? CHUNK_SIZE;
  const overlap = options.overlap ?? CHUNK_OVERLAP;

  const blocks = splitIntoBlocks(text);
  const chunks: string[] = [];

  let current: string[] = [];      // the chunk being filled right now
  let currentLength = 0;

  const flush = () => {
    if (current.length === 0) return;
    chunks.push(current.join("\n\n"));
    current = [];
    currentLength = 0;
  };

  for (const block of blocks) {
    for (const piece of block.length > size ? hardSplit(block, size) : [block]) {
      if (currentLength > 0 && currentLength + piece.length > size) flush();
      current.push(piece);
      currentLength += piece.length + 2;   // +2 for the "\n\n" join
    }
  }
  flush();

  return overlap > 0 ? addOverlap(chunks, overlap) : chunks;
}
```

The algorithm is a **greedy bin-packer over paragraphs**:

1. Split the document into "blocks" (paragraphs / whole code fences) — see `splitIntoBlocks`.
2. Keep appending blocks to the current chunk until adding the next one would exceed `size`; then close (`flush`) the chunk and start a new one.
3. A single block *bigger* than `size` on its own (a long code fence, say) can't be packed — `hardSplit` cuts it down first, so the packing loop only ever sees pieces that fit.
4. Finally, `addOverlap` stitches context across the seams.

Two subtleties worth noticing:

- **`currentLength > 0 &&`** in the overflow test: without it, a first piece larger than `size` would flush an *empty* chunk. A chunk always receives at least one piece.
- Because only **whole blocks** are appended, a chunk can never end mid-sentence — the property that makes each chunk embed as a complete thought.

#### `splitIntoBlocks(text)` — paragraph splitting that respects code fences

```ts
function splitIntoBlocks(text: string): string[] {
  const blocks: string[] = [];
  let buffer: string[] = [];
  let inFence = false;

  const flush = () => {
    const block = buffer.join("\n").trim();
    if (block) blocks.push(block);
    buffer = [];
  };

  for (const line of text.split("\n")) {
    if (line.trimStart().startsWith("```")) inFence = !inFence;

    if (line.trim() === "" && !inFence) {
      flush();
    } else {
      buffer.push(line);
    }
  }
  flush();

  return blocks;
}
```

The natural boundary in prose is the **blank line** — that's where a human would cut. But a blank line *inside* a fenced code block (` ``` ... ``` `) is not a paragraph break; cutting there would leave dangling half-programs in separate chunks. So the function walks line by line, toggling `inFence` every time it sees a fence marker, and ignores blank lines while inside a fence. Result: prose splits at paragraphs, code blocks travel as single unsplittable units.

#### `hardSplit(block, size)` — the last resort

```ts
function hardSplit(block: string, size: number): string[] {
  const pieces: string[] = [];
  let buffer: string[] = [];
  let length = 0;

  for (const line of block.split("\n")) {
    if (length > 0 && length + line.length > size) {
      pieces.push(buffer.join("\n"));
      buffer = [];
      length = 0;
    }
    buffer.push(line);
    length += line.length + 1;
  }
  if (buffer.length > 0) pieces.push(buffer.join("\n"));

  return pieces;
}
```

Called only when a single block already exceeds the target on its own. It applies the same greedy packing one level down — at **line** granularity — so even in the worst case no line is ever split in half.

#### `addOverlap(chunks, overlap)` — context across the seams

```ts
function addOverlap(chunks: string[], overlap: number): string[] {
  return chunks.map((chunk, i) => {
    if (i === 0) return chunk;                    // nothing precedes the first chunk

    const previous = chunks[i - 1];
    let tail = previous.slice(-overlap);          // last 150 chars of the previous chunk

    const breakAt = tail.indexOf("\n");           // snap forward to a line boundary
    if (breakAt !== -1) tail = tail.slice(breakAt + 1);

    return `${tail.trim()}\n\n${chunk}`;          // prepend tail to this chunk
  });
}
```

Each chunk (except the first) gets the tail of its predecessor prepended. The tail is trimmed **forward to the next line break** so the repeated text starts on a whole line rather than mid-sentence — the same "never embed half a thought" principle applied to the overlap itself.

#### `chunkLabel(chunk)` — cosmetic only

Returns a short human-readable label for console output: the first markdown heading in the chunk if there is one, else the first line, truncated to 40 characters. Used by all three entry points for pretty logging; plays no role in the pipeline itself.

---

### 6.2 `src/embed.ts` — text → vectors → Postgres

The **indexing** entry point (`npm run embed`) and home of the single most important export in the project: `embed()`.

#### Configuration

```ts
const SOURCE_DOCUMENT = new URL("../README.md", import.meta.url);
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
```

- `SOURCE_DOCUMENT` resolves relative to *this file's* location (not the working directory), so the script works no matter where you run it from. Swap this path for any text/markdown file — nothing else changes.
- `all-MiniLM-L6-v2` is a small, well-proven sentence-embedding model: 384 output dimensions, ~256 token input window, ~90 MB download on first run (then cached in `~/.cache`), runs on CPU via Transformers.js (`@xenova/transformers`). No API key, no network after the first download.

#### Model loading — once, lazily

```ts
const extractorPromise = pipeline("feature-extraction", EMBEDDING_MODEL);
```

`pipeline()` returns a promise that resolves to a function. Keeping the **promise** at module level means the model loads exactly once no matter how many times `embed()` is called; every call just awaits the same promise (already resolved after the first time).

#### `embed(text)` — the heart of the system

```ts
export async function embed(text: string): Promise<number[]> {
  const extractor = await extractorPromise;
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}
```

Three lines, two crucial options:

- **`pooling: "mean"`** — the model actually produces one vector *per word-piece*. Mean pooling averages them into a single sentence-level vector representing the whole text.
- **`normalize: true`** — scales the vector to unit length. With all vectors on the unit sphere, cosine distance is clean and comparable across rows.

The `Float32Array` result is converted to a plain `number[]` so the rest of the code (and `JSON.stringify`) can treat it as ordinary data.

#### `toVectorLiteral(embedding)` — formatting for pgvector

```ts
function toVectorLiteral(embedding: number[]): string {
  return JSON.stringify(embedding);
}
```

pgvector accepts vector literals shaped like `'[0.1,0.2,0.3]'` — which is exactly what `JSON.stringify` produces for a number array. A happy coincidence that makes the conversion a one-liner.

#### `main()` — the indexing run

```ts
async function main() {
  const db = new Client({ connectionString: process.env.DATABASE_URL });
  await db.connect();

  await db.query("TRUNCATE documents RESTART IDENTITY");   // idempotent re-runs

  const document = await readFile(SOURCE_DOCUMENT, "utf8");
  const chunks = chunkText(document);

  for (const content of chunks) {
    const embedding = await embed(content);
    await db.query(
      "INSERT INTO documents (content, embedding) VALUES ($1, $2)",
      [content, toVectorLiteral(embedding)],
    );
  }

  await db.end();
}
```

Step by step:

1. **Connect** using `DATABASE_URL` from `.env` (loaded by the `import "dotenv/config"` at the top of the file).
2. **`TRUNCATE ... RESTART IDENTITY`** — wipe the table and reset the id counter, so re-running the script replaces the index instead of appending duplicates. Indexing is *idempotent*.
3. **Read + chunk** the document (the only "new" step compared to a hand-written corpus — everything downstream treats a chunk exactly like a snippet).
4. **Embed and insert each chunk**, using a parameterized query (`$1, $2`) — never string interpolation — so content containing quotes or SQL-ish text is handled safely.
5. Console output shows each chunk's label, size, and the first 3 numbers of its vector so you can *see* the embeddings being produced.

#### The direct-run guard

```ts
import { fileURLToPath } from "node:url";
const isDirectRun = process.argv[1] === fileURLToPath(import.meta.url);

if (isDirectRun) {
  main().catch(...);
}
```

This file is both a **script** (`npm run embed`) and a **library** (`search.ts` imports `embed` from it). The guard compares the path of the file Node was told to execute (`process.argv[1]`) with this file's own path — `main()` only runs when they match. Without this, every search would silently re-index the whole database first. This is the ESM equivalent of Python's `if __name__ == "__main__"`.

---

### 6.3 `src/search.ts` — similarity search (Retrieve)

The **retrieval** entry point (`npm run search`), and — since the generation step was added — also a library exporting `search()` for `ask.ts`.

#### The exports

```ts
export const TOP_K = 3;                    // how many chunks to retrieve

export interface SearchHit {
  content: string;                         // the chunk's text
  distance: number;                        // cosine distance from the query (lower = closer)
}
```

`TOP_K = 3` balances two pressures: enough chunks that the answer is probably in there, few enough that the LLM's context stays focused (and, with a paid model, cheap).

#### `search(query, topK)` — the whole retrieval step

```ts
export async function search(query: string, topK: number = TOP_K): Promise<SearchHit[]> {
  const db = new Client({ connectionString: process.env.DATABASE_URL });
  await db.connect();

  // 1) Turn the query into a vector — same model/space as the stored documents.
  const queryVector = await embed(query);
  const queryLiteral = JSON.stringify(queryVector);

  // 2) Let Postgres do the comparison.
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
```

Dissecting the SQL — the core trick of the entire project:

- **`embedding <=> $1::vector`** — for each row, compute the cosine distance between that row's stored vector and the query vector. `$1` arrives as a string literal (`'[0.02,-0.04,...]'`); `::vector` casts it to the pgvector type.
- **`ORDER BY distance LIMIT $2`** — sort all rows by that distance, keep the closest `topK`. This is a **nearest-neighbour search expressed as ordinary SQL**. No special search engine, no vector database product — a `SELECT` with an unusual operator.
- The distance itself is returned alongside the content, so callers (and you, reading console output) can judge match quality. In practice: below ~0.6 is a real match; everything above ~0.8 means "the corpus doesn't cover this".

Note the deliberate symmetry with indexing: **`embed()` here is the imported one from `embed.ts`** — same model, same pooling, same normalization, same vector space.

#### The CLI block

The bottom of the file has the same `isDirectRun` guard as `embed.ts`. Run directly (`npm run search "..."`), it takes the query from the command line (or a default), calls `search()`, and pretty-prints each hit with its distance and label. Imported by `ask.ts`, the CLI block is skipped entirely.

---

### 6.4 `src/result.ts` — errors as values

**Zero dependencies.** A tiny generic type that lets a function *return* its failure instead of throwing it:

```ts
export type Result<T, E> =
  | { readonly ok: true;  readonly value: T }
  | { readonly ok: false; readonly error: E };

export const ok  = <T>(value: T): Result<T, never> => ({ ok: true, value });
export const err = <E>(error: E): Result<never, E> => ({ ok: false, error });
```

Throwing erases types (a `catch` block receives `unknown`) and relies on every caller remembering a `try/catch`. Returning a `Result` makes the compiler *force* the check: you cannot read `.value` until you have narrowed `if (result.ok)`. An unhandled failure becomes a **compile-time** error, not a runtime crash. The LLM calls — the parts most likely to fail — return `Result<string, LLMError>`, so every caller of `chat()` has to deal with failure explicitly.

---

### 6.5 `src/llm.ts` — the hardened LLM client

The home of `chat()`, the single function every LLM call in the project goes through (both generation in `ask.ts` and scoring in `rerank.ts`). It wraps a bare Ollama `fetch` in the three things a production-minded LLM client needs: a **timeout**, **error classification**, and **retry with backoff**. The call is local, but its failure *modes* (timeouts, 5xx while busy, 400 on a bad request) are identical to a hosted API — swap the URL and body shape and this code is unchanged.

#### Configuration

```ts
const OLLAMA_URL = "http://localhost:11434/api/chat";
const MODEL = "llama3.2";
export const DEFAULT_TIMEOUT_MS = 30_000;
```

[Ollama](https://ollama.com) is a local LLM runtime serving open-weight models over HTTP on port 11434. `llama3.2` is Meta's ~3-billion-parameter model (~2 GB on disk) — small enough for a laptop CPU, capable enough to read a few chunks and answer from them. Free, keyless, offline.

#### Operational vs. programmer errors

Every failure is classified into one of two kinds, and **only one is retried:**

| Kind | Meaning | Examples | Retry? |
|---|---|---|---|
| `operational` | Transient, outside our control | timeout, network blip, HTTP 429, HTTP 5xx | ✅ yes |
| `programmer` | *We* sent something wrong | HTTP 400/404 (bad body, unknown model), a 200 with no content | ❌ no |

Retrying a programmer error is actively harmful — it burns time hammering the server with a request that can never succeed — so it is surfaced loudly instead. The split lives in `chatOnce()`: a 4xx that isn't 429 is `programmer`; a 429 or 5xx is `operational`.

#### Timeout — `AbortController`

`fetch` has no built-in deadline, so a hung model would block forever. `chatOnce()` arms an `AbortController` that aborts the request after `timeoutMs`; an aborted fetch is turned into a clean `operational` "timed out" error rather than a hang. The re-ranker passes a longer budget (60s) because scoring 10 chunks as JSON — possibly including cold model-load time — is a bigger job than answering.

#### Retry with backoff — `withRetry()`

A small, reusable, **generic** wrapper over *any* `Result`-returning async function:

```ts
export async function withRetry<T, E>(
  fn: () => Promise<Result<T, E>>,
  opts: { retries?; baseDelayMs?; isRetryable: (e: E) => boolean; onRetry?; },
): Promise<Result<T, E>>
```

It retries up to 3 times, **doubling** the delay each time (500ms → 1s → 2s), and stops early on a non-retryable error. `chat()` is just `withRetry(chatOnce, { isRetryable: e => e.kind === "operational" })` — so a cold model that times out on the first call quietly recovers on a retry, while a malformed request fails fast.

#### `chat(messages, options)`

The public entry point. Takes an array of `{ role, content }` messages plus optional `{ timeoutMs, temperature, format }`, and returns `Result<string, LLMError>`. `format: "json"` asks Ollama to constrain output to valid JSON (used by the re-ranker); `temperature: 0` makes scoring deterministic.

#### `chatWithTools(messages, tools, options)` — the tool-calling variant

A second public entry point, added for the tool-calling feature (§6.9). It uses the **same** transport, timeout, error classification and retry as `chat()`, but differs in two ways: it sends a `tools` array in the request body so the model is allowed to call them, and it returns the **full assistant message** (`ChatMessage`, including any `tool_calls`) instead of just the text — because the caller needs the `tool_calls` to drive the agent loop. The non-tool path (`chat`/`chatOnce`) is untouched, so RAG is unaffected. The supporting types `Tool` (OpenAI-style function schema: `name`, `description`, JSON-Schema `parameters`) and `ToolCall` (`{ function: { name, arguments } }`) live here too, and `ChatMessage` gained an optional `tool_calls` field and a `"tool"` role for feeding results back.

---

### 6.6 `src/rerank.ts` — two-stage retrieval (Re-rank)

The **re-rank** entry point (`npm run rerank`) and home of `retrieveAndRerank()`, which `ask.ts` uses in place of a bare `search()`. It fixes a real weakness of pure vector search: *nearest by embedding* and *actually answers the question* are not the same thing. A chunk can sit close in vector space just because it **repeats the question's words** without answering it.

The fix is the standard production pattern — retrieve wide, then re-rank:

```
search(top 10 by distance)   →   LLM scores each 1–10   →   keep top 4
-------- cheap, fuzzy ------       ------ reads meaning ------
```

```ts
export const INITIAL_K = 10;   // cast a wide net cheaply
export const FINAL_K = 4;      // keep only the best after scoring
```

Stage 1 (`search(question, 10)`) is cheap and fuzzy. Stage 2 spends **one** `chat()` call with a "search relevance judge" system prompt, `temperature: 0`, `format: "json"`, and a 60s timeout, asking the model to score each candidate 1–10. The scores are parsed, the candidates sorted by them, and the top `FINAL_K` kept.

**Graceful degradation is deliberate.** Re-ranking is a *quality* improvement, not a *correctness* requirement. If the scoring call fails or returns unparseable JSON, `retrieveAndRerank()` logs a warning and falls back to the plain top-`FINAL_K` by distance — the pipeline still answers, it just skips the polish. It returns `Result<RankedHit[], LLMError>`, so a hard failure is still explicit, but in practice the fallback means callers almost always get chunks. A `RankedHit` is a `SearchHit` plus an optional `rerankScore`.

---

### 6.7 `src/ask.ts` — generation (Prompt + Context → LLM → Answer)

The **full-pipeline** entry point (`npm run ask`). Retrieval + re-ranking stop at "here are the 4 most *useful* chunks"; this file turns them into an actual answer. Since hardening, it is thin glue: it calls `retrieveAndRerank()`, builds a prompt, calls `chat()`, and handles the `Result` from each.

#### Configuration

```ts
const MODEL = "llama3.2";   // for display only — the real call lives in llm.ts
```

The Ollama URL and the timeout/retry logic moved to `llm.ts`; `ask.ts` keeps only the model name for its console output.

#### The system prompt — where grounding lives

```ts
const SYSTEM_PROMPT = `You are a helpful assistant answering questions about a codebase, using ONLY the context chunks provided in the user's message.

Rules:
- Base every claim on the context. Do not use outside knowledge to fill gaps.
- If the context does not contain the answer, say "I don't know based on the indexed documents" — do not guess.
- When you use a chunk, cite it inline like [1] or [2] matching the chunk ids.
- Be concise: a few sentences or a short list is usually enough.`;
```

Four rules, each earning its place:

| Rule | Failure mode it prevents |
|---|---|
| Answer only from context | The model answering from training data — plausible but unverifiable |
| Explicit "I don't know" phrasing | Hallucination when retrieval comes back with weak matches |
| Inline `[n]` citations | Untraceable claims — citations make grounding *inspectable* |
| Be concise | A 3B model rambling and drifting away from the source material |

The prompt is a constant — it never changes per question. (In a hosted setup this stability is also what would make it an ideal prompt-cache prefix.)

#### `buildPrompt(question, hits)` — the "Prompt + Context" stage

```ts
function buildPrompt(question: string, hits: SearchHit[]): string {
  const context = hits
    .map((hit, i) => `<chunk id="${i + 1}">\n${hit.content}\n</chunk>`)
    .join("\n\n");

  return `<context>\n${context}\n</context>\n\nQuestion: ${question}`;
}
```

The produced message looks like:

```
<context>
<chunk id="1">
...text of the closest chunk...
</chunk>

<chunk id="2">
...second...
</chunk>

<chunk id="3">
...third...
</chunk>

<chunk id="4">
...fourth...
</chunk>
</context>

Question: how do I choose the chunk size and overlap?
```

Design choices:

- **Tagged blocks** (`<chunk id="n">`) give the model unambiguous boundaries between chunks — markdown chunks contain headings, code fences, and blank lines of their own, so a naive concatenation would blur where one ends and the next begins. The `id` is what the `[n]` citations refer back to.
- **Question last.** LLMs weight recent tokens heavily; putting the question after the context keeps it "fresh" when generation starts.

#### Generation now goes through `chat()`

The old version had its own `generate()` that streamed tokens with a bare `fetch` and a buffering loop. That logic moved into `llm.ts` and became non-streaming — streaming is incompatible with clean retries (a retry that had already printed half an answer would double up), so the hardened call takes the whole reply atomically, then prints it. `ask.ts` just calls `chat()`:

```ts
const answer = await chat([
  { role: "system", content: SYSTEM_PROMPT },
  { role: "user", content: buildPrompt(question, hits) },
]);

if (!answer.ok) { /* handle the typed failure — see below */ }
process.stdout.write(answer.value + "\n");
```

#### `ask(question)` — orchestrating the whole pipeline

```ts
async function ask(question: string) {
  const retrieved = await retrieveAndRerank(question);   // ── Retrieve + Re-rank
  if (!retrieved.ok) {                                   // only a HARD failure lands here —
    console.error(`❌ Retrieval failed: ${retrieved.error.message}`);
    process.exit(1);                                     // re-rank degrades on its own
  }
  const hits = retrieved.value;

  console.log(`\n🔎 Question: "${question}"`);           // show the kept chunks with their
  hits.forEach((hit, i) => { /* score + distance */ });  // re-rank score + distance, so
                                                         // grounding is inspectable first
  const answer = await chat([                            // ── Prompt + Context → LLM → Answer
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: buildPrompt(question, hits) },
  ]);
  if (!answer.ok) { /* ...handle by error kind... */ }

  process.stdout.write(answer.value + "\n");
  console.log(`\n— sources: ...`);                       // which chunks fed the answer
}
```

The kept chunks, their re-rank scores, and their distances are printed **before** the answer on purpose: you can always see exactly what the model was given, which makes it obvious whether a bad answer is a *retrieval* problem (wrong chunks) or a *generation* problem (right chunks, wrong reading).

#### Error handling — reading the error `kind`

Because `chat()` returns a classified `Result`, `ask.ts` can give the right advice instead of a stack trace:

```ts
if (!answer.ok) {
  const { kind, message } = answer.error;
  console.error(`❌ Could not generate an answer (${kind} error): ${message}`);
  if (kind === "operational") {
    console.error("   This looks transient — check that `ollama serve` is running and try again.");
  } else {
    console.error("   This looks like a bug in the request — retrying won't help; check the code.");
  }
  process.exit(1);
}
```

An `operational` failure (timeout, server busy) tells the user to retry; a `programmer` failure (malformed request) tells them retrying won't help. The one predictable failure — Ollama not running — surfaces as a clean operational message, not a raw `ECONNREFUSED`.

---

### 6.8 `src/bookmarks.ts` — the mock bookmarks store

The data source behind the tools. The project had no real bookmarks store, so this mocks one with a fixed in-memory `Bookmark[]` (each is `{ title, url, tags }`) and three plain functions. It's deliberately simple — if a real store (a DB table, an API) shows up later, only the bodies of these functions change; the tool wiring in `tool-ask.ts` stays the same.

| Function | Behaviour |
|---|---|
| `getBookmarks(tag)` | Returns every bookmark whose tags include `tag`, **case-insensitively**. An unknown tag yields `[]` — a valid answer ("nothing under that tag"), not an error. |
| `addBookmark(url, title, tags)` | Appends a bookmark and returns it. Tags are lowercased and de-duplicated so `getBookmarks` stays case-insensitive. A **duplicate `url` throws** — which the tool layer turns into a readable tool-result error rather than a crash. |
| `countBookmarks()` | Returns the current total. |

The array is mutable (appended to by `addBookmark`), so within a single `npm run tools` run the three tools see each other's effects.

---

### 6.9 `src/tool-ask.ts` — tool calling (the agent loop)

The **tool-calling** entry point (`npm run tools`). Where RAG (§6.7) answers from text you retrieved and pasted into the prompt, tool calling is a different LLM mode: you hand the model **tools** it can choose to call, it asks for calls, *your* code runs them, you feed results back, and it loops until it returns plain text:

```
user question → LLM → (tool calls?) → your code runs them → tool results
              → LLM → … → final text answer
```

This feature is independent of the RAG pipeline — it imports only `chatWithTools()` from `llm.ts` (§6.5) and the store from `bookmarks.ts` (§6.8), so it cannot affect retrieval, re-ranking, or `ask.ts`.

#### The registry — one source of truth per tool

Each tool is one entry pairing a **Zod schema**, a description, and the implementation:

```ts
interface ToolSpec {
  description: string;                              // the model reads this to decide when to call
  schema: z.ZodObject<z.ZodRawShape>;               // validates arguments AND generates the JSON schema
  run: (args: Record<string, unknown>) => unknown;  // the actual code; returns any JSON-serialisable value
}
```

The Zod schema is the single source of truth for a tool's arguments. `z.toJSONSchema(spec.schema)` derives the JSON-Schema `parameters` the model is shown, and the same schema's `safeParse` validates what the model sends back — so the description the model sees and the validation we enforce can never drift apart. The three registered tools are `get_bookmarks(tag)`, `add_bookmark(url, title, tags)`, and `count_bookmarks()`.

#### `executeToolCall(call)` — validate, run, never throw

Runs one tool call and returns a `ToolOutcome` (`{ ok, content }`) whose `content` is always a JSON string ready to hand back as a `tool_result`. It **never throws**, so a bad call can't take the loop down. Three failure modes are handled distinctly, each as a readable error the model can recover from on its next turn:

| Failure | Handling |
|---|---|
| Unknown / hallucinated tool name | `{ error: "Unknown tool \"x\". Available tools: …" }` (lists the real ones) |
| Arguments fail Zod validation | `{ error: "Invalid arguments for x: <issues>" }` |
| The implementation throws (e.g. duplicate URL) | caught → `{ error: "x failed: <message>" }` |

`executeToolCalls(calls)` maps over a batch and returns all results in order — this is how **parallel** tool calls are handled: the model may request several at once, and we run them all and return one result per call.

#### `run(question)` — the loop

Seeds a transcript with a system prompt + the user question, then loops up to `MAX_TOOL_ITERATIONS = 5` (the cap against infinite loops). Each iteration calls `chatWithTools()`; if the reply has **no** `tool_calls` it's the final answer and is returned; otherwise every requested call is executed, each result is pushed back as a `role: "tool"` message, and the model gets another turn. It **returns** the final string (rather than `process.exit`-ing) so the tests can drive it; the thin CLI wrapper at the bottom — guarded by an `import.meta.url === pathToFileURL(process.argv[1])` check so importing the module in tests doesn't launch it — handles argv and exit codes. Every call, its arguments, a ✓/✗ marker, and its result are logged so the whole loop is inspectable.

---

### 6.10 `src/tool-ask.test.ts` — testing the tool layer

Tests for the tool layer, run with `npm test` (Node's built-in `node:test` via `tsx --test`). Two tiers:

- **Deterministic unit tests** on `executeToolCall` / `executeToolCalls` — they exercise argument validation, error handling, parallel execution, and unknown tools *without touching the LLM*, so they're fast and never flaky.
- **Live integration tests** that drive the real `run()` loop through Ollama. They're skipped automatically when Ollama isn't reachable on `localhost:11434`, so the suite still passes offline.

Between them they cover the five scenarios the feature promises: a no-tool turn, a one-tool request, multiple/parallel tools, invalid arguments, and an unknown tool. (The deterministic tests rely on the store being mutable, so a couple add a bookmark and assert the count moves.)

---

### 6.11 `src/eval.ts` — a lightweight eval

The **eval** entry point (`npm run eval`). Not a framework — the *first honest signal* that the pipeline actually works. It runs the real pipeline (retrieve → re-rank → generate) against 6 questions whose answers are known, and checks the two things that matter most in RAG:

1. When the answer **is** in the README — does it answer correctly?
2. When the answer is **not** — does it correctly **decline** ("I don't know…") instead of hallucinating?

Half the cases are deliberately unanswerable ("what is the capital of France?", "deploy to Kubernetes with autoscaling?"), because (2) is the failure people forget: a RAG system that always sounds confident is worse than useless, since the failure is invisible. Each case declares a `shouldAnswer` flag plus expected keywords, and a crude heuristic judges it:

| Case type | Passes when… |
|---|---|
| should **answer** | did *not* decline **and** an expected keyword appears |
| should **decline** | the answer contains an "I don't know / not in the context" marker |

For every case it logs the question, the retrieved chunk labels, the answer, and the judgment side by side, then prints `Score: N/6`.

> **The pass/fail is a heuristic, not ground truth.** Keyword presence is a proxy for correctness — a correct answer that phrases things differently can still "fail". The real value is the side-by-side log: *read the answers, don't just trust the ✅.* It's a regression tripwire, not a grade.

---

### 6.12 `src/snippets.ts` — the legacy corpus

The project's first version had no chunking: the corpus was **eight hand-written snippets**, each already chunk-sized (2–4 sentences describing a project — Mini Redis, a debounce helper, a memoize utility, ...). `chunk.ts` replaced this file by generating chunks automatically from a real document.

It's deliberately kept in the repo, unused: comparing a hand-written snippet with the chunks `chunkText()` produces is the clearest way to see *what chunking automates* — producing pieces of text "the right size to answer one question". Deleting it would change nothing.

### 6.13 Configuration files

**`package.json`** — four runtime dependencies, seven scripts:

```jsonc
{
  "type": "module",                          // ESM — enables import.meta.url etc.
  "scripts": {
    "embed":  "tsx src/embed.ts",            // index README.md into Postgres
    "search": "tsx src/search.ts",           // retrieval only (vector distance)
    "rerank": "tsx src/rerank.ts",           // retrieval + LLM re-rank (before/after)
    "ask":    "tsx src/ask.ts",              // full RAG: retrieve + re-rank + generate
    "tools":  "tsx src/tool-ask.ts",         // tool-calling agent loop (bookmarks)
    "eval":   "tsx src/eval.ts",             // 6-question pass/fail harness
    "test":   "tsx --test src/tool-ask.test.ts" // tests for the tool layer
  },
  "dependencies": {
    "@xenova/transformers": "^2.17.2",       // local embedding model runtime
    "dotenv": "^17.4.2",                     // loads .env into process.env
    "pg": "^8.22.0",                         // PostgreSQL client
    "zod": "^4.6.5"                          // tool-argument validation
  }
}
```

`tsx` (a dev dependency) runs TypeScript directly — no build step, instant feedback.

**`tsconfig.json`** — `strict: true` (full type checking), `noEmit: true` (type-check only; tsx handles execution), ES2022 modules.

**`.env`** (git-ignored) — one real value:

```
DATABASE_URL=postgres://localhost:5432/vector_demo
```

No API keys of any kind — every model in the pipeline runs locally.

**`.gitignore`** — `node_modules/`, `.env`, `.DS_Store`.

---

## 7. End-to-end trace of one question

What actually happens, in order, when you run:

```bash
npm run ask "how do I choose the chunk size and overlap?"
```

1. `tsx` executes `src/ask.ts`. `dotenv` loads `DATABASE_URL`. The question is read from `process.argv`.
2. `ask()` calls `retrieveAndRerank(question)`. **Stage 1 — retrieve:** `search(question, 10)`:
   - triggers the lazy model load in `embed.ts` (instant if the ~90 MB model is already cached),
   - runs the question through MiniLM → a 384-number unit vector,
   - runs `SELECT content, embedding <=> $1::vector AS distance FROM documents ORDER BY distance LIMIT 10`,
   - Postgres computes the cosine distance to **all stored vectors**, sorts, returns the 10 closest.
3. **Stage 2 — re-rank:** `retrieveAndRerank()` sends those 10 candidates to `chat()` with a relevance-judge prompt (`temperature: 0`, `format: "json"`, 60s timeout). The model scores each 1–10; the candidates are sorted by score and the top 4 kept. (If this call fails, it falls back to the top 4 by distance.)
4. The kept chunks print with their re-rank score and distance:
   ```
   [1] (score 10/10, distance 0.5652) -> "I don't know based on the indexed d…
   [2] (score  9/10, distance 0.5345) 4. Architecture & data flow
   [3] (score  9/10, distance 0.5675) -> section 3 "Choosing size and overlap…
   [4] (score  9/10, distance 0.5866) Choosing size and overlap
   ```
5. `buildPrompt()` wraps those 4 chunks in `<chunk id="n">` tags inside `<context>`, appends the question.
6. `chat()` (in `llm.ts`) POSTs `{model: "llama3.2", stream: false, messages: [system, user]}` to `localhost:11434/api/chat`, behind a 30s timeout and up to 3 retries. The full reply comes back atomically:
   > *According to chunk 3, [3], the recommended chunk size is 900 chars (~225 tokens), and the recommended overlap is 150 chars (~17%).*
7. The answer prints, then a sources line lists which chunks fed it.

Total cost: $0. Network traffic: none (everything is localhost).

And the negative case: ask *"what is the capital of France?"* and steps 2–3 still return 4 chunks — the *least bad* matches, at high distances — but step 6 produces *"I don't know based on the indexed documents"*, because the grounding rules forbid the model from using what it knows from training.

---

## 8. Design decisions and invariants

**The two invariants** — break either and the system silently returns garbage:

1. **One embedder, both sides.** Query vectors and document vectors must come from the same model with the same options. Enforced structurally: there is exactly one `embed()` in the codebase, and `search.ts` imports it.
2. **Dimensions must agree end-to-end.** Model output (384) = `VECTOR(384)` column = every stored row = every query vector. Changing the model means changing the schema and re-indexing.

**Deliberate choices, and what was traded away:**

| Choice | Why | Trade-off accepted |
|---|---|---|
| Local MiniLM embeddings, not a hosted API | Free, keyless, offline; identical pipeline shape | 384 dims / lower quality than e.g. `text-embedding-3-small` (1536) — irrelevant at 40 rows |
| Local llama3.2 (3B) via Ollama, not a frontier model | Free, keyless, offline | Noticeably weaker reasoning; fine for "summarize 3 chunks", occasionally garbles details |
| Hand-written chunker, not LangChain | The whole idea fits in 120 lines; writing it teaches what the parameters mean | None at this scale |
| Postgres + pgvector, not a dedicated vector DB | It's just SQL on a database you already know | Purpose-built vector DBs scale further — far beyond this project's needs |
| No HNSW/IVFFlat index | Brute force over 40 rows is instant | Would be needed at ~10k+ rows |
| `TRUNCATE` before each indexing run | Idempotent re-runs, no duplicate rows | Full re-embed each time — no incremental update |
| Sequential embedding loop (one chunk at a time) | Simple to read; local model calls are the bottleneck anyway | Parallelism would complicate without teaching anything |
| README.md as the corpus | Self-referential: the project can answer questions about itself | Editing the README changes retrieval results (re-run `npm run embed` after edits) |

**A structural pattern used repeatedly:** `embed.ts`, `search.ts`, and `rerank.ts` are each simultaneously a script and a library, using the `process.argv[1] === fileURLToPath(import.meta.url)` guard to run their CLI only when executed directly. This is what lets each pipeline stage be run and inspected *independently* (`npm run search`, `npm run rerank`) while still composing into the full pipeline that `ask.ts` drives.

---

## 9. Running it

### Prerequisites

- Node.js 18+ (built-in `fetch` is used)
- PostgreSQL with the pgvector extension
- [Ollama](https://ollama.com) with the llama3.2 model

### One-time setup

```bash
# 1. Database
psql -d postgres -c "CREATE DATABASE vector_demo;"
psql -d vector_demo -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d vector_demo -c "CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding VECTOR(384)
);"

# 2. LLM
ollama pull llama3.2        # ~2 GB, once

# 3. Project
npm install
echo 'DATABASE_URL=postgres://localhost:5432/vector_demo' > .env
```

### Usage

```bash
npm run embed                                  # index README.md (re-run after editing it)
npm run search "how does chunk overlap work?"  # retrieval only — raw chunks + distances
npm run rerank "how does chunk overlap work?"  # retrieval + re-rank — before/after order
npm run ask    "how does chunk overlap work?"  # full RAG — grounded, cited answer
npm run tools  "which bookmarks are tagged rag, and how many total?"  # tool-calling loop
npm run eval                                   # 6-question pass/fail harness
npm test                                       # tests for the tool layer
```

`npm run search` and `npm run rerank` are worth running on their own even though `ask` supersedes them: `search` shows the raw distances (intuition for what the vector space calls "similar"), and `rerank` shows the stage-1 vs stage-2 order side by side so you can *see* re-ranking move the genuinely useful chunk up past a mere keyword match. `npm run tools` is a separate LLM mode (not RAG) — see §6.9.

To type-check the whole project without running anything: `npx tsc --noEmit`.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ECONNREFUSED ... 5432` | Postgres isn't running | `brew services start postgresql` (or your platform's equivalent) |
| `relation "documents" does not exist` | Table never created | Run the `CREATE TABLE` from section 9 |
| `type "vector" does not exist` | pgvector extension missing | `CREATE EXTENSION vector;` in the `vector_demo` database |
| `expected 384 dimensions, got N` | Model/schema mismatch | Make `VECTOR(N)` match the model's output; re-run `npm run embed` |
| `Could not generate an answer (operational error): ... timed out` | Ollama slow or not running (often a cold model load) | `ollama serve`; the retry often recovers on its own. Raise `timeoutMs` if the model is large |
| `Could not generate an answer (programmer error): ...` | Malformed request / unknown model — retrying won't help | Check the model name and request body in `llm.ts` |
| `⚠️ Re-ranker unavailable ... falling back to vector order` | The re-rank scoring call failed | Harmless — the pipeline still answers by plain distance. Check Ollama if it persists |
| First `npm run embed` is slow | One-time ~90 MB model download | Wait once; it's cached afterwards |
| Search returns stale/odd chunks | README edited since last indexing | `npm run embed` again |
| Every distance > 0.8 | The corpus genuinely doesn't cover the topic | Expected behaviour — that's the "no good match" signal |

### Inspecting the database directly

```bash
psql -d vector_demo -c "SELECT id, left(content, 60) AS preview FROM documents;"
psql -d vector_demo -c "SELECT count(*) FROM documents;"
```

---

## 11. Extending the project

Each of these touches exactly one seam of the pipeline:

- **Index your own documents** — change `SOURCE_DOCUMENT` in `embed.ts` (or loop over a directory of files, adding a `source` column to the table). Nothing else changes.
- **Swap the LLM** — everything model-specific lives in `llm.ts` (`OLLAMA_URL`, `MODEL`). Point it at a bigger Ollama model (`ollama pull qwen2.5:7b`), or at a hosted API — the `Result`, timeout, and retry handling stay the same.
- **Swap the embedder** — change `EMBEDDING_MODEL` in `embed.ts`, update `VECTOR(n)` to the new dimension, re-run `npm run embed`. The one-embedder invariant keeps this a two-file change.
- **Tune retrieval** — `INITIAL_K` / `FINAL_K` in `rerank.ts` (how wide to cast, how many to keep); `TOP_K` in `search.ts` for bare search; `CHUNK_SIZE` / `CHUNK_OVERLAP` in `chunk.ts` (re-index after changing chunking).
- **Add a relevance threshold** — discard hits with distance > ~0.75 before prompting, so the LLM sees "no context" instead of misleading weak matches.
- **Re-add streaming** — the hardened `chat()` is non-streaming for clean retries. Stream again by only retrying failures that happen *before* the first byte.
- **Scale up** — at tens of thousands of chunks, add a vector index: `CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);`
- **Make it conversational** — keep a message history and re-retrieve per question; the retrieval and grounding logic is unchanged.
- **Grow the eval** — `eval.ts` is the seed. Log results to JSONL with timestamps to track quality over time, or replace the keyword heuristic with a separate LLM judge ("is this answer supported by the context?").
- **Add a tool** — add one entry to the `REGISTRY` in `tool-ask.ts`: a Zod `schema`, a `description`, and a `run` function. The JSON schema the model sees, the argument validation, and the dispatch all follow automatically — no other file changes. (Back it with a real data source by swapping the bodies in `bookmarks.ts`, or a new module, behind the same function signatures.)
- **Give RAG its own tool** — the tool loop and the RAG pipeline are independent today. To let the model *retrieve on demand*, wrap `retrieveAndRerank()` (from `rerank.ts`) as a `search_docs(query)` tool in the registry; the loop and validation are unchanged.
