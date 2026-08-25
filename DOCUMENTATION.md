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
   - [6.4 `src/ask.ts` — generation (Prompt + Context → LLM → Answer)](#64-srcaskts--generation-prompt--context--llm--answer)
   - [6.5 `src/snippets.ts` — the legacy corpus](#65-srcsnippetsts--the-legacy-corpus)
   - [6.6 Configuration files](#66-configuration-files)
7. [End-to-end trace of one question](#7-end-to-end-trace-of-one-question)
8. [Design decisions and invariants](#8-design-decisions-and-invariants)
9. [Running it](#9-running-it)
10. [Troubleshooting](#10-troubleshooting)
11. [Extending the project](#11-extending-the-project)

---

## 1. What this project is

This project answers natural-language questions about a document (the project's own `README.md`) by implementing every stage of a RAG pipeline by hand:

```
Document → Chunk → Embedding → pgvector → Retrieve → Prompt + Context → LLM → Answer
```

Ask it *"how do I choose the chunk size?"* and it will:

1. Convert your question into a 384-number vector,
2. Find the 3 chunks of the README whose vectors are closest in meaning,
3. Hand those chunks to a local LLM with strict instructions to answer **only** from them,
4. Stream back a grounded, cited answer.

The point is educational: every stage that frameworks like LangChain hide behind abstractions is written out here in plain TypeScript, small enough to read in one sitting (~450 lines across four files).

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
  embedding <=> $1 LIMIT 3      computes cosine distance to every stored row
      │
      ▼
  top 3 chunks                  the "Retrieve" step ends here
      │
      ▼
  buildPrompt()                 src/ask.ts — chunks wrapped in <chunk id="n">
      │                         tags + grounding rules + the question
      ▼
  Ollama /api/chat              llama3.2 running at localhost:11434
      │
      ▼
  streamed answer               grounded in the chunks, cited [1] [2] [3],
                                or "I don't know" if the chunks lack the answer
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
│   ├── ask.ts         # FULL RAG entry point (npm run ask). Retrieval + generation.
│   └── snippets.ts    # Legacy hand-written corpus. Unused; kept for comparison.
├── .env               # DATABASE_URL. Git-ignored.
├── .gitignore         # node_modules/, .env, .DS_Store
├── package.json       # Dependencies + the three npm scripts
├── tsconfig.json      # Strict TypeScript, ESM, no emit (tsx runs TS directly)
├── README.md          # Project README — and also the document being indexed
└── DOCUMENTATION.md   # This file
```

Dependency graph between source files (arrows mean "imports from"):

```
ask.ts ──► search.ts ──► embed.ts ──► chunk.ts
   │            │                        ▲
   └────────────┴────────────────────────┘   (chunkLabel, for console output)
```

`chunk.ts` sits at the bottom with zero imports; `ask.ts` sits at the top and touches everything.

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

### 6.4 `src/ask.ts` — generation (Prompt + Context → LLM → Answer)

The **full-pipeline** entry point (`npm run ask`). Everything before this file stops at "here are the 3 most relevant chunks"; this file turns them into an actual answer.

#### Configuration

```ts
const OLLAMA_URL = "http://localhost:11434/api/chat";
const MODEL = "llama3.2";
```

[Ollama](https://ollama.com) is a local LLM runtime: it downloads open-weight models and serves them over a simple HTTP API on port 11434. `llama3.2` is Meta's 3-billion-parameter model (~2 GB on disk) — small enough to run comfortably on a laptop CPU, capable enough for "read these three chunks and answer from them". Free, keyless, offline.

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
...second closest...
</chunk>

<chunk id="3">
...third closest...
</chunk>
</context>

Question: how do I choose the chunk size and overlap?
```

Design choices:

- **Tagged blocks** (`<chunk id="n">`) give the model unambiguous boundaries between chunks — markdown chunks contain headings, code fences, and blank lines of their own, so a naive concatenation would blur where one ends and the next begins. The `id` is what the `[n]` citations refer back to.
- **Question last.** LLMs weight recent tokens heavily; putting the question after the context keeps it "fresh" when generation starts.

#### `generate(question, hits)` — calling the local LLM

```ts
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

  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk as Uint8Array, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";                 // keep any incomplete trailing line
    for (const line of lines) {
      if (!line.trim()) continue;
      const part = JSON.parse(line);
      if (part.message?.content) process.stdout.write(part.message.content);
    }
  }
  process.stdout.write("\n");
}
```

Points of interest:

- **Plain `fetch`, no SDK.** Ollama's API is simple enough that Node's built-in fetch covers it — the project's dependency list stays at four packages.
- **`stream: true` → NDJSON.** Ollama replies with *newline-delimited JSON*: one JSON object per line, each carrying the next few generated tokens in `.message.content`, with a final `{"done": true}` object. Streaming means the answer appears word by word instead of after a long silence.
- **The buffering dance** (`buffer`, `lines.pop()`): network chunks don't align with line boundaries — a read might end mid-JSON-object. The code accumulates raw text, splits on newlines, processes every *complete* line, and carries the trailing partial line into the next iteration. `decoder.decode(chunk, { stream: true })` similarly handles multi-byte UTF-8 characters split across chunks.

#### `ask(question)` — orchestrating the whole pipeline

```ts
async function ask(question: string) {
  const hits = await search(question, TOP_K);         // ── Retrieve

  console.log(`\n🔎 Question: "${question}"`);        // show what was retrieved,
  hits.forEach((hit, i) => { ... });                  // with distances — so grounding
                                                      // is inspectable before the answer
  await generate(question, hits);                     // ── Prompt + Context → LLM → Answer

  console.log(`\n— sources: ...`);                    // which chunks fed the answer
}
```

The retrieved chunks and their distances are printed **before** the answer on purpose: you can always see exactly what the model was given, which makes it obvious whether a bad answer is a *retrieval* problem (wrong chunks) or a *generation* problem (right chunks, wrong reading).

#### Error handling

```ts
ask(question).catch((err) => {
  if (err instanceof TypeError && String(err.cause ?? "").includes("ECONNREFUSED")) {
    console.error("❌ Can't reach Ollama at localhost:11434.\n   Start it with:  ollama serve ...");
  } else {
    console.error("Ask error:", err);
  }
  process.exit(1);
});
```

The one predictable failure — Ollama isn't running — is caught specifically (Node's fetch throws a `TypeError` whose `cause` carries `ECONNREFUSED`) and turned into an actionable message instead of a stack trace.

---

### 6.5 `src/snippets.ts` — the legacy corpus

The project's first version had no chunking: the corpus was **eight hand-written snippets**, each already chunk-sized (2–4 sentences describing a project — Mini Redis, a debounce helper, a memoize utility, ...). `chunk.ts` replaced this file by generating chunks automatically from a real document.

It's deliberately kept in the repo, unused: comparing a hand-written snippet with the chunks `chunkText()` produces is the clearest way to see *what chunking automates* — producing pieces of text "the right size to answer one question". Deleting it would change nothing.

### 6.6 Configuration files

**`package.json`** — four runtime dependencies, three scripts:

```jsonc
{
  "type": "module",                          // ESM — enables import.meta.url etc.
  "scripts": {
    "embed":  "tsx src/embed.ts",            // index README.md into Postgres
    "search": "tsx src/search.ts",           // retrieval only
    "ask":    "tsx src/ask.ts"               // full RAG: retrieve + generate
  },
  "dependencies": {
    "@xenova/transformers": "^2.17.2",       // local embedding model runtime
    "dotenv": "^17.4.2",                     // loads .env into process.env
    "pg": "^8.22.0"                          // PostgreSQL client
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
2. `ask()` calls `search(question, 3)`, which:
   - triggers the lazy model load in `embed.ts` (instant if the ~90 MB model is already cached),
   - runs the question through MiniLM → a 384-number unit vector,
   - connects to Postgres and runs `SELECT content, embedding <=> $1::vector AS distance FROM documents ORDER BY distance LIMIT 3`,
   - Postgres computes the cosine distance from the query vector to **all 40 stored vectors**, sorts, returns the 3 closest.
3. The retrieved chunks print with their distances:
   ```
   [1] (distance 0.5345) 4. Architecture & data flow
   [2] (distance 0.5461) start a new one. Because we only ever a…
   [3] (distance 0.5866) Choosing size and overlap
   ```
4. `buildPrompt()` wraps those 3 chunks in `<chunk id="n">` tags inside `<context>`, appends the question.
5. `generate()` POSTs `{model: "llama3.2", stream: true, messages: [system, user]}` to `localhost:11434/api/chat`.
6. Ollama runs llama3.2 on the prompt; NDJSON fragments stream back and print as they arrive:
   > *According to chunk 3 ... Chunk size: **900 chars** (~225 tokens), Overlap: **150 chars** (~17%) ... designed for a model limit of 256 tokens.*
7. A sources line lists which chunks fed the answer.

Total cost: $0. Network traffic: none (everything is localhost).

And the negative case: ask *"what is the capital of France?"* and step 2 still returns 3 chunks — the *least bad* matches, at high distances — but step 6 produces *"I don't know based on the indexed documents"*, because the grounding rules forbid the model from using what it knows from training.

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

**A structural pattern used twice:** both `embed.ts` and `search.ts` are simultaneously scripts and libraries, using the `process.argv[1] === fileURLToPath(import.meta.url)` guard to run their CLI only when executed directly. This is what lets each pipeline stage be run and inspected *independently* while still composing into the full pipeline.

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
npm run search "how does chunk overlap work?"  # retrieval only — see chunks + distances
npm run ask "how does chunk overlap work?"     # full RAG — grounded, cited answer
```

`npm run search` is worth using on its own even though `ask` supersedes it: seeing the raw distances is how you develop intuition for what the vector space considers "similar".

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ECONNREFUSED ... 5432` | Postgres isn't running | `brew services start postgresql` (or your platform's equivalent) |
| `relation "documents" does not exist` | Table never created | Run the `CREATE TABLE` from section 9 |
| `type "vector" does not exist` | pgvector extension missing | `CREATE EXTENSION vector;` in the `vector_demo` database |
| `expected 384 dimensions, got N` | Model/schema mismatch | Make `VECTOR(N)` match the model's output; re-run `npm run embed` |
| `Can't reach Ollama at localhost:11434` | Ollama isn't running | `ollama serve` (and `ollama pull llama3.2` if needed) |
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
- **Swap the LLM** — everything model-specific lives in `generate()` in `ask.ts`. Point it at a bigger Ollama model (`ollama pull qwen2.5:7b`), or at a hosted API for stronger answers.
- **Swap the embedder** — change `EMBEDDING_MODEL` in `embed.ts`, update `VECTOR(n)` to the new dimension, re-run `npm run embed`. The one-embedder invariant keeps this a two-file change.
- **Tune retrieval** — `TOP_K` in `search.ts`; `CHUNK_SIZE` / `CHUNK_OVERLAP` in `chunk.ts` (re-index after changing chunking).
- **Add a relevance threshold** — discard hits with distance > ~0.75 before prompting, so the LLM sees "no context" instead of misleading weak matches.
- **Scale up** — at tens of thousands of chunks, add a vector index: `CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);`
- **Make it conversational** — keep a message history and re-retrieve per question; the retrieval and grounding logic is unchanged.
- **Evaluate it** — build a small set of (question, expected-chunk) pairs and measure how often the expected chunk appears in the top 3. This is how real RAG systems are tuned.
