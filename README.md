# Vector Search Demo — a minimal RAG pipeline

A small, **fully free, fully local** project that implements a complete
**RAG (Retrieval-Augmented Generation)** pipeline: make a real document
**searchable by meaning** instead of by keywords, then have an LLM **answer
questions from it** — grounded in the retrieved passages, nothing else.

You give it a natural-language question like *"how do I install pgvector on a
Mac?"*. It retrieves the most **semantically similar** passages — even when they
share no words with the question — and a local LLM (`llama3.2` via Ollama)
writes a cited answer from them, or says *"I don't know"* when the document
doesn't cover it.

> **This README is the corpus.** The pipeline indexes *this very file*. Every
> section below is chunked, embedded, and stored in Postgres — which means
> editing this README changes the search results, and you have to re-run
> `npm run embed` to see them. (That staleness is a real property of every RAG
> system, not a quirk of this one.)

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [The big idea: semantic search](#2-the-big-idea-semantic-search)
3. [Chunking: why a document must be cut up](#3-chunking-why-a-document-must-be-cut-up)
4. [Architecture & data flow](#4-architecture--data-flow)
5. [Tech stack — what and why](#5-tech-stack--what-and-why)
6. [Project structure](#6-project-structure)
7. [Prerequisites](#7-prerequisites)
8. [Setup from scratch](#8-setup-from-scratch)
9. [How to use it](#9-how-to-use-it)
10. [Code walkthrough](#10-code-walkthrough)
11. [How similarity search works](#11-how-similarity-search-works)
12. [Why it works (the semantic magic)](#12-why-it-works-the-semantic-magic)
13. [Troubleshooting](#13-troubleshooting)
14. [Where to go next](#14-where-to-go-next)

---

## 1. What this project does

The project is split into two phases, matching how every RAG system works.

**Offline (run once) — build the knowledge base:**

- Read a real document (`README.md`).
- **Chunk** it into ~40 overlapping passages of roughly 900 characters.
- Convert each chunk into an **embedding** — a list of 384 numbers that
  represents its meaning.
- Store the text + its embedding in a PostgreSQL table.

**Online (every search) — answer a query:**

- Take a user's question.
- Convert it into an embedding using the **same** model.
- Ask PostgreSQL for the chunks whose embeddings are **closest** to the
  question's embedding.
- Return the top 3.

That's the "R" (Retrieval) in RAG — `npm run search` stops there so you can
inspect the raw chunks and distances.

**Online (every question) — generate a grounded answer:**

- `npm run ask` runs the same retrieval, then wraps the top-3 chunks in a
  prompt with strict grounding rules.
- A **local LLM** (`llama3.2` served by Ollama) answers **only from those
  chunks**, citing them inline as [1] [2] [3] — or replies *"I don't know based
  on the indexed documents"* when they don't contain the answer.

That's the "G" (Generation): the knowledge lives in Postgres, the LLM only
reads and phrases it.

---

## 2. The big idea: semantic search

Traditional search matches **keywords**. If you search for "caching system" but
the document says "in-memory key-value store", keyword search finds **nothing** —
there are no shared words.

**Embeddings** solve this. An embedding model has read enormous amounts of text
and learned to place related concepts near each other in a high-dimensional
space. So it knows that:

```
cache  ↔  memory  ↔  Redis  ↔  key-value  ↔  fast lookup
```

...all point in a similar "direction". When we turn both the query and the
documents into vectors, *"caching system"* and *"in-memory key-value store"* end
up **close together** — even with zero shared words. Search then becomes a
geometry problem: **find the nearest vectors.**

---

## 3. Chunking: why a document must be cut up

An earlier version of this project used a hand-written corpus: eight short
snippets that were already the right size. A real document isn't. This README is
one ~29,000-character blob, and you **cannot** embed it as a single unit. Three
reasons, in increasing order of importance:

**1. The model has an input limit.** `all-MiniLM-L6-v2` reads at most ~256
word-pieces (roughly 1,000 characters) and *silently discards the rest*. Embed the
whole file and you'd be indexing the table of contents while the other 95% of the
document vanishes. No error, no warning — just a knowledge base that's mostly
empty.

**2. One vector per document is a blurry average.** An embedding represents the
meaning of its input as a single point. Feed it a document covering installation,
SQL operators, and troubleshooting, and you get a point somewhere in the middle
of all three — close to nothing in particular. The query *"how do I install
pgvector?"* would match that average weakly instead of matching the install
section strongly.

**3. Retrieval returns whole rows.** Whatever you stored is what comes back. Small
rows mean precise context — which matters enormously once you feed those rows to
an LLM, where every irrelevant sentence is both a distraction and a cost.

So we cut the document up first. That's the entire new step:

```
README.md  ->  chunkText()  ->  [chunk 1, chunk 2, chunk 3, ... chunk 40]
```

Everything downstream is untouched, because **a chunk is just a string** — the
same shape a hand-written snippet was.

### The two rules that make chunking "smart"

A naive implementation is `text.slice(0, 900)`, repeated. That's bad, because it
cuts wherever the character count runs out — usually mid-sentence, sometimes
mid-word. A chunk ending in *"the distance operators are"* embeds as half a
thought, and half a thought has a meaningless vector.

**Rule 1 — cut only at natural boundaries.** We split at blank lines (the
paragraph breaks a human would use), then greedily pack whole paragraphs into a
chunk until adding one more would exceed the target size. A chunk therefore ends
where a paragraph ends. One exception matters in a technical document: a blank
line *inside* a fenced code block (` ``` `) is not a paragraph break, and
splitting there would leave dangling half-programs — so the splitter tracks
whether it's inside a fence and ignores blank lines while it is.

**Rule 2 — overlap the seams.** Even with clean cuts, an idea that spans two
paragraphs gets divided, and *neither* chunk carries it fully. So each chunk
repeats the last ~150 characters of the one before it. The cost is a few
duplicated tokens; the benefit is that a fact sitting on a boundary still lands
somewhere intact. This is why chunk 2 of this README starts with the tail of
chunk 1.

### Choosing size and overlap

| Setting | Too small | Too large | This project |
|---------|-----------|-----------|--------------|
| **Chunk size** | Loses context — a chunk saying "it uses cosine" without saying what "it" is | Blurry average again; may exceed the model's input limit | **900 chars** (~225 tokens, just under MiniLM's 256 limit) |
| **Overlap** | Ideas fall through the cracks between chunks | Wasted storage and duplicate hits in results | **150 chars** (~17%) |

There's no universally correct answer — it depends on the document and the
model. The useful instinct: **a chunk should be the smallest piece of text that
still answers a question on its own.**

---

## 4. Architecture & data flow

```
OFFLINE  (run once: `npm run embed`)
────────────────────────────────────
   README.md         (one ~29,000-character document)
          │
          ▼
   chunkText()       ◄── THE NEW STEP (src/chunk.ts)
          │
          ▼
   40 chunks         (~900 chars each, 150-char overlap)
          │
          ▼
   embed()           (local model: all-MiniLM-L6-v2)
          │
          ▼
   384-number vector per chunk
          │
          ▼
   INSERT INTO documents (content, embedding)     ──►  PostgreSQL + pgvector
                                                        (the knowledge base)

ONLINE  (every question: `npm run ask "..."`)
──────────────────────────────────────────────
   user query (a sentence)
          │
          ▼
   embed()           ◄── the SAME function/model as above
          │
          ▼
   384-number query vector
          │
          ▼
   SELECT ... ORDER BY embedding <=> queryVector LIMIT 3   ──►  PostgreSQL
          │
          ▼
   top 3 most similar chunks     ◄── `npm run search` stops here (Retrieve)
          │
          ▼
   buildPrompt()     (src/ask.ts — chunks wrapped in <chunk id="n"> tags,
          │           grounding rules, the question last)
          ▼
   llama3.2          (local LLM served by Ollama at localhost:11434)
          │
          ▼
   grounded, cited answer — or "I don't know" if the chunks lack it
```

The single most important rule: **the query and the documents must be embedded by
the exact same model**, or their vectors live in different "spaces" and the
distances are meaningless. That's why `search.ts` imports `embed()` from
`embed.ts` rather than duplicating it.

---

## 5. Tech stack — what and why

| Tool | Role | Why this choice |
|------|------|-----------------|
| **TypeScript** | Language | Types make the shapes (a vector is `number[]`, a row is `{ content, distance }`) explicit and catch mistakes at compile time. |
| **Node.js** | Runtime | Runs the scripts; huge ecosystem for DB + ML libraries. |
| **tsx** | Runs TS directly | No separate build step — `tsx src/embed.ts` just runs the TypeScript. Fast feedback while learning. |
| **@xenova/transformers** (Transformers.js) | Embedding model, **local** | Runs `all-MiniLM-L6-v2` entirely on your machine. **No API key, no cost, no rate limits, works offline.** Perfect for learning. Downloads ~90 MB once, then cached. |
| **all-MiniLM-L6-v2** | The embedding model | A small, proven sentence-embedding model. Outputs **384 dimensions**; reads ~256 word-pieces of input, which is what sets the chunk size. |
| **PostgreSQL** | Database | A real, production-grade database you likely already know — no special vector DB needed. |
| **pgvector** | Postgres extension | Adds a `vector` column type and distance operators (`<->`, `<=>`) so similarity search is just SQL. |
| **pg** | Node ↔ Postgres driver | Standard, well-documented PostgreSQL client for Node. |
| **dotenv** | Config loading | Loads `DATABASE_URL` from `.env` so connection details aren't hard-coded. |
| **Ollama** | Local LLM runtime | Serves open-weight models over HTTP at `localhost:11434`. **No API key, no cost, works offline** — same ethos as the embeddings. |
| **llama3.2 (3B)** | The generation model | Small enough for a laptop CPU, capable enough to answer from 3 supplied chunks. Called with plain `fetch` — no SDK dependency. |

Chunking has **no dependency** — it's ~120 lines of plain string handling in
`src/chunk.ts`. Libraries exist (LangChain's `RecursiveCharacterTextSplitter` is
the well-known one), but the whole idea is small enough to write yourself, and
writing it yourself is how you learn what the parameters actually do.

> **No paid dependencies.** An earlier version used the OpenAI embedding API;
> that package is gone. The entire pipeline — embeddings, database, LLM — now
> runs locally with no API key.

### Why local embeddings instead of a paid API?

This is a concept-learning project, not production. The local model gives you the
identical RAG pipeline (chunk → embed → store → search) at **zero cost and zero
setup friction**. The only trade-off is a smaller vector size (384 vs. OpenAI's
1536) and slightly lower quality — completely irrelevant at this scale. For
production with thousands of documents, a hosted model like OpenAI
`text-embedding-3-small` would be a reasonable upgrade (and would only require
changing `embed()` and the column dimension).

---

## 6. Project structure

```
vector-search-demo/
├── src/
│   ├── chunk.ts      # Splits a document into overlapping chunks. Exports chunkText().
│   ├── embed.ts      # OFFLINE: chunks README.md, embeds each chunk, stores it. Exports embed().
│   ├── search.ts     # ONLINE: embeds a query + finds the top 3 matching chunks. Exports search().
│   ├── ask.ts        # ONLINE: full RAG — retrieval + a grounded answer from a local LLM.
│   └── snippets.ts   # The OLD hand-written corpus. No longer used — kept for reference.
├── .env              # DATABASE_URL (no API key needed). Not committed.
├── .gitignore        # ignores node_modules/ and .env
├── package.json      # deps + `npm run embed` / `npm run search` scripts
├── tsconfig.json     # TypeScript config
├── README.md         # this file — and also the document being indexed
└── DOCUMENTATION.md  # deep-dive docs: every file and function explained
```

`snippets.ts` is deliberately left in place: comparing it with the chunks that
come out of `chunk.ts` is the clearest way to see what chunking automates. You
can delete it with no effect on the pipeline.

The database table `documents` lives inside PostgreSQL (not in this folder):

```
Table "public.documents"
  Column   |    Type     | notes
-----------+-------------+---------------------------
 id        | integer     | auto-incrementing primary key
 content   | text        | the chunk text
 embedding | vector(384) | the chunk's embedding
```

---

## 7. Prerequisites

- **Node.js** (v18+; v20+ recommended)
- **PostgreSQL** (v14+ used here)
- **pgvector** extension installed for your PostgreSQL version
- **Ollama** with the `llama3.2` model (only for `npm run ask` — search works without it)

---

## 8. Setup from scratch

If you're recreating this on a fresh machine:

### 8.1 Install and start PostgreSQL (macOS / Homebrew)

```bash
brew install postgresql@14
brew services start postgresql@14
```

### 8.2 Install pgvector

The Homebrew `pgvector` bottle may only ship files for the newest Postgres
versions. If `CREATE EXTENSION vector` fails with "could not open extension
control file", build it from source against your Postgres:

```bash
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git /tmp/pgvector
cd /tmp/pgvector
make        PG_CONFIG=/opt/homebrew/opt/postgresql@14/bin/pg_config
make install PG_CONFIG=/opt/homebrew/opt/postgresql@14/bin/pg_config
```

### 8.3 Create the database and table

```bash
psql -d postgres -c "CREATE DATABASE vector_demo;"
psql -d vector_demo -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d vector_demo -c "
  CREATE TABLE documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT NOT NULL,
    embedding VECTOR(384)
  );
"
```

> The `384` must match the embedding model's output size. If you switch models,
> change this number to match (and re-run the embedding step).

### 8.4 Install Node dependencies

```bash
cd ai-practice/vector-search-demo
npm install
```

### 8.5 Configure `.env`

```
DATABASE_URL=postgres://localhost:5432/vector_demo
```

No API key of any kind — every model in the pipeline runs locally.

### 8.6 Install Ollama and pull the LLM

```bash
brew install ollama
brew services start ollama   # or run `ollama serve` in a terminal
ollama pull llama3.2         # ~2 GB, downloaded once
```

---

## 9. How to use it

### Step 1 — Build the knowledge base (offline, run once)

```bash
npm run embed
```

This reads the README, chunks it, embeds every chunk, and stores them. Expected
output:

```
Read 29520 characters from README.md
Split into 40 chunks (target 900 chars, 150 overlap)

Embedding 40 chunks with Xenova/all-MiniLM-L6-v2...

✅ Vector Search Demo — a minimal RAG retr…    824 chars -> [-0.035, 0.014, -0.005, ...] (384 dims)
✅ Table of contents                          1025 chars -> [-0.065, -0.007, 0.052, ...] (384 dims)
✅ 1. What this project does                   935 chars -> [-0.048, 0.031, -0.021, ...] (384 dims)
...
Done. Every chunk of README.md now has a vector in the documents table.
```

Re-run it any time the README changes — it `TRUNCATE`s the table first, so no
duplicates. **The chunk count and the numbers above will differ from what you
see**, because editing this README changes the very document being indexed.

### Step 2 — Search (online, any time)

```bash
npm run search "how do I install pgvector on a mac?"
```

Example output (truncated):

```
🔎 Query: "how do I install pgvector on a mac?"

Top 3 most similar chunks:

1. [distance 0.3676]  Step 2 — Search (online, any time)
   ...
2. [distance 0.4177]  8. Setup from scratch
   ### 8.1 Install and start PostgreSQL (macOS / Homebrew)

   brew install postgresql@14
   ...
   ### 8.2 Install pgvector

   The Homebrew `pgvector` bottle may only ship files for the newest Postgres
   versions. If `CREATE EXTENSION vector` fails with...
```

Try any question about this project:

```bash
npm run search "what does the overlap between chunks do"
# -> section 3 "Choosing size and overlap", distance ~0.45

npm run search "how do I bake sourdough bread"
# -> nothing relevant: every distance above 0.8
```

**Notice how much tighter the distances got.** With the hand-written snippets,
the best match for a good query was ~0.58. With chunks, well-targeted queries
land in the 0.34–0.45 range — because a chunk's vector represents *one* topic
instead of averaging eight.

### Step 3 — Ask (the full RAG loop, any time)

```bash
npm run ask "how do I choose the chunk size and overlap?"
```

This runs the same retrieval as Step 2, then hands the top 3 chunks to
`llama3.2` with strict grounding rules and streams the answer:

```
🔎 Question: "how do I choose the chunk size and overlap?"
📚 Retrieved 3 chunks:
   [1] (distance 0.5345) 4. Architecture & data flow
   [2] (distance 0.5461) start a new one. Because we only ever a…
   [3] (distance 0.5866) Choosing size and overlap

🤖 llama3.2 (local via Ollama):

According to chunk 3, ... Chunk size: 900 chars (~225 tokens),
Overlap: 150 chars (~17%) ... [3]

— sources: [1] 4. Architecture & data flow  ·  [2] …  ·  [3] Choosing size and overlap
```

The grounding is testable: ask something this README doesn't cover, and the
model refuses to fall back on its training knowledge:

```bash
npm run ask "what is the capital of France?"
# -> "I don't know based on the indexed documents. The context does not
#     provide information about the capital of France."
```

llama3.2 obviously *knows* the answer — the refusal proves the response comes
from the retrieved chunks, not from the model's memory.

### A self-reference artifact worth understanding

Look at the top hit above. The best match for *"how do I install pgvector on a
mac?"* is **not** the installation instructions — it's this very section, because
the code block a few lines up contains that exact question as an example. The
chunk that literally *asks* the question beat the chunk that *answers* it,
~0.36 to ~0.42.

That's not a bug in the pipeline; it's embeddings doing precisely what they're
built to do. A passage quoting a question is genuinely, semantically close to
that question. But it's useless as an answer, and it's a real failure mode in
production RAG: FAQ pages, changelogs, and support-ticket archives are full of
text that echoes user questions without resolving them.

Two standard mitigations, neither implemented here:

- **Retrieve more, then re-rank.** Pull the top 10 by vector distance, then score
  them with a cross-encoder or an LLM on *"does this actually answer the
  question?"* — a judgement pure vector distance cannot make.
- **Exclude the wrong material at index time.** Don't chunk the sections that
  only demonstrate queries.

It also explains why the numbers in this README drift: the examples are part of
the corpus they describe. Change one and you change the results it reports.

### Inspect the database directly

```bash
psql -d vector_demo -c "SELECT id, left(content,40), vector_dims(embedding) FROM documents;"
```

---

## 10. Code walkthrough

### `src/chunk.ts` — cut the document into pieces

The only new file. It exports one function plus its two tuning constants:

```ts
export const CHUNK_SIZE = 900;     // target characters per chunk
export const CHUNK_OVERLAP = 150;  // characters repeated from the previous chunk

export function chunkText(text: string, options: ChunkOptions = {}): string[]
```

It works in three passes.

**Pass 1 — split into blocks at paragraph boundaries.** Walk the document line by
line and cut at blank lines. The `inFence` flag is the important detail: it flips
every time a ` ``` ` line goes by, and while it's `true` blank lines are *not*
treated as breaks, so code samples survive intact.

```ts
for (const line of text.split("\n")) {
  if (line.trimStart().startsWith("```")) inFence = !inFence;

  if (line.trim() === "" && !inFence) {
    flush();            // end the current block
  } else {
    buffer.push(line);
  }
}
```

**Pass 2 — greedily pack blocks into chunks.** Keep adding whole paragraphs to
the current chunk; the moment one would push it past `size`, close the chunk and
start a new one. Because we only ever append *whole* blocks, a chunk can never end
mid-sentence.

```ts
for (const block of blocks) {
  for (const piece of block.length > size ? hardSplit(block, size) : [block]) {
    if (currentLength > 0 && currentLength + piece.length > size) flush();

    current.push(piece);
    currentLength += piece.length + 2; // +2 for the "\n\n" join
  }
}
```

The inner loop handles the edge case: a single block that's already larger than
the target (a long code fence, say) can't be packed at all, so `hardSplit()` cuts
it at line boundaries first. That guarantees the packing loop only ever sees
pieces small enough to fit.

**Pass 3 — add the overlap.** Prepend the tail of each chunk to the one after it,
snapped forward to the next line break so the tail reads as whole lines:

```ts
return chunks.map((chunk, i) => {
  if (i === 0) return chunk;                     // nothing precedes the first

  let tail = chunks[i - 1].slice(-overlap);
  const breakAt = tail.indexOf("\n");
  if (breakAt !== -1) tail = tail.slice(breakAt + 1);  // drop the partial line

  return `${tail.trim()}\n\n${chunk}`;
});
```

There's also `chunkLabel()`, a small helper that finds the nearest markdown
heading in a chunk. It's used purely for readable console output — it has no role
in the pipeline.

### `src/embed.ts` — build the knowledge base (and export `embed`)

**Where the corpus comes from.** This is the only line that changed conceptually:
the corpus is now a file on disk rather than an array in a `.ts` file.

```ts
const SOURCE_DOCUMENT = new URL("../README.md", import.meta.url);
```

Point it at any text or markdown file and nothing else in the pipeline needs to
change.

**Loading the model (once):**

```ts
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const extractorPromise = pipeline("feature-extraction", EMBEDDING_MODEL);
```

`pipeline(...)` downloads (first run) and loads the model. It returns a **promise**
we keep and `await` whenever we embed. Loading once and reusing is far faster than
reloading per call.

**Turning text into a vector:**

```ts
export async function embed(text: string): Promise<number[]> {
  const extractor = await extractorPromise;
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}
```

- `pooling: "mean"` averages the model's per-word vectors into a single
  chunk-level vector.
- `normalize: true` scales the vector to **length 1**. This is important: with
  unit-length vectors, cosine distance is clean and well-behaved (see
  [section 11](#11-how-similarity-search-works)).
- `output.data` is a `Float32Array`; we convert it to a plain `number[]`.
- `export` — so `search.ts` can reuse the **exact same** embedding logic.

**Formatting for pgvector:**

```ts
function toVectorLiteral(embedding: number[]): string {
  return JSON.stringify(embedding); // [0.1,0.2,...] — exactly what pgvector wants
}
```

**Chunking, embedding, and storing:**

```ts
async function main() {
  const db = new Client({ connectionString: process.env.DATABASE_URL });
  await db.connect();
  await db.query("TRUNCATE documents RESTART IDENTITY"); // start clean

  // --- THE NEW STEP: one document in, many chunks out. ---
  const document = await readFile(SOURCE_DOCUMENT, "utf8");
  const chunks = chunkText(document);

  // --- UNCHANGED: a chunk is just a string, like a snippet was. ---
  for (const content of chunks) {
    const embedding = await embed(content);              // text -> 384 numbers
    await db.query(
      "INSERT INTO documents (content, embedding) VALUES ($1, $2)",
      [content, toVectorLiteral(embedding)]              // store text + vector
    );
  }
  await db.end();
}
```

Compare this to the previous version and the point of the exercise becomes
obvious: **two lines were inserted at the top of the loop, and nothing else
moved.** Chunking is a preprocessing step, not a change to the architecture.

**Run-only-when-called guard:**

```ts
import { fileURLToPath } from "node:url";
const isDirectRun = process.argv[1] === fileURLToPath(import.meta.url);
if (isDirectRun) {
  main().catch((err) => { console.error(err); process.exit(1); });
}
```

This ensures the full indexing (`main`) runs **only** when you execute
`npm run embed`. When `search.ts` imports `embed`, this block is skipped — so a
search doesn't accidentally re-index everything.

### `src/search.ts` — answer a query

Functionally unchanged from the snippet version. It embeds the query with the
same `embed()` and lets Postgres rank every stored vector against it:

```ts
import { embed } from "./embed.js"; // SAME embedder used to index the documents
export const TOP_K = 3;

export async function search(query: string, topK: number = TOP_K) {
  const db = new Client({ connectionString: process.env.DATABASE_URL });
  await db.connect();

  const queryVector  = await embed(query);          // query -> 384 numbers
  const queryLiteral = JSON.stringify(queryVector); // '[...]' for pgvector

  const result = await db.query(
    `SELECT content,
            embedding <=> $1::vector AS distance
     FROM documents
     ORDER BY distance
     LIMIT $2`,
    [queryLiteral, TOP_K]
  );

  await db.end();
  return result.rows as { content: string; distance: number }[];
}
```

`search()` is exported (with the same direct-run guard as `embed.ts`) so that
`ask.ts` can reuse retrieval without triggering the CLI. Printing uses
`chunkLabel()` for the heading and indents the multi-line chunk body.

### `src/ask.ts` — generate the answer

The generation half of RAG. It reuses `search()` for retrieval, then does three
things:

**1. Prompt + Context.** The chunks are wrapped in tagged blocks so the model
can tell them apart and cite them by id, with the question placed last:

```ts
function buildPrompt(question: string, hits: SearchHit[]): string {
  const context = hits
    .map((hit, i) => `<chunk id="${i + 1}">\n${hit.content}\n</chunk>`)
    .join("\n\n");
  return `<context>\n${context}\n</context>\n\nQuestion: ${question}`;
}
```

**2. Grounding rules.** A fixed system prompt orders the model to base every
claim on the context, cite chunks inline as [1]/[2], and say *"I don't know
based on the indexed documents"* instead of guessing. This is the line between
RAG and a plain chatbot: the model's own knowledge is deliberately off-limits.

**3. The LLM call.** Plain `fetch` to Ollama's `/api/chat` on localhost — no
SDK. With `stream: true`, Ollama replies with NDJSON (one JSON object per
line, each carrying a few tokens); the code buffers partial lines and prints
fragments as they arrive, so the answer streams word by word:

```ts
const response = await fetch("http://localhost:11434/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "llama3.2",
    stream: true,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: buildPrompt(question, hits) },
    ],
  }),
});
```

The retrieved chunks and their distances are printed *before* the answer on
purpose: a bad answer is then immediately diagnosable — wrong chunks means a
retrieval problem; right chunks but a wrong reading means a generation problem.

---

## 11. How similarity search works

The heart of the search is one line of SQL:

```sql
ORDER BY embedding <=> $1::vector   -- sort chunks by distance to the query
LIMIT 3                             -- keep only the closest three
```

Read it as English: **"Sort chunks by vector distance; return the closest
three."** PostgreSQL computes the distance from the query vector to every row's
embedding, sorts smallest-first, and returns the top matches. **Smaller distance =
more similar meaning.**

### The distance operators (pgvector)

| Operator | Metric | Meaning |
|----------|--------|---------|
| `<->` | Euclidean (L2) | Straight-line distance between vector tips |
| `<=>` | Cosine | Angle between the vectors |
| `<#>` | Negative inner product | — |

This project uses **`<=>` (cosine)**, the conventional choice for text embeddings.

**Important:** because `embed()` sets `normalize: true` (all vectors have length
1), **`<->` and `<=>` produce the same ranking.** They're mathematically linked
for unit vectors (`euclidean² = 2 × cosine_distance`). So if your notes use `<->`,
you'll get identical top-3 results. Only the distance *numbers* differ, not the
order.

### Reading the distances

Rough guide for this model and corpus:

| Distance | Interpretation |
|----------|----------------|
| **< 0.5** | Strong match — the chunk almost certainly answers the query |
| **0.5 – 0.7** | Related; often useful context |
| **> 0.7** | Weak. Frequently means *the document has no answer*, and you're seeing the least-bad chunk |

That last row is the one to watch. Similarity search **always** returns something —
`ORDER BY ... LIMIT 3` can't return "nothing relevant." Asking this corpus
*"how do I bake sourdough bread"* still returns three chunks; they just sit
above 0.8. Nothing in the *ranking* tells you the answer isn't
there — only the absolute distance does. `ask.ts` defends against this with
its grounding prompt — the LLM says "I don't know" when the chunks don't
contain the answer — but a hard relevance threshold *before* prompting is the
stronger fix (see [section 14](#14-where-to-go-next)).

---

## 12. Why it works (the semantic magic)

Consider:

- **Chunk:** section 8.2, *"The Homebrew `pgvector` bottle may only ship files for
  the newest Postgres versions... build it from source"*
- **Query:** *"how do I install pgvector on a mac?"*

The words "install" and "Mac" barely appear in that chunk — it says "Homebrew",
"bottle", "build from source". Yet it's the highest-ranked chunk that actually
answers the question, at distance ~0.42. (The chunk above it merely quotes the
question — see [section 9](#a-self-reference-artifact-worth-understanding).)

Why? The embedding model learned during training that these concepts are related:

```
install  ↔  Homebrew  ↔  brew  ↔  build from source  ↔  macOS
```

So both texts produce vectors pointing in nearly the **same direction** → small
distance → top match. **That semantic understanding — matching meaning, not
words — is the entire reason vector search (and RAG) works.**

---

## 13. Troubleshooting

| Problem | Cause / Fix |
|---------|-------------|
| `could not open extension control file ".../vector.control"` | pgvector isn't installed for your Postgres version. Build from source (see [8.2](#82-install-pgvector)). |
| `expected 384 dimensions, not 1536` on insert | The `documents.embedding` column size doesn't match the model. `ALTER TABLE documents ALTER COLUMN embedding TYPE vector(384);` |
| `ECONNREFUSED` connecting to Postgres | Postgres isn't running. `brew services start postgresql@14`. Check `DATABASE_URL` in `.env`. |
| First `npm run embed` is slow / downloads a lot | Expected — it downloads the ~90 MB model once, then caches it. Subsequent runs are fast and offline. |
| Search results are stale / don't reflect a README edit | The README **is** the corpus. Re-run `npm run embed` after editing it. |
| Every result has distance > 0.7 | Usually correct behaviour: the document has no answer to that query. See [section 11](#reading-the-distances). |
| A result starts mid-sentence | That's the 150-char overlap from the previous chunk, working as intended. |
| Top hit is an example *asking* the question, not answering it | Expected — see [section 9](#a-self-reference-artifact-worth-understanding). Fix by re-ranking, or by not indexing example blocks. |
| Chunks look too big or too small | Tune `CHUNK_SIZE` / `CHUNK_OVERLAP` in `src/chunk.ts`, then re-run `npm run embed`. Going far above ~1000 chars will exceed MiniLM's input window and silently truncate. |

---

## 14. Where to go next

The full RAG loop now runs end-to-end. To extend it:

- ~~**Complete the RAG loop (add "Generation")**~~ — **done**: `src/ask.ts`
  feeds the top-3 chunks to a local LLM under grounding rules. Its own next
  steps: pair it with the threshold below, or swap `llama3.2` for a bigger
  model — everything model-specific lives in one `generate()` function.
- **Add a relevance threshold:** drop results above ~0.7 distance, so an
  unanswerable query returns "no good match" instead of the least-bad chunk.
- **Store chunk metadata:** add `source` and `chunk_index` columns so results can
  cite *where* they came from ("README.md § 8.2"). Essential once the corpus has
  more than one file.
- **Index more documents:** the pipeline is file-agnostic. Glob a directory,
  chunk every file, and store them all — nothing else changes.
- **Re-rank the results** with a cross-encoder or an LLM, to fix the
  question-matches-question artifact described in [section 9](#a-self-reference-artifact-worth-understanding).
- **Add a vector index** for speed on large corpora:
  `CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);`
  (Irrelevant at 40 rows; essential at 100k+.)
- **Try a bigger model / hosted API** (e.g. OpenAI `text-embedding-3-small`,
  1536 dims) if you want higher-quality embeddings and a larger input window —
  change `embed()` and the column dimension.
