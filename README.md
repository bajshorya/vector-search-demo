# Vector Search Demo — a minimal RAG retrieval pipeline

A small, **fully free, fully local** project that demonstrates the core of a
**RAG (Retrieval-Augmented Generation)** system: how to make a collection of text
documents **searchable by meaning** instead of by keywords.

You give it a natural-language question like *"How can I build a caching server?"*
and it returns the most **semantically similar** documents — even when they share
no words with the question.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [The big idea: semantic search](#2-the-big-idea-semantic-search)
3. [Architecture & data flow](#3-architecture--data-flow)
4. [Tech stack — what and why](#4-tech-stack--what-and-why)
5. [Project structure](#5-project-structure)
6. [Prerequisites](#6-prerequisites)
7. [Setup from scratch](#7-setup-from-scratch)
8. [How to use it](#8-how-to-use-it)
9. [Code walkthrough](#9-code-walkthrough)
10. [How similarity search works](#10-how-similarity-search-works)
11. [Why it works (the semantic magic)](#11-why-it-works-the-semantic-magic)
12. [Troubleshooting](#12-troubleshooting)
13. [Where to go next](#13-where-to-go-next)

---

## 1. What this project does

The project is split into two phases, matching how every RAG system works.

**Offline (run once) — build the knowledge base:**

- Take 8–10 short text snippets (the "corpus").
- Convert each into an **embedding** — a list of 384 numbers that represents its
  meaning.
- Store the text + its embedding in a PostgreSQL table.

**Online (every search) — answer a query:**

- Take a user's question.
- Convert it into an embedding using the **same** model.
- Ask PostgreSQL for the documents whose embeddings are **closest** to the
  question's embedding.
- Return the top 3.

That's the "R" (Retrieval) in RAG. This project stops at retrieval — it finds the
right documents. Adding an LLM to *generate* an answer from them would complete
the full RAG loop (see [section 13](#13-where-to-go-next)).

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

## 3. Architecture & data flow

```
OFFLINE  (run once: `npm run embed`)
────────────────────────────────────
   src/snippets.ts   (8 text snippets = the corpus)
          │
          ▼
   embed()           (local model: all-MiniLM-L6-v2)
          │
          ▼
   384-number vector per snippet
          │
          ▼
   INSERT INTO documents (content, embedding)     ──►  PostgreSQL + pgvector
                                                        (the knowledge base)

ONLINE  (every search: `npm run search "..."`)
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
   top 3 most similar snippets
```

The single most important rule: **the query and the documents must be embedded by
the exact same model**, or their vectors live in different "spaces" and the
distances are meaningless. That's why `search.ts` imports `embed()` from
`embed.ts` rather than duplicating it.

---

## 4. Tech stack — what and why

| Tool | Role | Why this choice |
|------|------|-----------------|
| **TypeScript** | Language | Types make the shapes (a vector is `number[]`, a row is `{ content, distance }`) explicit and catch mistakes at compile time. |
| **Node.js** | Runtime | Runs the scripts; huge ecosystem for DB + ML libraries. |
| **tsx** | Runs TS directly | No separate build step — `tsx src/embed.ts` just runs the TypeScript. Fast feedback while learning. |
| **@xenova/transformers** (Transformers.js) | Embedding model, **local** | Runs `all-MiniLM-L6-v2` entirely on your machine. **No API key, no cost, no rate limits, works offline.** Perfect for learning. Downloads ~90 MB once, then cached. |
| **all-MiniLM-L6-v2** | The embedding model | A small, proven sentence-embedding model. Outputs **384 dimensions**. More than good enough to rank 8 short snippets clearly. |
| **PostgreSQL** | Database | A real, production-grade database you likely already know — no special vector DB needed. |
| **pgvector** | Postgres extension | Adds a `vector` column type and distance operators (`<->`, `<=>`) so similarity search is just SQL. |
| **pg** | Node ↔ Postgres driver | Standard, well-documented PostgreSQL client for Node. |
| **dotenv** | Config loading | Loads `DATABASE_URL` from `.env` so connection details aren't hard-coded. |

> **Note:** `openai` is still listed in `package.json` from an earlier version
> that used the OpenAI embedding API. It is **no longer used** — embeddings are
> now fully local. You can remove it with `npm uninstall openai`.

### Why local embeddings instead of a paid API?

This is a concept-learning project, not production. The local model gives you the
identical RAG pipeline (embed → store → search) at **zero cost and zero setup
friction**. The only trade-off is a smaller vector size (384 vs. OpenAI's 1536)
and slightly lower quality — completely irrelevant at this scale. For production
with thousands of documents, a hosted model like OpenAI `text-embedding-3-small`
would be a reasonable upgrade (and would only require changing `embed()` and the
column dimension).

---

## 5. Project structure

```
vector-search-demo/
├── src/
│   ├── snippets.ts   # The corpus: an array of 8 text snippets to search.
│   ├── embed.ts      # OFFLINE: embeds each snippet + stores it. Exports embed().
│   └── search.ts     # ONLINE: embeds a query + finds the top 3 matches.
├── .env              # DATABASE_URL (no API key needed).
├── .gitignore        # ignores node_modules/ and .env
├── package.json      # deps + `npm run embed` / `npm run search` scripts
├── tsconfig.json     # TypeScript config
└── README.md         # this file
```

The database table `documents` lives inside PostgreSQL (not in this folder):

```
Table "public.documents"
  Column   |    Type     | notes
-----------+-------------+---------------------------
 id        | integer     | auto-incrementing primary key
 content   | text        | the snippet text
 embedding | vector(384) | the snippet's embedding
```

---

## 6. Prerequisites

- **Node.js** (v18+; v20+ recommended)
- **PostgreSQL** (v14+ used here)
- **pgvector** extension installed for your PostgreSQL version

---

## 7. Setup from scratch

If you're recreating this on a fresh machine:

### 7.1 Install and start PostgreSQL (macOS / Homebrew)

```bash
brew install postgresql@14
brew services start postgresql@14
```

### 7.2 Install pgvector

The Homebrew `pgvector` bottle may only ship files for the newest Postgres
versions. If `CREATE EXTENSION vector` fails with "could not open extension
control file", build it from source against your Postgres:

```bash
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git /tmp/pgvector
cd /tmp/pgvector
make        PG_CONFIG=/opt/homebrew/opt/postgresql@14/bin/pg_config
make install PG_CONFIG=/opt/homebrew/opt/postgresql@14/bin/pg_config
```

### 7.3 Create the database and table

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

### 7.4 Install Node dependencies

```bash
cd ai-practice/vector-search-demo
npm install
```

### 7.5 Configure `.env`

```
DATABASE_URL=postgres://localhost:5432/vector_demo
```

---

## 8. How to use it

### Step 1 — Build the knowledge base (offline, run once)

```bash
npm run embed
```

This embeds all snippets and stores them. Expected output:

```
Embedding 8 snippets with Xenova/all-MiniLM-L6-v2...

✅ Mini Redis               -> [-0.028, 0.010, -0.072, ...] (384 dims)
✅ Idea Radar               -> [-0.035, -0.081, -0.033, ...] (384 dims)
...
Done. Every snippet now has a vector stored in the documents table.
```

Re-run it any time you change `snippets.ts` — it `TRUNCATE`s the table first, so
no duplicates.

### Step 2 — Search (online, any time)

```bash
npm run search "How can I build a caching server?"
```

Example output:

```
🔎 Query: "How can I build a caching server?"

Top 3 most similar snippets:

1. [distance 0.5847]  Mini Redis
   Mini Redis. An in-memory key-value database supporting TTL, persistence...

2. [distance 0.7100]  Memoize Utility
   Memoize Utility. A higher-order JavaScript function that caches results...

3. [distance 0.8108]  Collab Docs
   Collab Docs. A real-time collaborative document editor using WebSockets...
```

Try any question:

```bash
npm run search "javascript inheritance without using classes"
# -> top hit: Prototype Chain Explorer
```

### Inspect the database directly

```bash
psql -d vector_demo -c "SELECT id, left(content,25), vector_dims(embedding) FROM documents;"
```

---

## 9. Code walkthrough

### `src/snippets.ts` — the corpus

Just an exported array of strings. This is the raw material to be made
searchable. Keep each snippet 2–4 sentences; aim for 8–10.

```ts
export const snippets: string[] = [
  `Mini Redis. An in-memory key-value database supporting TTL, persistence, and pub/sub. ...`,
  `Idea Radar. An application that recommends startup ideas using market trends. ...`,
  // ...8 total
];
```

### `src/embed.ts` — build the knowledge base (and export `embed`)

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
  sentence-level vector.
- `normalize: true` scales the vector to **length 1**. This is important: with
  unit-length vectors, cosine distance is clean and well-behaved (see
  [section 10](#10-how-similarity-search-works)).
- `output.data` is a `Float32Array`; we convert it to a plain `number[]`.
- `export` — so `search.ts` can reuse the **exact same** embedding logic.

**Formatting for pgvector:**

```ts
function toVectorLiteral(embedding: number[]): string {
  return JSON.stringify(embedding); // [0.1,0.2,...] — exactly what pgvector wants
}
```

**Embedding + storing every snippet:**

```ts
async function main() {
  const db = new Client({ connectionString: process.env.DATABASE_URL });
  await db.connect();
  await db.query("TRUNCATE documents RESTART IDENTITY"); // start clean

  for (const content of snippets) {
    const embedding = await embed(content);              // text -> 384 numbers
    await db.query(
      "INSERT INTO documents (content, embedding) VALUES ($1, $2)",
      [content, toVectorLiteral(embedding)]              // store text + vector
    );
  }
  await db.end();
}
```

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

```ts
import { embed } from "./embed.js"; // SAME embedder used to index the documents
const TOP_K = 3;

async function search(query: string) {
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

The query is read from the command line (`process.argv`), embedded, and compared
against every stored vector by the SQL below.

---

## 10. How similarity search works

The heart of the search is one line of SQL:

```sql
ORDER BY embedding <=> $1::vector   -- sort documents by distance to the query
LIMIT 3                             -- keep only the closest three
```

Read it as English: **"Sort documents by vector distance; return the closest
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

---

## 11. Why it works (the semantic magic)

Consider:

- **Document:** "Mini Redis. An in-memory key-value store."
- **Query:** "How can I build a caching server?"

They share **no keywords** — "cache", "caching", and "server" don't appear in the
document. A keyword search would return nothing. Yet this project ranks Mini Redis
**first**.

Why? The embedding model learned during training that these concepts are related:

```
caching  ↔  in-memory  ↔  key-value  ↔  Redis  ↔  fast lookup
```

So both texts produce vectors pointing in nearly the **same direction** → small
distance → top match. **That semantic understanding — matching meaning, not
words — is the entire reason vector search (and RAG) works.**

---

## 12. Troubleshooting

| Problem | Cause / Fix |
|---------|-------------|
| `could not open extension control file ".../vector.control"` | pgvector isn't installed for your Postgres version. Build from source (see [7.2](#72-install-pgvector)). |
| `expected 384 dimensions, not 1536` on insert | The `documents.embedding` column size doesn't match the model. `ALTER TABLE documents ALTER COLUMN embedding TYPE vector(384);` |
| `ECONNREFUSED` connecting to Postgres | Postgres isn't running. `brew services start postgresql@14`. Check `DATABASE_URL` in `.env`. |
| First `npm run embed` is slow / downloads a lot | Expected — it downloads the ~90 MB model once, then caches it. Subsequent runs are fast and offline. |
| Search returns weird / unrelated results | Make sure you re-ran `npm run embed` after changing snippets, and that query + documents use the same model. |

---

## 13. Where to go next

This project implements **retrieval**. To extend it:

- **Complete the RAG loop (add "Generation"):** feed the top-3 snippets to an LLM
  as context and ask it to answer the user's question using them. That turns
  *retrieval* into *retrieval-augmented generation*.
- **Add an index** for speed on large corpora:
  `CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);`
  (Irrelevant at 8 rows; essential at 100k+.)
- **Add a relevance threshold:** ignore results with distance above, say, 0.9, so
  irrelevant queries return "no good match" instead of the least-bad snippet.
- **Grow the corpus:** the pipeline is identical whether you have 8 snippets or
  8,000.
- **Try a bigger model / hosted API** (e.g. OpenAI `text-embedding-3-small`,
  1536 dims) if you want higher-quality embeddings — change `embed()` and the
  column dimension.
```
