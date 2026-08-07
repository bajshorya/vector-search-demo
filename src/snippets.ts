// Step 5 — The Corpus.
//
// This is the collection of documents we want to make searchable.
// Just plain text for now — no embeddings yet. Later, embed.ts will turn each
// of these into a vector and store it in the `documents` table.
//
// Keep each snippet 2–4 sentences. Aim for 8–10 total.
// Replace the placeholders below with real projects from your resume.

export const snippets: string[] = [
  // --- examples from your notes ---
  `Mini Redis. An in-memory key-value database supporting TTL, persistence, and pub/sub. Built in TypeScript to explore how real caches handle expiry and durability.`,

  `Idea Radar. An application that recommends startup ideas using market trends. It scrapes signals from public sources and ranks opportunities by momentum.`,

  `Collab Docs. A real-time collaborative document editor using WebSockets. Multiple users can edit the same document simultaneously with live cursors and conflict resolution.`,

  `Memoize Utility. A higher-order JavaScript function that caches results of a pure function by its arguments using a closure over a null-prototype cache object. Tested against a naive recursive Fibonacci to show repeated calls return instantly instead of recomputing.`,

  `Debounce Helper. A fully typed TypeScript debounce that delays running a function until calls stop arriving for a set interval, cancelling any pending run on each new call. Uses generics (Parameters<F>) so the wrapper preserves the original function's argument types.`,

  `Closure Loop-Bug Demo. A teaching example reproducing the classic var loop-capture bug where every closure reads the same final counter value, then fixing it two ways: block-scoped let, and a var-based IIFE that captures a fresh copy per iteration.`,

  `Prototype Chain Explorer. A no-class inheritance demo building a three-level animal to dog to puppy chain with Object.create, each level overriding or adding one method. Uses getPrototypeOf and hasOwnProperty to prove where each property lives along the chain.`,

  `Vector Search Demo. A small retrieval-augmented-generation pipeline that embeds a corpus of project snippets and stores the vectors in PostgreSQL with pgvector. A user query is embedded and compared by vector similarity to return the top matching snippets.`,
];
