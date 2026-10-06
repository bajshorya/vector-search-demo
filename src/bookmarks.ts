// A tiny bookmarks store — the data source behind the get_bookmarks tool.
//
// There was no real bookmarks store in the project yet, so this mocks one with a
// fixed in-memory list. It's deliberately plain: a `Bookmark[]` and one lookup
// function. If a real store (a DB table, an API) shows up later, only the body
// of getBookmarks has to change — the tool wiring in tool-ask.ts stays the same.

export interface Bookmark {
  title: string;
  url: string;
  tags: string[];
}

// `let` (not `const`) isn't needed — we mutate the array in place with push —
// but the list is no longer frozen: add_bookmark appends to it at runtime.
const BOOKMARKS: Bookmark[] = [
  {
    title: "Ollama tool calling docs",
    url: "https://github.com/ollama/ollama/blob/main/docs/api.md#tools",
    tags: ["ollama", "llm", "tools"],
  },
  {
    title: "pgvector: open-source vector similarity search for Postgres",
    url: "https://github.com/pgvector/pgvector",
    tags: ["postgres", "vector", "rag"],
  },
  {
    title: "Chunking strategies for RAG",
    url: "https://www.pinecone.io/learn/chunking-strategies/",
    tags: ["rag", "chunking"],
  },
  {
    title: "TypeScript handbook: everyday types",
    url: "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html",
    tags: ["typescript"],
  },
  {
    title: "The Result type pattern in TypeScript",
    url: "https://imhoff.blog/posts/using-results-in-typescript",
    tags: ["typescript", "errors"],
  },
];

/**
 * Return every bookmark carrying `tag` (case-insensitive). An unknown tag just
 * yields an empty list — that's a valid answer, not an error, so the caller (and
 * the model) can say "nothing bookmarked under that tag".
 */
export function getBookmarks(tag: string): Bookmark[] {
  const needle = tag.trim().toLowerCase();
  return BOOKMARKS.filter((b) =>
    b.tags.some((t) => t.toLowerCase() === needle),
  );
}

/**
 * Save a new bookmark and return it. Tags are lowercased and de-duplicated so
 * getBookmarks stays case-insensitive. A duplicate URL is rejected (thrown) so
 * the tool layer can turn it into a readable tool_result error.
 */
export function addBookmark(
  url: string,
  title: string,
  tags: string[],
): Bookmark {
  if (BOOKMARKS.some((b) => b.url === url)) {
    throw new Error(`A bookmark with url "${url}" already exists.`);
  }
  const normalizedTags = [
    ...new Set(tags.map((t) => t.trim().toLowerCase()).filter(Boolean)),
  ];
  const bookmark: Bookmark = { title, url, tags: normalizedTags };
  BOOKMARKS.push(bookmark);
  return bookmark;
}

/** How many bookmarks are stored right now. */
export function countBookmarks(): number {
  return BOOKMARKS.length;
}
