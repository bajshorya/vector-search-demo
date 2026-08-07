// Step 9 — Chunking.
//
// Yesterday the corpus was hand-written: 8 snippets that were already the right
// size. A real document isn't. README.md is one ~15,000-character blob, and you
// cannot feed that to the embedder as a single unit:
//
//   1. Models have an input limit. all-MiniLM-L6-v2 only looks at the first
//      ~256 word-pieces and silently throws away the rest.
//   2. Even if it fit, one vector for a whole document is a *blurry average* of
//      every topic in it. "How do I install pgvector?" would match the whole
//      README weakly instead of matching the install section strongly.
//   3. Retrieval returns whole rows. Small rows = precise context for the LLM.
//
// So we cut the document into pieces first:
//
//     README.md  ->  chunkText()  ->  [chunk 1, chunk 2, chunk 3, ...]
//
// Everything downstream is unchanged: each chunk is just a string, exactly like
// a snippet was.

/** Target size of a chunk, in characters. */
export const CHUNK_SIZE = 900;

/**
 * How many characters of the previous chunk to repeat at the start of the next
 * one. Without overlap, a sentence that straddles a boundary gets cut in half
 * and *neither* chunk carries its full meaning. A little repetition is cheap
 * insurance — it costs a few extra tokens and saves answers from falling
 * through the cracks between chunks.
 */
export const CHUNK_OVERLAP = 150;

export interface ChunkOptions {
  size?: number;
  overlap?: number;
}

/**
 * Split a document into overlapping chunks of roughly `size` characters.
 *
 * The rule that makes this "smart" rather than a blind `slice()`: we only ever
 * cut at *natural boundaries* (blank lines between paragraphs, and never inside
 * a fenced code block). A chunk that ends mid-sentence embeds badly, because the
 * vector then represents half a thought.
 */
export function chunkText(text: string, options: ChunkOptions = {}): string[] {
  const size = options.size ?? CHUNK_SIZE;
  const overlap = options.overlap ?? CHUNK_OVERLAP;

  const blocks = splitIntoBlocks(text);
  const chunks: string[] = [];

  // `current` is the chunk we're currently filling up.
  let current: string[] = [];
  let currentLength = 0;

  const flush = () => {
    if (current.length === 0) return;
    chunks.push(current.join("\n\n"));
    current = [];
    currentLength = 0;
  };

  for (const block of blocks) {
    // A single block bigger than the target (a long code fence, say) can't be
    // packed — cut it down first so the loop below always sees small pieces.
    for (const piece of block.length > size ? hardSplit(block, size) : [block]) {
      // Adding this piece would overflow the target -> close the current chunk
      // and start a new one.
      if (currentLength > 0 && currentLength + piece.length > size) flush();

      current.push(piece);
      currentLength += piece.length + 2; // +2 for the "\n\n" join
    }
  }
  flush();

  return overlap > 0 ? addOverlap(chunks, overlap) : chunks;
}

/**
 * Cut the document at blank lines — the paragraph boundaries a human would use.
 *
 * The one exception is fenced code blocks (``` ... ```): a blank line *inside*
 * a fence is not a paragraph break, and splitting there would leave dangling
 * half-programs. So we track whether we're inside a fence and ignore blank
 * lines while we are.
 */
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

/**
 * Last resort for a block that is already too big on its own. Cut it at line
 * boundaries so we at least never split a line in half.
 */
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

/**
 * Prepend the tail of each chunk to the one after it, so context flows across
 * the seam. We trim the tail forward to the next line break to avoid starting a
 * chunk mid-sentence.
 */
function addOverlap(chunks: string[], overlap: number): string[] {
  return chunks.map((chunk, i) => {
    if (i === 0) return chunk;

    const previous = chunks[i - 1];
    let tail = previous.slice(-overlap);

    // Snap to a line boundary if there is one, so the tail reads as whole lines.
    const breakAt = tail.indexOf("\n");
    if (breakAt !== -1) tail = tail.slice(breakAt + 1);

    return `${tail.trim()}\n\n${chunk}`;
  });
}

/**
 * A short human-readable label for a chunk, used only for console output.
 * Prefers the nearest markdown heading; otherwise the first line of text.
 */
export function chunkLabel(chunk: string): string {
  const heading = chunk.split("\n").find((line) => line.startsWith("#"));
  const line = (heading ?? chunk.split("\n")[0] ?? "").replace(/^#+\s*/, "");
  return line.length > 40 ? `${line.slice(0, 39)}…` : line;
}
