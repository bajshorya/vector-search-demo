// Day 4 concept, applied — a generic Result<T, E> type.
//
// Instead of throwing (which erases types and forces every caller to remember a
// try/catch), a function can *return* its failure as a value. The compiler then
// forces the caller to check `if (result.ok)` before touching `.value`, so an
// unhandled error becomes a type error at compile time instead of a crash at
// runtime.
//
//     Result<string, LLMError>   // "either a string, or an LLMError — never both"
//
// This is the same shape as Rust's Result and Go's (value, err) pair, written
// out in plain TypeScript. It's used by llm.ts and rerank.ts so the LLM calls
// (the parts most likely to fail) hand their errors back as data.

/** A success carrying a value, or a failure carrying an error. Exactly one. */
export type Result<T, E> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

/** Wrap a success value. */
export const ok = <T>(value: T): Result<T, never> => ({ ok: true, value });

/** Wrap a failure value. */
export const err = <E>(error: E): Result<never, E> => ({ ok: false, error });

//explain this code in simpler words:
//This code defines a generic type called `Result<T, E>` that can represent either a successful outcome or a failure.
//- If the operation is successful, it will have a property `ok` set to `true` and a `value` of type `T`.
//- If the operation fails, it will have a property `ok` set to `false` and an `error` of type `E`.

//The code also provides two helper functions:
//- `ok(value)` creates a successful result with the given value.
//- `err(error)` creates a failed result with the given error.
