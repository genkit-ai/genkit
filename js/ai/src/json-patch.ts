/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * A tiny, dependency-free RFC 6902 (JSON Patch) implementation.
 *
 * This module is intentionally self-contained and browser-safe (no Node APIs,
 * no runtime dependencies) so it can be shared by the in-process server agent
 * and the browser-facing agent client.
 *
 * Genkit uses JSON Patch to stream incremental changes to a session's custom
 * state (`AgentStreamChunk.customPatch`). The {@link diff} helper only emits
 * `add` / `remove` / `replace` operations (a valid RFC 6902 subset - `move` /
 * `copy` are optimizations we deliberately skip), while {@link applyPatch}
 * understands the full operation set for interoperability.
 *
 * @module json-patch
 */

/**
 * A single RFC 6902 (JSON Patch) operation.
 */
export interface JsonPatchOperation {
  op: 'add' | 'remove' | 'replace' | 'move' | 'copy' | 'test';
  /** A JSON Pointer (RFC 6901) to the target location, e.g. `"/agentStatus"`. */
  path: string;
  /** Source pointer; required for `move` and `copy`. */
  from?: string;
  /** New value; required for `add`, `replace`, and `test`. */
  value?: any;
}

/**
 * An RFC 6902 JSON Patch: an ordered list of operations.
 */
export type JsonPatch = JsonPatchOperation[];

/**
 * Escapes a single JSON Pointer reference token per RFC 6901 (`~` → `~0`,
 * `/` → `~1`).
 */
function escapeToken(token: string): string {
  return token.replace(/~/g, '~0').replace(/\//g, '~1');
}

/**
 * Unescapes a single JSON Pointer reference token per RFC 6901.
 */
function unescapeToken(token: string): string {
  return token.replace(/~1/g, '/').replace(/~0/g, '~');
}

/**
 * Reference tokens that could be used to pollute `Object.prototype` (or an
 * object's constructor) when walking into an existing object. We reject these
 * outright since patches may originate from untrusted, server-sent data.
 */
const FORBIDDEN_TOKENS = new Set(['__proto__', 'prototype', 'constructor']);

/**
 * Parses a JSON Pointer string into its reference tokens.
 *
 * The root pointer (`""`) parses to an empty array.
 *
 * Reference tokens that could lead to prototype pollution (`__proto__`,
 * `prototype`, `constructor`) are rejected.
 */
function parsePointer(pointer: string): string[] {
  if (pointer === '') return [];
  if (pointer[0] !== '/') {
    throw new Error(`Invalid JSON Pointer: "${pointer}" must start with "/".`);
  }
  const tokens = pointer.slice(1).split('/').map(unescapeToken);
  for (const token of tokens) {
    if (FORBIDDEN_TOKENS.has(token)) {
      throw new Error(
        `Invalid JSON Pointer: "${pointer}" contains forbidden token "${token}".`
      );
    }
  }
  return tokens;
}

/**
 * Returns `true` for values that are plain JSON objects (not arrays / null).
 */
function isObject(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Lists the keys of a plain object whose values are not `undefined`.
 *
 * JSON has no `undefined`: `JSON.stringify` drops such members entirely, so
 * `{ a: undefined }` and `{}` serialize identically. We mirror that here so
 * that diffing/equality match how state is actually persisted (and so we never
 * emit `undefined`-valued `add`/`replace` ops, which serialize to a valueless,
 * meaningless operation).
 */
function definedKeys(obj: Record<string, any>): string[] {
  return Object.keys(obj).filter((key) => obj[key] !== undefined);
}

/**
 * Deep structural equality for JSON-serializable values.
 *
 * Object members whose value is `undefined` are treated as absent, matching
 * JSON semantics (see {@link definedKeys}).
 */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }
  if (isObject(a) && isObject(b)) {
    const aKeys = definedKeys(a);
    const bKeys = definedKeys(b);
    if (aKeys.length !== bKeys.length) return false;
    for (const key of aKeys) {
      if (b[key] === undefined) return false;
      if (!deepEqual(a[key], b[key])) return false;
    }
    return true;
  }
  return false;
}

/**
 * Structured clone fallback that works across runtimes (uses the global
 * `structuredClone` when available, otherwise JSON round-trips).
 */
function clone<T>(value: T): T {
  if (value === undefined) return value;
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

/**
 * Computes an RFC 6902 JSON Patch that transforms `from` into `to`.
 *
 * The diff is rooted at the document, so pointers are bare (e.g. `/agentStatus`,
 * `/items/0`). Only `add` / `remove` / `replace` operations are emitted.
 *
 * When the two documents differ at the root in a way that cannot be expressed
 * as member-level changes (e.g. an object becomes an array, or a primitive
 * changes), a single whole-document `replace` at path `""` is returned.
 */
export function diff(from: unknown, to: unknown): JsonPatch {
  const patch: JsonPatch = [];
  diffRecursive(from, to, '', patch);
  return patch;
}

function diffRecursive(
  from: unknown,
  to: unknown,
  pointer: string,
  patch: JsonPatch
): void {
  if (deepEqual(from, to)) return;

  // Both plain objects - recurse member by member. Members whose value is
  // `undefined` are treated as absent (JSON has no `undefined`); this avoids
  // emitting valueless `add`/`replace` ops and correctly turns "set to
  // undefined" into a `remove`.
  if (isObject(from) && isObject(to)) {
    const keys = new Set([...Object.keys(from), ...Object.keys(to)]);
    for (const key of keys) {
      const childPointer = `${pointer}/${escapeToken(key)}`;
      const inFrom = from[key] !== undefined;
      const inTo = to[key] !== undefined;
      if (inFrom && !inTo) {
        patch.push({ op: 'remove', path: childPointer });
      } else if (!inFrom && inTo) {
        patch.push({ op: 'add', path: childPointer, value: clone(to[key]) });
      } else if (inFrom && inTo) {
        diffRecursive(from[key], to[key], childPointer, patch);
      }
    }
    return;
  }

  // Both arrays - recurse by index, then add/remove the tail difference.
  if (Array.isArray(from) && Array.isArray(to)) {
    const min = Math.min(from.length, to.length);
    for (let i = 0; i < min; i++) {
      diffRecursive(from[i], to[i], `${pointer}/${i}`, patch);
    }
    if (to.length > from.length) {
      for (let i = from.length; i < to.length; i++) {
        // Appends use the "-" end-of-array token per RFC 6902.
        patch.push({ op: 'add', path: `${pointer}/-`, value: clone(to[i]) });
      }
    } else if (from.length > to.length) {
      // Remove from the tail backwards so indices stay valid as we go.
      for (let i = from.length - 1; i >= to.length; i--) {
        patch.push({ op: 'remove', path: `${pointer}/${i}` });
      }
    }
    return;
  }

  // Type mismatch or differing primitives - replace at this location.
  patch.push({ op: 'replace', path: pointer, value: clone(to) });
}

/**
 * Applies an RFC 6902 JSON Patch to `document`, returning the new value.
 *
 * The input is not mutated; a clone is patched and returned. Operating on the
 * root pointer (`""`) replaces / adds the whole document.
 *
 * Apply is intentionally lenient to keep streaming robust: applying an `add` /
 * `replace` whose parent container is missing initializes the parent as an
 * object, and a `remove` / `replace` targeting a missing member is a no-op
 * rather than an error. `test` operations are honored and throw on mismatch.
 */
export function applyPatch<T = any>(document: T, patch: JsonPatch): T {
  let doc: any = clone(document);
  for (const op of patch) {
    doc = applyOperation(doc, op);
  }
  return doc as T;
}

function applyOperation(doc: any, op: JsonPatchOperation): any {
  const tokens = parsePointer(op.path);

  // Root operations replace / set the entire document.
  if (tokens.length === 0) {
    switch (op.op) {
      case 'add':
      case 'replace':
        return clone(op.value);
      case 'remove':
        return undefined;
      case 'test':
        if (!deepEqual(doc, op.value)) {
          throw new Error(`JSON Patch 'test' failed at root.`);
        }
        return doc;
      case 'move':
      case 'copy': {
        const value = clone(getValue(doc, parsePointer(op.from!)));
        return value;
      }
      default:
        throw new Error(`Unsupported JSON Patch op: ${(op as any).op}`);
    }
  }

  // Lenient: initialize a missing root container so member-level adds/replaces
  // still land (e.g. applying `/status` onto an undefined document).
  if (doc == null && (op.op === 'add' || op.op === 'replace')) {
    doc = {};
  }

  switch (op.op) {
    case 'add':
      setValue(doc, tokens, clone(op.value), /* isAdd= */ true);
      return doc;
    case 'replace':
      setValue(doc, tokens, clone(op.value), /* isAdd= */ false);
      return doc;
    case 'remove':
      removeValue(doc, tokens);
      return doc;
    case 'test': {
      const actual = getValue(doc, tokens);
      if (!deepEqual(actual, op.value)) {
        throw new Error(`JSON Patch 'test' failed at "${op.path}".`);
      }
      return doc;
    }
    case 'move': {
      const fromTokens = parsePointer(op.from!);
      const value = clone(getValue(doc, fromTokens));
      removeValue(doc, fromTokens);
      setValue(doc, tokens, value, /* isAdd= */ true);
      return doc;
    }
    case 'copy': {
      const value = clone(getValue(doc, parsePointer(op.from!)));
      setValue(doc, tokens, value, /* isAdd= */ true);
      return doc;
    }
    default:
      throw new Error(`Unsupported JSON Patch op: ${(op as any).op}`);
  }
}

/**
 * Reads the value at `tokens`, returning `undefined` for any missing segment.
 */
function getValue(doc: any, tokens: string[]): any {
  let cur = doc;
  for (const token of tokens) {
    if (cur == null) return undefined;
    cur = Array.isArray(cur) ? cur[Number(token)] : cur[token];
  }
  return cur;
}

/**
 * Sets the value at `tokens`, creating intermediate object containers as
 * needed. When `isAdd` is true and the parent is an array, the special `-`
 * token appends and a numeric token inserts at that index.
 */
function setValue(
  doc: any,
  tokens: string[],
  value: any,
  isAdd: boolean
): void {
  const parent = ensureParent(doc, tokens);
  if (parent == null) return; // Lenient: nothing to set onto.
  const last = tokens[tokens.length - 1];
  if (Array.isArray(parent)) {
    if (last === '-') {
      parent.push(value);
      return;
    }
    const idx = Number(last);
    if (Number.isNaN(idx)) return;
    if (isAdd) {
      parent.splice(idx, 0, value);
    } else {
      parent[idx] = value;
    }
    return;
  }
  parent[last] = value;
}

/**
 * Removes the value at `tokens`. Missing members are a no-op.
 */
function removeValue(doc: any, tokens: string[]): void {
  const parent = getValue(doc, tokens.slice(0, -1));
  if (parent == null) return;
  const last = tokens[tokens.length - 1];
  if (Array.isArray(parent)) {
    const idx = Number(last);
    if (!Number.isNaN(idx) && idx >= 0 && idx < parent.length) {
      parent.splice(idx, 1);
    }
    return;
  }
  if (isObject(parent)) {
    delete parent[last];
  }
}

/**
 * Walks to the parent container of `tokens`, lazily creating intermediate
 * objects for missing segments so leniently-applied patches still land.
 */
function ensureParent(doc: any, tokens: string[]): any {
  let cur = doc;
  for (let i = 0; i < tokens.length - 1; i++) {
    const token = tokens[i];
    if (cur == null) return undefined;
    const next = Array.isArray(cur) ? cur[Number(token)] : cur[token];
    if (next == null || typeof next !== 'object') {
      const created: any = {};
      if (Array.isArray(cur)) {
        cur[Number(token)] = created;
      } else {
        cur[token] = created;
      }
      cur = created;
    } else {
      cur = next;
    }
  }
  return cur;
}
