/**
 * Copyright 2025 Google LLC
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
 * Some OpenAI-compatible hosts (OpenAI itself, Azure OpenAI, and gateways that
 * forward to them) only accept a `json_schema` response format whose root is
 * `type: "object"`. For those hosts, a schema with any other root (array,
 * enum, scalar, `anyOf`, ...) is sent wrapped in an object and the reply is
 * unwrapped before it reaches the caller.
 */

/** Property that holds the caller's value inside the wrapper object. */
export const WRAPPER_KEY = 'value';

type JsonSchema = Record<string, unknown>;

/**
 * Whether the schema's root is explicitly an object. Anything else, including
 * `properties` without a `type` (which also allows non-object values), is
 * wrapped.
 */
export function hasObjectRoot(schema: JsonSchema): boolean {
  return schema.type === 'object';
}

/** Where the caller's schema sits once wrapped, as a JSON pointer. */
const WRAPPED_ROOT = `#/properties/${WRAPPER_KEY}`;

/** Keywords whose values are data, not subschemas, so `$ref` there is not a
 * reference. */
const DATA_KEYWORDS = new Set(['const', 'enum', 'default', 'examples']);

/**
 * Re-points local references into the caller's schema (`#`, `#/items/...`) at
 * its new place under the wrapper. References into `$defs` and `definitions`
 * are left alone, because those stay at the root.
 */
function rewriteRefs(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(rewriteRefs);
  if (node === null || typeof node !== 'object') return node;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(node)) {
    if (key === '$ref' && typeof value === 'string') {
      out[key] = rewriteRef(value);
    } else if (DATA_KEYWORDS.has(key)) {
      out[key] = value;
    } else {
      out[key] = rewriteRefs(value);
    }
  }
  return out;
}

function rewriteRef(ref: string): string {
  if (ref === '#') return WRAPPED_ROOT;
  if (
    !ref.startsWith('#/') ||
    ref === '#/$defs' ||
    ref.startsWith('#/$defs/') ||
    ref === '#/definitions' ||
    ref.startsWith('#/definitions/')
  ) {
    return ref;
  }
  return WRAPPED_ROOT + ref.slice(1);
}

/**
 * Wraps `schema` as the single required property of an object schema.
 * Definitions stay at the root so `#/$defs/...` references resolve, `$schema`
 * is only meaningful there, and other local references are rewritten to the
 * schema's new place.
 */
export function wrapSchemaRoot(schema: JsonSchema): JsonSchema {
  const { $schema, $defs, definitions, ...inner } = rewriteRefs(
    schema
  ) as JsonSchema & {
    $schema?: unknown;
    $defs?: unknown;
    definitions?: unknown;
  };
  return {
    ...($schema !== undefined ? { $schema } : {}),
    type: 'object',
    properties: { [WRAPPER_KEY]: inner },
    required: [WRAPPER_KEY],
    additionalProperties: false,
    ...($defs !== undefined ? { $defs } : {}),
    ...(definitions !== undefined ? { definitions } : {}),
  };
}

/**
 * Returns the wrapped value, or `parsed` unchanged when it isn't a wrapper
 * (a host that ignored the schema may reply with the bare value).
 */
export function unwrapValue(parsed: unknown): unknown {
  if (
    parsed !== null &&
    typeof parsed === 'object' &&
    !Array.isArray(parsed) &&
    WRAPPER_KEY in parsed
  ) {
    return (parsed as Record<string, unknown>)[WRAPPER_KEY];
  }
  return parsed;
}

/** The wrapper's opening, as the tokens it is made of. Whitespace may sit
 * before and between them. */
const PREFIX_TOKENS = ['{', `"${WRAPPER_KEY}"`, ':'];

/**
 * Matches the start of `text` against the wrapper's opening. Returns the
 * length of the opening when it is all there, `'partial'` when `text` could
 * still become it, and `'no'` when it cannot.
 */
function matchPrefix(text: string): number | 'partial' | 'no' {
  let i = 0;
  for (const token of PREFIX_TOKENS) {
    while (i < text.length && /\s/.test(text[i])) i++;
    const rest = text.slice(i, i + token.length);
    if (rest !== token) {
      return token.startsWith(rest) && i + rest.length === text.length
        ? 'partial'
        : 'no';
    }
    i += token.length;
  }
  while (i < text.length && /\s/.test(text[i])) i++;
  return i;
}

/**
 * Strips the `{"value":` prefix and the closing `}` from streamed text so
 * chunks read as the caller's value (e.g. `[{...},` for an array) and the
 * output format's chunk parser sees what it expects.
 *
 * Text is held back for as long as it could still be the start of the
 * wrapper, so the result does not depend on how the stream is split. Trailing
 * whitespace and braces are then held until more text arrives, because the
 * last `}` belongs to the wrapper only if nothing follows it. `flush` drops
 * that final brace. A reply that is not a wrapper passes through unchanged.
 */
export class WrapperStreamUnwrapper {
  private started = false;
  private wrapped = false;
  private head = '';
  private held = '';

  push(delta: string): string {
    if (this.started) return this.release(delta);
    this.head += delta;
    const match = matchPrefix(this.head);
    if (match === 'partial') return '';
    this.started = true;
    let out = this.head;
    this.head = '';
    if (match !== 'no') {
      this.wrapped = true;
      out = out.slice(match);
    }
    return this.release(out);
  }

  private release(text: string): string {
    const all = this.held + text;
    const tail = /[\s}]*$/.exec(all)![0];
    this.held = tail;
    return all.slice(0, all.length - tail.length);
  }

  /** Emits what is left: anything held except the wrapper's closing brace
   * (and the whitespace around it). */
  flush(): string {
    const text = this.started ? this.held : this.head;
    this.head = '';
    this.held = '';
    if (!this.started || !this.wrapped) return text;
    const close = text.lastIndexOf('}');
    return (close === -1 ? text : text.slice(0, close)).trimEnd();
  }
}
