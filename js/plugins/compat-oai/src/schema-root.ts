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

/** Keywords whose values are data, not subschemas, so `$ref` there is not a
 * reference. */
const DATA_KEYWORDS = new Set(['const', 'enum', 'default', 'examples']);

type Json = unknown;

function isRecord(node: Json): node is Record<string, Json> {
  return node !== null && typeof node === 'object' && !Array.isArray(node);
}

/** What follows `#/$defs/` or `#/definitions/` in `ref`, if either starts it. */
function defsPath(ref: string): string | undefined {
  for (const prefix of ['#/$defs/', '#/definitions/']) {
    if (ref.startsWith(prefix)) return ref.slice(prefix.length);
  }
  return undefined;
}

/**
 * Whether `ref` is a local reference that strict hosts can't follow. A
 * reference to a whole top-level definition is fine. One into the middle of a
 * definition is not, and nor, when `local` is set, is any other local
 * reference.
 */
function hoistable(ref: string, local: boolean): boolean {
  if (ref !== '#' && !ref.startsWith('#/')) return false;
  const path = defsPath(ref);
  if (path !== undefined) return path.includes('/');
  return local && ref !== '#/$defs' && ref !== '#/definitions';
}

/** Calls `found` with every `$ref` in `node`. */
function collectRefs(node: Json, found: (ref: string) => void): void {
  if (Array.isArray(node)) {
    node.forEach((v) => collectRefs(v, found));
  } else if (isRecord(node)) {
    for (const [key, value] of Object.entries(node)) {
      if (key === '$ref' && typeof value === 'string') {
        found(value);
      } else if (!DATA_KEYWORDS.has(key)) {
        collectRefs(value, found);
      }
    }
  }
}

/** Whether the JSON pointer `ref` names a schema inside `schema`. */
function resolvesToSchema(schema: JsonSchema, ref: string): boolean {
  let node: Json = schema;
  for (const raw of ref.replace(/^#/, '').split('/').slice(1)) {
    const seg = raw.replace(/~1/g, '/').replace(/~0/g, '~');
    if (Array.isArray(node)) {
      const i = Number(seg);
      if (!/^\d+$/.test(seg) || i >= node.length) return false;
      node = node[i];
    } else if (isRecord(node) && Object.hasOwn(node, seg)) {
      node = node[seg];
    } else {
      return false;
    }
  }
  return isRecord(node);
}

function escapePointer(key: string): string {
  return key.replace(/~/g, '~0').replace(/\//g, '~1');
}

/**
 * Copies a schema, moving the subschemas that references point at into `defs`
 * and pointing both their old place and the references there. It moves the
 * targets of references into the middle of a definition and, when `local` is
 * set, those of any other local reference outside the definitions.
 */
class RefHoister {
  /** Each referenced JSON pointer, and its name in `defs`. */
  private readonly targets = new Map<string, string>();
  readonly defs: Record<string, Json> = {};

  constructor(schema: JsonSchema, local: boolean) {
    const taken = new Set(
      isRecord(schema.$defs) ? Object.keys(schema.$defs) : []
    );
    const refs = new Set<string>();
    collectRefs(schema, (ref) => {
      if (hoistable(ref, local) && resolvesToSchema(schema, ref)) refs.add(ref);
    });
    // A pointer into a definition is named after its path there
    // (`Person_properties_name`), and any other after its path under the
    // wrapper (`value_items`).
    for (const ref of [...refs].sort()) {
      const path = defsPath(ref) ?? WRAPPER_KEY + ref.slice(1);
      const base = path.replace(/[^\w-]/g, '_');
      let name = base;
      for (let i = 2; taken.has(name); i++) name = `${base}_${i}`;
      taken.add(name);
      this.targets.set(ref, name);
    }
  }

  get hoists(): boolean {
    return this.targets.size > 0;
  }

  /** Copies the `$defs` map into `defs`. */
  walkDefs($defs: Record<string, Json>): void {
    for (const [name, def] of Object.entries($defs)) {
      this.defs[name] = this.walk(def, `#/$defs/${escapePointer(name)}`);
    }
  }

  /** Copies `node`, found at JSON pointer `ptr` in the caller's schema. */
  walk(node: Json, ptr: string): Json {
    const name = this.targets.get(ptr);
    if (name === undefined) return this.copy(node, ptr);
    this.defs[name] = this.copy(node, ptr);
    return { $ref: `#/$defs/${name}` };
  }

  private copy(node: Json, ptr: string): Json {
    if (Array.isArray(node)) {
      return node.map((v, i) => this.walk(v, `${ptr}/${i}`));
    }
    if (!isRecord(node)) return node;
    const out: Record<string, Json> = {};
    for (const [key, value] of Object.entries(node)) {
      if (key === '$ref' && typeof value === 'string') {
        const name = this.targets.get(value);
        out[key] = name === undefined ? value : `#/$defs/${name}`;
      } else if (DATA_KEYWORDS.has(key)) {
        out[key] = value;
      } else {
        out[key] = this.walk(value, `${ptr}/${escapePointer(key)}`);
      }
    }
    return out;
  }
}

/**
 * Wraps `schema` as the single required property of an object schema.
 * `$schema`, `$defs` and `definitions` stay at the root, so references into
 * the definitions still resolve.
 *
 * Any other local reference (`#`, `#/items/...`) pointed into the caller's
 * schema, which is no longer the root. Strict hosts accept references only to
 * top-level definitions, so each referenced subschema moves into `$defs` and
 * both its old place and every reference to it point there instead.
 */
export function wrapSchemaRoot(schema: JsonSchema): JsonSchema {
  const { $schema, $defs, definitions, ...inner } = schema;
  const hoister = new RefHoister(schema, true);
  if (isRecord($defs)) hoister.walkDefs($defs);
  const value = hoister.walk(inner, '#');
  // Walk definitions before deciding on $defs, since walking them can hoist.
  const walkedDefinitions =
    definitions !== undefined
      ? hoister.walk(definitions, '#/definitions')
      : undefined;
  const defs =
    isRecord($defs) || Object.keys(hoister.defs).length > 0
      ? hoister.defs
      : $defs;
  return {
    ...($schema !== undefined ? { $schema } : {}),
    type: 'object',
    properties: { [WRAPPER_KEY]: value },
    required: [WRAPPER_KEY],
    additionalProperties: false,
    ...(defs !== undefined ? { $defs: defs } : {}),
    ...(walkedDefinitions !== undefined
      ? { definitions: walkedDefinitions }
      : {}),
  };
}

/**
 * Returns `schema` with every reference into the middle of a definition
 * (`#/$defs/Person/properties/name`) pointed at a new top-level definition
 * instead, since strict hosts resolve only whole top-level definitions. A
 * schema without such references comes back unchanged.
 */
export function hoistDefRefs(schema: JsonSchema): JsonSchema {
  const hoister = new RefHoister(schema, false);
  if (!hoister.hoists) return schema;
  const out: JsonSchema = {};
  for (const [key, value] of Object.entries(schema)) {
    if (key === '$defs' && isRecord(value)) {
      hoister.walkDefs(value);
    } else {
      out[key] = hoister.walk(value, `#/${escapePointer(key)}`);
    }
  }
  out.$defs = hoister.defs;
  return out;
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
