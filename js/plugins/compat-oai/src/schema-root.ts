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

/** Whether the schema's root is an object, which every host accepts. */
export function hasObjectRoot(schema: JsonSchema): boolean {
  if (schema.type === 'object') return true;
  return schema.type === undefined && schema.properties !== undefined;
}

/** Wraps `schema` as the single required property of an object schema. */
export function wrapSchemaRoot(schema: JsonSchema): JsonSchema {
  // Definitions must stay at the root so `#/$defs/...` references resolve,
  // and `$schema` is only meaningful at the root.
  const { $schema, $defs, definitions, ...inner } = schema as JsonSchema & {
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

const PREFIX = new RegExp(`^\\s*\\{\\s*"${WRAPPER_KEY}"\\s*:\\s*`);

/**
 * Strips the `{"value":` prefix and the closing `}` from streamed text so
 * chunks read as the caller's value (e.g. `[{...},` for an array) and the
 * output format's chunk parser sees what it expects.
 *
 * Text is held back until the prefix is seen, and trailing whitespace and
 * braces are held until more text arrives, because the last `}` belongs to the
 * wrapper only if nothing follows it. `flush` drops that final brace.
 */
export class WrapperStreamUnwrapper {
  private started = false;
  private wrapped = false;
  private head = '';
  private held = '';

  push(delta: string): string {
    if (!this.started) {
      this.head += delta;
      const match = PREFIX.exec(this.head);
      if (!match) {
        // Not a wrapper (or not enough text yet to tell): keep buffering
        // unless it can no longer become one.
        const lead = this.head.trimStart();
        if (lead && (lead[0] !== '{' || this.head.length > 40)) {
          this.started = true;
          const out = this.head;
          this.head = '';
          return this.release(out);
        }
        return '';
      }
      this.started = true;
      this.wrapped = true;
      const rest = this.head.slice(match[0].length);
      this.head = '';
      return this.release(rest);
    }
    return this.release(delta);
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
