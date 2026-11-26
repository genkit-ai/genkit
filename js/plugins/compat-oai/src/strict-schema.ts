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

type JsonSchema = Record<string, unknown>;

// Only keywords where closing each subschema on its own keeps the schema's
// meaning. Under allOf, not and if/then/else it would not, and strict mode
// rejects those anyway.

/** Keywords whose value is a list of subschemas. `items` is one in its
 * array form (a tuple). */
const SCHEMA_LIST_KEYWORDS = ['anyOf', 'oneOf', 'prefixItems', 'items'];
/** Keywords whose value maps names to subschemas. */
const SCHEMA_MAP_KEYWORDS = ['properties', '$defs', 'definitions'];

function isObjectSchema(schema: JsonSchema): boolean {
  const type = schema.type;
  return (
    type === 'object' ||
    (Array.isArray(type) && type.includes('object')) ||
    (type === undefined && schema.properties !== undefined)
  );
}

function isSchema(value: unknown): value is JsonSchema {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * OpenAI strict structured output requires `additionalProperties: false` on
 * every object. A schema-valued `additionalProperties` is kept, and objects
 * inside it are closed.
 */
export function closeObjectSchemas(schema: JsonSchema): JsonSchema {
  const out: JsonSchema = { ...schema };
  if (isSchema(out.items)) out.items = closeObjectSchemas(out.items);
  for (const key of SCHEMA_LIST_KEYWORDS) {
    const list = out[key];
    if (Array.isArray(list)) {
      out[key] = list.map((s) => (isSchema(s) ? closeObjectSchemas(s) : s));
    }
  }
  for (const key of SCHEMA_MAP_KEYWORDS) {
    const map = out[key];
    if (isSchema(map)) {
      out[key] = Object.fromEntries(
        Object.entries(map).map(([name, s]) => [
          name,
          isSchema(s) ? closeObjectSchemas(s) : s,
        ])
      );
    }
  }
  if (isSchema(out.additionalProperties)) {
    out.additionalProperties = closeObjectSchemas(out.additionalProperties);
  } else if (
    isObjectSchema(out) &&
    (out.additionalProperties === undefined ||
      out.additionalProperties === true)
  ) {
    out.additionalProperties = false;
  }
  return out;
}
