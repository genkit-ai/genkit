/**
 * Copyright 2024 Google LLC
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

import { stringify } from 'yaml';
import type { MessageData, Part } from '../types/model';
import type { PromptFrontmatter } from '../types/prompt';

/** A JSON Schema-ish object. */
type JsonSchema = Record<string, any>;

function scalarType(schema: JsonSchema | null | undefined): string {
  switch (schema?.type) {
    case 'string':
    case 'integer':
    case 'number':
    case 'boolean':
    case 'null':
      return schema.type;
    default:
      return 'any';
  }
}

/** Wraps a type kind with an optional description, e.g. `(array, the tags)`. */
function wrap(kind: string, description?: string): string {
  return description ? `(${kind}, ${description})` : `(${kind})`;
}

/**
 * Encodes a single property as a Picoschema key suffix and value. Scalars carry
 * their description after a comma in the value (`string, the title`); wrapped
 * kinds (object, array, enum) carry it inside the parentheses on the key.
 */
function picoEntry(schema: JsonSchema): { suffix: string; value: any } {
  const description =
    typeof schema?.description === 'string' ? schema.description : undefined;

  if (Array.isArray(schema?.enum)) {
    return { suffix: wrap('enum', description), value: schema.enum };
  }
  if (schema?.type === 'array') {
    const items: JsonSchema = schema.items ?? {};
    const value =
      items.type === 'object' || items.properties
        ? picoObject(items)
        : scalarType(items);
    return { suffix: wrap('array', description), value };
  }
  if (schema?.type === 'object' || schema?.properties) {
    return { suffix: wrap('object', description), value: picoObject(schema) };
  }
  const type = scalarType(schema);
  return { suffix: '', value: description ? `${type}, ${description}` : type };
}

/** Converts an object JSON Schema into a Picoschema object structure. */
function picoObject(schema: JsonSchema): Record<string, any> {
  const required = new Set<string>(
    Array.isArray(schema.required) ? schema.required : []
  );
  const out: Record<string, any> = {};
  for (const [name, propSchema] of Object.entries<any>(
    schema.properties ?? {}
  )) {
    const optional = required.has(name) ? '' : '?';
    const { suffix, value } = picoEntry(propSchema);
    out[`${name}${optional}${suffix}`] = value;
  }
  const additional = schema.additionalProperties;
  if (additional && typeof additional === 'object') {
    const { suffix, value } = picoEntry(additional);
    out[`(*)${suffix}`] = value;
  }
  return out;
}

/**
 * Converts a JSON Schema into the equivalent Picoschema for a `.prompt` file.
 * Object schemas become the compact Picoschema form. Non-object top-level
 * schemas (a bare array or scalar) have no Picoschema form, so the JSON Schema
 * is returned unchanged; Dotprompt accepts raw JSON Schema there too.
 */
export function jsonSchemaToPicoschema(schema: unknown): any {
  if (!schema || typeof schema !== 'object') {
    return schema;
  }
  const s = schema as JsonSchema;
  if (s.type === 'object' || s.properties) {
    return picoObject(s);
  }
  return schema;
}

/**
 * Maps a generate request's output config onto `.prompt` frontmatter, converting
 * the JSON Schema to Picoschema. The frontmatter format is limited to
 * json/text/media, so the JSON-producing formats (json, jsonl, array, enum) map
 * onto `json`. Returns undefined when there is nothing to record.
 */
export function toFrontmatterOutput(output?: {
  format?: string;
  jsonSchema?: unknown;
  schema?: unknown;
}): PromptFrontmatter['output'] | undefined {
  if (!output) return undefined;
  const result: NonNullable<PromptFrontmatter['output']> = {};
  if (output.format === 'text') {
    result.format = 'text';
  } else if (output.format === 'media') {
    result.format = 'media';
  } else if (output.format) {
    result.format = 'json';
  }
  const schema = output.jsonSchema ?? output.schema;
  if (schema && typeof schema === 'object') {
    result.schema = jsonSchemaToPicoschema(schema);
  }
  return result.format || result.schema ? result : undefined;
}

export function fromMessages(
  frontmatter: PromptFrontmatter,
  messages: MessageData[]
): string {
  const cleanFrontmatter = cleanupFrontmatter(frontmatter);
  const { rendered: renderedMessages, anyOmitted } = renderMessages(messages);

  const header = `---
${stringify(cleanFrontmatter, {
  collectionStyle: 'block',
  aliasDuplicateObjects: false,
}).trim()}
---`;

  if (anyOmitted) {
    return (
      `${header}

{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}

${renderedMessages}`.trimEnd() + '\n'
    );
  }

  return (
    `${header}

${renderedMessages}`.trimEnd() + '\n'
  );
}

/**
 * Renders an array of message data into a Dotprompt template string.
 */
function renderMessages(messages: MessageData[]): {
  rendered: string;
  anyOmitted: boolean;
} {
  let anyOmitted = false;
  let rendered = '';

  messages.forEach((message) => {
    const hasToolRequest = message.content.some((p) => 'toolRequest' in p);
    const hasToolResponse = message.content.some((p) => 'toolResponse' in p);
    const hasSupportedPart =
      message.content.length === 0 ||
      message.content.some((p) => 'text' in p || 'media' in p);
    const hasUnsupportedPart = message.content.some(
      (p) => !('text' in p) && !('media' in p)
    );

    if (hasToolRequest || hasToolResponse || !hasSupportedPart) {
      anyOmitted = true;
      let reason = 'unsupported content';
      if (hasToolRequest) {
        reason = 'toolRequest';
      } else if (hasToolResponse) {
        reason = 'toolResponse';
      }
      rendered += `{{! message with role "${message.role}" omitted (${reason}). }}\n\n`;
    } else {
      if (hasUnsupportedPart) {
        anyOmitted = true;
      }
      rendered += `{{role "${message.role}"}}\n`;
      rendered += message.content.map(partToString).join('');
      rendered += '\n\n';
    }
  });

  return { rendered, anyOmitted };
}

/**
 * Removes empty arrays, empty objects, and null/undefined values from the
 * frontmatter to ensure the generated YAML is clean and idiomatic.
 */
function cleanupFrontmatter(frontmatter: PromptFrontmatter): any {
  return recursiveCleanup(frontmatter) || {};
}

function recursiveCleanup(val: any): any {
  if (Array.isArray(val)) {
    const cleaned = val
      .map(recursiveCleanup)
      .filter((v) => v !== undefined && v !== null);
    return cleaned.length > 0 ? cleaned : undefined;
  }
  if (val !== null && typeof val === 'object' && !(val instanceof Date)) {
    const cleaned: any = {};
    let hasProps = false;
    for (const key in val) {
      const v = recursiveCleanup(val[key]);
      if (v !== undefined && v !== null) {
        cleaned[key] = v;
        hasProps = true;
      }
    }
    return hasProps ? cleaned : undefined;
  }
  return val === null || val === undefined ? undefined : val;
}

function partToString(part: Part): string {
  if ('text' in part && part.text !== undefined) {
    return part.text;
  } else if ('media' in part && part.media !== undefined) {
    return `{{media url:${part.media.url}}}`;
  }

  const type =
    Object.keys(part).find(
      (k) => k !== 'metadata' && part[k as keyof Part] !== undefined
    ) || 'unknown';
  return `{{! ${type} part omitted }}`;
}
