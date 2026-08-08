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

const PICOSCHEMA_SCALAR_TYPES = new Set([
  'boolean',
  'integer',
  'null',
  'number',
  'string',
]);

/**
 * Converts JSON Schema into Dotprompt's compact Picoschema notation.
 * Picoschema is a subset of JSON Schema, so constraints without a Picoschema
 * equivalent are omitted as part of this opt-in, best-effort conversion.
 */
export function jsonSchemaToPicoschema(schema: unknown): unknown {
  if (!isRecord(schema) || Object.keys(schema).length === 0) return undefined;
  return convertJsonSchemaNode(schema);
}

function convertJsonSchemaNode(schema: Record<string, unknown>): unknown {
  const description =
    typeof schema.description === 'string' ? schema.description : undefined;
  const type = scalarType(schema.type);

  if (Array.isArray(schema.enum)) {
    return schema.enum.filter((value) => value !== null);
  }

  if (type && PICOSCHEMA_SCALAR_TYPES.has(type)) {
    return description ? `${type}, ${description}` : type;
  }

  if (!type && schema.properties === undefined && schema.items === undefined) {
    return description ? `any, ${description}` : 'any';
  }

  if (type === 'array') {
    return isRecord(schema.items) && Object.keys(schema.items).length > 0
      ? convertJsonSchemaNode(schema.items)
      : 'any';
  }

  if (type === 'object' || isRecord(schema.properties)) {
    return convertJsonSchemaObject(schema);
  }

  if (type) {
    return description ? `${type}, ${description}` : type;
  }

  return undefined;
}

function convertJsonSchemaObject(
  schema: Record<string, unknown>
): Record<string, unknown> | undefined {
  const properties = isRecord(schema.properties) ? schema.properties : {};
  const requiredProperties = new Set(
    Array.isArray(schema.required)
      ? schema.required.filter(
          (propertyName): propertyName is string =>
            typeof propertyName === 'string'
        )
      : []
  );
  const result: Record<string, unknown> = {};

  for (const [propertyName, value] of Object.entries(properties)) {
    if (!isPicoschemaPropertyName(propertyName) || !isRecord(value)) {
      return undefined;
    }

    const key = requiredProperties.has(propertyName)
      ? propertyName
      : `${propertyName}?`;
    const type = scalarType(value.type);

    if (Array.isArray(value.enum)) {
      result[`${key}(enum)`] = value.enum.filter(
        (enumValue) => enumValue !== null
      );
    } else if (type === 'array') {
      result[`${key}(array)`] =
        isRecord(value.items) && Object.keys(value.items).length > 0
          ? convertJsonSchemaNode(value.items)
          : 'any';
    } else if (type === 'object' || isRecord(value.properties)) {
      const nestedObject = convertJsonSchemaObject(value);
      if (!nestedObject) return undefined;
      result[`${key}(object)`] = nestedObject;
    } else {
      const converted = convertJsonSchemaNode(value);
      if (converted === undefined) return undefined;
      result[key] = converted;
    }
  }

  if (schema.additionalProperties === true) {
    result['(*)'] = 'any';
  } else if (isRecord(schema.additionalProperties)) {
    const additionalProperties = convertJsonSchemaNode(
      schema.additionalProperties
    );
    if (additionalProperties === undefined) return undefined;
    result['(*)'] = additionalProperties;
  }

  return result;
}

function scalarType(type: unknown): string | undefined {
  if (typeof type === 'string') return type;
  if (!Array.isArray(type) || !type.includes('null')) return undefined;
  const nonNullTypes = type.filter(
    (value): value is string => typeof value === 'string' && value !== 'null'
  );
  return nonNullTypes.length === 1 ? nonNullTypes[0] : undefined;
}

function isPicoschemaPropertyName(propertyName: string): boolean {
  return (
    propertyName.length > 0 &&
    propertyName !== '(*)' &&
    !['__proto__', 'constructor', 'prototype'].includes(propertyName) &&
    !propertyName.includes('(') &&
    !propertyName.endsWith('?')
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && !Array.isArray(value) && typeof value === 'object';
}

function toFrontmatterSchema(
  config?: {
    schema?: unknown;
    jsonSchema?: unknown;
  },
  picoSchema = false
): unknown | undefined {
  const schema = config?.jsonSchema ?? config?.schema;
  if (!schema || typeof schema !== 'object') return undefined;
  if (!picoSchema) return schema;
  return jsonSchemaToPicoschema(schema) ?? schema;
}

/**
 * Maps a generate request's output config onto `.prompt` frontmatter. The
 * frontmatter format is limited to json/text/media, so the JSON-producing
 * formats (json, jsonl, array, enum) map onto `json`. Returns undefined when
 * there is nothing to record.
 */
export function toFrontmatterOutput(
  output?: {
    format?: string;
    jsonSchema?: unknown;
    schema?: unknown;
  },
  picoSchema = false
): PromptFrontmatter['output'] | undefined {
  if (!output) return undefined;
  const result: NonNullable<PromptFrontmatter['output']> = {};
  if (output.format === 'text') {
    result.format = 'text';
  } else if (output.format === 'media') {
    result.format = 'media';
  } else if (output.format) {
    result.format = 'json';
  }
  const schema = toFrontmatterSchema(output, picoSchema);
  if (schema !== undefined) {
    result.schema = schema;
  }
  return result.format || result.schema ? result : undefined;
}

/**
 * Maps a request's input config onto `.prompt` frontmatter. Returns undefined
 * when there is nothing to record.
 */
export function toFrontmatterInput(
  input?: {
    schema?: unknown;
    jsonSchema?: unknown;
    default?: unknown;
  },
  picoSchema = false
): PromptFrontmatter['input'] | undefined {
  if (!input) return undefined;
  const result: NonNullable<PromptFrontmatter['input']> = {
    schema: toFrontmatterSchema(input, picoSchema),
    default: input.default,
  };
  return result.schema || result.default !== undefined ? result : undefined;
}

/**
 * Converts a prompt creation request into a complete `.prompt` template file string.
 */
export function toPromptFile(request: {
  model: string;
  messages: MessageData[];
  picoSchema?: boolean;
  config?: Record<string, unknown>;
  tools?: { name: string }[];
  use?: PromptFrontmatter['use'];
  input?: {
    schema?: unknown;
    jsonSchema?: unknown;
    default?: unknown;
  };
  output?: {
    format?: string;
    jsonSchema?: unknown;
    schema?: unknown;
  };
}): string {
  const frontmatter: PromptFrontmatter = {
    model: request.model.replace('/model/', ''),
    config: request.config,
    tools: request.tools?.map((toolDefinition) => toolDefinition.name),
    use: request.use,
    input: toFrontmatterInput(request.input, request.picoSchema),
    output: toFrontmatterOutput(request.output, request.picoSchema),
  };
  return renderPromptFile(frontmatter, request.messages);
}

export function renderPromptFile(
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
