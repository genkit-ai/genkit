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

function toFrontmatterSchema(config?: {
  schema?: unknown;
  jsonSchema?: unknown;
}): unknown | undefined {
  const schema = config?.jsonSchema ?? config?.schema;
  return schema && typeof schema === 'object' ? schema : undefined;
}

/**
 * Maps a generate request's output config onto `.prompt` frontmatter. The
 * frontmatter format is limited to json/text/media, so the JSON-producing
 * formats (json, jsonl, array, enum) map onto `json`. Returns undefined when
 * there is nothing to record.
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
  const schema = toFrontmatterSchema(output);
  if (schema !== undefined) {
    result.schema = schema;
  }
  return result.format || result.schema ? result : undefined;
}

/**
 * Maps a request's input config onto `.prompt` frontmatter. Returns undefined
 * when there is nothing to record.
 */
export function toFrontmatterInput(input?: {
  schema?: unknown;
  jsonSchema?: unknown;
  default?: unknown;
}): PromptFrontmatter['input'] | undefined {
  if (!input) return undefined;
  const result: NonNullable<PromptFrontmatter['input']> = {
    schema: toFrontmatterSchema(input),
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
    input: toFrontmatterInput(request.input),
    output: toFrontmatterOutput(request.output),
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
