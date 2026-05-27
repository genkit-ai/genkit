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

import { GenkitError } from '@genkit-ai/core';
import JSON5 from 'json5';
import { extractJson } from '../extract.js';
import type { Formatter } from './types.js';

function objectLines(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('{'));
}

export const jsonlFormatter: Formatter<unknown[], unknown[]> = {
  name: 'jsonl',
  config: {
    contentType: 'application/jsonl',
  },
  handler: (schema) => {
    if (
      schema &&
      (schema.type !== 'array' || schema.items?.type !== 'object')
    ) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Must supply an 'array' schema type containing 'object' items when using the 'jsonl' parser format.`,
      });
    }

    let instructions: string | undefined;
    if (schema?.items) {
      instructions = `Output should be JSONL format, a sequence of JSON objects (one per line) separated by a newline \`\\n\` character. Each line should be a JSON object conforming to the following schema:

\`\`\`
${JSON.stringify(schema.items)}
\`\`\`
    `;
    }

    return {
      parseChunk: (chunk) => {
        const results: unknown[] = [];

        const text = chunk.accumulatedText;

        let startIndex = 0;
        if (chunk.previousChunks?.length) {
          const lastNewline = chunk.previousText.lastIndexOf('\n');
          if (lastNewline !== -1) {
            startIndex = lastNewline + 1;
          }
        }

        const lines = text.slice(startIndex).split('\n');

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('{')) {
            try {
              const result = JSON5.parse(trimmed);
              if (result) {
                results.push(result);
              }
            } catch (e) {
              break;
            }
          }
        }

        return results;
      },

      parseMessage: (message) => {
        const items = objectLines(message.text)
          .map((l) => extractJson(l))
          .filter((l) => !!l);

        return items;
      },

      instructions,
    };
  },
};
