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
import { extractItems } from '../extract.js';
import type { Formatter } from './types.js';

export const arrayFormatter: Formatter<unknown[], unknown[]> = {
  name: 'array',
  config: {
    contentType: 'application/json',
    constrained: true,
  },
  handler: (schema) => {
    if (schema && schema.type !== 'array') {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Must supply an 'array' schema type when using the 'items' parser format.`,
      });
    }

    let instructions: string | undefined;
    if (schema) {
      instructions = `Output should be a JSON array conforming to the following schema:
    
\`\`\`
${JSON.stringify(schema)}
\`\`\`
    `;
    }

    return {
      parseChunk: (chunk) => {
        // first, determine the cursor position from the previous chunks
        const cursor = chunk.previousChunks?.length
          ? extractItems(chunk.previousText).cursor
          : 0;
        // then, extract the items starting at that cursor
        const { items } = extractItems(chunk.accumulatedText, cursor);

        return items;
      },

      parseMessage: (message) => {
        const { items } = extractItems(message.text, 0);
        return items;
      },

      instructions,
    };
  },
};
