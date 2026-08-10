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

import { extractJson } from '../extract.js';
import type { Formatter } from './types.js';

export const jsonFormatter: Formatter<unknown, unknown> = {
  name: 'json',
  config: {
    format: 'json',
    contentType: 'application/json',
    constrained: true,
    defaultInstructions: false,
  },
  handler: (schema) => {
    let instructions: string | undefined;

    if (schema) {
      instructions = `Output should be in JSON format and conform to the following schema:

\`\`\`
${JSON.stringify(schema)}
\`\`\`
`;
    }
    return {
      parseChunk: (chunk) => {
        return extractJson(chunk.accumulatedText);
      },

      parseMessage: (message) => {
        return extractJson(message.text);
      },

      instructions,
    };
  },
};
