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
      parseChunk: (() => {
        let lineParts: string[] = [];
        let lineType: 'object' | 'other' | undefined;
        let depth = 0;
        let quote: '"' | "'" | undefined;
        let escaped = false;
        let comment: 'line' | 'block' | undefined;
        let blockCommentStar = false;
        let pendingSlash = false;
        let rootClosed = false;
        let invalidLine = false;
        let lineEmitted = false;
        let parseAttempted = false;
        const processedContents: object[] = [];
        let cachedResults: unknown[] = [];

        const resetLine = () => {
          lineParts = [];
          lineType = undefined;
          depth = 0;
          quote = undefined;
          escaped = false;
          comment = undefined;
          blockCommentStar = false;
          pendingSlash = false;
          rootClosed = false;
          invalidLine = false;
          lineEmitted = false;
          parseAttempted = false;
        };

        const scanSegment = (segment: string) => {
          if (lineEmitted) return;
          lineParts.push(segment);

          for (const char of segment) {
            if (!lineType) {
              if (/\s/u.test(char)) continue;
              lineType = char === '{' ? 'object' : 'other';
              if (lineType === 'object') depth = 1;
              continue;
            }
            if (lineType === 'other') continue;

            if (quote) {
              if (escaped) {
                escaped = false;
              } else if (char === '\\') {
                escaped = true;
              } else if (char === quote) {
                quote = undefined;
              }
              continue;
            }
            if (comment === 'line') continue;
            if (comment === 'block') {
              if (blockCommentStar && char === '/') {
                comment = undefined;
                blockCommentStar = false;
              } else {
                blockCommentStar = char === '*';
              }
              continue;
            }
            if (pendingSlash) {
              pendingSlash = false;
              if (char === '/') {
                comment = 'line';
                continue;
              }
              if (char === '*') {
                comment = 'block';
                blockCommentStar = false;
                continue;
              }
              invalidLine = true;
            }
            if (char === '/') {
              pendingSlash = true;
              continue;
            }
            if (rootClosed) {
              if (!/\s/u.test(char)) invalidLine = true;
              continue;
            }
            if (char === '"' || char === "'") {
              quote = char;
              continue;
            }
            if (char === '{' || char === '[') {
              depth++;
            } else if (char === '}' || char === ']') {
              depth--;
              if (depth === 0) rootClosed = true;
            }
          }
        };

        const parseLine = (): unknown[] => {
          if (
            lineEmitted ||
            parseAttempted ||
            lineType !== 'object' ||
            !rootClosed ||
            quote ||
            comment === 'block' ||
            pendingSlash ||
            invalidLine
          ) {
            return [];
          }

          parseAttempted = true;
          try {
            const result = JSON5.parse(lineParts.join('').trim());
            if (result) {
              lineEmitted = true;
              return [result];
            }
          } catch {
            // A closed root object cannot become valid by appending to the line.
          }
          return [];
        };

        const processText = (text: string): unknown[] => {
          const results: unknown[] = [];
          const segments = text.split('\n');
          let emitResults = true;

          for (let index = 0; index < segments.length; index++) {
            scanSegment(segments[index]);
            if (emitResults) results.push(...parseLine());

            if (index < segments.length - 1) {
              if (lineType === 'object' && !lineEmitted) {
                emitResults = false;
              }
              resetLine();
            }
          }
          return results;
        };

        const resetStream = () => {
          resetLine();
          processedContents.length = 0;
          cachedResults = [];
        };

        const hasMatchingPrefix = (
          previousChunks: readonly { content: object }[],
          length: number
        ) => {
          if (previousChunks.length < length) return false;
          for (let index = 0; index < length; index++) {
            if (previousChunks[index]?.content !== processedContents[index]) {
              return false;
            }
          }
          return true;
        };

        return (chunk) => {
          const previousChunks = chunk.previousChunks ?? [];

          if (
            processedContents.length === previousChunks.length + 1 &&
            processedContents[processedContents.length - 1] === chunk.content &&
            hasMatchingPrefix(previousChunks, previousChunks.length)
          ) {
            return cachedResults;
          }

          if (!hasMatchingPrefix(previousChunks, processedContents.length)) {
            resetStream();
          }

          for (
            let index = processedContents.length;
            index < previousChunks.length;
            index++
          ) {
            processText(
              previousChunks[index].content
                .map((part) => part.text || '')
                .join('')
            );
            processedContents.push(previousChunks[index].content);
          }

          cachedResults = processText(chunk.text);
          processedContents.push(chunk.content);
          return cachedResults;
        };
      })(),

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
