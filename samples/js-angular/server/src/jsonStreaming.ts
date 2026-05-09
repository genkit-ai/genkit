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

import { googleAI } from '@genkit-ai/google-genai';
import { z } from 'genkit';
import { Allow, parse } from 'partial-json';
import { ai } from './genkit.js';

const GameCharactersSchema = z.object({
  characters: z
    .array(
      z
        .object({
          name: z.string().describe('Name of a character'),
          abilities: z
            .array(z.string())
            .describe('Various abilities (strength, magic, archery, etc.)'),
        })
        .describe('Game character')
    )
    .describe('Characters'),
});

export const streamCharacters = ai.defineFlow(
  {
    name: 'streamCharacters',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: GameCharactersSchema,
  },
  async (count, { sendChunk }) => {
    const { response, stream } = await ai.generateStream({
      model: googleAI.model('gemini-flash-latest'),
      output: {
        format: 'json',
        schema: GameCharactersSchema,
      },
      config: {
        temperature: 1,
      },
      prompt: `Respond as JSON only. Generate ${count} different RPG game characters.`,
    });

    let buffer = '';
    for await (const chunk of stream) {
      buffer += chunk.content[0].text!;
      if (buffer.length > 10) {
        sendChunk(parse(maybeStripMarkdown(buffer), Allow.ALL));
      }
    }

    return (await response).text;
  }
);

const markdownRegex = /^\s*(```json)?((.|\n)*?)(```)?\s*$/i;
function maybeStripMarkdown(withMarkdown: string) {
  const mdMatch = markdownRegex.exec(withMarkdown);
  if (!mdMatch) {
    return withMarkdown;
  }
  return mdMatch[2];
}
