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

import { googleAI } from '@genkit-ai/google-genai';
import { genkit, z } from 'genkit';

const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-flash-latest'),
});

const lookupUsers = ai.defineTool(
  {
    name: 'lookupUsers',
    description: 'use this tool to list users',
    outputSchema: z.array(z.object({ name: z.string(), id: z.number() })),
  },
  async () => [
    { id: 123, name: 'Michael Bleigh' },
    { id: 456, name: 'Pavel Jbanov' },
    { id: 789, name: 'Chris Gill' },
    { id: 1122, name: 'Marissa Christy' },
  ]
);

async function main() {
  const { stream } = await ai.generateStream({
    prompt:
      'use the lookupUsers tool and generate silly nicknames for each, then generate 50 fake users in the same format. return a JSON array.',
    output: {
      format: 'json',
      schema: z.array(
        z.object({ id: z.number(), name: z.string(), nickname: z.string() })
      ),
    },
    tools: [lookupUsers],
  });

  for await (const chunk of stream) {
    console.log('raw:', chunk);
    console.log('output:', chunk.output);
  }
}
main();
