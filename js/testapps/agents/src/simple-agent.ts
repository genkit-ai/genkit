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

import { z } from 'genkit';
import { ai } from './genkit.js';

export const simpleAgent = ai.defineCustomAgent(
  { name: 'simpleAgent' },
  async (sess, { sendChunk }) => {
    await sess.run(async (input) => {
      sendChunk({
        modelChunk: {
          content: [
            { text: 'Processing: ' + input.messages?.[0]?.content[0]?.text },
          ],
        },
      });
    });
    return {
      message: { role: 'model', content: [{ text: 'Session finished' }] },
    };
  }
);

export const testSimpleAgent = ai.defineFlow(
  {
    name: 'testSimpleAgent',
    inputSchema: z.string().default('Hello, how are you?'),
    outputSchema: z.any(),
  },
  async (text) => {
    const res = await simpleAgent.run(
      {
        messages: [{ role: 'user', content: [{ text }] }],
      },
      { init: {} }
    );
    return res.result;
  }
);
