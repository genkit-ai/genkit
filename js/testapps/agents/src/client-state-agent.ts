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

import { MessageSchema, z } from 'genkit';
import { ai } from './genkit.js';

export const clientStateAgent = ai.defineSessionFlow(
  { name: 'clientStateAgent' }, // No store!
  async (sess, { sendChunk }) => {
    await sess.run(async (input) => {
      const text =
        input.messages?.[input.messages.length - 1]?.content[0]?.text || '';
      const currentCustom = (sess.session.getCustom() as any) || { count: 0 };

      const newCustom = { ...currentCustom };
      newCustom.count = (currentCustom.count || 0) + 1;
      sess.session.updateCustom(() => newCustom);

      sendChunk({
        modelChunk: {
          content: [
            {
              text: `Processed: ${text}. In-memory Turn count: ${newCustom.count}`,
            },
          ],
        },
      });
    });

    const msgs = sess.session.getMessages();
    return {
      message: msgs[msgs.length - 1],
    };
  }
);

export const testClientStateAgent = ai.defineFlow(
  {
    name: 'testClientStateAgent',
    inputSchema: z.object({
      state: z.any().optional(),
      messages: z.array(MessageSchema).default([
        {
          role: 'user',
          content: [{ text: 'foo' }],
        },
        {
          role: 'model',
          content: [{ text: 'bar' }],
        },
      ]),
      text: z.string().default('Hello, increment the turn count!'),
    }),

    outputSchema: z.any(),
  },
  async (input, { sendChunk }) => {
    const res = await clientStateAgent.run(
      {
        messages: [{ role: 'user' as const, content: [{ text: input.text }] }],
      },
      {
        init: {
          state: {
            ...input.state,
            messages: input.messages,
          },
        },
        onChunk: sendChunk,
      }
    );
    return {
      init: {
        state: {
          ...input.state,
          messages: input.messages,
        },
      },
      result: res,
    };
  }
);
