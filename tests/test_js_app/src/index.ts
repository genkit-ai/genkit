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

import { MessageData, genkit, z } from 'genkit';
import { getBasicUsageStats } from 'genkit/model';

const ai = genkit({
  plugins: [],
});

ai.defineModel(
  {
    name: 'customReflector',
  },
  async (input) => {
    // In Go, JSON object properties are output in sorted order.
    // JSON.stringify uses the order they appear in the program.
    // So swap the order here to match Go.
    const m = input.messages[0];
    input.messages[0] = { content: m.content, role: m.role };
    const message: MessageData = {
      role: 'model',
      content: [
        {
          text: JSON.stringify(input),
        },
      ],
    };
    return {
      finishReason: 'stop',
      message,
      usage: getBasicUsageStats(input.messages, message),
    };
  }
);

export const testFlow = ai.defineFlow(
  { name: 'testFlow', inputSchema: z.string(), outputSchema: z.string() },
  async (subject) => {
    const response = await ai.generate({
      model: 'customReflector',
      prompt: subject,
    });

    const want = `{"messages":[{"content":[{"text":"${subject}"}],"role":"user"}],"tools":[],"output":{"format":"text"}}`;
    if (response.text !== want) {
      throw new Error(`Expected ${want} but got ${response.text}`);
    }

    return 'Test flow passed';
  }
);

// genkit flow:run streamy 5 -s
export const streamy = ai.defineFlow(
  {
    name: 'streamy',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: z.object({ count: z.number() }),
  },
  async (count, { sendChunk }) => {
    let i = 0;
    for (; i < count; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      sendChunk({ count: i });
    }
    return `done: ${count}, streamed: ${i} times`;
  }
);
