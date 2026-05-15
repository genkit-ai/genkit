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

import { openAI } from '@genkit-ai/compat-oai/openai';

(async () => {
  const oai = openAI();
  const gpt4o = await oai.model('gpt-4o');
  const response = await gpt4o({
    messages: [
      {
        role: 'user',
        content: [{ text: 'what is a gablorken of 4!' }],
      },
    ],
    tools: [
      {
        name: 'gablorken',
        description: 'calculates a gablorken',
        inputSchema: {
          type: 'object',
          properties: {
            value: {
              type: 'number',
              description: 'the value to calculate gablorken for',
            },
          },
        },
      },
    ],
  });

  console.log(JSON.stringify(response.message, undefined, 2));
})();
