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

import { ExecutablePrompt, z } from 'genkit';
import { triageAgent } from '../agents';
import { ai } from '../config/genkit';

export const flow = ai.defineFlow(
  {
    name: 'multiAgentMultiModel',
    inputSchema: z.object({
      userInput: z
        .string()
        .default('I want to buy an UltraBook Air, can you help me with this?'),
    }),
    outputSchema: z.string(),
  },
  async (input) => {
    let currentAgent: ExecutablePrompt<any> = triageAgent;
    let textInput = input.userInput;
    const history: any[] = [];

    while (true) {
      history.push({ role: 'user' as const, content: [{ text: textInput }] });
      const response = await currentAgent(textInput, { messages: history });

      if (response.finishReason === 'interrupted') {
        const interrupt = response.interrupts.find(
          (i) => i.toolRequest?.name === 'transferToAgent'
        );
        if (interrupt) {
          const agentName = (interrupt.toolRequest.input as any).agentName;

          // Resolve the target prompt specialist agent
          currentAgent = ai.prompt(agentName);
          textInput = 'Please continue with the new specialist.';
          continue;
        }
      }

      return response.text;
    }
  }
);
