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
import { InMemorySessionStore } from 'genkit/beta';
import { ai } from './genkit.js';

const store = new InMemorySessionStore<{}>();

export const userApproval = ai.defineInterrupt({
  name: 'userApproval',
  description:
    'Ask the user for approval before proceeding with a sensitive action.',
  inputSchema: z.object({
    action: z.string().describe('The action to be approved'),
    details: z.string().describe('Details about the action'),
  }),
  outputSchema: z.object({
    approved: z.boolean().describe('Whether the user approved the action'),
    feedback: z.string().optional().describe('Optional feedback from the user'),
  }),
});

export const transferMoney = ai.defineTool(
  {
    name: 'transferMoney',
    description: 'Transfer money to a specified account.',
    inputSchema: z.object({
      amount: z.number(),
      toAccount: z.string(),
    }),
    outputSchema: z.object({
      success: z.boolean(),
      transactionId: z.string(),
    }),
  },
  async (input) => {
    return { success: true, transactionId: `txn-${Date.now()}` };
  }
);

export const bankingAgent = ai.defineAgent({
  name: 'bankingAgent',
  system:
    'You are a helpful banking assistant. If the user wants to transfer money, ALWAYS use the userApproval interrupt to confirm the details before executing the transferMoney tool.',
  tools: [userApproval, transferMoney],
  store,
});

export const testBankingAgent = ai.defineFlow(
  {
    name: 'testBankingAgent',
    inputSchema: z.string().default('Transfer $500 to my savings account.'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    // Start a chat and send the user's request.
    const chat = bankingAgent.chat();
    const turn = chat.sendStream(text);
    for await (const chunk of turn.stream) {
      sendChunk(chunk.raw);
    }
    let res = await turn.response;

    // Check if the agent paused on the userApproval interrupt. `res.interrupts`
    // surfaces the paused tool requests with `respond`/`restart` builders.
    const approval = res.interrupts.find((i) => i.name === 'userApproval');

    if (approval) {
      sendChunk({ status: 'Agent interrupted! Requesting user approval...' });

      // Resume the same chat (it tracks the snapshot automatically). Use the
      // interrupt's `respond` builder to provide the tool output directly
      // without re-executing the tool.
      const resumeTurn = chat.resumeStream({
        respond: [approval.respond({ approved: true, feedback: 'Looks good' })],
      });
      for await (const chunk of resumeTurn.stream) {
        sendChunk(chunk.raw);
      }
      res = await resumeTurn.response;
    }

    return res.raw;
  }
);
