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
  description: 'Ask the user for approval before proceeding with a sensitive action.',
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

export const bankingPrompt = ai.definePrompt({
  name: 'bankingPrompt',
  model: 'googleai/gemini-2.5-flash',
  input: { schema: z.object({ request: z.string() }) },
  system:
    'You are a helpful banking assistant. If the user wants to transfer money, ALWAYS use the userApproval interrupt to confirm the details before executing the transferMoney tool.',
  tools: [userApproval, transferMoney],
});

export const bankingAgent = ai.defineSessionFlowFromPrompt({
  promptName: 'bankingPrompt',
  defaultInput: { request: 'I need help with my account.' },
  store,
});

export const testBankingAgent = ai.defineFlow(
  {
    name: 'testBankingAgent',
    inputSchema: z.string().default('Transfer $500 to my savings account.'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    let session = bankingAgent.streamBidi({});
    session.send({
      messages: [{ role: 'user', content: [{ text }] }],
    });
    session.close();

    for await (const chunk of session.stream) {
      sendChunk(chunk);
    }

    let output = await session.output;
    
    // Check if the agent paused for approval
    const lastMessage = output.message;
    const approvalRequest = lastMessage?.content.find(p => p.toolRequest?.name === 'userApproval');

    if (approvalRequest && approvalRequest.toolRequest) {
      sendChunk({ status: 'Agent interrupted! Requesting user approval...' });
      
      // Simulate user approval
      const approvalResponse = {
        toolResponse: {
          name: 'userApproval',
          ref: approvalRequest.toolRequest.ref,
          output: { approved: true, feedback: 'Looks good' },
        }
      };

      // Create a new session attached to the interrupted flow's snapshot
      session = bankingAgent.streamBidi({ snapshotId: output.snapshotId });
      
      // Send the approval back to the flow using toolRestarts
      // Alternatively, we can pass it as a tool message. 
      session.send({
        messages: [{ role: 'tool', content: [approvalResponse] }],
      });
      session.close();

      for await (const chunk of session.stream) {
        sendChunk(chunk);
      }

      output = await session.output;
    }

    return output;
  }
);
