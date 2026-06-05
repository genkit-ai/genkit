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

import { googleAI } from '@genkit-ai/google-genai';
import { z } from 'genkit';
import { ai } from '../genkit.js';

//
// ATM Interrupt Showcase
//

const depositCheck = ai.defineTool(
  {
    name: 'depositCheck',
    description: "Deposits a check into the user's account.",
    inputSchema: z.object({
      amount: z.number().describe('The check amount in dollars'),
      accountType: z
        .enum(['checking', 'savings'])
        .describe('The destination account'),
    }),
    outputSchema: z.object({
      success: z.boolean(),
      message: z.string(),
    }),
  },
  async (input) => {
    return {
      success: true,
      message: `Deposited $${input.amount} into your ${input.accountType} account.`,
    };
  }
);

const askAtmPin = ai.defineInterrupt({
  name: 'askAtmPin',
  description:
    'Prompts the user to enter their 4-digit ATM PIN to authorize a sensitive transaction.',
  inputSchema: z.object({
    promptMessage: z.string().describe('The message requesting the PIN'),
  }),
  outputSchema: z.string().describe('The 4-digit PIN'),
});

const withdrawMoney = ai.defineTool(
  {
    name: 'withdrawMoney',
    description: "Withdraws money from the user's bank account.",
    inputSchema: z.object({
      amount: z.number().describe('The amount of money to withdraw in dollars'),
      accountType: z
        .enum(['checking', 'savings'])
        .describe('The account to withdraw from'),
    }),
    outputSchema: z.object({
      success: z.boolean(),
      message: z.string(),
    }),
  },
  async (input, { interrupt, resumed }) => {
    const resumedStatus = (resumed as Record<string, any>)?.status;

    // If the user rejected the withdrawal during the interrupt phase
    if (resumedStatus === 'REJECTED') {
      return {
        success: false,
        message: `Withdrawal of $${input.amount} was rejected by the user.`,
      };
    }

    // Trigger an interrupt to confirm if amount > $200
    if (resumedStatus !== 'APPROVED' && input.amount > 200) {
      interrupt({
        message: `Withdrawing $${input.amount} is a large transaction. Please approve this withdrawal from your ${input.accountType} account.`,
      });
    }

    // If approved, or if amount <= 200, perform withdrawal
    return {
      success: true,
      message: `Successfully withdrew $${input.amount} from your ${input.accountType} account.`,
    };
  }
);

const transferMoney = ai.defineTool(
  {
    name: 'transferMoney',
    description:
      'Transfers money from a source account to a destination account.',
    inputSchema: z.object({
      source: z.enum(['checking', 'savings']).describe('The source account'),
      destination: z
        .enum(['checking', 'savings'])
        .describe('The destination account'),
      amount: z.number().describe('The amount to transfer in dollars'),
    }),
    outputSchema: z.object({
      success: z.boolean(),
      message: z.string(),
    }),
  },
  async (input, { interrupt, resumed }) => {
    const resumedStatus = (resumed as Record<string, any>)?.status;

    if (resumedStatus === 'REJECTED') {
      return {
        success: false,
        message: `Transfer of $${input.amount} was rejected by the user.`,
      };
    }

    // Large transfers (e.g. > $100) require approval
    if (resumedStatus !== 'APPROVED' && input.amount > 100) {
      interrupt({
        message: `Please confirm you want to transfer $${input.amount} from ${input.source} to ${input.destination}.`,
      });
    }

    return {
      success: true,
      message: `Successfully transferred $${input.amount} from ${input.source} to ${input.destination}.`,
    };
  }
);

export const atmAgentPrompt = ai.definePrompt({
  name: 'atmAgentPrompt',
  description: 'ATM Agent prompt showcasing interrupts and tools.',
  model: googleAI.model('gemini-flash-latest'),
  input: {
    schema: z.object({
      message: z
        .string()
        .describe('The user request (e.g. "Withdraw $300 from checking")'),
    }),
  },
  output: {
    format: 'text',
  },
  config: {
    temperature: 1.0,
  },
  tools: [depositCheck, askAtmPin, withdrawMoney, transferMoney],
  system: `
    You are a helpful and secure ATM banking assistant.

    Guidelines:
    1. If the user wants to deposit a check, call the \`depositCheck\` tool.
       - Note: Deposits of more than $1000 require the user to enter their PIN first. If the deposit is over $1000, you MUST use the \`askAtmPin\` tool to request and verify their PIN before performing the deposit.
    2. If the user wants to withdraw money, call the \`withdrawMoney\` tool.
    3. If the user wants to transfer money between accounts, call the \`transferMoney\` tool.
  `,
  prompt: 'Help the user with their request: "{{message}}"',
});
