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

/**
 * Agent Transfer / Handoff Demo — demonstrates the `handoff` middleware.
 *
 * This sample models a customer-service experience where the user always talks
 * to a single agent, but the *persona* driving that agent changes as the
 * conversation evolves:
 *
 *   • The `triage` persona greets the user, figures out their issue, and
 *     transfers them to a specialist.
 *   • The `refund` persona handles refunds and has access to order/refund
 *     tools (`lookupOrder`, `issueRefund`).
 *   • The `billing` persona answers invoice/billing questions and has access
 *     to `getInvoice`.
 *
 * Unlike the `agents` (delegation) middleware — where an orchestrator fires off
 * a one-shot subtask and gets a result back — `handoff` *transfers control*.
 * When the model calls a `transfer_to_<persona>` tool, the middleware swaps the
 * active system prompt and the visible toolset for all subsequent turns, so the
 * user keeps talking directly to the specialist. Specialists can transfer back
 * to triage or to each other.
 *
 * Try it:
 *   - "I was charged twice for order #1234, I want a refund"
 *       → triage transfers to refund, which looks up the order and refunds it.
 *   - "Why is my latest invoice so high?"
 *       → triage transfers to billing.
 */

import { handoff } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { ai, defaultModel } from './genkit.js';

// ---------------------------------------------------------------------------
// Tools — the specialist personas use these. Only the *active* persona's tools
// are exposed to the model at any given turn (the handoff middleware gates
// tool visibility by name).
// ---------------------------------------------------------------------------

export const lookupOrder = ai.defineTool(
  {
    name: 'lookupOrder',
    description: 'Look up an order by its ID to inspect its details.',
    inputSchema: z.object({
      orderId: z.string().describe('The order ID, e.g. "1234".'),
    }),
    outputSchema: z.object({
      orderId: z.string(),
      item: z.string(),
      amount: z.number(),
      status: z.string(),
    }),
  },
  async ({ orderId }) => ({
    orderId,
    item: 'Wireless Headphones',
    amount: 79.99,
    status: 'delivered',
  })
);

export const issueRefund = ai.defineTool(
  {
    name: 'issueRefund',
    description: 'Issue a refund for an order.',
    inputSchema: z.object({
      orderId: z.string().describe('The order ID to refund.'),
      amount: z.number().describe('The amount to refund.'),
    }),
    outputSchema: z.object({
      refundId: z.string(),
      orderId: z.string(),
      amount: z.number(),
    }),
  },
  async ({ orderId, amount }) => ({
    refundId: `rf-${Date.now()}`,
    orderId,
    amount,
  })
);

export const getInvoice = ai.defineTool(
  {
    name: 'getInvoice',
    description: 'Fetch the most recent invoice for the current customer.',
    inputSchema: z.object({}),
    outputSchema: z.object({
      invoiceId: z.string(),
      total: z.number(),
      lineItems: z.array(z.object({ name: z.string(), amount: z.number() })),
    }),
  },
  async () => ({
    invoiceId: 'inv-2026-04',
    total: 129.97,
    lineItems: [
      { name: 'Subscription (Pro)', amount: 49.99 },
      { name: 'Overage charges', amount: 79.98 },
    ],
  })
);

// ---------------------------------------------------------------------------
// The host agent — a single agent that hosts multiple personas via `handoff`.
//
// The agent-level `system` prompt holds the shared brand voice; the active
// persona's instructions are layered on top each turn by the middleware.
// ---------------------------------------------------------------------------

export const customerServiceAgent = ai.defineAgent({
  name: 'customerServiceAgent',
  model: defaultModel,
  system:
    'You are Acme Corp customer support. Always be warm, concise, and helpful.',
  maxTurns: 10,
  use: [
    handoff({
      personas: [
        {
          name: 'triage',
          description:
            "Greets the user, figures out the user's issue, and routes them to the right specialist.",
          system:
            'You are the first point of contact. Greet the user, ask a clarifying ' +
            'question if needed to understand their issue, then transfer them to the ' +
            'specialist best suited to help. Do not try to resolve refund or billing ' +
            'issues yourself — transfer instead.',
        },
        {
          name: 'refund',
          description: 'Handles refund and return requests.',
          system:
            'You are the refunds specialist. Always look up the order with ' +
            'lookupOrder before issuing a refund, confirm the amount, then use ' +
            'issueRefund. If the user has a non-refund question, transfer them back ' +
            'to triage.',
          tools: ['lookupOrder', 'issueRefund'],
        },
        {
          name: 'billing',
          description: 'Answers billing and invoice questions.',
          system:
            'You are the billing specialist. Use getInvoice to inspect the ' +
            "customer's latest invoice and explain charges clearly. If the user " +
            'wants a refund, transfer them to the refund agent.',
          tools: ['getInvoice'],
        },
      ],
      defaultPersona: 'triage',
      maxTransfers: 5,
    }),
  ],
});

// ---------------------------------------------------------------------------
// Test flow — a single user turn (triage will typically transfer and the
// specialist will respond in the same generate call).
// ---------------------------------------------------------------------------

export const testCustomerServiceAgent = ai.defineFlow(
  {
    name: 'testCustomerServiceAgent',
    inputSchema: z
      .string()
      .default(
        'I was charged twice for order #1234 and I would like a refund.'
      ),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const chat = customerServiceAgent.chat();
    const turn = chat.sendStream(text);
    for await (const chunk of turn.stream) {
      sendChunk(chunk.raw);
    }
    const res = await turn.response;
    return res.raw;
  }
);

// ---------------------------------------------------------------------------
// Test flow — multi-turn conversation that flows across personas. The active
// persona persists across turns (the same chat/session is reused).
// ---------------------------------------------------------------------------

export const testCustomerServiceConversation = ai.defineFlow(
  {
    name: 'testCustomerServiceConversation',
    inputSchema: z.any(),
    outputSchema: z.any(),
  },
  async (_input, { sendChunk }) => {
    const chat = customerServiceAgent.chat();

    const say = async (text: string) => {
      sendChunk({ user: text });
      const turn = chat.sendStream(text);
      for await (const chunk of turn.stream) {
        sendChunk(chunk.raw);
      }
      return (await turn.response).raw;
    };

    // Triage → refund.
    await say('Hi, I need a refund for order #1234.');
    // Still in the refund persona; ask a billing question → refund transfers
    // back to triage, which routes to billing.
    const last = await say(
      'Thanks! While I have you — why was my latest invoice so high?'
    );
    return last;
  }
);
