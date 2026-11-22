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

import {
  choice,
  noul,
  ranked,
  responseInfo,
  score,
  systemOne,
  type Answer,
} from '@genkit-ai/systemone';
import { genkit, z } from 'genkit';

// jev, reached through OpenRouter's Decisions API, with OPENROUTER_API_KEY.
// Without it the plugin is created with apiKey: false, so the Dev UI still
// starts and lists the models, and a flow fails with an error that says how
// to give the key.
const model = 'openrouter-decisions/typesafe/jev-1.13';

const ai = genkit({
  plugins: [
    systemOne.openRouter({ apiKey: process.env.OPENROUTER_API_KEY || false }),
  ],
  promptDir: 'prompts',
});

// The decision: one question per field. The options of a choice and the
// levels of a score belong to the question, with what each one means.
const Triage = ai.defineSchema(
  'Triage',
  z.object({
    department: choice(
      {
        billing: 'Payments, invoicing, refunds',
        technical: 'Bugs, outages, integrations',
        sales: 'Plans, pricing, upgrades',
        other: 'None of the above',
      },
      {
        guidance: {
          billing: {
            not_for: 'Where an order is, or when it arrives',
            examples: ['I was charged twice for one order.'],
          },
        },
      }
    ).describe('Which team should handle this?'),
    isUrgent: noul({
      yes: 'Names a deadline, or says now or today',
      no: 'No time pressure is expressed',
    }).describe('Does the ticket explicitly communicate time pressure?'),
    frustration: score(['Calm', 'Concerned but civil', 'Very angry']).describe(
      'How frustrated is the customer?'
    ),
  })
);

const TicketInput = z.object({
  ticket: z.string(),
  accountTier: z.string().default('free'),
});

// Three questions about one ticket in a single call, routed in code behind a
// confidence gate. The thresholds are the application's, next to the
// questions; the model returns probabilities.
export const triageFlow = ai.defineFlow(
  { name: 'triageFlow', inputSchema: TicketInput },
  async ({ ticket, accountTier }) => {
    const response = await ai.generate({
      model,
      system: `The state is a support ticket from a customer on the ${accountTier} plan.`,
      prompt: ticket,
      output: { schema: Triage },
    });
    const out = response.output!;
    const route =
      (out.department.confidence ?? 0) < 0.6 ? 'human' : out.department.choice;
    return {
      route,
      urgent: out.isUrgent.noul >= 0.7,
      frustration: out.frustration.score,
      answeredBy: responseInfo(response).model,
      decision: out,
    };
  }
);

// The same decision from a prompt file that renders the state as JSON.
export const triagePromptFlow = ai.defineFlow(
  { name: 'triagePromptFlow', inputSchema: TicketInput },
  async (input) => {
    const triage = ai.prompt<typeof TicketInput, typeof Triage>('triage');
    const { output } = await triage(input);
    return output;
  }
);

// A yes/no battery over a message, reduced to pass, review, or block.
const Screen = z.object({
  personalData: noul().describe(
    'Does the message contain personal data such as an address, a phone number, or a card number?'
  ),
  abusive: noul().describe('Is the message abusive toward a person?'),
  offTopic: noul().describe(
    'Is the message about something other than our product?'
  ),
});

export const screenFlow = ai.defineFlow(
  { name: 'screenFlow', inputSchema: z.string() },
  async (message) => {
    const { output } = await ai.generate({
      model,
      prompt: message,
      output: { schema: Screen },
    });
    const worst = Math.max(...Object.values(output!).map((a) => a.noul));
    return {
      verdict: worst >= 0.8 ? 'block' : worst >= 0.5 ? 'review' : 'pass',
      answers: output,
    };
  }
);

// Scores every passage of a shortlist against a query, one call per passage,
// and orders them. A decision model has no per-item questions.
const Relevance = z.object({
  relevance: score([
    'Unrelated',
    'Mentions the topic',
    'Answers part of the query',
    'Answers the query',
  ]).describe('How well does the passage answer the query?'),
});

export const rankFlow = ai.defineFlow(
  {
    name: 'rankFlow',
    inputSchema: z.object({ query: z.string(), passages: z.array(z.string()) }),
  },
  async ({ query, passages }) => {
    const scored = await Promise.all(
      passages.map(async (passage) => {
        const { output } = await ai.generate({
          model,
          prompt: [{ data: { query, passage } }],
          output: { schema: Relevance },
        });
        return { passage, score: output!.relevance.score };
      })
    );
    return scored.sort((a, b) => b.score - a.score);
  }
);

// The built-in enum output format: the system message is the question, and
// the values are the options.
export const teamFlow = ai.defineFlow(
  { name: 'teamFlow', inputSchema: z.string() },
  async (ticket) => {
    const response = await ai.generate({
      model,
      system: 'Which team should handle this ticket?',
      prompt: ticket,
      output: {
        format: 'enum',
        schema: z.enum(['billing', 'technical', 'sales']),
      },
    });
    return response.output;
  }
);

// A choice whose options, the tools on hand, are only known at run time.
const tools = [
  { name: 'search', description: 'Looks facts up on the web' },
  { name: 'calendar', description: "Reads and books the user's meetings" },
  { name: 'calculator', description: 'Does arithmetic' },
  { name: 'none', description: 'No tool is needed' },
];

export const toolFlow = ai.defineFlow(
  { name: 'toolFlow', inputSchema: z.string() },
  async (request) => {
    const response = await ai.generate({
      model,
      prompt: request,
      output: {
        jsonSchema: systemOne.schema({
          tool: {
            type: 'choice',
            instructions: 'Which tool serves the request?',
            options: tools.map((t) => ({
              name: t.name,
              criteria: t.description,
            })),
          },
        }),
      },
    });
    const { tool } = response.output as Record<string, Answer>;
    return {
      tool: (tool.confidence ?? 0) >= 0.8 ? tool.choice : 'ask the user',
      ranked: ranked({
        choice: tool.choice!,
        probabilities: tool.probabilities,
        confidence: tool.confidence,
      }),
    };
  }
);
