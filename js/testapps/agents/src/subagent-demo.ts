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
 * Sub-Agent Delegation Demo — demonstrates the `agents` middleware
 *
 * This sample showcases how to use the `agents` middleware to let a main
 * orchestrator agent delegate tasks to specialized sub-agents:
 *
 *   • The `researcher` sub-agent handles research tasks (description
 *     auto-discovered from the agent's `description` field).
 *   • The `coder` sub-agent handles code generation (with an explicit
 *     description override in the middleware config).
 *   • The `orchestrator` main agent decides which sub-agent to delegate to.
 *
 * The middleware injects a dedicated tool per sub-agent (e.g.
 * `delegate_to_researcher`, `delegate_to_coder`). When the orchestrator
 * model calls one of these tools the middleware intercepts the call, runs
 * the appropriate sub-agent, and returns its response as the tool result.
 *
 * Key features demonstrated:
 *   - Per-agent delegation tools (one tool per agent, richer descriptions)
 *   - Auto-discovered agent descriptions from registry metadata
 *   - Explicit description overrides via config
 *   - `maxDelegations` guard rail to prevent runaway loops
 *   - `historyLength` to forward conversation context to sub-agents
 */

import { agents, retry } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { ai, defaultModel } from './genkit.js';

// ---------------------------------------------------------------------------
// Sub-Agent 1: Researcher — answers research questions
// ---------------------------------------------------------------------------

const researcher = ai.defineAgent({
  name: 'researcher',
  description:
    'A thorough research assistant that searches the web and provides well-sourced answers.',
  model: defaultModel,
  config: {
    tools: [{ googleSearch: {} }],
  },
  system:
    'You are a thorough research assistant. When asked a question, use the getWebResults tool to find information, then provide a clear, well-sourced answer.',
  maxTurns: 10,
  use: [retry()],
});

// ---------------------------------------------------------------------------
// Sub-Agent 2: Coder — generates and explains code
// ---------------------------------------------------------------------------

const coder = ai.defineAgent({
  name: 'coder',
  description: 'An expert programmer that writes clean, well-commented code.',
  maxTurns: 10,
  system:
    'You are an expert programmer. When asked to write code, provide clean, well-commented code with explanations. Use TypeScript by default unless asked otherwise.',
  use: [retry()],
});

// ---------------------------------------------------------------------------
// Main Orchestrator Agent — delegates to sub-agents
//
// Note: the system prompt no longer needs to describe the sub-agents — the
// middleware auto-discovers descriptions from the agent registry and injects
// them into the system prompt automatically.
// ---------------------------------------------------------------------------

export const orchestratorAgent = ai.defineAgent({
  name: 'orchestrator',
  system: `You are a helpful project assistant.

Analyze the user's request and delegate to the appropriate sub-agent.
If the request requires both research AND code, call them sequentially.
After receiving sub-agent responses, synthesize a final answer for the user.`,
  use: [
    agents({
      agents: [
        // Auto-discover description from the agent's registry metadata:
        'researcher',
        // Override the description for the orchestrator's context:
        {
          name: 'coder',
          description:
            'Writes, debugs, and explains code. Use for any programming tasks.',
        },
      ],
      maxDelegations: 5,
      historyLength: 4,
    }),
    retry(),
  ],
});

// ---------------------------------------------------------------------------
// Test flow — demonstrates sub-agent delegation
// ---------------------------------------------------------------------------

export const testSubAgentDemo = ai.defineFlow(
  {
    name: 'testSubAgentDemo',
    inputSchema: z
      .string()
      .default(
        'Research the best sorting algorithms and then write a TypeScript implementation of quicksort.'
      ),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const res = await orchestratorAgent.run(
      { messages: [{ role: 'user' as const, content: [{ text }] }] },
      { init: {}, onChunk: sendChunk }
    );
    return res.result;
  }
);

// ---------------------------------------------------------------------------
// Test flow — simple delegation to a single sub-agent
// ---------------------------------------------------------------------------

export const testSubAgentSimple = ai.defineFlow(
  {
    name: 'testSubAgentSimple',
    inputSchema: z
      .string()
      .default('Write a function that calculates the fibonacci sequence.'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const res = await orchestratorAgent.run(
      { messages: [{ role: 'user' as const, content: [{ text }] }] },
      { init: {}, onChunk: sendChunk }
    );
    return res.result;
  }
);
