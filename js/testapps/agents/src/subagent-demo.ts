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
 *   • The `researcher` sub-agent handles research tasks
 *   • The `coder` sub-agent handles code generation tasks
 *   • The `orchestrator` main agent decides which sub-agent to delegate to
 *
 * The middleware injects a `call_agent` tool that the orchestrator model can
 * call. When it does, the middleware intercepts the call, runs the appropriate
 * sub-agent, and returns the sub-agent's response as the tool result.
 */

import { agents } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Sub-Agent 1: Researcher — answers research questions
// ---------------------------------------------------------------------------

const getWebResults = ai.defineTool(
  {
    name: 'getWebResults',
    description: 'Search the web for information on a topic.',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ results: z.string() }),
  },
  async (input) => {
    // Simulated web search results.
    return {
      results: `Top results for "${input.query}": [1] Wikipedia article about ${input.query}. [2] Recent research paper on ${input.query}.`,
    };
  }
);

const researcher = ai.defineAgent({
  name: 'researcher',
  model: 'googleai/gemini-flash-latest',
  system:
    'You are a thorough research assistant. When asked a question, use the getWebResults tool to find information, then provide a clear, well-sourced answer.',
  tools: [getWebResults],
});

// ---------------------------------------------------------------------------
// Sub-Agent 2: Coder — generates and explains code
// ---------------------------------------------------------------------------

const coder = ai.defineAgent({
  name: 'coder',
  model: 'googleai/gemini-flash-latest',
  system:
    'You are an expert programmer. When asked to write code, provide clean, well-commented code with explanations. Use TypeScript by default unless asked otherwise.',
});

// ---------------------------------------------------------------------------
// Main Orchestrator Agent — delegates to sub-agents
// ---------------------------------------------------------------------------

export const orchestratorAgent = ai.defineAgent({
  name: 'orchestrator',
  model: 'googleai/gemini-flash-latest',
  system: `You are a helpful project assistant. You have access to specialized sub-agents:

- **researcher**: Use for research questions, fact-finding, looking up information
- **coder**: Use for writing code, debugging, code explanations

Analyze the user's request and delegate to the appropriate sub-agent using the call_agent tool.
If the request requires both research AND code, call them sequentially.
After receiving sub-agent responses, synthesize a final answer for the user.`,
  use: [agents({ agents: ['researcher', 'coder'] })],
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
