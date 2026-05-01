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

import * as assert from 'assert';
import { genkit } from 'genkit/beta';
import { describe, it } from 'node:test';
import { agents } from '../src/agents.js';

describe('agents middleware', () => {
  it('injects call_agent tool and system prompt with agent list', async () => {
    const ai = genkit({});

    // Define a mock model for the sub-agent.
    const researcherModel = ai.defineModel(
      { name: 'researcher-model-' + Math.random() },
      async () => ({
        message: {
          role: 'model' as const,
          content: [{ text: 'Research result: quantum computing is cool.' }],
        },
      })
    );

    // Define a sub-agent using defineAgent (registers at /agent/researcher).
    ai.defineAgent({
      name: 'researcher',
      model: researcherModel,
      system: 'You are a research assistant.',
    });

    let modelTurn = 0;
    const mainModel = ai.defineModel(
      { name: 'main-model-' + Math.random() },
      async (req) => {
        modelTurn++;
        if (modelTurn === 1) {
          // Verify system prompt contains sub-agents instructions.
          const systemMsg = req.messages?.find((m) => m.role === 'system');
          assert.ok(systemMsg, 'System message should exist');
          const hasAgentInstructions = systemMsg!.content.some((p) =>
            p.text?.includes('<sub-agents>')
          );
          assert.ok(
            hasAgentInstructions,
            'System should contain sub-agent instructions'
          );

          // Model calls the call_agent tool.
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'call_agent',
                    input: {
                      agent: 'researcher',
                      task: 'Explain quantum computing briefly.',
                    },
                  },
                },
              ],
            },
          };
        }
        // Second turn: model produces final text.
        return {
          message: {
            role: 'model' as const,
            content: [
              { text: 'Based on the research: quantum computing uses qubits.' },
            ],
          },
        };
      }
    );

    const result = await ai.generate({
      model: mainModel,
      prompt: 'Tell me about quantum computing',
      use: [agents({ agents: ['researcher'] })],
    });

    assert.ok(result.text.includes('quantum computing'));

    // Verify the tool message came back with the sub-agent's response.
    const toolMsg = result.messages.find((m) => m.role === 'tool');
    assert.ok(toolMsg, 'Should have a tool response message');
    const toolResponse = toolMsg!.content.find((p) => p.toolResponse);
    assert.ok(toolResponse, 'Should have a tool response part');
    assert.strictEqual(toolResponse!.toolResponse!.name, 'call_agent');
    const toolOutput = toolResponse!.toolResponse!.output as {
      response: string;
    };
    assert.ok(
      toolOutput.response.includes('quantum computing'),
      'Sub-agent response should be in tool output'
    );
  });

  it('returns error message for unknown agent name', async () => {
    const ai = genkit({});

    // Define a mock model for the coder sub-agent.
    const coderModel = ai.defineModel(
      { name: 'coder-model-' + Math.random() },
      async () => ({
        message: {
          role: 'model' as const,
          content: [{ text: 'code result' }],
        },
      })
    );

    // Register a sub-agent so the middleware can resolve at least one.
    ai.defineAgent({
      name: 'coder',
      model: coderModel,
      system: 'You write code.',
    });

    let modelTurn = 0;
    const mainModel = ai.defineModel(
      { name: 'main-err-' + Math.random() },
      async () => {
        modelTurn++;
        if (modelTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'call_agent',
                    input: {
                      agent: 'nonexistent',
                      task: 'do something',
                    },
                  },
                },
              ],
            },
          };
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'handled error' }],
          },
        };
      }
    );

    // 'nonexistent' is in the agents list (so the enum validates) but
    // has no corresponding agent registered — the middleware should return
    // an error as tool output instead of throwing.
    const result = await ai.generate({
      model: mainModel,
      prompt: 'test',
      use: [agents({ agents: ['coder', 'nonexistent'] })],
    });

    // The model should still get a response (error was returned as tool output, not thrown).
    assert.ok(result.text);
  });

  it('supports custom tool name via toolName option', async () => {
    const ai = genkit({});

    const helperModel = ai.defineModel(
      { name: 'helper-model-' + Math.random() },
      async () => ({
        message: {
          role: 'model' as const,
          content: [{ text: 'helped!' }],
        },
      })
    );

    ai.defineAgent({
      name: 'helper',
      model: helperModel,
      system: 'You help.',
    });

    let modelTurn = 0;
    const mainModel = ai.defineModel(
      { name: 'main-custom-' + Math.random() },
      async (req) => {
        modelTurn++;
        if (modelTurn === 1) {
          // Verify custom tool name in system prompt.
          const systemMsg = req.messages?.find((m) => m.role === 'system');
          const hasCustomName = systemMsg?.content.some((p) =>
            p.text?.includes('delegate_task')
          );
          assert.ok(hasCustomName, 'System should reference custom tool name');

          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_task',
                    input: { agent: 'helper', task: 'help me' },
                  },
                },
              ],
            },
          };
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'final' }],
          },
        };
      }
    );

    const result = await ai.generate({
      model: mainModel,
      prompt: 'test custom name',
      use: [agents({ agents: ['helper'], toolName: 'delegate_task' })],
    });

    assert.ok(result.text);
  });

  it('throws if no agents provided', () => {
    const ai = genkit({});

    assert.throws(() => {
      // Instantiating the middleware should throw.
      agents.instantiate({
        config: { agents: [] },
        ai,
        pluginConfig: undefined,
      });
    }, /at least one agent name/);
  });
});
