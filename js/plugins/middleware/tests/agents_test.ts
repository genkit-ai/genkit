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
import { GenkitError, z, type MessageData } from 'genkit';
import {
  InMemorySessionStore,
  Session,
  genkit,
  type SessionSnapshot,
  type SessionStore,
} from 'genkit/beta';
import { describe, it } from 'node:test';
import { agents } from '../src/agents.js';
import { artifacts } from '../src/artifacts.js';

describe('agents middleware', () => {
  it('injects per-agent delegation tools and system prompt', async () => {
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

          // Verify per-agent tool name appears in instructions.
          const hasToolName = systemMsg!.content.some((p) =>
            p.text?.includes('delegate_to_researcher')
          );
          assert.ok(hasToolName, 'System should reference per-agent tool name');

          // Model calls the per-agent delegation tool.
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_researcher',
                    input: {
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
    assert.strictEqual(
      toolResponse!.toolResponse!.name,
      'delegate_to_researcher'
    );
    const toolOutput = toolResponse!.toolResponse!.output as {
      response: string;
    };
    assert.ok(
      toolOutput.response.includes('quantum computing'),
      'Sub-agent response should be in tool output'
    );
  });

  it('returns error message for unregistered agent', async () => {
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
          // Call the tool for an agent that is in config but not registered.
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_nonexistent',
                    input: {
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

    // 'nonexistent' is in the agents list (so its tool exists) but has
    // no corresponding agent registered — the middleware should return an
    // error as tool output instead of throwing.
    const result = await ai.generate({
      model: mainModel,
      prompt: 'test',
      use: [agents({ agents: ['coder', 'nonexistent'] })],
    });

    // The model should still get a response (error was returned as tool output).
    assert.ok(result.text);
  });

  it('supports custom tool prefix', async () => {
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
            p.text?.includes('ask_helper')
          );
          assert.ok(hasCustomName, 'System should reference custom tool name');

          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'ask_helper',
                    input: { task: 'help me' },
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
      prompt: 'test custom prefix',
      use: [agents({ agents: ['helper'], toolPrefix: 'ask' })],
    });

    assert.ok(result.text);
  });

  it('uses agent description objects in config', async () => {
    const ai = genkit({});

    const helperModel = ai.defineModel(
      { name: 'desc-model-' + Math.random() },
      async () => ({
        message: {
          role: 'model' as const,
          content: [{ text: 'I helped with code!' }],
        },
      })
    );

    ai.defineAgent({
      name: 'myagent',
      description: 'Registry description (should be overridden).',
      model: helperModel,
      system: 'You help.',
    });

    let modelTurn = 0;
    const mainModel = ai.defineModel(
      { name: 'main-desc-' + Math.random() },
      async (req) => {
        modelTurn++;
        if (modelTurn === 1) {
          // Verify the override description appears in system prompt.
          const systemMsg = req.messages?.find((m) => m.role === 'system');
          const hasOverrideDesc = systemMsg?.content.some((p) =>
            p.text?.includes('Custom override description')
          );
          assert.ok(
            hasOverrideDesc,
            'System should contain the override description'
          );

          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_myagent',
                    input: { task: 'do it' },
                  },
                },
              ],
            },
          };
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'done' }],
          },
        };
      }
    );

    const result = await ai.generate({
      model: mainModel,
      prompt: 'test descriptions',
      use: [
        agents({
          agents: [
            {
              name: 'myagent',
              description: 'Custom override description for tests.',
            },
          ],
        }),
      ],
    });

    assert.ok(result.text);
  });

  it('auto-discovers agent descriptions from registry', async () => {
    const ai = genkit({});

    const model = ai.defineModel(
      { name: 'autodesc-model-' + Math.random() },
      async () => ({
        message: {
          role: 'model' as const,
          content: [{ text: 'discovered!' }],
        },
      })
    );

    ai.defineAgent({
      name: 'smartagent',
      description: 'A very smart agent that knows everything.',
      model,
      system: 'You know things.',
    });

    let modelTurn = 0;
    const mainModel = ai.defineModel(
      { name: 'main-autodesc-' + Math.random() },
      async (req) => {
        modelTurn++;
        if (modelTurn === 1) {
          // Verify the auto-discovered description appears.
          const systemMsg = req.messages?.find((m) => m.role === 'system');
          const hasAutoDesc = systemMsg?.content.some((p) =>
            p.text?.includes('A very smart agent that knows everything')
          );
          assert.ok(
            hasAutoDesc,
            'System should contain auto-discovered description'
          );

          return {
            message: {
              role: 'model' as const,
              content: [{ text: 'no tools needed' }],
            },
          };
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'ok' }],
          },
        };
      }
    );

    const result = await ai.generate({
      model: mainModel,
      prompt: 'test auto-discovery',
      use: [agents({ agents: ['smartagent'] })],
    });

    assert.ok(result.text);
  });

  it('enforces maxDelegations limit', async () => {
    const ai = genkit({});

    const subModel = ai.defineModel(
      { name: 'sub-limit-' + Math.random() },
      async () => ({
        message: {
          role: 'model' as const,
          content: [{ text: 'sub result' }],
        },
      })
    );

    ai.defineAgent({
      name: 'worker',
      model: subModel,
      system: 'You work.',
    });

    let modelTurn = 0;
    const mainModel = ai.defineModel(
      { name: 'main-limit-' + Math.random() },
      async () => {
        modelTurn++;
        if (modelTurn <= 3) {
          // Keep trying to delegate (should hit limit after 2).
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_worker',
                    input: { task: `task ${modelTurn}` },
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
      prompt: 'test max delegations',
      use: [agents({ agents: ['worker'], maxDelegations: 2 })],
    });

    // The third delegation should have been rejected with a limit message.
    const toolMsgs = result.messages.filter((m) => m.role === 'tool');
    assert.ok(toolMsgs.length >= 3, 'Should have at least 3 tool responses');

    // Find the tool response that mentions the limit.
    const limitResponse = toolMsgs.find((m) =>
      m.content.some((p) => {
        const output = p.toolResponse?.output as { response?: string };
        return output?.response?.includes('Delegation limit reached');
      })
    );
    assert.ok(limitResponse, 'Should have a delegation limit response');
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
    }, /at least one agent/);
  });

  it('inline artifactStrategy includes artifact content in tool result', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-inline-artifacts' });

    // Sub-agent model: calls write_artifact, then responds.
    let subTurn = 0;
    const subModel = ai.defineModel(
      { name: 'sub-inline-' + Math.random() },
      async () => {
        subTurn++;
        if (subTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'write_artifact',
                    input: {
                      name: 'result.md',
                      content: '# Research Results\nSome findings.',
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
            content: [{ text: 'Here are my research results.' }],
          },
        };
      }
    );

    ai.defineAgent({
      name: 'inlineResearcher',
      model: subModel,
      system: 'You are a researcher.',
      use: [artifacts()],
    });

    // Main model: delegates to inlineResearcher, then produces final text.
    let mainTurn = 0;
    let capturedToolOutput: any;
    const mainModel = ai.defineModel(
      { name: 'main-inline-' + Math.random() },
      async (req) => {
        mainTurn++;
        if (mainTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_inlineResearcher',
                    input: { task: 'Research something.' },
                  },
                },
              ],
            },
          };
        }
        // Capture tool output from the delegation result.
        const toolMsg = req.messages?.find((m: any) => m.role === 'tool');
        if (toolMsg) {
          const toolResp = toolMsg.content.find((p: any) => p.toolResponse);
          capturedToolOutput = toolResp?.toolResponse?.output;
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'Synthesis complete.' }],
          },
        };
      }
    );

    await session.run(async () => {
      await ai.generate({
        model: mainModel,
        prompt: 'Research and summarize',
        use: [
          agents({
            agents: ['inlineResearcher'],
            artifactStrategy: 'inline',
          }),
        ],
      });
    });

    // Verify tool output contains artifact with content (inline strategy).
    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.ok(capturedToolOutput.artifacts, 'Should have artifacts in output');
    assert.ok(
      capturedToolOutput.artifacts.length > 0,
      'Should have at least one artifact'
    );

    const artifact = capturedToolOutput.artifacts[0];
    assert.ok(
      artifact.name.includes('inlineResearcher'),
      'Artifact name should be namespaced with agent name'
    );
    assert.ok(
      artifact.name.includes('result.md'),
      'Should contain original name'
    );
    assert.ok(
      artifact.content.includes('Research Results'),
      'Inline strategy should include content in tool result'
    );

    // Verify artifacts were also merged into parent session.
    const sessionArtifacts = session.getArtifacts();
    assert.ok(
      sessionArtifacts.length > 0,
      'Session should have merged artifacts'
    );
    assert.ok(
      sessionArtifacts[0].metadata?.source === 'inlineResearcher',
      'Merged artifact should have source metadata'
    );
  });

  it('session artifactStrategy includes only names in tool result', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-session-artifacts' });

    // Sub-agent model: writes artifact then responds.
    let subTurn = 0;
    const subModel = ai.defineModel(
      { name: 'sub-session-' + Math.random() },
      async () => {
        subTurn++;
        if (subTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'write_artifact',
                    input: {
                      name: 'code.ts',
                      content: 'console.log("hello world")',
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
            content: [{ text: 'Here is the code.' }],
          },
        };
      }
    );

    ai.defineAgent({
      name: 'sessionCoder',
      model: subModel,
      system: 'You write code.',
      use: [artifacts()],
    });

    let mainTurn = 0;
    let capturedToolOutput: any;
    const mainModel = ai.defineModel(
      { name: 'main-session-' + Math.random() },
      async (req) => {
        mainTurn++;
        if (mainTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_sessionCoder',
                    input: { task: 'Write hello world.' },
                  },
                },
              ],
            },
          };
        }
        const toolMsg = req.messages?.find((m: any) => m.role === 'tool');
        if (toolMsg) {
          const toolResp = toolMsg.content.find((p: any) => p.toolResponse);
          capturedToolOutput = toolResp?.toolResponse?.output;
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'Done.' }],
          },
        };
      }
    );

    await session.run(async () => {
      await ai.generate({
        model: mainModel,
        prompt: 'Write some code',
        use: [
          agents({
            agents: ['sessionCoder'],
            artifactStrategy: 'session',
          }),
        ],
      });
    });

    // Verify tool output has artifact name but NOT content (session strategy).
    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.ok(capturedToolOutput.artifacts, 'Should have artifacts in output');
    assert.ok(
      capturedToolOutput.artifacts.length > 0,
      'Should have at least one artifact'
    );

    const artifact = capturedToolOutput.artifacts[0];
    assert.ok(
      artifact.name.includes('sessionCoder'),
      'Artifact name should be namespaced with agent name'
    );
    assert.ok(
      artifact.name.includes('code.ts'),
      'Should contain original name'
    );
    // Session strategy should NOT have content in the tool result.
    assert.strictEqual(
      artifact.content,
      undefined,
      'Session strategy should not include content in tool result'
    );

    // Verify artifacts were merged into parent session.
    const sessionArtifacts = session.getArtifacts();
    assert.ok(
      sessionArtifacts.length > 0,
      'Session should have merged artifacts'
    );
    assert.ok(
      sessionArtifacts[0].metadata?.invocationId,
      'Merged artifact should have invocationId metadata'
    );
  });

  it('artifact names are namespaced with invocation ID pattern', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-namespace' });

    // Sub-agent writes an artifact.
    let subTurn = 0;
    const subModel = ai.defineModel(
      { name: 'sub-ns-' + Math.random() },
      async () => {
        subTurn++;
        if (subTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'write_artifact',
                    input: { name: 'output.txt', content: 'hello' },
                  },
                },
              ],
            },
          };
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'done' }],
          },
        };
      }
    );

    ai.defineAgent({
      name: 'nsAgent',
      model: subModel,
      system: 'You produce output.',
      use: [artifacts()],
    });

    let mainTurn = 0;
    const mainModel = ai.defineModel(
      { name: 'main-ns-' + Math.random() },
      async () => {
        mainTurn++;
        if (mainTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_nsAgent',
                    input: { task: 'produce output' },
                  },
                },
              ],
            },
          };
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'ok' }],
          },
        };
      }
    );

    await session.run(async () => {
      await ai.generate({
        model: mainModel,
        prompt: 'test namespace',
        use: [agents({ agents: ['nsAgent'] })],
      });
    });

    // Verify the artifact name follows the pattern: {agentName}_{random4}/{artifactName}
    const sessionArtifacts = session.getArtifacts();
    assert.ok(
      sessionArtifacts.length > 0,
      'Should have merged artifacts in session'
    );

    const name = sessionArtifacts[0].name!;
    // Pattern: nsAgent_{4chars}/output.txt
    const namePattern = /^nsAgent_[a-z0-9]{4}\/output\.txt$/;
    assert.ok(
      namePattern.test(name),
      `Artifact name "${name}" should match pattern nsAgent_XXXX/output.txt`
    );
  });

  it('returns a tool response (does not propagate) when a sub-agent interrupts', async () => {
    const ai = genkit({});

    // A tool that always interrupts (never resolves to a value).
    const approvalTool = ai.defineInterrupt({
      name: 'needs_approval',
      description: 'Requires human approval before proceeding.',
      inputSchema: z.object({}),
    });

    // Sub-agent model calls the interrupting tool.
    const subModel = ai.defineModel(
      { name: 'sub-interrupt-' + Math.random() },
      async () => ({
        message: {
          role: 'model' as const,
          content: [
            {
              toolRequest: {
                name: 'needs_approval',
                input: {},
              },
            },
          ],
        },
      })
    );

    ai.defineAgent({
      name: 'interrupter',
      model: subModel,
      system: 'You require approval.',
      tools: [approvalTool],
    });

    // Main model delegates, then produces final text after the delegation
    // tool resolves (the sub-agent interrupt must NOT halt the parent loop).
    let mainTurn = 0;
    let capturedToolOutput: any;
    const mainModel = ai.defineModel(
      { name: 'main-interrupt-' + Math.random() },
      async (req) => {
        mainTurn++;
        if (mainTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_interrupter',
                    input: { task: 'do something requiring approval' },
                  },
                },
              ],
            },
          };
        }
        const toolMsg = req.messages?.find((m: any) => m.role === 'tool');
        if (toolMsg) {
          const toolResp = toolMsg.content.find((p: any) => p.toolResponse);
          capturedToolOutput = toolResp?.toolResponse?.output;
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'acknowledged the interrupt' }],
          },
        };
      }
    );

    const result = await ai.generate({
      model: mainModel,
      prompt: 'delegate to an agent that interrupts',
      use: [agents({ agents: ['interrupter'] })],
    });

    // The sub-agent interrupt should be reported as a normal tool response,
    // NOT propagated as an interrupt to the orchestrator.
    assert.notStrictEqual(
      result.finishReason,
      'interrupted',
      'Orchestrator should not be interrupted by a sub-agent interrupt'
    );
    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.match(
      capturedToolOutput.response,
      /interrupt/i,
      'Tool response should indicate the sub-agent interrupted'
    );
    assert.ok(
      result.text.includes('acknowledged'),
      'Orchestrator should continue after the interrupt is reported'
    );
  });

  it('forwards recent history (text only) to sub-agents via historyLength', async () => {
    const ai = genkit({});

    // Capture what the sub-agent model actually receives.
    let capturedSubMessages: any[] | undefined;
    const subModel = ai.defineModel(
      { name: 'sub-history-' + Math.random() },
      async (req) => {
        capturedSubMessages = req.messages;
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'sub done' }],
          },
        };
      }
    );

    ai.defineAgent({
      name: 'historyWorker',
      model: subModel,
      system: 'You are a worker.',
    });

    let mainTurn = 0;
    const mainModel = ai.defineModel(
      { name: 'main-history-' + Math.random() },
      async () => {
        mainTurn++;
        if (mainTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_historyWorker',
                    input: { task: 'do the main task' },
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

    // Provide conversation history that includes a complete tool exchange.
    // The model message with a `toolRequest` part (and the `tool` message)
    // must NOT be forwarded to the sub-agent — only text user/model parts.
    await ai.generate({
      model: mainModel,
      messages: [
        { role: 'user', content: [{ text: 'please search for X' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'search', ref: '1', input: {} } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 'search', ref: '1', output: {} } }],
        },
        { role: 'model', content: [{ text: 'I found the answer.' }] },
        { role: 'user', content: [{ text: 'now do the work' }] },
      ],
      use: [agents({ agents: ['historyWorker'], historyLength: 10 })],
    });

    assert.ok(capturedSubMessages, 'Sub-agent should have received messages');

    // No forwarded part should be a tool/tool-request part.
    const hasToolParts = capturedSubMessages!.some((m: any) =>
      m.content?.some((p: any) => p.toolRequest || p.toolResponse)
    );
    assert.ok(
      !hasToolParts,
      'Forwarded history must not contain tool/tool-request parts'
    );

    // No forwarded message should be a `tool` role message.
    const hasToolRole = capturedSubMessages!.some(
      (m: any) => m.role === 'tool'
    );
    assert.ok(!hasToolRole, 'Forwarded history must not contain tool messages');

    // The text from the history should be forwarded.
    const allText = capturedSubMessages!
      .flatMap((m: any) => m.content ?? [])
      .map((p: any) => p.text ?? '')
      .join('\n');
    assert.ok(
      allText.includes('please search for X'),
      'User text history should be forwarded'
    );
    assert.ok(
      allText.includes('I found the answer.'),
      'Model text history should be forwarded'
    );
    assert.ok(
      allText.includes('do the main task'),
      'The delegated task should be present'
    );
  });

  it('returns sub-agent failure as an error tool response', async () => {
    const ai = genkit({});

    // Sub-agent model throws, causing the agent to resolve with
    // finishReason: 'failed' and a structured error.
    const subModel = ai.defineModel(
      { name: 'sub-failing-' + Math.random() },
      async () => {
        throw new Error('sub-agent boom');
      }
    );

    ai.defineAgent({
      name: 'failer',
      model: subModel,
      system: 'You fail.',
    });

    let mainTurn = 0;
    let capturedToolOutput: any;
    const mainModel = ai.defineModel(
      { name: 'main-failing-' + Math.random() },
      async (req) => {
        mainTurn++;
        if (mainTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_failer',
                    input: { task: 'do the impossible' },
                  },
                },
              ],
            },
          };
        }
        const toolMsg = req.messages?.find((m: any) => m.role === 'tool');
        if (toolMsg) {
          const toolResp = toolMsg.content.find((p: any) => p.toolResponse);
          capturedToolOutput = toolResp?.toolResponse?.output;
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'recovered from failure' }],
          },
        };
      }
    );

    const result = await ai.generate({
      model: mainModel,
      prompt: 'delegate to a failing agent',
      use: [agents({ agents: ['failer'] })],
    });

    // The failure should be returned as tool output (not thrown), so the
    // orchestrator can recover.
    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.match(
      capturedToolOutput.response,
      /Error calling agent 'failer'/,
      'Tool response should surface the sub-agent error'
    );
    assert.ok(
      result.text.includes('recovered'),
      'Orchestrator should be able to recover after the failure'
    );
  });

  it('returns an aborted sub-agent run as an error tool response', async () => {
    const ai = genkit({});

    // A timeout out of the sub-agent's model is a stop, not a break: the
    // agent resolves with finishReason: 'aborted' and the error that stopped
    // it, and carries no message.
    const subModel = ai.defineModel(
      { name: 'sub-timeout-' + Math.random() },
      async () => {
        const err = new Error('sub-agent timed out');
        err.name = 'TimeoutError';
        throw err;
      }
    );

    ai.defineAgent({
      name: 'slowpoke',
      model: subModel,
      system: 'You are slow.',
    });

    let mainTurn = 0;
    let capturedToolOutput: any;
    const mainModel = ai.defineModel(
      { name: 'main-timeout-' + Math.random() },
      async (req) => {
        mainTurn++;
        if (mainTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_slowpoke',
                    input: { task: 'take your time' },
                  },
                },
              ],
            },
          };
        }
        const toolMsg = req.messages?.find((m: any) => m.role === 'tool');
        if (toolMsg) {
          const toolResp = toolMsg.content.find((p: any) => p.toolResponse);
          capturedToolOutput = toolResp?.toolResponse?.output;
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'recovered from the stop' }],
          },
        };
      }
    );

    const result = await ai.generate({
      model: mainModel,
      prompt: 'delegate to a slow agent',
      use: [agents({ agents: ['slowpoke'] })],
    });

    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.match(
      capturedToolOutput.response,
      /Error calling agent 'slowpoke': sub-agent timed out/,
      'Tool response should surface what stopped the sub-agent'
    );
    assert.ok(
      result.text.includes('recovered'),
      'Orchestrator should be able to recover after the stop'
    );
  });

  it("reports every non-answer finish reason as a failure that keeps the agent's last words", async () => {
    const ai = genkit({});
    const reasons = ['failed', 'blocked', 'length', 'aborted'] as const;
    for (const reason of reasons) {
      // A custom sub-agent that ends its turn on `reason` after saying
      // something partial, without throwing.
      ai.defineCustomAgent({ name: `ender_${reason}` }, async (sess) => {
        await sess.run(async () => {
          sess.addMessages([
            {
              role: 'model',
              content: [{ text: 'partial notes: found 3 of 5 sources' }],
            },
          ]);
          return { finishReason: reason };
        });
        const msgs = sess.getMessages();
        return { message: msgs[msgs.length - 1], finishReason: reason };
      });
    }

    for (const reason of reasons) {
      let mainTurn = 0;
      let capturedToolOutput: any;
      const mainModel = ai.defineModel(
        { name: `main-reason-${reason}-` + Math.random() },
        async (req) => {
          mainTurn++;
          if (mainTurn === 1) {
            return {
              message: {
                role: 'model' as const,
                content: [
                  {
                    toolRequest: {
                      name: `delegate_to_ender_${reason}`,
                      input: { task: 'dig' },
                    },
                  },
                ],
              },
            };
          }
          const toolMsg = req.messages?.find((m: any) => m.role === 'tool');
          capturedToolOutput = toolMsg?.content.find((p: any) => p.toolResponse)
            ?.toolResponse?.output;
          return {
            message: { role: 'model' as const, content: [{ text: 'ok' }] },
          };
        }
      );
      await ai.generate({
        model: mainModel,
        prompt: 'go',
        use: [agents({ agents: [`ender_${reason}`] })],
      });
      // Reported as a failure that names the reason and keeps the agent's
      // last words: they explain the outcome, and losing them leaves the
      // model with nothing it can act on.
      assert.match(capturedToolOutput.response, /Error calling agent/);
      assert.ok(
        capturedToolOutput.response.includes(`'${reason}'`),
        `response should name the finish reason: ${capturedToolOutput.response}`
      );
      assert.ok(
        capturedToolOutput.response.includes('found 3 of 5 sources'),
        `response should keep the last message: ${capturedToolOutput.response}`
      );
    }
  });

  it('says outright when a sub-agent completed without a final message', async () => {
    const ai = genkit({});
    // Ends on a model message holding only a tool request, after saving one
    // artifact: there is no answer text, and the result is in the artifact.
    ai.defineCustomAgent({ name: 'silent' }, async (sess) => {
      await sess.run(async () => {
        sess.addArtifacts([{ name: 'report.md', parts: [{ text: 'body' }] }]);
        sess.addMessages([
          {
            role: 'model',
            content: [{ toolRequest: { name: 'search', input: { q: 'x' } } }],
          },
        ]);
        return { finishReason: 'stop' };
      });
      const msgs = sess.getMessages();
      return {
        message: msgs[msgs.length - 1],
        artifacts: sess.getArtifacts(),
        finishReason: 'stop',
      };
    });

    let mainTurn = 0;
    let capturedToolOutput: any;
    const mainModel = ai.defineModel(
      { name: 'main-silent-' + Math.random() },
      async (req) => {
        mainTurn++;
        if (mainTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'delegate_to_silent',
                    input: { task: 'go' },
                  },
                },
              ],
            },
          };
        }
        const toolMsg = req.messages?.find((m: any) => m.role === 'tool');
        capturedToolOutput = toolMsg?.content.find((p: any) => p.toolResponse)
          ?.toolResponse?.output;
        return {
          message: { role: 'model' as const, content: [{ text: 'ok' }] },
        };
      }
    );
    await ai.generate({
      model: mainModel,
      prompt: 'go',
      use: [agents({ agents: ['silent'] })],
    });
    assert.match(capturedToolOutput.response, /completed/);
    assert.match(capturedToolOutput.response, /no final message/);
    assert.match(capturedToolOutput.response, /one artifact/);
    assert.strictEqual(capturedToolOutput.artifacts.length, 1);
  });
});

// ---------------------------------------------------------------------------
// Background delegation (`async: true`)
// ---------------------------------------------------------------------------

const CHECK_TOOL = 'check_background_tasks';
const WAIT_TOOL = 'wait_for_background_tasks';
const ABORT_TOOL = 'abort_background_tasks';

/** Outputs of every tool response named `toolName` in `messages`. */
function toolOutputs(messages: MessageData[] | undefined, toolName: string) {
  return (messages ?? [])
    .flatMap((m) => m.content)
    .filter((p) => p.toolResponse?.name === toolName)
    .map((p) => p.toolResponse!.output as any);
}

/** The text of the system message in `messages`, if any. */
function systemText(messages: MessageData[] | undefined): string {
  return (messages ?? [])
    .filter((m) => m.role === 'system')
    .flatMap((m) => m.content)
    .map((p) => p.text ?? '')
    .join('\n');
}

function toolRequest(name: string, input: unknown) {
  return {
    message: {
      role: 'model' as const,
      content: [{ toolRequest: { name, input } }],
    },
  };
}

function textResponse(text: string) {
  return { message: { role: 'model' as const, content: [{ text }] } };
}

/**
 * The conversation background delegations leave behind: one delegation tool
 * result per `[agent, taskId]`, as an orchestrator's history carries its
 * launches.
 */
function launchHistory(launches: [string, string][]): MessageData[] {
  return launches.map(([agent, taskId]) => ({
    role: 'tool' as const,
    content: [
      {
        toolResponse: { name: `delegate_to_${agent}`, output: { taskId } },
      },
    ],
  }));
}

/**
 * Instantiates the middleware the way one generate call does, with `history`
 * as the conversation its tools see, so the background-task tools accept the
 * task IDs that history launched.
 */
async function instantiateWith(
  ai: ReturnType<typeof genkit>,
  config: Parameters<typeof agents.instantiate>[0]['config'],
  history: MessageData[]
) {
  const def = agents.instantiate({ config, ai, pluginConfig: undefined });
  await def.generate!(
    { request: { messages: history }, currentTurn: 0, messageIndex: 0 } as any,
    {} as any,
    async () => textResponse('') as any
  );
  return def;
}

/** A gate a test opens to let a sub-agent turn finish. */
function makeGate() {
  let release!: () => void;
  const opened = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { opened, release };
}

/**
 * Defines a store-backed custom sub-agent whose single turn waits for `gate`
 * (or the invocation's abort), then says `text` and saves the given
 * artifacts. Modeled on the gated agents the Go conformance tests use.
 */
function defineGatedResearcher(
  ai: ReturnType<typeof genkit>,
  name: string,
  gate: Promise<void>,
  opts: {
    text?: string;
    artifacts?: { name: string; parts: { text: string }[] }[];
    finishReason?: 'stop' | 'failed' | 'aborted';
    onAbort?: () => void;
    store?: InMemorySessionStore;
    maxSnapshotWaitMs?: number;
  } = {}
) {
  return ai.defineCustomAgent(
    {
      name,
      store: opts.store ?? new InMemorySessionStore(),
      maxSnapshotWaitMs: opts.maxSnapshotWaitMs,
    },
    async (sess, { abortSignal }) => {
      await sess.run(async () => {
        await Promise.race([
          gate,
          new Promise<never>((_, reject) =>
            abortSignal?.addEventListener(
              'abort',
              () => {
                opts.onAbort?.();
                reject(new Error('aborted'));
              },
              { once: true }
            )
          ),
        ]);
        if (opts.artifacts) sess.addArtifacts(opts.artifacts);
        sess.addMessages([
          {
            role: 'model',
            content: [{ text: opts.text ?? 'research complete' }],
          },
        ]);
        return { finishReason: opts.finishReason ?? 'stop' };
      });
      const msgs = sess.getMessages();
      return {
        message: msgs[msgs.length - 1],
        artifacts: sess.getArtifacts(),
        finishReason: opts.finishReason ?? 'stop',
      };
    }
  );
}

describe('agents middleware (async)', () => {
  it('launches, checks, and collects a background delegation in one generate call', async () => {
    const ai = genkit({});
    const gate = makeGate();
    defineGatedResearcher(ai, 'researcher', gate.opened, {
      artifacts: [
        { name: 'findings.md', parts: [{ text: 'the findings body' }] },
      ],
    });

    // Scripted orchestrator: launch in background, check, release the gate,
    // wait, then finish. Each step keys off the tool responses so far.
    let capturedSystem = '';
    const orchestrator = ai.defineModel(
      { name: 'orch-async-' + Math.random() },
      async (req) => {
        capturedSystem = systemText(req.messages);
        const launches = toolOutputs(req.messages, 'delegate_to_researcher');
        const checks = toolOutputs(req.messages, CHECK_TOOL);
        const waits = toolOutputs(req.messages, WAIT_TOOL);
        if (launches.length === 0) {
          return toolRequest('delegate_to_researcher', {
            task: 'dig into X',
            background: true,
          });
        }
        if (checks.length === 0) {
          return toolRequest(CHECK_TOOL, { taskIds: [launches[0].taskId] });
        }
        if (waits.length === 0) {
          gate.release();
          return toolRequest(WAIT_TOOL, { taskIds: [launches[0].taskId] });
        }
        return textResponse('done');
      }
    );

    const result = await ai.generate({
      model: orchestrator,
      prompt: 'research X',
      maxTurns: 10,
      use: [agents({ agents: ['researcher'], async: true })],
    });
    assert.strictEqual(result.text, 'done');

    for (const want of ['background', CHECK_TOOL, WAIT_TOOL, ABORT_TOOL]) {
      assert.ok(
        capturedSystem.includes(want),
        `async system prompt should mention ${want}: ${capturedSystem}`
      );
    }

    const [launch] = toolOutputs(result.messages, 'delegate_to_researcher');
    assert.strictEqual(launch.status, 'pending');
    assert.ok(launch.taskId.startsWith('researcher:'), launch.taskId);
    assert.match(launch.response, /Background task .* started/);

    const [check] = toolOutputs(result.messages, CHECK_TOOL);
    assert.strictEqual(check.tasks.length, 1);
    assert.strictEqual(check.tasks[0].status, 'pending');

    const [wait] = toolOutputs(result.messages, WAIT_TOOL);
    assert.strictEqual(wait.tasks.length, 1);
    const task = wait.tasks[0];
    assert.strictEqual(task.status, 'completed');
    assert.strictEqual(task.agent, 'researcher');
    assert.strictEqual(task.response, 'research complete');
    const snapshotId = launch.taskId.slice('researcher:'.length);
    assert.strictEqual(
      task.artifacts[0].name,
      `researcher_${snapshotId.slice(0, 8)}/findings.md`
    );
    assert.ok(task.artifacts[0].content.includes('the findings body'));
    assert.strictEqual(wait.timedOut, undefined);
  });

  it('collects a task launched by an earlier generate call from its history', async () => {
    const ai = genkit({});
    const subModel = ai.defineModel(
      { name: 'researcher-bg-' + Math.random() },
      async () => textResponse('background answer')
    );
    ai.defineAgent({
      name: 'researcher',
      model: subModel,
      system: 'You research.',
      store: new InMemorySessionStore(),
    });

    // First call: launch in the background and stop without waiting.
    const launcher = ai.defineModel(
      { name: 'orch-launch-' + Math.random() },
      async (req) =>
        toolOutputs(req.messages, 'delegate_to_researcher').length === 0
          ? toolRequest('delegate_to_researcher', {
              task: 'long dig',
              background: true,
            })
          : textResponse('launched')
    );
    const first = await ai.generate({
      model: launcher,
      prompt: 'go',
      use: [agents({ agents: ['researcher'], async: true })],
    });
    const [launch] = toolOutputs(first.messages, 'delegate_to_researcher');
    assert.ok(launch.taskId);

    // Second call, fresh middleware instance, on the first call's history:
    // wait on the recorded task ID plus a missing snapshot and an
    // unconfigured agent, which must be reported in isolation from one
    // another.
    const waiter = ai.defineModel(
      { name: 'orch-wait-' + Math.random() },
      async (req) =>
        toolOutputs(req.messages, WAIT_TOOL).length === 0
          ? toolRequest(WAIT_TOOL, {
              taskIds: [
                launch.taskId,
                'researcher:no-such-snapshot',
                'ghost:whatever',
              ],
            })
          : textResponse('collected')
    );
    const second = await ai.generate({
      model: waiter,
      messages: [
        ...first.messages,
        ...launchHistory([['researcher', 'researcher:no-such-snapshot']]),
        { role: 'user', content: [{ text: 'collect' }] },
      ],
      use: [agents({ agents: ['researcher'], async: true })],
    });
    const [wait] = toolOutputs(second.messages, WAIT_TOOL);
    assert.strictEqual(wait.tasks.length, 3);
    assert.strictEqual(wait.tasks[0].status, 'completed');
    assert.strictEqual(wait.tasks[0].response, 'background answer');
    assert.strictEqual(wait.tasks[1].status, 'unknown');
    assert.match(wait.tasks[1].error, /not found/);
    assert.match(wait.tasks[1].error, /Delegate the task again/);
    assert.strictEqual(wait.tasks[2].status, 'unknown');
    assert.match(wait.tasks[2].error, /does not match any configured agent/);
    assert.strictEqual(wait.timedOut, undefined);
  });

  it('reports a task that committed without an answer as failed', async () => {
    const ai = genkit({});
    const gate = makeGate();
    // The turn commits (the row is `completed`) but declares `failed`.
    defineGatedResearcher(ai, 'researcher', gate.opened, {
      text: 'partial notes',
      finishReason: 'failed',
    });
    const orchestrator = ai.defineModel(
      { name: 'orch-no-answer-' + Math.random() },
      async (req) => {
        const launches = toolOutputs(req.messages, 'delegate_to_researcher');
        if (launches.length === 0) {
          return toolRequest('delegate_to_researcher', {
            task: 'dig',
            background: true,
          });
        }
        if (toolOutputs(req.messages, WAIT_TOOL).length === 0) {
          gate.release();
          return toolRequest(WAIT_TOOL, { taskIds: [launches[0].taskId] });
        }
        return textResponse('done');
      }
    );
    const result = await ai.generate({
      model: orchestrator,
      prompt: 'go',
      use: [agents({ agents: ['researcher'], async: true })],
    });
    const [wait] = toolOutputs(result.messages, WAIT_TOOL);
    const task = wait.tasks[0];
    assert.strictEqual(task.status, 'failed');
    assert.ok(task.error, 'the report must explain why there is no answer');
    assert.match(task.error, /partial notes/);
    assert.strictEqual(task.response, undefined);
  });

  it('reports a task that committed as aborted under that status', async () => {
    const ai = genkit({});
    const gate = makeGate();
    // The turn commits (the row is `completed`) but declares `aborted`.
    defineGatedResearcher(ai, 'researcher', gate.opened, {
      text: 'stopped early',
      finishReason: 'aborted',
    });
    const orchestrator = ai.defineModel(
      { name: 'orch-aborted-reason-' + Math.random() },
      async (req) => {
        const launches = toolOutputs(req.messages, 'delegate_to_researcher');
        if (launches.length === 0) {
          return toolRequest('delegate_to_researcher', {
            task: 'dig',
            background: true,
          });
        }
        if (toolOutputs(req.messages, WAIT_TOOL).length === 0) {
          gate.release();
          return toolRequest(WAIT_TOOL, { taskIds: [launches[0].taskId] });
        }
        return textResponse('done');
      }
    );
    const result = await ai.generate({
      model: orchestrator,
      prompt: 'go',
      use: [agents({ agents: ['researcher'], async: true })],
    });
    const [wait] = toolOutputs(result.messages, WAIT_TOOL);
    const task = wait.tasks[0];
    assert.strictEqual(task.status, 'aborted');
    assert.match(task.error, /stopped early/);
    assert.strictEqual(task.response, undefined);
  });

  it('reports a settled task on a deadline that beats the follow', async () => {
    const ai = genkit({});
    const researcher = defineGatedResearcher(
      ai,
      'researcher',
      Promise.resolve()
    );
    const task = await researcher.chat().detach('dig');
    await task.wait();

    const def = await instantiateWith(
      ai,
      { agents: ['researcher'], async: true },
      launchHistory([['researcher', `researcher:${task.snapshotId}`]])
    );
    const waitTool = def.tools!.find((t) => t.__action.name === WAIT_TOOL)!;
    // Whichever wins, the deadline or the follow's first read, the row is
    // settled and the report must say so: a timeout returns the current
    // statuses, not the follow's last look at them.
    const out = await waitTool({
      taskIds: [`researcher:${task.snapshotId}`],
      timeoutSeconds: 0.000001,
    });
    assert.strictEqual(out.tasks[0].status, 'completed');
    assert.strictEqual(out.tasks[0].response, 'research complete');
    assert.strictEqual(out.timedOut, undefined);
  });

  it('fails a cancelled wait without re-reading its tasks', async () => {
    const ai = genkit({});
    const gate = makeGate();
    const researcher = defineGatedResearcher(ai, 'researcher', gate.opened);
    const task = await researcher.chat().detach('dig');

    // Count the middleware's plain reads; the follow itself waits in the
    // store and does not go through this action.
    let reads = 0;
    const readAction = researcher.getSnapshotDataAction;
    const run = readAction.run.bind(readAction);
    readAction.run = ((...args: Parameters<typeof run>) => {
      reads++;
      return run(...args);
    }) as typeof run;

    const def = await instantiateWith(
      ai,
      { agents: ['researcher'], async: true },
      launchHistory([['researcher', `researcher:${task.snapshotId}`]])
    );
    const waitTool = def.tools!.find((t) => t.__action.name === WAIT_TOOL)!;
    const controller = new AbortController();
    const waiting = waitTool(
      { taskIds: [`researcher:${task.snapshotId}`] },
      { abortSignal: controller.signal }
    );
    setTimeout(() => controller.abort(), 20);
    await assert.rejects(waiting);
    // The call fails as a whole, so a report refreshed for it is discarded.
    assert.strictEqual(reads, 0, 'a cancelled wait must not re-read its tasks');
    gate.release();
  });

  it('stops a synchronous sub-agent when the delegation tool call is cancelled', async () => {
    const ai = genkit({});
    const gate = makeGate();
    let aborted = false;
    defineGatedResearcher(ai, 'researcher', gate.opened, {
      onAbort: () => {
        aborted = true;
      },
    });

    const def = agents.instantiate({
      config: { agents: ['researcher'], async: true },
      ai,
      pluginConfig: undefined,
    });
    const delegate = def.tools!.find(
      (t) => t.__action.name === 'delegate_to_researcher'
    )!;
    const controller = new AbortController();
    const delegating = delegate(
      { task: 'dig' },
      { abortSignal: controller.signal }
    );
    setTimeout(() => controller.abort(), 20);
    // A sub-agent that never sees the stop finishes once the gate opens, so a
    // regression fails the assertion rather than hanging the test.
    const fallback = setTimeout(gate.release, 500);
    await delegating;
    clearTimeout(fallback);
    assert.ok(aborted, 'the sub-agent turn must observe the stop');
    gate.release();
  });

  it('fails a cancelled check or abort call without stopping the task', async () => {
    const ai = genkit({});
    const gate = makeGate();
    const researcher = defineGatedResearcher(ai, 'researcher', gate.opened);
    const task = await researcher.chat().detach('dig');
    const def = agents.instantiate({
      config: { agents: ['researcher'], async: true },
      ai,
      pluginConfig: undefined,
    });
    const taskIds = [`researcher:${task.snapshotId}`];
    const cancelled = AbortSignal.abort();
    for (const name of [CHECK_TOOL, ABORT_TOOL]) {
      const t = def.tools!.find((t) => t.__action.name === name)!;
      await assert.rejects(
        t({ taskIds }, { abortSignal: cancelled }),
        (e: any) => e?.name === 'AbortError',
        `${name} must fail as a whole`
      );
    }
    const row = await researcher.getSnapshotData({
      snapshotId: task.snapshotId,
    });
    assert.strictEqual(row?.status, 'pending', 'the task must keep running');
    gate.release();
  });

  it('rejects an agent reference without a name', () => {
    assert.throws(
      () =>
        agents.instantiate({
          config: { agents: [''] },
          ai: genkit({}),
          pluginConfig: undefined,
        }),
      (e: any) => e.status === 'INVALID_ARGUMENT'
    );
  });

  it('says when an abort cannot reach the worker', async () => {
    const ai = genkit({});
    const gate = makeGate();
    // A store without a change feed: the runtime can flip the row but has no
    // way to signal the worker, and publishes the agent as not abortable.
    const store = new InMemorySessionStore();
    Object.defineProperty(store, 'onSnapshotStateChange', { value: undefined });
    const researcher = defineGatedResearcher(ai, 'researcher', gate.opened, {
      store,
    });
    const task = await researcher.chat().detach('dig');

    const def = await instantiateWith(
      ai,
      { agents: ['researcher'], async: true },
      launchHistory([['researcher', `researcher:${task.snapshotId}`]])
    );
    const abortTool = def.tools!.find((t) => t.__action.name === ABORT_TOOL)!;
    const waitTool = def.tools!.find((t) => t.__action.name === WAIT_TOOL)!;
    const taskIds = [`researcher:${task.snapshotId}`];
    const out = await abortTool({ taskIds });
    assert.strictEqual(out.tasks[0].status, 'aborting');
    assert.match(out.tasks[0].error, /cannot signal its worker/);

    // The worker runs to its end, and only then does the row settle.
    gate.release();
    const settled = await waitTool({ taskIds });
    assert.strictEqual(settled.tasks[0].status, 'aborted');
    assert.match(settled.tasks[0].error, /cannot signal its worker/);
    const row = await researcher.getSnapshotData({
      snapshotId: task.snapshotId,
    });
    const messages = row?.state?.messages ?? [];
    assert.strictEqual(
      messages[messages.length - 1]?.content[0].text,
      'research complete',
      'the settled row must keep the work the run finished'
    );
  });

  it('does not advertise background work on a synchronous instance', async () => {
    const ai = genkit({});
    ai.defineAgent({
      name: 'researcher',
      model: ai.defineModel(
        { name: 'researcher-sync-schema-' + Math.random() },
        async () => textResponse('unused')
      ),
      system: 'unused',
    });
    let delegateDef: any;
    const orchestrator = ai.defineModel(
      { name: 'orch-sync-schema-' + Math.random() },
      async (req) => {
        delegateDef = req.tools?.find(
          (t) => t.name === 'delegate_to_researcher'
        );
        return textResponse('done');
      }
    );
    await ai.generate({
      model: orchestrator,
      prompt: 'go',
      use: [agents({ agents: ['researcher'] })],
    });
    assert.ok(delegateDef, 'the delegation tool must reach the model');
    // The schemas are what the model reads: nothing in them may point at
    // background tasks or the tools that collect them, which this instance
    // does not have. The task handle stays: it is what continue_task spends.
    const inputSchema = JSON.stringify(delegateDef.inputSchema);
    const outputSchema = JSON.stringify(delegateDef.outputSchema);
    assert.ok(!inputSchema.includes('background'), inputSchema);
    assert.ok(!outputSchema.includes('background'), outputSchema);
    assert.ok(!outputSchema.includes(CHECK_TOOL), outputSchema);
    assert.ok(outputSchema.includes('continue_task'), outputSchema);
  });

  it('times out a wait, reporting running tasks as pending and keeping unresolvable errors', async () => {
    const ai = genkit({});
    const gate = makeGate();
    // Never released: the task is pending for the whole wait.
    defineGatedResearcher(ai, 'researcher', gate.opened);
    const orchestrator = ai.defineModel(
      { name: 'orch-timeout-' + Math.random() },
      async (req) => {
        const launches = toolOutputs(req.messages, 'delegate_to_researcher');
        if (launches.length === 0) {
          return toolRequest('delegate_to_researcher', {
            task: 'dig',
            background: true,
          });
        }
        if (toolOutputs(req.messages, WAIT_TOOL).length === 0) {
          return toolRequest(WAIT_TOOL, {
            taskIds: [launches[0].taskId, 'ghost:whatever'],
            timeoutSeconds: 0.2,
          });
        }
        return textResponse('done');
      }
    );
    const result = await ai.generate({
      model: orchestrator,
      prompt: 'go',
      use: [agents({ agents: ['researcher'], async: true })],
    });
    const [wait] = toolOutputs(result.messages, WAIT_TOOL);
    assert.strictEqual(wait.timedOut, true);
    assert.strictEqual(wait.tasks.length, 2);
    assert.strictEqual(wait.tasks[0].status, 'pending');
    assert.strictEqual(wait.tasks[0].error, undefined);
    // Nothing about the deadline makes an unresolvable handle more likely to
    // settle later; reporting it as pending would send the model back to
    // re-check it forever.
    assert.strictEqual(wait.tasks[1].status, 'unknown');
    assert.match(wait.tasks[1].error, /does not match any configured agent/);
    gate.release();
  });

  it('treats a timeout too large for a timer as unbounded', async () => {
    const ai = genkit({});
    ai.defineAgent({
      name: 'researcher',
      model: ai.defineModel(
        { name: 'researcher-overflow-' + Math.random() },
        async () => textResponse('unused')
      ),
      system: 'unused',
      store: new InMemorySessionStore(),
    });
    // A missing snapshot settles on the first pass (NOT_FOUND is a dead end),
    // so the wait returns without waiting out the absurd timeout; a deadline
    // that overflowed into an immediate timer would instead come back timed
    // out with a read that never happened.
    const waiter = ai.defineModel(
      { name: 'orch-overflow-' + Math.random() },
      async (req) =>
        toolOutputs(req.messages, WAIT_TOOL).length === 0
          ? toolRequest(WAIT_TOOL, {
              taskIds: ['researcher:no-such-snapshot'],
              timeoutSeconds: 10_000_000_000,
            })
          : textResponse('collected')
    );
    const result = await ai.generate({
      model: waiter,
      messages: [
        ...launchHistory([['researcher', 'researcher:no-such-snapshot']]),
        { role: 'user', content: [{ text: 'go' }] },
      ],
      use: [agents({ agents: ['researcher'], async: true })],
    });
    const [wait] = toolOutputs(result.messages, WAIT_TOOL);
    assert.strictEqual(wait.timedOut, undefined);
    assert.strictEqual(wait.tasks[0].status, 'unknown');
    assert.match(wait.tasks[0].error, /not found/);
  });

  it('returns on the first settled task with waitFor "first"', async () => {
    const ai = genkit({});
    ai.defineAgent({
      name: 'quick',
      model: ai.defineModel({ name: 'quick-' + Math.random() }, async () =>
        textResponse('quick answer')
      ),
      system: 'unused',
      store: new InMemorySessionStore(),
    });
    // The slow sub-agent finishes only when released, so the race can only
    // be won by the quick one.
    const gate = makeGate();
    defineGatedResearcher(ai, 'slow', gate.opened);

    const orchestrator = ai.defineModel(
      { name: 'orch-race-' + Math.random() },
      async (req) => {
        const slow = toolOutputs(req.messages, 'delegate_to_slow');
        const quick = toolOutputs(req.messages, 'delegate_to_quick');
        if (slow.length === 0) {
          return toolRequest('delegate_to_slow', {
            task: 'dig forever',
            background: true,
          });
        }
        if (quick.length === 0) {
          return toolRequest('delegate_to_quick', {
            task: 'answer fast',
            background: true,
          });
        }
        if (toolOutputs(req.messages, WAIT_TOOL).length === 0) {
          // The slow task first in the list, so a settled result in slot 1
          // proves the join raced instead of following input order.
          return toolRequest(WAIT_TOOL, {
            taskIds: [slow[0].taskId, quick[0].taskId],
            waitFor: 'first',
          });
        }
        return textResponse('done');
      }
    );
    const result = await ai.generate({
      model: orchestrator,
      prompt: 'go',
      maxTurns: 10,
      use: [agents({ agents: ['quick', 'slow'], async: true })],
    });
    const [wait] = toolOutputs(result.messages, WAIT_TOOL);
    assert.strictEqual(wait.timedOut, undefined, 'a won race is not a timeout');
    assert.strictEqual(wait.tasks[0].status, 'pending');
    assert.strictEqual(wait.tasks[1].status, 'completed');
    assert.strictEqual(wait.tasks[1].response, 'quick answer');
    assert.match(wait.note, /first settled/);
    gate.release();
  });

  it('answers an unknown waitFor value with guidance', async () => {
    const ai = genkit({});
    ai.defineAgent({
      name: 'quick',
      model: ai.defineModel({ name: 'quick2-' + Math.random() }, async () =>
        textResponse('unused')
      ),
      system: 'unused',
      store: new InMemorySessionStore(),
    });
    const orchestrator = ai.defineModel(
      { name: 'orch-badjoin-' + Math.random() },
      async (req) =>
        toolOutputs(req.messages, WAIT_TOOL).length === 0
          ? toolRequest(WAIT_TOOL, {
              taskIds: ['quick:whatever'],
              waitFor: 'any',
            })
          : textResponse('done')
    );
    const result = await ai.generate({
      model: orchestrator,
      prompt: 'go',
      use: [agents({ agents: ['quick'], async: true })],
    });
    const [wait] = toolOutputs(result.messages, WAIT_TOOL);
    assert.match(wait.note, /'first'/);
    assert.strictEqual(wait.tasks, undefined);
  });

  it('aborts a running background task, which winds down and settles as aborted', async () => {
    const ai = genkit({});
    const gate = makeGate();
    let stopped = false;
    // Never released: the abort is the only thing that can end this task.
    defineGatedResearcher(ai, 'researcher', gate.opened, {
      onAbort: () => {
        stopped = true;
      },
    });
    const orchestrator = ai.defineModel(
      { name: 'orch-abort-' + Math.random() },
      async (req) => {
        const launches = toolOutputs(req.messages, 'delegate_to_researcher');
        if (launches.length === 0) {
          return toolRequest('delegate_to_researcher', {
            task: 'dig',
            background: true,
          });
        }
        if (toolOutputs(req.messages, ABORT_TOOL).length === 0) {
          return toolRequest(ABORT_TOOL, { taskIds: [launches[0].taskId] });
        }
        if (toolOutputs(req.messages, WAIT_TOOL).length === 0) {
          return toolRequest(WAIT_TOOL, { taskIds: [launches[0].taskId] });
        }
        return textResponse('done');
      }
    );
    const result = await ai.generate({
      model: orchestrator,
      prompt: 'go',
      maxTurns: 10,
      use: [agents({ agents: ['researcher'], async: true })],
    });
    // The abort answers once the stop is durable, without waiting for the
    // finalize: the task is winding down.
    const [aborting] = toolOutputs(result.messages, ABORT_TOOL);
    assert.strictEqual(aborting.tasks[0].status, 'aborting');
    assert.match(aborting.tasks[0].error, /winding down/);
    // The row said aborting; the runtime observes that flip and cancels the
    // work, which is the half a status write alone would not prove.
    assert.strictEqual(stopped, true, 'the sub-agent was never cancelled');
    // A wait follows the wind-down to the settled row.
    const [settled] = toolOutputs(result.messages, WAIT_TOOL);
    assert.strictEqual(settled.tasks[0].status, 'aborted');
  });

  it('reports the result when aborting a task that had already finished', async () => {
    const ai = genkit({});
    const researcher = defineGatedResearcher(
      ai,
      'researcher',
      Promise.resolve(),
      {
        artifacts: [
          { name: 'findings.md', parts: [{ text: 'the findings body' }] },
        ],
      }
    );
    // Launch and settle the task outside the middleware, so nothing has
    // cached its report before the abort tool runs.
    const task = await researcher.chat().detach('dig into X');
    await task.wait();

    const def = await instantiateWith(
      ai,
      { agents: ['researcher'], async: true },
      launchHistory([['researcher', `researcher:${task.snapshotId}`]])
    );
    const abortTool = def.tools!.find((t) => t.__action.name === ABORT_TOOL)!;
    const out = await abortTool({ taskIds: [`researcher:${task.snapshotId}`] });
    const report = out.tasks[0];
    assert.strictEqual(report.status, 'completed');
    assert.strictEqual(report.response, 'research complete');
    assert.strictEqual(
      report.artifacts[0].name,
      `researcher_${task.snapshotId.slice(0, 8)}/findings.md`
    );
    assert.ok(report.artifacts[0].content.includes('the findings body'));
  });

  it('refuses a background launch on a sub-agent without a store and refunds the slot', async () => {
    const ai = genkit({});
    ai.defineAgent({
      name: 'researcher',
      model: ai.defineModel(
        { name: 'researcher-nostore-' + Math.random() },
        async () => textResponse('synchronous answer')
      ),
      system: 'unused',
    });
    // Launch in the background (refused), then synchronously: with a cap of
    // one, the retry only succeeds if the refusal returned its slot.
    const orchestrator = ai.defineModel(
      { name: 'orch-nostore-' + Math.random() },
      async (req) => {
        const results = toolOutputs(req.messages, 'delegate_to_researcher');
        if (results.length === 0) {
          return toolRequest('delegate_to_researcher', {
            task: 'dig',
            background: true,
          });
        }
        if (results.length === 1) {
          return toolRequest('delegate_to_researcher', { task: 'dig' });
        }
        return textResponse('done');
      }
    );
    const result = await ai.generate({
      model: orchestrator,
      prompt: 'go',
      use: [agents({ agents: ['researcher'], async: true, maxDelegations: 1 })],
    });
    const [refused, retried] = toolOutputs(
      result.messages,
      'delegate_to_researcher'
    );
    assert.strictEqual(refused.taskId, undefined);
    assert.match(refused.response, /Error calling agent/);
    assert.match(refused.response, /no session store/);
    assert.match(refused.response, /without "background"/);
    assert.strictEqual(retried.response, 'synchronous answer');
  });

  it('lets two async instances with distinct prefixes share one generate call', async () => {
    const ai = genkit({});
    ai.defineAgent({
      name: 'researcher',
      model: ai.defineModel(
        { name: 'researcher-coexist-' + Math.random() },
        async () => textResponse('unused')
      ),
      system: 'unused',
      store: new InMemorySessionStore(),
    });
    const model = ai.defineModel(
      { name: 'orch-coexist-' + Math.random() },
      async () => textResponse('done')
    );
    const result = await ai.generate({
      model,
      prompt: 'go',
      use: [
        agents({ agents: ['researcher'], toolPrefix: 'research', async: true }),
        agents({ agents: ['researcher'], toolPrefix: 'code', async: true }),
      ],
    });
    assert.strictEqual(result.text, 'done');
  });

  it('rejects colliding tool names at instantiation', () => {
    const ai = genkit({});
    assert.throws(
      () =>
        agents.instantiate({
          config: {
            agents: ['check_background_tasks'],
            toolPrefix: '',
            async: true,
          },
          ai,
          pluginConfig: undefined,
        }),
      /collides/
    );
  });

  it('parses task IDs against the longest configured agent name', async () => {
    const ai = genkit({});
    const def = await instantiateWith(
      ai,
      { agents: ['a', 'a:b'], async: true },
      launchHistory([
        ['a:b', 'a:b:1234'],
        ['a', 'a:5678'],
      ])
    );
    const checkTool = def.tools!.find((t) => t.__action.name === CHECK_TOOL)!;
    // Neither agent is registered, so each report fails at resolution and
    // names the agent the handle was parsed to.
    const out = await checkTool({ taskIds: ['a:b:1234', 'a:5678', 'a:'] });
    assert.strictEqual(out.tasks[0].agent, 'a:b');
    assert.strictEqual(out.tasks[1].agent, 'a');
    assert.match(out.tasks[1].error, /not registered/);
    assert.strictEqual(out.tasks[2].status, 'unknown');
    assert.match(out.tasks[2].error, /does not match any configured agent/);
  });

  it('bounds a wait the model left unbounded with maxWaitSeconds', async () => {
    const ai = genkit({});
    const gate = makeGate();
    const researcher = defineGatedResearcher(ai, 'researcher', gate.opened);
    const task = await researcher.chat().detach('dig');
    const taskId = `researcher:${task.snapshotId}`;

    const def = await instantiateWith(
      ai,
      { agents: ['researcher'], async: true, maxWaitSeconds: 0.05 },
      launchHistory([['researcher', taskId]])
    );
    const waitTool = def.tools!.find((t) => t.__action.name === WAIT_TOOL)!;
    // No timeoutSeconds: "until every task settles", which the bound caps.
    const out = await waitTool({ taskIds: [taskId] });
    assert.strictEqual(out.timedOut, true);
    assert.strictEqual(out.tasks[0].status, 'pending');
    gate.release();
  });

  it('follows a sub-agent wait that answers before the task settles', async () => {
    const ai = genkit({});
    const gate = makeGate();
    // The sub-agent's wait action answers with the pending row every 10ms.
    const researcher = defineGatedResearcher(ai, 'researcher', gate.opened, {
      maxSnapshotWaitMs: 10,
    });
    const task = await researcher.chat().detach('dig');
    const taskId = `researcher:${task.snapshotId}`;

    const def = await instantiateWith(
      ai,
      { agents: ['researcher'], async: true },
      launchHistory([['researcher', taskId]])
    );
    const waitTool = def.tools!.find((t) => t.__action.name === WAIT_TOOL)!;
    const waiting = waitTool({ taskIds: [taskId] });
    setTimeout(gate.release, 50);
    const out = await waiting;
    assert.strictEqual(out.tasks[0].status, 'completed');
    assert.strictEqual(out.timedOut, undefined);
  });

  it('refuses a task ID this conversation did not launch', async () => {
    const ai = genkit({});
    const gate = makeGate();
    const researcher = defineGatedResearcher(ai, 'researcher', gate.opened);
    // Another conversation's task: its ID reaches this one only as text.
    const task = await researcher.chat().detach('dig');

    const def = await instantiateWith(
      ai,
      { agents: ['researcher'], async: true },
      []
    );
    const abortTool = def.tools!.find((t) => t.__action.name === ABORT_TOOL)!;
    const out = await abortTool({ taskIds: [`researcher:${task.snapshotId}`] });
    assert.strictEqual(out.tasks[0].status, 'unknown');
    assert.match(out.tasks[0].error, /not issued in this conversation/);
    const row = await researcher.getSnapshotData({
      snapshotId: task.snapshotId,
    });
    assert.strictEqual(row?.status, 'pending', 'the abort must not reach it');
    gate.release();
  });

  it('answers a background-task tool called without task IDs with guidance', async () => {
    const ai = genkit({});
    const def = agents.instantiate({
      config: { agents: ['researcher'], async: true },
      ai,
      pluginConfig: undefined,
    });
    for (const name of [CHECK_TOOL, WAIT_TOOL, ABORT_TOOL]) {
      const t = def.tools!.find((t) => t.__action.name === name);
      assert.ok(t, `tool ${name} should be registered`);
      // An omitted taskIds must decode: a required field would fail the whole
      // generate call rather than a turn the model can correct.
      const out = await t!({});
      assert.match(out.note, /No task IDs given/);
    }
  });

  it('reports what the sub-agent last said, not whatever the transcript ends on', async () => {
    const ai = genkit({});
    const gate = makeGate();
    // The transcript ends on a tool response after the model spoke.
    ai.defineCustomAgent(
      { name: 'researcher', store: new InMemorySessionStore() },
      async (sess) => {
        await sess.run(async () => {
          await gate.opened;
          sess.addMessages([
            { role: 'model', content: [{ text: 'working on it' }] },
            {
              role: 'tool',
              content: [
                { toolResponse: { name: 'search', output: 'raw results' } },
              ],
            },
          ]);
          return { finishReason: 'stop' };
        });
        return { artifacts: sess.getArtifacts(), finishReason: 'stop' };
      }
    );
    const orchestrator = ai.defineModel(
      { name: 'orch-last-model-' + Math.random() },
      async (req) => {
        const launches = toolOutputs(req.messages, 'delegate_to_researcher');
        if (launches.length === 0) {
          return toolRequest('delegate_to_researcher', {
            task: 'dig',
            background: true,
          });
        }
        if (toolOutputs(req.messages, WAIT_TOOL).length === 0) {
          gate.release();
          return toolRequest(WAIT_TOOL, { taskIds: [launches[0].taskId] });
        }
        return textResponse('done');
      }
    );
    const result = await ai.generate({
      model: orchestrator,
      prompt: 'go',
      use: [agents({ agents: ['researcher'], async: true })],
    });
    const [wait] = toolOutputs(result.messages, WAIT_TOOL);
    assert.strictEqual(wait.tasks[0].status, 'completed');
    assert.strictEqual(wait.tasks[0].response, 'working on it');
    assert.strictEqual(wait.tasks[0].error, undefined);
  });
});

const CONTINUE_TOOL = 'continue_task';

type Genkit = ReturnType<typeof genkit>;

/** The text of every message, joined, for "the run saw X" assertions. */
function joinedText(messages: MessageData[]): string {
  return messages
    .flatMap((m) => m.content)
    .map((p) => p.text ?? '')
    .join('\n');
}

/**
 * A sub-agent model that fails its first `n` calls and then answers `text`,
 * recording each request's messages in `seen`.
 */
function failNTimesModel(
  ai: Genkit,
  n: number,
  text: string,
  seen?: MessageData[][]
) {
  let calls = 0;
  return ai.defineModel({ name: 'flaky-' + Math.random() }, async (req) => {
    seen?.push(req.messages);
    calls++;
    if (calls <= n) throw new Error('model melted');
    return textResponse(text);
  });
}

/** Defines a store-backed prompt agent on `model`. */
function defineKeeper(
  ai: Genkit,
  name: string,
  model: ReturnType<typeof failNTimesModel>,
  store: SessionStore = new InMemorySessionStore()
) {
  return ai.defineAgent({ name, model, system: 'You keep going.', store });
}

/**
 * Runs an orchestrator whose `step` picks the next model response from the
 * conversation so far, with the agents middleware configured by `config`.
 * `history` is the conversation before this call (see {@link mintedBy}).
 */
async function orchestrate(
  ai: Genkit,
  config: Parameters<typeof agents>[0],
  step: (
    messages: MessageData[]
  ) => ReturnType<typeof toolRequest> | ReturnType<typeof textResponse>,
  history: MessageData[] = []
) {
  const model = ai.defineModel(
    { name: 'orch-continue-' + Math.random() },
    async (req) => step(req.messages)
  );
  return ai.generate({
    model,
    messages: history,
    prompt: 'go',
    maxTurns: 10,
    use: [agents(config)],
  });
}

/** The newest output of `toolName` in `messages`, if any. */
function lastOutput(messages: MessageData[], toolName: string): any {
  const outs = toolOutputs(messages, toolName);
  return outs[outs.length - 1];
}

/**
 * A conversation in which an earlier call to `toolName` returned `taskId`,
 * which is what lets the middleware's tools accept a handle the test planted
 * in the store.
 */
function mintedBy(taskId: string, toolName: string): MessageData[] {
  return [
    {
      role: 'model',
      content: [{ toolRequest: { name: toolName, ref: 'earlier', input: {} } }],
    },
    {
      role: 'tool',
      content: [
        {
          toolResponse: {
            name: toolName,
            ref: 'earlier',
            output: { response: '', taskId },
          },
        },
      ],
    },
  ];
}

/**
 * An orchestrator step that delegates `task` once, then continues the
 * delegation's taskId (with `instructions` when given), then finishes.
 */
function delegateThenContinue(
  delegateTool: string,
  task: string,
  instructions?: string
) {
  return (messages: MessageData[]) => {
    const continued = lastOutput(messages, CONTINUE_TOOL);
    if (continued) return textResponse('done: ' + continued.response);
    const delegated = lastOutput(messages, delegateTool);
    if (delegated) {
      return toolRequest(CONTINUE_TOOL, {
        taskId: delegated.taskId,
        ...(instructions && { instructions }),
      });
    }
    return toolRequest(delegateTool, { task });
  };
}

/**
 * Writes a dead worker's pending row the way a detach mints one: created
 * now, with a heartbeat that went stale.
 */
async function saveDeadPendingRow(
  store: SessionStore,
  sessionId: string,
  parentId?: string
): Promise<string> {
  const now = new Date().toISOString();
  const stale = new Date(Date.now() - 10 * 60_000).toISOString();
  return (await store.saveSnapshot(undefined, () => ({
    createdAt: now,
    updatedAt: now,
    heartbeatAt: stale,
    status: 'pending',
    sessionId,
    ...(parentId && { parentId }),
    state: { sessionId },
  })))!;
}

/**
 * Defines a store-backed "keeper" sub-agent on `store`, runs one delegation
 * to commit a conversation ("start X"), and plants a dead worker's pending
 * row on top of it. Returns the dead task's handle and the committed
 * (parent) snapshot ID.
 */
async function seedDeadKeeperTask(
  ai: Genkit,
  store: SessionStore,
  model: ReturnType<typeof failNTimesModel>
): Promise<{ deadTask: string; committedId: string; pendingId: string }> {
  defineKeeper(ai, 'keeper', model, store);
  const first = await orchestrate(ai, { agents: ['keeper'] }, (messages) =>
    lastOutput(messages, 'delegate_to_keeper')
      ? textResponse('seeded')
      : toolRequest('delegate_to_keeper', { task: 'start X' })
  );
  const seeded = lastOutput(first.messages, 'delegate_to_keeper');
  assert.ok(
    seeded?.taskId,
    `seeded delegation has a handle: ${JSON.stringify(seeded)}`
  );
  const committedId = seeded.taskId.slice('keeper:'.length);
  const committed = await store.getSnapshot({ snapshotId: committedId });
  const sessionId = committed?.sessionId ?? committed?.state?.sessionId;
  assert.ok(sessionId, 'the committed row names its session');
  const pendingId = await saveDeadPendingRow(store, sessionId!, committedId);
  return { deadTask: `keeper:${pendingId}`, committedId, pendingId };
}

/**
 * Wraps an in-memory store so a test can fail writes to a snapshot ID or
 * rewrite reads of one.
 */
function flakyStore() {
  const base = new InMemorySessionStore();
  const failSave = new Map<string, Error>();
  let getHook:
    | ((
        id: string | undefined,
        snap: SessionSnapshot | undefined
      ) => SessionSnapshot | undefined | Promise<SessionSnapshot | undefined>)
    | undefined;
  const store: SessionStore = {
    async getSnapshot(opts) {
      const snap = await base.getSnapshot(opts);
      return getHook ? getHook(opts.snapshotId, snap) : snap;
    },
    async saveSnapshot(id, mutator, options) {
      const failure = id ? failSave.get(id) : undefined;
      if (failure) throw failure;
      return base.saveSnapshot(id, mutator, options);
    },
    onSnapshotStateChange: (id, callback, options) =>
      base.onSnapshotStateChange(id, callback, options),
  };
  return {
    store,
    failSave,
    setGetHook(hook: typeof getHook) {
      getHook = hook;
    },
  };
}

describe('agents middleware (continue)', () => {
  it('stamps a synchronous delegation to a store-backed sub-agent with its task handle', async () => {
    const ai = genkit({});
    defineKeeper(ai, 'keeper', failNTimesModel(ai, 0, 'kept'));
    const resp = await orchestrate(ai, { agents: ['keeper'] }, (messages) =>
      lastOutput(messages, 'delegate_to_keeper')
        ? textResponse('done')
        : toolRequest('delegate_to_keeper', { task: 'keep X' })
    );
    const [got] = toolOutputs(resp.messages, 'delegate_to_keeper');
    assert.strictEqual(got.response, 'kept');
    assert.match(got.taskId, /^keeper:.+/);
    assert.strictEqual(got.status, 'completed');
  });

  it('stamps a failed synchronous delegation with its handle and the continue hint', async () => {
    const ai = genkit({});
    defineKeeper(ai, 'flaky', failNTimesModel(ai, 99, ''));
    const resp = await orchestrate(ai, { agents: ['flaky'] }, (messages) =>
      lastOutput(messages, 'delegate_to_flaky')
        ? textResponse('done')
        : toolRequest('delegate_to_flaky', { task: 'try X' })
    );
    const [got] = toolOutputs(resp.messages, 'delegate_to_flaky');
    assert.match(got.response, /model melted/);
    assert.match(got.response, /continue_task/);
    assert.match(got.taskId, /^flaky:.+/);
    assert.strictEqual(got.status, 'failed');
  });

  it('merges a synchronous run and a re-check of its handle under one artifact namespace', async () => {
    const ai = genkit({});
    const gate = makeGate();
    gate.release();
    defineGatedResearcher(ai, 'researcher', gate.opened, {
      artifacts: [{ name: 'notes.md', parts: [{ text: 'the notes' }] }],
    });
    const resp = await orchestrate(
      ai,
      { agents: ['researcher'], async: true },
      (messages) => {
        const delegated = lastOutput(messages, 'delegate_to_researcher');
        if (!delegated) {
          return toolRequest('delegate_to_researcher', { task: 'dig' });
        }
        if (!lastOutput(messages, CHECK_TOOL)) {
          return toolRequest(CHECK_TOOL, { taskIds: [delegated.taskId] });
        }
        return textResponse('done');
      }
    );
    const [delegated] = toolOutputs(resp.messages, 'delegate_to_researcher');
    const [check] = toolOutputs(resp.messages, CHECK_TOOL);
    assert.strictEqual(check.tasks[0].status, 'completed');
    assert.strictEqual(
      check.tasks[0].artifacts[0].name,
      delegated.artifacts[0].name,
      'one run, one namespace, whichever path folds it'
    );
  });

  it('gives a client-managed delegation no handle and refuses to continue it', async () => {
    const ai = genkit({});
    ai.defineAgent({
      name: 'ephemeral',
      model: failNTimesModel(ai, 0, 'done here'),
      system: 'unused',
    });
    // A store-backed sibling keeps the continue tool in front of the model.
    defineKeeper(ai, 'keeper', failNTimesModel(ai, 0, 'kept'));
    const resp = await orchestrate(
      ai,
      { agents: ['ephemeral', 'keeper'] },
      (messages) => {
        // The planted history holds one delegate_to_ephemeral result.
        if (toolOutputs(messages, 'delegate_to_ephemeral').length < 2) {
          return toolRequest('delegate_to_ephemeral', { task: 'do X' });
        }
        if (!lastOutput(messages, CONTINUE_TOOL)) {
          return toolRequest(CONTINUE_TOOL, { taskId: 'ephemeral:whatever' });
        }
        return textResponse('done');
      },
      mintedBy('ephemeral:whatever', 'delegate_to_ephemeral')
    );
    const [, got] = toolOutputs(resp.messages, 'delegate_to_ephemeral');
    assert.strictEqual(got.response, 'done here');
    assert.strictEqual(got.taskId, undefined);
    const [refused] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(refused.response, /manages its state on the client/);
  });

  it('refuses to continue a task this conversation did not mint, without spending a slot', async () => {
    const ai = genkit({});
    const store = new InMemorySessionStore();
    const { deadTask, pendingId } = await seedDeadKeeperTask(
      ai,
      store,
      failNTimesModel(ai, 0, 'kept going')
    );
    // The handle reaches this conversation only as text.
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'], maxDelegations: 1 },
      (messages) => {
        if (!lastOutput(messages, CONTINUE_TOOL)) {
          return toolRequest(CONTINUE_TOOL, {
            taskId: deadTask,
            instructions: 'continue',
          });
        }
        if (!lastOutput(messages, 'delegate_to_keeper')) {
          return toolRequest('delegate_to_keeper', { task: 'fresh' });
        }
        return textResponse('done');
      }
    );
    const [refused] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(refused.response, /not issued in this conversation/);
    const row = await store.getSnapshot({ snapshotId: pendingId });
    assert.strictEqual(row?.status, 'pending', 'no fence was written');
    const [delegated] = toolOutputs(resp.messages, 'delegate_to_keeper');
    assert.strictEqual(delegated.response, 'kept going');
  });

  it('accepts a handle that a continue result in the history minted', async () => {
    const ai = genkit({});
    defineKeeper(ai, 'keeper', failNTimesModel(ai, 0, 'kept'));
    const resp = await orchestrate(ai, { agents: ['keeper'] }, (messages) =>
      lastOutput(messages, 'delegate_to_keeper')
        ? textResponse('done')
        : toolRequest('delegate_to_keeper', { task: 'keep X' })
    );
    const { taskId } = lastOutput(resp.messages, 'delegate_to_keeper');
    // A later call sees the handle only in a continue tool's result.
    const def = await instantiateWith(
      ai,
      { agents: ['keeper'], async: true },
      mintedBy(taskId, CONTINUE_TOOL)
    );
    const checkTool = def.tools!.find((t) => t.__action.name === CHECK_TOOL)!;
    const out = await checkTool({ taskIds: [taskId] });
    assert.strictEqual(out.tasks[0].status, 'completed');
  });

  it('withholds the continue tool when no sub-agent can be continued', async () => {
    const ai = genkit({});
    ai.defineAgent({
      name: 'ephemeral',
      model: failNTimesModel(ai, 0, 'unused'),
      system: 'unused',
    });
    let toolNames: string[] = [];
    let system = '';
    await orchestrate(ai, { agents: ['ephemeral'] }, (messages) => {
      system = systemText(messages);
      return textResponse('done');
    });
    // The model hook sees the request after the generate hook, so read the
    // tool list there.
    const model = ai.defineModel(
      { name: 'orch-tools-' + Math.random() },
      async (req) => {
        toolNames = (req.tools ?? []).map((t) => t.name);
        return textResponse('done');
      }
    );
    await ai.generate({
      model,
      prompt: 'go',
      use: [agents({ agents: ['ephemeral'] })],
    });
    assert.ok(
      !toolNames.includes(CONTINUE_TOOL),
      `continue_task must not reach the model: ${toolNames}`
    );
    assert.ok(
      toolNames.includes('delegate_to_ephemeral'),
      `the delegation tool must reach the model: ${toolNames}`
    );
    assert.ok(
      !system.includes(CONTINUE_TOOL),
      `the system prompt must not mention continue_task: ${system}`
    );
  });

  it('retries a failed task from its saved progress', async () => {
    const ai = genkit({});
    const seen: MessageData[][] = [];
    defineKeeper(ai, 'flaky', failNTimesModel(ai, 1, 'recovered', seen));
    const resp = await orchestrate(
      ai,
      { agents: ['flaky'] },
      delegateThenContinue('delegate_to_flaky', 'try X')
    );
    const [failure] = toolOutputs(resp.messages, 'delegate_to_flaky');
    assert.strictEqual(failure.status, 'failed');
    const [continued] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.strictEqual(continued.response, 'recovered');
    assert.strictEqual(continued.status, 'completed');
    assert.match(continued.taskId, /^flaky:.+/);
    // The retry ran on the committed conversation: same task, no new input.
    assert.strictEqual(seen.length, 2, 'the sub-agent model runs twice');
    const retry = seen[1];
    assert.match(retry[retry.length - 1].content[0].text ?? '', /try X/);
  });

  it('steers a retry with instructions', async () => {
    const ai = genkit({});
    const seen: MessageData[][] = [];
    defineKeeper(ai, 'flaky', failNTimesModel(ai, 1, 'steered', seen));
    const resp = await orchestrate(
      ai,
      { agents: ['flaky'] },
      delegateThenContinue(
        'delegate_to_flaky',
        'try X',
        'skip the flaky source'
      )
    );
    const [continued] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.strictEqual(continued.response, 'steered');
    const retry = seen[1];
    const last = retry[retry.length - 1];
    assert.strictEqual(last.role, 'user');
    assert.match(last.content[0].text ?? '', /skip the flaky source/);
    assert.match(joinedText(retry), /try X/);
  });

  it('refuses an instructions-less follow-up on a completed task and refunds its slot', async () => {
    const ai = genkit({});
    const seen: MessageData[][] = [];
    defineKeeper(ai, 'helper', failNTimesModel(ai, 0, 'answered', seen));
    // The delegation and the corrected follow-up spend both slots, so the
    // refusal in between must return the one it reserved.
    const resp = await orchestrate(
      ai,
      { agents: ['helper'], maxDelegations: 2 },
      (messages) => {
        const delegated = lastOutput(messages, 'delegate_to_helper');
        const continues = toolOutputs(messages, CONTINUE_TOOL);
        if (!delegated) {
          return toolRequest('delegate_to_helper', { task: 'answer X' });
        }
        if (continues.length === 0) {
          return toolRequest(CONTINUE_TOOL, { taskId: delegated.taskId });
        }
        if (continues.length === 1) {
          return toolRequest(CONTINUE_TOOL, {
            taskId: delegated.taskId,
            instructions: 'now also cover Y',
          });
        }
        return textResponse('done');
      }
    );
    const continues = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(continues[0].response, /already completed/);
    assert.strictEqual(continues[1].response, 'answered');
    assert.strictEqual(continues[1].status, 'completed');
    const followUp = joinedText(seen[1]);
    assert.match(followUp, /answer X/);
    assert.match(followUp, /now also cover Y/);
  });

  it('carries the delegation label onto results, reports, and continuations', async () => {
    const ai = genkit({});
    defineKeeper(ai, 'flaky', failNTimesModel(ai, 1, 'recovered'));
    const resp = await orchestrate(
      ai,
      { agents: ['flaky'], async: true },
      (messages) => {
        const delegated = lastOutput(messages, 'delegate_to_flaky');
        if (!delegated) {
          return toolRequest('delegate_to_flaky', {
            task: 'try X',
            name: 'second-try',
          });
        }
        if (!lastOutput(messages, CHECK_TOOL)) {
          return toolRequest(CHECK_TOOL, { taskIds: [delegated.taskId] });
        }
        if (!lastOutput(messages, CONTINUE_TOOL)) {
          return toolRequest(CONTINUE_TOOL, { taskId: delegated.taskId });
        }
        return textResponse('done');
      }
    );
    const [failure] = toolOutputs(resp.messages, 'delegate_to_flaky');
    assert.strictEqual(failure.name, 'second-try');
    const [check] = toolOutputs(resp.messages, CHECK_TOOL);
    assert.strictEqual(check.tasks[0].name, 'second-try');
    const [continued] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.strictEqual(continued.response, 'recovered');
    assert.strictEqual(continued.name, 'second-try');
  });

  it('echoes a background launch label on its report', async () => {
    const ai = genkit({});
    defineKeeper(ai, 'quick', failNTimesModel(ai, 0, 'quick answer'));
    const resp = await orchestrate(
      ai,
      { agents: ['quick'], async: true },
      (messages) => {
        const launch = lastOutput(messages, 'delegate_to_quick');
        if (!launch) {
          return toolRequest('delegate_to_quick', {
            task: 'answer fast',
            background: true,
            name: 'fast-lane',
          });
        }
        if (!lastOutput(messages, WAIT_TOOL)) {
          return toolRequest(WAIT_TOOL, { taskIds: [launch.taskId] });
        }
        return textResponse('done');
      }
    );
    const [launch] = toolOutputs(resp.messages, 'delegate_to_quick');
    assert.strictEqual(launch.name, 'fast-lane');
    const [wait] = toolOutputs(resp.messages, WAIT_TOOL);
    assert.strictEqual(wait.tasks[0].name, 'fast-lane');
  });

  it('counts a continuation against maxDelegations', async () => {
    const ai = genkit({});
    defineKeeper(ai, 'flaky', failNTimesModel(ai, 99, ''));
    const resp = await orchestrate(
      ai,
      { agents: ['flaky'], maxDelegations: 1 },
      delegateThenContinue('delegate_to_flaky', 'try X')
    );
    const [continued] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(continued.response, /Delegation limit reached/);
  });

  it('recovers an expired task from its committed progress behind a fence', async () => {
    const ai = genkit({});
    const store = new InMemorySessionStore();
    const seen: MessageData[][] = [];
    const { deadTask, pendingId } = await seedDeadKeeperTask(
      ai,
      store,
      failNTimesModel(ai, 0, 'kept going', seen)
    );
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'] },
      (messages) =>
        lastOutput(messages, CONTINUE_TOOL)
          ? textResponse('done')
          : toolRequest(CONTINUE_TOOL, {
              taskId: deadTask,
              instructions: 'continue',
            }),
      mintedBy(deadTask, 'delegate_to_keeper')
    );
    const [continued] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.strictEqual(continued.response, 'kept going');
    // The fence flipped the dead row so a slow worker cannot race the
    // recovered session; no worker is left to finalize it.
    const fenced = await store.getSnapshot({ snapshotId: pendingId });
    assert.strictEqual(fenced?.status, 'aborting');
    const recovered = joinedText(seen[seen.length - 1]);
    assert.match(recovered, /start X/);
    assert.match(recovered, /continue/);
  });

  it('refuses to continue an expired task that saved nothing', async () => {
    const ai = genkit({});
    const store = new InMemorySessionStore();
    defineKeeper(ai, 'keeper', failNTimesModel(ai, 0, 'unused'), store);
    const pendingId = await saveDeadPendingRow(store, 'sess-dead');
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'] },
      (messages) =>
        lastOutput(messages, CONTINUE_TOOL)
          ? textResponse('done')
          : toolRequest(CONTINUE_TOOL, { taskId: `keeper:${pendingId}` }),
      mintedBy(`keeper:${pendingId}`, 'delegate_to_keeper')
    );
    const [refused] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(refused.response, /saved no progress to continue from/);
    assert.match(refused.response, /Delegate the task again/);
  });

  it('gates an expired task whose parent finished on instructions', async () => {
    const ai = genkit({});
    const { deadTask } = await seedDeadKeeperTask(
      ai,
      new InMemorySessionStore(),
      failNTimesModel(ai, 0, 'kept')
    );
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'] },
      (messages) =>
        lastOutput(messages, CONTINUE_TOOL)
          ? textResponse('done')
          : toolRequest(CONTINUE_TOOL, { taskId: deadTask }),
      mintedBy(deadTask, 'delegate_to_keeper')
    );
    const [refused] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(refused.response, /last finished turn/);
  });

  it('continues a failed task in the background', async () => {
    const ai = genkit({});
    defineKeeper(ai, 'flaky', failNTimesModel(ai, 1, 'recovered later'));
    const resp = await orchestrate(
      ai,
      { agents: ['flaky'], async: true },
      (messages) => {
        if (lastOutput(messages, WAIT_TOOL)) return textResponse('done');
        const continued = lastOutput(messages, CONTINUE_TOOL);
        if (continued) {
          return toolRequest(WAIT_TOOL, { taskIds: [continued.taskId] });
        }
        const delegated = lastOutput(messages, 'delegate_to_flaky');
        if (delegated) {
          return toolRequest(CONTINUE_TOOL, {
            taskId: delegated.taskId,
            background: true,
          });
        }
        return toolRequest('delegate_to_flaky', { task: 'try X' });
      }
    );
    const [continued] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.strictEqual(continued.status, 'pending');
    assert.match(continued.taskId, /^flaky:.+/);
    const [wait] = toolOutputs(resp.messages, WAIT_TOOL);
    assert.strictEqual(wait.tasks[0].status, 'completed');
    assert.strictEqual(wait.tasks[0].response, 'recovered later');
  });

  it('refuses to recover an expired task its store cannot fence', async () => {
    const ai = genkit({});
    // A store without a change feed: the flip would land, but a worker that
    // is only late would never see it.
    const store = new InMemorySessionStore();
    Object.defineProperty(store, 'onSnapshotStateChange', { value: undefined });
    const { deadTask, pendingId } = await seedDeadKeeperTask(
      ai,
      store,
      failNTimesModel(ai, 0, 'kept going')
    );
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'] },
      (messages) =>
        lastOutput(messages, CONTINUE_TOOL)
          ? textResponse('done')
          : toolRequest(CONTINUE_TOOL, {
              taskId: deadTask,
              instructions: 'continue',
            }),
      mintedBy(deadTask, 'delegate_to_keeper')
    );
    const [refused] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(refused.response, /cannot signal its worker/);
    const row = await store.getSnapshot({ snapshotId: pendingId });
    assert.strictEqual(row?.status, 'pending', 'no fence was written');
  });

  it('fails a cancelled continuation without fencing the task', async () => {
    const ai = genkit({});
    const store = new InMemorySessionStore();
    const { deadTask, pendingId } = await seedDeadKeeperTask(
      ai,
      store,
      failNTimesModel(ai, 0, 'kept going')
    );
    const def = await instantiateWith(
      ai,
      { agents: ['keeper'] },
      mintedBy(deadTask, 'delegate_to_keeper')
    );
    const continueTool = def.tools!.find(
      (t) => t.__action.name === CONTINUE_TOOL
    )!;
    await assert.rejects(
      continueTool(
        { taskId: deadTask, instructions: 'continue' },
        { abortSignal: AbortSignal.abort() }
      ),
      (e: any) => e?.name === 'AbortError'
    );
    const row = await store.getSnapshot({ snapshotId: pendingId });
    assert.strictEqual(row?.status, 'pending', 'no fence was written');
  });

  it('refuses a recovery whose fence fails transiently and refunds its slot', async () => {
    const ai = genkit({});
    const flaky = flakyStore();
    const { deadTask, pendingId } = await seedDeadKeeperTask(
      ai,
      flaky.store,
      failNTimesModel(ai, 0, 'kept going')
    );
    flaky.failSave.set(pendingId, new Error('store blip'));
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'], maxDelegations: 1 },
      (messages) => {
        const continues = toolOutputs(messages, CONTINUE_TOOL);
        if (continues.length === 1) flaky.failSave.delete(pendingId);
        return continues.length < 2
          ? toolRequest(CONTINUE_TOOL, {
              taskId: deadTask,
              instructions: 'continue',
            })
          : textResponse('done');
      },
      mintedBy(deadTask, 'delegate_to_keeper')
    );
    const continues = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(continues[0].response, /could not fence/);
    assert.match(continues[0].response, /Try again later/);
    assert.strictEqual(continues[1].response, 'kept going');
  });

  for (const tc of [
    {
      name: 'a fence the store refuses',
      refusal: /could not fence/,
      arm: (flaky: ReturnType<typeof flakyStore>, pendingId: string) =>
        flaky.failSave.set(
          pendingId,
          new GenkitError({
            status: 'FAILED_PRECONDITION',
            message: 'store cannot fence',
          })
        ),
    },
    {
      name: 'a row gone after the fence',
      refusal: /could not read/,
      arm: (flaky: ReturnType<typeof flakyStore>, pendingId: string) =>
        flaky.setGetHook((id, snap) =>
          id === pendingId && snap?.status === 'aborting' ? undefined : snap
        ),
    },
  ]) {
    it(`keeps the slot on a dead end: ${tc.name}`, async () => {
      const ai = genkit({});
      const flaky = flakyStore();
      const { deadTask, pendingId } = await seedDeadKeeperTask(
        ai,
        flaky.store,
        failNTimesModel(ai, 0, 'kept going')
      );
      tc.arm(flaky, pendingId);
      const resp = await orchestrate(
        ai,
        { agents: ['keeper'], maxDelegations: 1 },
        (messages) =>
          toolOutputs(messages, CONTINUE_TOOL).length < 2
            ? toolRequest(CONTINUE_TOOL, {
                taskId: deadTask,
                instructions: 'continue',
              })
            : textResponse('done'),
        mintedBy(deadTask, 'delegate_to_keeper')
      );
      const continues = toolOutputs(resp.messages, CONTINUE_TOOL);
      assert.match(continues[0].response, tc.refusal);
      assert.doesNotMatch(continues[0].response, /Try again later/);
      assert.match(continues[1].response, /Delegation limit reached/);
    });
  }

  it('refunds the slot when the parent read blips', async () => {
    const ai = genkit({});
    const flaky = flakyStore();
    const { deadTask, committedId } = await seedDeadKeeperTask(
      ai,
      flaky.store,
      failNTimesModel(ai, 0, 'kept going')
    );
    flaky.setGetHook((id, snap) => {
      if (id === committedId) throw new Error('parent read blip');
      return snap;
    });
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'], maxDelegations: 1 },
      (messages) => {
        const continues = toolOutputs(messages, CONTINUE_TOOL);
        if (continues.length === 1) flaky.setGetHook(undefined);
        return continues.length < 2
          ? toolRequest(CONTINUE_TOOL, {
              taskId: deadTask,
              instructions: 'continue',
            })
          : textResponse('done');
      },
      mintedBy(deadTask, 'delegate_to_keeper')
    );
    const continues = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(continues[0].response, /could not be read/);
    assert.match(continues[0].response, /Try again later/);
    assert.strictEqual(continues[1].response, 'kept going');
  });

  it('refuses a fenced task whose live worker is winding down, then continues it once it settles', async () => {
    const ai = genkit({});
    const flaky = flakyStore();
    const { deadTask, pendingId } = await seedDeadKeeperTask(
      ai,
      flaky.store,
      failNTimesModel(ai, 0, 'kept going')
    );
    // While the hook is set, the fenced row reads with a fresh heartbeat: a
    // live worker's beats keep it so while it drains.
    flaky.setGetHook((id, snap) =>
      id === pendingId && snap?.status === 'aborting'
        ? { ...snap, heartbeatAt: new Date().toISOString() }
        : snap
    );
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'], maxDelegations: 1 },
      (messages) => {
        const continues = toolOutputs(messages, CONTINUE_TOOL);
        // The worker dies without finalizing: the row goes stale again, and
        // the same handle recovers through the parent.
        if (continues.length === 1) flaky.setGetHook(undefined);
        return continues.length < 2
          ? toolRequest(CONTINUE_TOOL, {
              taskId: deadTask,
              instructions: 'continue',
            })
          : textResponse('done');
      },
      mintedBy(deadTask, 'delegate_to_keeper')
    );
    const continues = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(continues[0].response, /winding down/);
    assert.strictEqual(continues[1].response, 'kept going');
  });

  it('gates a fenced task whose completed finalize won the race on instructions', async () => {
    const ai = genkit({});
    const flaky = flakyStore();
    const { deadTask, pendingId } = await seedDeadKeeperTask(
      ai,
      flaky.store,
      failNTimesModel(ai, 0, 'kept going')
    );
    flaky.setGetHook((id, snap) =>
      id === pendingId && snap?.status === 'aborting'
        ? {
            ...snap,
            status: 'completed',
            finishReason: 'stop',
            heartbeatAt: undefined,
            state: {
              messages: [{ role: 'user', content: [{ text: 'finished' }] }],
            },
          }
        : snap
    );
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'] },
      (messages) =>
        lastOutput(messages, CONTINUE_TOOL)
          ? textResponse('done')
          : toolRequest(CONTINUE_TOOL, { taskId: deadTask }),
      mintedBy(deadTask, 'delegate_to_keeper')
    );
    const [refused] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(refused.response, /already completed/);
  });

  it('reports an aborting task as winding down and refuses to continue it with a refund', async () => {
    const ai = genkit({});
    const store = new InMemorySessionStore();
    const { committedId } = await seedDeadKeeperTask(
      ai,
      store,
      failNTimesModel(ai, 0, 'kept going')
    );
    const committed = await store.getSnapshot({ snapshotId: committedId });
    const now = new Date().toISOString();
    const abortingId = (await store.saveSnapshot(undefined, () => ({
      createdAt: now,
      updatedAt: now,
      heartbeatAt: now,
      status: 'aborting',
      sessionId: committed!.sessionId,
      parentId: committedId,
      state: { sessionId: committed!.sessionId },
    })))!;
    const task = `keeper:${abortingId}`;
    const resp = await orchestrate(
      ai,
      { agents: ['keeper'], async: true, maxDelegations: 1 },
      (messages) => {
        if (!lastOutput(messages, CHECK_TOOL)) {
          return toolRequest(CHECK_TOOL, { taskIds: [task] });
        }
        if (!lastOutput(messages, CONTINUE_TOOL)) {
          return toolRequest(CONTINUE_TOOL, {
            taskId: task,
            instructions: 'go on',
          });
        }
        // The planted history holds one delegate_to_keeper result.
        if (toolOutputs(messages, 'delegate_to_keeper').length < 2) {
          return toolRequest('delegate_to_keeper', { task: 'more' });
        }
        return textResponse('done');
      },
      mintedBy(task, 'delegate_to_keeper')
    );
    const [check] = toolOutputs(resp.messages, CHECK_TOOL);
    assert.strictEqual(check.tasks[0].status, 'aborting');
    assert.match(check.tasks[0].error, /winding down/);
    const [refused] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(refused.response, /winding down/);
    // The refusal returned its slot: under a cap of one, a delegation still
    // runs afterwards.
    const [, delegated] = toolOutputs(resp.messages, 'delegate_to_keeper');
    assert.strictEqual(delegated.response, 'kept going');
  });

  it('reports an interrupted task without a continue hint and refuses to continue it', async () => {
    const ai = genkit({});
    const gate = makeGate();
    ai.defineCustomAgent(
      { name: 'researcher', store: new InMemorySessionStore() },
      async (sess) => {
        await sess.run(async () => {
          await gate.opened;
          sess.addMessages([
            { role: 'model', content: [{ text: 'need a human' }] },
          ]);
          return { finishReason: 'interrupted' as const };
        });
        const msgs = sess.getMessages();
        return {
          message: msgs[msgs.length - 1],
          finishReason: 'interrupted' as const,
        };
      }
    );
    const resp = await orchestrate(
      ai,
      { agents: ['researcher'], async: true },
      (messages) => {
        const launch = lastOutput(messages, 'delegate_to_researcher');
        if (!launch) {
          return toolRequest('delegate_to_researcher', {
            task: 'dig into X',
            background: true,
          });
        }
        if (!lastOutput(messages, WAIT_TOOL)) {
          gate.release();
          return toolRequest(WAIT_TOOL, { taskIds: [launch.taskId] });
        }
        if (!lastOutput(messages, CONTINUE_TOOL)) {
          return toolRequest(CONTINUE_TOOL, {
            taskId: launch.taskId,
            instructions: 'the answer is 42',
          });
        }
        return textResponse('done');
      }
    );
    const [wait] = toolOutputs(resp.messages, WAIT_TOOL);
    assert.strictEqual(wait.tasks[0].status, 'failed');
    assert.match(wait.tasks[0].error, /interrupted/);
    assert.doesNotMatch(wait.tasks[0].error, /continue_task/);
    const [refused] = toolOutputs(resp.messages, CONTINUE_TOOL);
    assert.match(refused.response, /stopped on an interrupt/);
  });
});
