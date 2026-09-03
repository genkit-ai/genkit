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
import { z, type MessageData } from 'genkit';
import { InMemorySessionStore, Session, genkit } from 'genkit/beta';
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
  } = {}
) {
  return ai.defineCustomAgent(
    { name, store: opts.store ?? new InMemorySessionStore() },
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

  it('collects a task launched by an earlier generate call from its ID alone', async () => {
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

    // Second call, fresh middleware instance: wait on the recorded task ID
    // plus a missing snapshot and an unconfigured agent, which must be
    // reported in isolation from one another.
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
      prompt: 'collect',
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

    const def = agents.instantiate({
      config: { agents: ['researcher'], async: true },
      ai,
      pluginConfig: undefined,
    });
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

    const def = agents.instantiate({
      config: { agents: ['researcher'], async: true },
      ai,
      pluginConfig: undefined,
    });
    const abortTool = def.tools!.find((t) => t.__action.name === ABORT_TOOL)!;
    const out = await abortTool({ taskIds: [`researcher:${task.snapshotId}`] });
    assert.strictEqual(out.tasks[0].status, 'aborted');
    assert.match(out.tasks[0].error, /cannot signal its worker/);
    gate.release();
  });

  it('does not advertise task handles on a synchronous instance', async () => {
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
    // does not have.
    const inputSchema = JSON.stringify(delegateDef.inputSchema);
    const outputSchema = JSON.stringify(delegateDef.outputSchema);
    assert.ok(!inputSchema.includes('background'));
    assert.ok(!outputSchema.includes('taskId'));
    assert.ok(!outputSchema.includes('background'));
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
      prompt: 'go',
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

  it('aborts a running background task and reports it aborted afterwards', async () => {
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
        if (toolOutputs(req.messages, CHECK_TOOL).length === 0) {
          return toolRequest(CHECK_TOOL, { taskIds: [launches[0].taskId] });
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
    const [aborted] = toolOutputs(result.messages, ABORT_TOOL);
    assert.strictEqual(aborted.tasks[0].status, 'aborted');
    assert.ok(aborted.tasks[0].error);
    // The row said aborted; the runtime observes that flip and cancels the
    // work, which is the half a status write alone would not prove.
    assert.strictEqual(stopped, true, 'the sub-agent was never cancelled');
    const [checked] = toolOutputs(result.messages, CHECK_TOOL);
    assert.strictEqual(checked.tasks[0].status, 'aborted');
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

    const def = agents.instantiate({
      config: { agents: ['researcher'], async: true },
      ai,
      pluginConfig: undefined,
    });
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
    const def = agents.instantiate({
      config: { agents: ['a', 'a:b'], async: true },
      ai,
      pluginConfig: undefined,
    });
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
