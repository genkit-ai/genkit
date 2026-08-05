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
import { z } from 'genkit';
import { genkit, Session } from 'genkit/beta';
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
});
