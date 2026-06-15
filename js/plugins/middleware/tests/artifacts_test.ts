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
import { genkit, Session } from 'genkit/beta';
import { describe, it } from 'node:test';
import { artifacts } from '../src/artifacts.js';

describe('artifacts middleware', () => {
  // Helper: create a mock model that calls a tool on the first turn,
  // then responds with text on the second.
  function createToolModel(
    ai: any,
    toolName: string,
    input: any,
    namePrefix = 'art'
  ) {
    let turn = 0;
    return ai.defineModel(
      { name: `${namePrefix}-model-${Math.random()}` },
      async () => {
        turn++;
        if (turn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [{ toolRequest: { name: toolName, input } }],
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
  }

  // Helper: create a model that calls multiple tools sequentially.
  function createMultiToolModel(
    ai: any,
    calls: { toolName: string; input: any }[],
    namePrefix = 'multi'
  ) {
    let turn = 0;
    return ai.defineModel(
      { name: `${namePrefix}-model-${Math.random()}` },
      async () => {
        turn++;
        if (turn <= calls.length) {
          const call = calls[turn - 1];
          return {
            message: {
              role: 'model' as const,
              content: [
                { toolRequest: { name: call.toolName, input: call.input } },
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
  }

  it('write_artifact creates an artifact in the session', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-write' });

    const model = createToolModel(ai, 'write_artifact', {
      name: 'hello.txt',
      content: 'Hello, world!',
    });

    await session.run(async () => {
      const result = await ai.generate({
        model,
        prompt: 'create a file',
        use: [artifacts()],
      });

      assert.ok(result.text);
    });

    // Verify artifact was stored in the session.
    const stored = session.getArtifacts();
    assert.strictEqual(stored.length, 1, 'Should have 1 artifact');
    assert.strictEqual(stored[0].name, 'hello.txt');
    assert.strictEqual(stored[0].parts[0].text, 'Hello, world!');
  });

  it('read_artifact reads an existing artifact from the session', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-read' });

    // Pre-populate session with an artifact.
    session.addArtifacts([
      {
        name: 'poem.txt',
        parts: [{ text: 'Roses are red, violets are blue.' }],
      },
    ]);

    let capturedToolOutput: any;
    let modelTurn = 0;
    const model = ai.defineModel(
      { name: 'read-model-' + Math.random() },
      async (req: any) => {
        modelTurn++;
        if (modelTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'read_artifact',
                    input: { name: 'poem.txt' },
                  },
                },
              ],
            },
          };
        }
        // Capture the tool response from the previous turn.
        const toolMsg = req.messages?.find((m: any) => m.role === 'tool');
        if (toolMsg) {
          const toolResp = toolMsg.content.find((p: any) => p.toolResponse);
          capturedToolOutput = toolResp?.toolResponse?.output;
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'read it' }],
          },
        };
      }
    );

    await session.run(async () => {
      await ai.generate({
        model,
        prompt: 'read poem',
        use: [artifacts()],
      });
    });

    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.strictEqual(capturedToolOutput.found, true);
    assert.strictEqual(capturedToolOutput.name, 'poem.txt');
    assert.ok(capturedToolOutput.content.includes('Roses are red'));
  });

  it('read_artifact returns found=false for missing artifact', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-missing' });

    let capturedToolOutput: any;
    let modelTurn = 0;
    const model = ai.defineModel(
      { name: 'missing-model-' + Math.random() },
      async (req: any) => {
        modelTurn++;
        if (modelTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'read_artifact',
                    input: { name: 'nonexistent.txt' },
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
            content: [{ text: 'not found' }],
          },
        };
      }
    );

    await session.run(async () => {
      await ai.generate({
        model,
        prompt: 'read missing',
        use: [artifacts()],
      });
    });

    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.strictEqual(capturedToolOutput.found, false);
  });

  it('readonly mode does not provide write_artifact tool', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-readonly' });

    // Model tries to call write_artifact, which should not be available.
    // In readonly mode, only read_artifact is injected.
    // The model calling a non-existent tool should cause an error or
    // the tool should not be in the available tools.
    let capturedTools: any[];
    const model = ai.defineModel(
      { name: 'readonly-model-' + Math.random() },
      async (req: any) => {
        capturedTools = req.tools ?? [];
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
        model,
        prompt: 'test readonly',
        use: [artifacts({ readonly: true })],
      });
    });

    // Verify only read_artifact is in the tools.
    const toolNames = capturedTools!.map((t: any) => t.name);
    assert.ok(
      toolNames.includes('read_artifact'),
      'Should have read_artifact tool'
    );
    assert.ok(
      !toolNames.includes('write_artifact'),
      'Should NOT have write_artifact tool in readonly mode'
    );
  });

  it('default mode provides both read_artifact and write_artifact', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-readwrite' });

    let capturedTools: any[];
    const model = ai.defineModel(
      { name: 'rw-model-' + Math.random() },
      async (req: any) => {
        capturedTools = req.tools ?? [];
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
        model,
        prompt: 'test readwrite',
        use: [artifacts()],
      });
    });

    const toolNames = capturedTools!.map((t: any) => t.name);
    assert.ok(
      toolNames.includes('read_artifact'),
      'Should have read_artifact tool'
    );
    assert.ok(
      toolNames.includes('write_artifact'),
      'Should have write_artifact tool'
    );
  });

  it('injects <artifacts> listing into system prompt', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-listing' });

    // Pre-populate with some artifacts.
    session.addArtifacts([
      { name: 'report.md', parts: [{ text: '# Report\nSome content' }] },
      {
        name: 'data.json',
        parts: [{ text: '{"key": "value"}' }],
        metadata: { source: 'researcher' },
      },
    ]);

    let capturedMessages: any[];
    const model = ai.defineModel(
      { name: 'listing-model-' + Math.random() },
      async (req: any) => {
        capturedMessages = req.messages;
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
        model,
        prompt: 'list artifacts',
        use: [artifacts()],
      });
    });

    const systemMsg = capturedMessages!.find((m: any) => m.role === 'system');
    assert.ok(systemMsg, 'System message should exist');

    const artifactBlock = systemMsg.content.find(
      (p: any) => p.text && p.text.includes('<artifacts>')
    );
    assert.ok(artifactBlock, 'System should contain <artifacts> block');
    assert.ok(
      artifactBlock.text.includes('report.md'),
      'Listing should include report.md'
    );
    assert.ok(
      artifactBlock.text.includes('data.json'),
      'Listing should include data.json'
    );
    assert.ok(
      artifactBlock.text.includes('[from: researcher]'),
      'Listing should include source metadata'
    );
  });

  it('shows empty listing when no artifacts in session', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-empty' });

    let capturedMessages: any[];
    const model = ai.defineModel(
      { name: 'empty-model-' + Math.random() },
      async (req: any) => {
        capturedMessages = req.messages;
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
        model,
        prompt: 'check empty',
        use: [artifacts()],
      });
    });

    const systemMsg = capturedMessages!.find((m: any) => m.role === 'system');
    assert.ok(systemMsg, 'System message should exist');

    const artifactBlock = systemMsg.content.find(
      (p: any) => p.text && p.text.includes('<artifacts>')
    );
    assert.ok(artifactBlock, 'System should contain <artifacts> block');
    assert.ok(
      artifactBlock.text.includes('No artifacts are currently available'),
      'Should indicate no artifacts'
    );
  });

  it('write then read in same session', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-write-read' });

    let capturedToolOutput: any;

    const model = createMultiToolModel(ai, [
      {
        toolName: 'write_artifact',
        input: { name: 'code.ts', content: 'console.log("hello")' },
      },
      {
        toolName: 'read_artifact',
        input: { name: 'code.ts' },
      },
    ]);

    // Override model to capture tool output on third turn.
    let turnCount = 0;
    const captureModel = ai.defineModel(
      { name: 'wr-model-' + Math.random() },
      async (req: any) => {
        turnCount++;
        if (turnCount === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'write_artifact',
                    input: {
                      name: 'code.ts',
                      content: 'console.log("hello")',
                    },
                  },
                },
              ],
            },
          };
        }
        if (turnCount === 2) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'read_artifact',
                    input: { name: 'code.ts' },
                  },
                },
              ],
            },
          };
        }
        // Third turn: capture the read result.
        const toolMsgs = req.messages?.filter((m: any) => m.role === 'tool');
        if (toolMsgs && toolMsgs.length >= 2) {
          const lastToolMsg = toolMsgs[toolMsgs.length - 1];
          const toolResp = lastToolMsg.content.find((p: any) => p.toolResponse);
          capturedToolOutput = toolResp?.toolResponse?.output;
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'done' }],
          },
        };
      }
    );

    await session.run(async () => {
      await ai.generate({
        model: captureModel,
        prompt: 'write then read',
        use: [artifacts()],
      });
    });

    assert.ok(capturedToolOutput, 'Should capture read result');
    assert.strictEqual(capturedToolOutput.found, true);
    assert.strictEqual(capturedToolOutput.name, 'code.ts');
    assert.ok(capturedToolOutput.content.includes('console.log'));
  });

  it('write_artifact deduplicates by name (replaces existing)', async () => {
    const ai = genkit({});
    const session = new Session({ sessionId: 'test-dedup' });

    let turnCount = 0;
    const model = ai.defineModel(
      { name: 'dedup-model-' + Math.random() },
      async () => {
        turnCount++;
        if (turnCount === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'write_artifact',
                    input: { name: 'file.txt', content: 'version 1' },
                  },
                },
              ],
            },
          };
        }
        if (turnCount === 2) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'write_artifact',
                    input: { name: 'file.txt', content: 'version 2' },
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

    await session.run(async () => {
      await ai.generate({
        model,
        prompt: 'overwrite file',
        use: [artifacts()],
      });
    });

    const stored = session.getArtifacts();
    assert.strictEqual(
      stored.length,
      1,
      'Should have 1 artifact (deduplicated)'
    );
    assert.strictEqual(stored[0].name, 'file.txt');
    assert.strictEqual(
      stored[0].parts[0].text,
      'version 2',
      'Should have latest version'
    );
  });

  it('gracefully handles no active session', async () => {
    const ai = genkit({});

    // No session.run() wrapper, so ai.currentSession() throws inside the
    // tool. The middleware catches that and the read_artifact tool must
    // resolve deterministically with found=false and an explanatory message
    // (rather than throwing out of the generate call).
    let capturedToolOutput: any;
    let modelTurn = 0;
    const model = ai.defineModel(
      { name: 'nosession-model-' + Math.random() },
      async (req: any) => {
        modelTurn++;
        if (modelTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'read_artifact',
                    input: { name: 'anything' },
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
            content: [{ text: 'done' }],
          },
        };
      }
    );

    const result = await ai.generate({
      model,
      prompt: 'no session',
      use: [artifacts()],
    });

    // The generate call must complete (no thrown error) and the tool must
    // report the missing session deterministically.
    assert.ok(result.text.includes('done'));
    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.strictEqual(
      capturedToolOutput.found,
      false,
      'read_artifact should report found=false with no active session'
    );
    assert.match(
      capturedToolOutput.content,
      /no active session/i,
      'read_artifact should explain there is no active session'
    );
  });

  it('write_artifact reports no active session deterministically', async () => {
    const ai = genkit({});

    let capturedToolOutput: any;
    let modelTurn = 0;
    const model = ai.defineModel(
      { name: 'nosession-write-' + Math.random() },
      async (req: any) => {
        modelTurn++;
        if (modelTurn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'write_artifact',
                    input: { name: 'x.txt', content: 'hello' },
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
            content: [{ text: 'done' }],
          },
        };
      }
    );

    const result = await ai.generate({
      model,
      prompt: 'no session write',
      use: [artifacts()],
    });

    assert.ok(result.text.includes('done'));
    assert.ok(capturedToolOutput, 'Tool output should be captured');
    assert.match(
      capturedToolOutput.status,
      /no active session/i,
      'write_artifact should explain there is no active session'
    );
  });
});
