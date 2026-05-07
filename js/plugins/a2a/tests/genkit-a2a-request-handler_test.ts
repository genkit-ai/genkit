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
import { describe, it } from 'node:test';

import type { AgentCard, MessageSendParams } from '@a2a-js/sdk';

import {
  GenkitA2ARequestHandler,
  type GenkitAgentLike,
} from '../src/genkit-a2a-request-handler.js';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function createTestCard(overrides?: Partial<AgentCard>): AgentCard {
  return {
    name: 'Test Agent',
    description: 'A test agent',
    url: 'https://test.example.com/a2a',
    version: '1.0.0',
    protocolVersion: '0.2.2',
    capabilities: {},
    defaultInputModes: ['text/plain'],
    defaultOutputModes: ['text/plain'],
    skills: [],
    ...overrides,
  } as AgentCard;
}

function createMockAgent(overrides?: Partial<GenkitAgentLike>): GenkitAgentLike {
  return {
    __action: { name: 'testAgent', description: 'A test agent' },
    run: async () => ({
      result: {
        message: { role: 'model', content: [{ text: 'Hello!' }] },
        state: {
          messages: [
            { role: 'user', content: [{ text: 'Hi' }] },
            { role: 'model', content: [{ text: 'Hello!' }] },
          ],
          artifacts: [],
          custom: {},
        },
      },
    }),
    getSnapshotData: async () => undefined,
    abort: async () => undefined,
    ...overrides,
  };
}

function sendParams(
  text: string,
  extras?: Record<string, unknown>
): MessageSendParams {
  return {
    message: {
      kind: 'message',
      messageId: crypto.randomUUID(),
      role: 'user',
      parts: [{ kind: 'text', text }],
      ...extras,
    },
  } as MessageSendParams;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GenkitA2ARequestHandler', () => {
  // -----------------------------------------------------------------------
  // Agent Card
  // -----------------------------------------------------------------------

  describe('getAgentCard', () => {
    it('returns the configured agent card', async () => {
      const card = createTestCard({ name: 'My Agent' });
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card,
      });

      const result = await handler.getAgentCard();
      assert.strictEqual(result.name, 'My Agent');
      assert.strictEqual(result.protocolVersion, '0.2.2');
    });

    it('getAuthenticatedExtendedAgentCard returns same card', async () => {
      const card = createTestCard();
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card,
      });

      const result = await handler.getAuthenticatedExtendedAgentCard();
      assert.deepStrictEqual(result, card);
    });

    it('derives card from agent metadata when card is omitted', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          __action: { name: 'myWeatherAgent', description: 'Weather helper' },
        }),
        url: 'http://localhost:3000',
      });

      const result = await handler.getAgentCard();
      assert.strictEqual(result.name, 'myWeatherAgent');
      assert.strictEqual(result.description, 'Weather helper');
      assert.strictEqual(result.url, 'http://localhost:3000');
      assert.strictEqual(result.protocolVersion, '0.2.2');
    });

    it('uses default description when agent has none', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          __action: { name: 'silentAgent' },
        }),
        url: 'http://localhost:4000',
      });

      const result = await handler.getAgentCard();
      assert.strictEqual(result.description, 'silentAgent A2A agent');
    });

    it('throws when neither card nor url is provided', () => {
      assert.throws(() => {
        new GenkitA2ARequestHandler({
          agent: createMockAgent(),
        });
      }, /either.*card.*url/i);
    });
  });

  // -----------------------------------------------------------------------
  // sendMessage
  // -----------------------------------------------------------------------

  describe('sendMessage', () => {
    it('returns a Task with kind "task"', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      const result = await handler.sendMessage(sendParams('Hello'));
      assert.strictEqual(result.kind, 'task');
    });

    it('returns a completed task with history', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      const result = await handler.sendMessage(sendParams('Hi'));
      assert.strictEqual(result.kind, 'task');
      const task = result as any;
      assert.strictEqual(task.status.state, 'completed');
      assert.ok(task.history);
      assert.strictEqual(task.history.length, 2);
      assert.strictEqual(task.history[0].role, 'user');
      assert.strictEqual(task.history[1].role, 'agent');
    });

    it('maps Genkit parts to A2A parts in history', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          run: async () => ({
            result: {
              message: {
                role: 'model',
                content: [{ text: 'The answer is 42.' }],
              },
              state: {
                messages: [
                  { role: 'user', content: [{ text: 'What is the answer?' }] },
                  {
                    role: 'model',
                    content: [{ text: 'The answer is 42.' }],
                  },
                ],
                artifacts: [],
                custom: {},
              },
            },
          }),
        }),
        card: createTestCard(),
      });

      const result = (await handler.sendMessage(
        sendParams('What is the answer?')
      )) as any;

      const lastMsg = result.history[result.history.length - 1];
      assert.strictEqual(lastMsg.parts[0].kind, 'text');
      assert.strictEqual(lastMsg.parts[0].text, 'The answer is 42.');
    });

    it('passes the incoming message to agent.run', async () => {
      let receivedInput: any;

      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          run: async (input) => {
            receivedInput = input;
            return {
              result: {
                state: { messages: [], artifacts: [], custom: {} },
              },
            };
          },
        }),
        card: createTestCard(),
      });

      await handler.sendMessage(sendParams('Test message'));

      assert.ok(receivedInput);
      assert.strictEqual(receivedInput.messages.length, 1);
      assert.strictEqual(receivedInput.messages[0].role, 'user');
      const parts = receivedInput.messages[0].content;
      assert.ok(parts.some((p: any) => p.text === 'Test message'));
    });

    it('detects tool response parts and sets role to "tool"', async () => {
      let receivedInput: any;

      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          run: async (input) => {
            receivedInput = input;
            return {
              result: {
                state: { messages: [], artifacts: [], custom: {} },
              },
            };
          },
        }),
        card: createTestCard(),
      });

      await handler.sendMessage({
        message: {
          kind: 'message',
          messageId: 'msg-1',
          role: 'user',
          parts: [
            {
              kind: 'data',
              data: { id: 'c1', name: 'myTool', response: { ok: true } },
              metadata: { genkit_type: 'function_response' },
            } as any,
          ],
        },
      });

      assert.strictEqual(receivedInput.messages[0].role, 'tool');
    });

    it('uses incoming contextId and taskId when provided', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      const result = (await handler.sendMessage(
        sendParams('Hi', { contextId: 'ctx-42', taskId: 'task-99' })
      )) as any;

      assert.strictEqual(result.contextId, 'ctx-42');
      assert.strictEqual(result.id, 'task-99');
    });

    it('generates contextId and taskId when not provided', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      const result = (await handler.sendMessage(sendParams('Hi'))) as any;

      assert.ok(result.contextId, 'Should generate a contextId');
      assert.ok(result.id, 'Should generate a taskId');
    });

    it('passes taskId as init.snapshotId for continuity', async () => {
      let receivedInit: any;

      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          run: async (_input, options) => {
            receivedInit = options.init;
            return {
              result: {
                state: { messages: [], artifacts: [], custom: {} },
              },
            };
          },
        }),
        card: createTestCard(),
      });

      await handler.sendMessage(sendParams('Hi', { taskId: 'prev-task-123' }));

      assert.strictEqual(receivedInit.snapshotId, 'prev-task-123');
      assert.strictEqual(receivedInit.newSnapshotId, 'prev-task-123');
    });

    it('includes artifacts in the task response', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          run: async () => ({
            result: {
              state: {
                messages: [
                  { role: 'model', content: [{ text: 'Here is your report' }] },
                ],
                artifacts: [
                  {
                    name: 'report-1',
                    parts: [{ text: 'Report content...' }],
                  },
                ],
                custom: {},
              },
            },
          }),
        }),
        card: createTestCard(),
      });

      const result = (await handler.sendMessage(
        sendParams('Generate report')
      )) as any;

      assert.ok(result.artifacts);
      assert.strictEqual(result.artifacts.length, 1);
      assert.strictEqual(result.artifacts[0].artifactId, 'report-1');
    });

    it('omits artifacts when none are present', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          run: async () => ({
            result: {
              state: {
                messages: [
                  { role: 'model', content: [{ text: 'No artifacts' }] },
                ],
                artifacts: [],
                custom: {},
              },
            },
          }),
        }),
        card: createTestCard(),
      });

      const result = (await handler.sendMessage(sendParams('Hi'))) as any;
      assert.strictEqual(result.artifacts, undefined);
    });
  });

  // -----------------------------------------------------------------------
  // sendMessageStream
  // -----------------------------------------------------------------------

  describe('sendMessageStream', () => {
    it('yields the same result as sendMessage (blocking fallback)', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      const events: any[] = [];
      for await (const event of handler.sendMessageStream(sendParams('Hi'))) {
        events.push(event);
      }

      assert.strictEqual(events.length, 1);
      assert.strictEqual(events[0].kind, 'task');
    });
  });

  // -----------------------------------------------------------------------
  // getTask
  // -----------------------------------------------------------------------

  describe('getTask', () => {
    it('returns a Task from snapshot data', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          getSnapshotData: async (snapshotId) => ({
            snapshotId,
            createdAt: '2026-05-07T12:00:00Z',
            event: 'turnEnd',
            status: 'done',
            state: {
              messages: [
                { role: 'user', content: [{ text: 'Question' }] },
                { role: 'model', content: [{ text: 'Answer' }] },
              ],
              artifacts: [],
              custom: {},
            },
          }),
        }),
        card: createTestCard(),
      });

      const result = await handler.getTask({ id: 'snap-123' });

      assert.strictEqual(result.kind, 'task');
      assert.strictEqual(result.id, 'snap-123');
      assert.strictEqual(result.status.state, 'completed');
      assert.ok(result.history);
      assert.strictEqual(result.history!.length, 2);
    });

    it('throws taskNotFound when snapshot does not exist', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      await assert.rejects(
        () => handler.getTask({ id: 'nonexistent' }),
        (err: any) => {
          assert.ok(err.message.includes('nonexistent'));
          return true;
        }
      );
    });

    it('respects historyLength parameter', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          getSnapshotData: async (snapshotId) => ({
            snapshotId,
            createdAt: '2026-05-07T12:00:00Z',
            event: 'turnEnd',
            status: 'done',
            state: {
              messages: [
                { role: 'user', content: [{ text: 'Q1' }] },
                { role: 'model', content: [{ text: 'A1' }] },
                { role: 'user', content: [{ text: 'Q2' }] },
                { role: 'model', content: [{ text: 'A2' }] },
              ],
              artifacts: [],
              custom: {},
            },
          }),
        }),
        card: createTestCard(),
      });

      const result = await handler.getTask({
        id: 'snap-123',
        historyLength: 2,
      });

      assert.strictEqual(result.history!.length, 2);
      // Should be the last 2 messages
      assert.ok(
        result.history![0].parts.some((p: any) => p.text === 'Q2')
      );
    });

    it('maps snapshot status correctly', async () => {
      const statuses = [
        { genkit: 'done', a2a: 'completed' },
        { genkit: 'pending', a2a: 'working' },
        { genkit: 'failed', a2a: 'failed' },
        { genkit: 'aborted', a2a: 'canceled' },
      ];

      for (const { genkit, a2a } of statuses) {
        const handler = new GenkitA2ARequestHandler({
          agent: createMockAgent({
            getSnapshotData: async (snapshotId) => ({
              snapshotId,
              createdAt: '2026-05-07T12:00:00Z',
              event: 'turnEnd',
              status: genkit,
              state: { messages: [], artifacts: [], custom: {} },
            }),
          }),
          card: createTestCard(),
        });

        const result = await handler.getTask({ id: 'snap-1' });
        assert.strictEqual(
          result.status.state,
          a2a,
          `Genkit status "${genkit}" should map to A2A state "${a2a}"`
        );
      }
    });
  });

  // -----------------------------------------------------------------------
  // cancelTask
  // -----------------------------------------------------------------------

  describe('cancelTask', () => {
    it('aborts the agent and returns the canceled task', async () => {
      let abortCalled = false;

      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent({
          abort: async () => {
            abortCalled = true;
            return 'pending';
          },
          getSnapshotData: async (snapshotId) => ({
            snapshotId,
            createdAt: '2026-05-07T12:00:00Z',
            event: 'turnEnd',
            status: 'aborted',
            state: { messages: [], artifacts: [], custom: {} },
          }),
        }),
        card: createTestCard(),
      });

      const result = await handler.cancelTask({ id: 'task-to-cancel' });

      assert.ok(abortCalled, 'abort should have been called');
      assert.strictEqual(result.status.state, 'canceled');
    });

    it('throws taskNotFound when abort returns undefined', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      await assert.rejects(
        () => handler.cancelTask({ id: 'nonexistent' }),
        (err: any) => {
          assert.ok(err.message.includes('nonexistent'));
          return true;
        }
      );
    });
  });

  // -----------------------------------------------------------------------
  // Push notifications — unsupported
  // -----------------------------------------------------------------------

  describe('push notifications (unsupported)', () => {
    it('setTaskPushNotificationConfig throws', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      await assert.rejects(() =>
        handler.setTaskPushNotificationConfig({} as any)
      );
    });

    it('getTaskPushNotificationConfig throws', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      await assert.rejects(() =>
        handler.getTaskPushNotificationConfig({ id: 'x' })
      );
    });

    it('listTaskPushNotificationConfigs throws', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      await assert.rejects(() =>
        handler.listTaskPushNotificationConfigs({ taskId: 'x' } as any)
      );
    });

    it('deleteTaskPushNotificationConfig throws', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      await assert.rejects(() =>
        handler.deleteTaskPushNotificationConfig({ taskId: 'x' } as any)
      );
    });
  });

  // -----------------------------------------------------------------------
  // resubscribe — unsupported
  // -----------------------------------------------------------------------

  describe('resubscribe (unsupported)', () => {
    it('throws unsupported operation', async () => {
      const handler = new GenkitA2ARequestHandler({
        agent: createMockAgent(),
        card: createTestCard(),
      });

      await assert.rejects(async () => {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        for await (const _ of handler.resubscribe({ id: 'x' })) {
          // should not reach here
        }
      });
    });
  });
});
