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

import type {
  Message as A2AMessage,
  MessageSendParams,
  Task,
  TaskArtifactUpdateEvent,
  TaskStatusUpdateEvent,
} from '@a2a-js/sdk';
import type {
  AgentChunk,
  AgentInput,
  AgentResponse,
  AgentStreamChunk,
  Part,
} from 'genkit/beta';
import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  GenkitA2ARequestHandler,
  type GenkitAgent,
} from '../src/request-handler.js';

type A2AEvent =
  | A2AMessage
  | Task
  | TaskStatusUpdateEvent
  | TaskArtifactUpdateEvent;

/**
 * Builds a minimal fake Genkit agent whose single turn streams the given
 * `chunks` and resolves with the given finish state.
 */
function fakeAgent(opts: {
  chunks: AgentStreamChunk[];
  response: Partial<AgentResponse> & {
    finishReason: AgentResponse['finishReason'];
  };
  capture?: (input: AgentInput, init?: { sessionId?: string }) => void;
}): GenkitAgent {
  return {
    __action: { name: 'fakeAgent', description: 'A fake agent.' },
    chat(init?: { sessionId?: string }) {
      return {
        sendStream(input: AgentInput) {
          opts.capture?.(input, init);
          const stream = (async function* (): AsyncIterable<AgentChunk> {
            for (const raw of opts.chunks) {
              yield { raw } as AgentChunk;
            }
          })();
          return {
            stream,
            response: Promise.resolve(opts.response as AgentResponse),
            abort() {},
          };
        },
      } as unknown as ReturnType<GenkitAgent['chat']>;
    },
  };
}

function userMessage(
  parts: A2AMessage['parts'],
  extra?: Partial<A2AMessage>
): MessageSendParams {
  return {
    message: {
      kind: 'message',
      messageId: 'u1',
      role: 'user',
      parts,
      ...extra,
    },
  };
}

function modelChunk(content: Part[]): AgentStreamChunk {
  return { modelChunk: { role: 'model', content } };
}

async function collect(
  gen: AsyncGenerator<A2AEvent, void, undefined>
): Promise<A2AEvent[]> {
  const out: A2AEvent[] = [];
  for await (const e of gen) out.push(e);
  return out;
}

describe('GenkitA2ARequestHandler.sendMessageStream', () => {
  it('emits task, working, artifact, and completed events for a text turn', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeAgent({
        chunks: [
          modelChunk([{ text: 'Hello' }]),
          modelChunk([{ text: ' world' }]),
        ],
        response: {
          finishReason: 'stop',
          message: { role: 'model', content: [{ text: 'Hello world' }] },
        },
      }),
      url: 'http://localhost:3000',
    });

    const events = await collect(
      handler.sendMessageStream(userMessage([{ kind: 'text', text: 'hi' }]))
    );

    assert.strictEqual(events[0].kind, 'task');
    assert.strictEqual(
      (events[1] as TaskStatusUpdateEvent).status.state,
      'working'
    );

    const artifacts = events.filter(
      (e) => e.kind === 'artifact-update'
    ) as TaskArtifactUpdateEvent[];
    assert.strictEqual(artifacts.length, 2);
    assert.strictEqual(artifacts[0].append, false);
    assert.strictEqual(artifacts[1].append, true);
    // Both chunks stream against the same artifact id.
    assert.strictEqual(
      artifacts[0].artifact.artifactId,
      artifacts[1].artifact.artifactId
    );

    const final = events[events.length - 1] as TaskStatusUpdateEvent;
    assert.strictEqual(final.kind, 'status-update');
    assert.strictEqual(final.status.state, 'completed');
    assert.strictEqual(final.final, true);
    assert.deepStrictEqual(final.status.message?.parts, [
      { kind: 'text', text: 'Hello world' },
    ]);
  });

  it('drops tool-call parts from streamed artifacts (only user-facing content)', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeAgent({
        chunks: [
          // Tool mechanics: must NOT be streamed as artifact content.
          modelChunk([
            { toolRequest: { ref: 'r1', name: 'getWeather', input: {} } },
          ]),
          modelChunk([
            {
              toolResponse: {
                ref: 'r1',
                name: 'getWeather',
                output: { weather: 'sunny' },
              },
            },
          ]),
          // User-facing text: SHOULD be streamed.
          modelChunk([{ text: 'It is sunny.' }]),
        ],
        response: {
          finishReason: 'stop',
          message: { role: 'model', content: [{ text: 'It is sunny.' }] },
        },
      }),
      url: 'http://localhost:3000',
    });

    const events = await collect(
      handler.sendMessageStream(userMessage([{ kind: 'text', text: 'weather?' }]))
    );

    const artifacts = events.filter(
      (e) => e.kind === 'artifact-update'
    ) as TaskArtifactUpdateEvent[];
    // Only the single text chunk produces an artifact update.
    assert.strictEqual(artifacts.length, 1);
    assert.deepStrictEqual(artifacts[0].artifact.parts, [
      { kind: 'text', text: 'It is sunny.' },
    ]);
    // The accumulated task artifact likewise contains no tool data parts.
    const task = await handler.getTask({ id: (events[0] as Task).id });
    const allParts = (task.artifacts ?? []).flatMap((a) => a.parts);
    assert.ok(allParts.every((p) => p.kind === 'text'));
  });

  it('maps an interrupted turn to input-required with the interrupt parts', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeAgent({
        chunks: [],
        response: {
          finishReason: 'interrupted',
          message: {
            role: 'model',
            content: [
              {
                toolRequest: { ref: 'r1', name: 'approve', input: {} },
                metadata: { interrupt: { reason: 'confirm' } },
              },
            ],
          },
        },
      }),
      url: 'http://localhost:3000',
    });

    const events = await collect(
      handler.sendMessageStream(userMessage([{ kind: 'text', text: 'go' }]))
    );
    const final = events[events.length - 1] as TaskStatusUpdateEvent;
    assert.strictEqual(final.status.state, 'input-required');
    assert.strictEqual(final.final, true);
    const part = final.status.message!.parts[0];
    assert.strictEqual(part.kind, 'data');
  });

  it('maps a failed turn to a failed status', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeAgent({
        chunks: [],
        response: { finishReason: 'failed', finishMessage: 'boom' },
      }),
      url: 'http://localhost:3000',
    });
    const events = await collect(
      handler.sendMessageStream(userMessage([{ kind: 'text', text: 'go' }]))
    );
    const final = events[events.length - 1] as TaskStatusUpdateEvent;
    assert.strictEqual(final.status.state, 'failed');
    assert.deepStrictEqual(final.status.message?.parts, [
      { kind: 'text', text: 'boom' },
    ]);
  });

  it('uses the A2A contextId as the Genkit sessionId', async () => {
    let captured: { sessionId?: string } | undefined;
    const handler = new GenkitA2ARequestHandler({
      agent: fakeAgent({
        chunks: [],
        response: {
          finishReason: 'stop',
          message: { role: 'model', content: [{ text: 'ok' }] },
        },
        capture: (_input, init) => {
          captured = init;
        },
      }),
      url: 'http://localhost:3000',
    });
    await collect(
      handler.sendMessageStream(
        userMessage([{ kind: 'text', text: 'hi' }], { contextId: 'ctx-123' })
      )
    );
    assert.strictEqual(captured?.sessionId, 'ctx-123');
  });
});

describe('GenkitA2ARequestHandler.getTask', () => {
  it('returns the accumulated task after a turn', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeAgent({
        chunks: [modelChunk([{ text: 'hi' }])],
        response: {
          finishReason: 'stop',
          message: { role: 'model', content: [{ text: 'hi' }] },
        },
      }),
      url: 'http://localhost:3000',
    });
    const events = await collect(
      handler.sendMessageStream(
        userMessage([{ kind: 'text', text: 'hi' }], { taskId: 't-1' })
      )
    );
    const task = events[0] as Task;
    const fetched = await handler.getTask({ id: task.id });
    assert.strictEqual(fetched.id, task.id);
    assert.strictEqual(fetched.status.state, 'completed');
    assert.strictEqual(fetched.artifacts?.length, 1);
  });
});
