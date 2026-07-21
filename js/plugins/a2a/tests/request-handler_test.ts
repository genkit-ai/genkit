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
  MessageData,
  Part,
  SessionSnapshot,
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
 * Builds a minimal fake **client-managed** Genkit agent (no store, no
 * `getSnapshot`) whose single turn streams the given `chunks` and resolves with
 * the given finish state.
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

/** A single programmed turn for {@link fakeServerAgent}. */
interface FakeTurn {
  /** The snapshot id this turn is persisted under (its `turnStart` id). */
  snapshotId: string;
  finishReason: AgentResponse['finishReason'];
  /** The final model message (also appended to the snapshot's history). */
  message?: MessageData;
  /** User-facing model chunks streamed before completion. */
  chunks?: Part[][];
}

/**
 * Builds a fake **server-managed** Genkit agent (declares
 * `stateManagement: 'server'` and implements `getSnapshot`). Each `sendStream`
 * consumes the next programmed {@link FakeTurn}: it emits a `turnStart` chunk
 * carrying the reserved snapshot id, streams any model chunks, records a
 * snapshot readable via `getSnapshot`, and resolves the response with the
 * snapshot id + finish reason. Accumulates the user message and the turn's
 * model message into the snapshot history.
 */
function fakeServerAgent(opts: {
  turns: FakeTurn[];
  capture?: (
    input: AgentInput,
    init?: { sessionId?: string; snapshotId?: string }
  ) => void;
}): GenkitAgent {
  const snapshots = new Map<string, SessionSnapshot>();
  let history: MessageData[] = [];
  let turnIdx = 0;

  return {
    __action: {
      name: 'fakeServerAgent',
      description: 'A fake server-managed agent.',
      metadata: { agent: { stateManagement: 'server' } },
    },
    async getSnapshot(
      lookup: string | { snapshotId: string } | { sessionId: string }
    ): Promise<SessionSnapshot | undefined> {
      if (typeof lookup === 'string') return snapshots.get(lookup);
      if ('snapshotId' in lookup) return snapshots.get(lookup.snapshotId);
      // sessionId lookup: return the latest snapshot for the session.
      let latest: SessionSnapshot | undefined;
      for (const s of snapshots.values()) {
        if (s.sessionId === lookup.sessionId) latest = s;
      }
      return latest;
    },
    chat(init?: { sessionId?: string; snapshotId?: string }) {
      const sessionId = init?.sessionId ?? 'session-1';
      return {
        sendStream(input: AgentInput) {
          opts.capture?.(input, init);
          const turn = opts.turns[turnIdx++];
          if (!turn) throw new Error('fakeServerAgent: no turn programmed.');

          // Accumulate history: user message (if any) then model message.
          if (input.message) history = [...history, input.message];
          if (turn.message) history = [...history, turn.message];

          snapshots.set(turn.snapshotId, {
            snapshotId: turn.snapshotId,
            sessionId,
            status: 'completed',
            finishReason: turn.finishReason,
            createdAt: new Date().toISOString(),
            state: { sessionId, messages: [...history] },
          });

          const chunks = turn.chunks ?? [];
          const stream = (async function* (): AsyncIterable<AgentChunk> {
            yield {
              snapshotId: turn.snapshotId,
              raw: { turnStart: { snapshotId: turn.snapshotId } },
            } as unknown as AgentChunk;
            for (const content of chunks) {
              yield {
                raw: { modelChunk: { role: 'model', content } },
              } as AgentChunk;
            }
          })();

          return {
            stream,
            response: Promise.resolve({
              snapshotId: turn.snapshotId,
              finishReason: turn.finishReason,
              message: turn.message,
            } as AgentResponse),
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

describe('GenkitA2ARequestHandler.sendMessageStream (client-managed)', () => {
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
      handler.sendMessageStream(
        userMessage([{ kind: 'text', text: 'weather?' }])
      )
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

  it('returns the accumulated task after a turn via the in-memory cache', async () => {
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
      handler.sendMessageStream(userMessage([{ kind: 'text', text: 'hi' }]))
    );
    const task = events[0] as Task;
    const fetched = await handler.getTask({ id: task.id });
    assert.strictEqual(fetched.id, task.id);
    assert.strictEqual(fetched.status.state, 'completed');
    assert.strictEqual(fetched.artifacts?.length, 1);
  });
});

describe('GenkitA2ARequestHandler.sendMessageStream (server-managed)', () => {
  it('uses the A2A contextId as the Genkit sessionId', async () => {
    let captured: { sessionId?: string; snapshotId?: string } | undefined;
    const handler = new GenkitA2ARequestHandler({
      agent: fakeServerAgent({
        turns: [
          {
            snapshotId: 'snap-1',
            finishReason: 'stop',
            message: { role: 'model', content: [{ text: 'ok' }] },
          },
        ],
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

  it('uses the turnStart snapshotId as the A2A taskId', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeServerAgent({
        turns: [
          {
            snapshotId: 'snap-1',
            finishReason: 'stop',
            message: { role: 'model', content: [{ text: 'ok' }] },
          },
        ],
      }),
      url: 'http://localhost:3000',
    });
    const events = await collect(
      handler.sendMessageStream(userMessage([{ kind: 'text', text: 'hi' }]))
    );
    const task = events[0] as Task;
    assert.strictEqual(task.id, 'snap-1');
  });

  it('reads getTask straight from the agent snapshot (no task-store entry)', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeServerAgent({
        turns: [
          {
            snapshotId: 'snap-1',
            finishReason: 'stop',
            message: { role: 'model', content: [{ text: 'the answer' }] },
          },
        ],
      }),
      url: 'http://localhost:3000',
    });
    await collect(
      handler.sendMessageStream(
        userMessage([{ kind: 'text', text: 'q' }], { contextId: 'ctx-1' })
      )
    );

    const fetched = await handler.getTask({ id: 'snap-1' });
    assert.strictEqual(fetched.id, 'snap-1');
    assert.strictEqual(fetched.contextId, 'ctx-1');
    assert.strictEqual(fetched.status.state, 'completed');
    assert.deepStrictEqual(fetched.status.message?.parts, [
      { kind: 'text', text: 'the answer' },
    ]);
  });

  it('resumes an interrupted task and advances the task pointer', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeServerAgent({
        turns: [
          {
            snapshotId: 'snap-1',
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
          {
            snapshotId: 'snap-2',
            finishReason: 'stop',
            message: { role: 'model', content: [{ text: 'done' }] },
          },
        ],
      }),
      url: 'http://localhost:3000',
    });

    // Phase 1: fresh turn interrupts -> input-required, taskId = snap-1.
    const first = await collect(
      handler.sendMessageStream(
        userMessage([{ kind: 'text', text: 'go' }], { contextId: 'ctx-1' })
      )
    );
    const task = first[0] as Task;
    assert.strictEqual(task.id, 'snap-1');
    assert.strictEqual(
      (first[first.length - 1] as TaskStatusUpdateEvent).status.state,
      'input-required'
    );

    // Phase 2: resume by sending the tool response against the same taskId.
    const second = await collect(
      handler.sendMessageStream(
        userMessage(
          [
            {
              kind: 'data',
              data: { ref: 'r1', name: 'approve', output: { ok: true } },
              metadata: { 'genkit:type': 'toolResponse' },
            },
          ],
          { contextId: 'ctx-1', taskId: 'snap-1' }
        )
      )
    );
    const resumedFinal = second[second.length - 1] as TaskStatusUpdateEvent;
    assert.strictEqual(resumedFinal.status.state, 'completed');

    // The task pointer advanced: getTask(snap-1) now resolves to snap-2.
    const fetched = await handler.getTask({ id: 'snap-1' });
    assert.strictEqual(fetched.status.state, 'completed');
    assert.deepStrictEqual(fetched.status.message?.parts, [
      { kind: 'text', text: 'done' },
    ]);
  });

  it('fails loudly when continuing an unknown task', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeServerAgent({ turns: [] }),
      url: 'http://localhost:3000',
    });
    await assert.rejects(
      collect(
        handler.sendMessageStream(
          userMessage([{ kind: 'text', text: 'hi' }], {
            contextId: 'ctx-1',
            taskId: 'does-not-exist',
          })
        )
      ),
      /snapshot does-not-exist does not exist/
    );
  });

  it('fails loudly when getTask targets an unknown task', async () => {
    const handler = new GenkitA2ARequestHandler({
      agent: fakeServerAgent({ turns: [] }),
      url: 'http://localhost:3000',
    });
    await assert.rejects(
      handler.getTask({ id: 'nope' }),
      /snapshot nope does not exist/
    );
  });
});
