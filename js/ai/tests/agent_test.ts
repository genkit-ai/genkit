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

import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import { enableTelemetry } from '@genkit-ai/core/tracing';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { describe, it } from 'node:test';

import { z } from '@genkit-ai/core';
import { TestSpanExporter } from '../../core/tests/utils.js';
import { AgentError } from '../src/agent-core.js';
import {
  AgentStreamChunk,
  SessionRunner,
  defineAgent,
  defineCustomAgent,
  definePromptAgent,
} from '../src/agent.js';
import { definePrompt } from '../src/prompt.js';
import { InMemorySessionStore } from '../src/session-stores.js';
import {
  Session,
  reserveSnapshotId,
  type SessionSnapshot,
} from '../src/session.js';
import { ToolInterruptError, defineTool, interrupt } from '../src/tool.js';
import {
  defineEchoModel,
  defineProgrammableModel,
  type ProgrammableModel,
} from './helpers.js';

initNodeFeatures();

const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessors: [new SimpleSpanProcessor(spanExporter)],
});

/**
 * Returns a Promise that resolves once the given snapshotId reaches targetStatus
 * in the store. Rejects after timeoutMs if the status is never reached.
 */
function waitForSnapshotStatus<S>(
  store: InMemorySessionStore<S>,
  snapshotId: string,
  targetStatus: NonNullable<SessionSnapshot<S>['status']>,
  timeoutMs = 5000
): Promise<SessionSnapshot<S>> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () =>
        reject(
          new Error(
            `Timed out waiting for snapshot ${snapshotId} to reach status "${targetStatus}"`
          )
        ),
      timeoutMs
    );

    const unsubscribeFn = store.onSnapshotStateChange(snapshotId, (snap) => {
      if (snap.status === targetStatus) {
        clearTimeout(timer);
        if (typeof unsubscribeFn === 'function') unsubscribeFn();
        resolve(snap);
      }
    });

    // Check in case already at the target status.
    store.getSnapshot({ snapshotId: snapshotId }).then((snap) => {
      if (snap?.status === targetStatus) {
        clearTimeout(timer);
        if (typeof unsubscribeFn === 'function') unsubscribeFn();
        resolve(snap);
      }
    });
  });
}

describe('Agent', () => {
  describe('Session', () => {
    it('should maintain custom state', () => {
      const session = new Session<{ foo: string }>({ custom: { foo: 'bar' } });
      assert.strictEqual(session.getCustom()?.foo, 'bar');

      session.updateCustom((c) => ({ ...c!, foo: 'baz' }));
      assert.strictEqual(session.getCustom()?.foo, 'baz');
    });

    it('should add and set messages', () => {
      const session = new Session({});
      session.addMessages([{ role: 'user', content: [{ text: 'hi' }] }]);
      assert.strictEqual(session.getMessages().length, 1);
      assert.strictEqual(session.getMessages()[0].role, 'user');

      session.setMessages([{ role: 'model', content: [{ text: 'hello' }] }]);
      assert.strictEqual(session.getMessages().length, 1);
      assert.strictEqual(session.getMessages()[0].role, 'model');
    });

    it('should add and deduplicate artifacts', () => {
      const session = new Session({});
      session.addArtifacts([{ name: 'art1', parts: [{ text: 'content1' }] }]);
      assert.strictEqual(session.getArtifacts().length, 1);

      // Add with same name should replace
      session.addArtifacts([{ name: 'art1', parts: [{ text: 'content2' }] }]);
      assert.strictEqual(session.getArtifacts().length, 1);
      assert.deepStrictEqual(session.getArtifacts()[0].parts, [
        { text: 'content2' },
      ]);

      // Add with different name should append
      session.addArtifacts([{ name: 'art2', parts: [{ text: 'content3' }] }]);
      assert.strictEqual(session.getArtifacts().length, 2);
    });

    it('should process all artifacts in a batch without dropping any', () => {
      const session = new Session({});
      session.addArtifacts([{ name: 'art1', parts: [{ text: 'v1' }] }]);

      // Replace art1 and add art2 and art3 in the same batch.
      session.addArtifacts([
        { name: 'art1', parts: [{ text: 'v2' }] },
        { name: 'art2', parts: [{ text: 'new' }] },
        { name: 'art3', parts: [{ text: 'another' }] },
      ]);

      const arts = session.getArtifacts();
      assert.strictEqual(arts.length, 3);
      assert.strictEqual(
        arts.find((a) => a.name === 'art1')?.parts[0].text,
        'v2'
      );
      assert.strictEqual(
        arts.find((a) => a.name === 'art2')?.parts[0].text,
        'new'
      );
      assert.strictEqual(
        arts.find((a) => a.name === 'art3')?.parts[0].text,
        'another'
      );
    });

    it('should emit artifactAdded for new and artifactUpdated for replaced', () => {
      const session = new Session({});
      const added: string[] = [];
      const updated: string[] = [];
      session.on('artifactAdded', (a: { name?: string }) =>
        added.push(a.name ?? '')
      );
      session.on('artifactUpdated', (a: { name?: string }) =>
        updated.push(a.name ?? '')
      );

      session.addArtifacts([{ name: 'art1', parts: [] }]);
      session.addArtifacts([
        { name: 'art1', parts: [] }, // replace
        { name: 'art2', parts: [] }, // new
      ]);

      assert.deepStrictEqual(added, ['art1', 'art2']);
      assert.deepStrictEqual(updated, ['art1']);
    });

    it('should increment version on mutation', () => {
      const session = new Session({});
      const v0 = session.getVersion();

      session.addMessages([{ role: 'user', content: [{ text: 'hi' }] }]);
      const v1 = session.getVersion();
      assert.ok(v1 > v0);

      session.updateCustom((c) => c);
      const v2 = session.getVersion();
      assert.ok(v2 > v1);

      session.addArtifacts([{ name: 'a', parts: [] }]);
      const v3 = session.getVersion();
      assert.ok(v3 > v2);
    });
  });

  describe('SessionRunner', () => {
    it('should loop over inputs and call handler', async () => {
      const session = new Session({});
      const inputs = [
        { message: { role: 'user' as const, content: [{ text: 'hi' }] } },
        { message: { role: 'user' as const, content: [{ text: 'bye' }] } },
      ];

      async function* inputGen() {
        for (const input of inputs) {
          yield input;
        }
      }

      const runner = new SessionRunner(session, inputGen());
      let turns = 0;
      const seenInputs: any[] = [];

      await runner.run(async (input) => {
        turns++;
        seenInputs.push(input);
      });

      assert.strictEqual(turns, 2);
      assert.deepStrictEqual(seenInputs, inputs);
      assert.strictEqual(session.getMessages().length, 2);
    });

    it('should trigger snapshots if store is present', async () => {
      const store = new InMemorySessionStore();
      const session = new Session({});
      const inputs = [
        { message: { role: 'user' as const, content: [{ text: 'hi' }] } },
      ];

      async function* inputGen() {
        for (const input of inputs) {
          yield input;
        }
      }

      let turnEnded = false;
      let turnSnapshotId: string | undefined;

      const runner = new SessionRunner(session, inputGen(), {
        store,
        onEndTurn: (snapshotId) => {
          turnEnded = true;
          turnSnapshotId = snapshotId;
        },
      });

      await runner.run(async () => {});

      assert.ok(turnEnded);
      assert.ok(turnSnapshotId);

      const saved = await store.getSnapshot({ snapshotId: turnSnapshotId! });
      assert.ok(saved);
      assert.strictEqual(saved?.snapshotId, turnSnapshotId);
    });

    it('reserves the turn snapshotId at turn start and persists under it', async () => {
      const store = new InMemorySessionStore();
      const session = new Session({});

      async function* inputGen() {
        yield {
          message: { role: 'user' as const, content: [{ text: 'hi' }] },
        };
      }

      let ctxSnapshotId: string | undefined;
      let ctxParentSnapshotId: string | undefined;
      let ctxTurnIndex: number | undefined;
      let endTurnSnapshotId: string | undefined;

      const runner = new SessionRunner(session, inputGen(), {
        store,
        onEndTurn: (snapshotId) => {
          endTurnSnapshotId = snapshotId;
        },
      });

      await runner.run(async (_input, ctx) => {
        ctxSnapshotId = ctx.snapshotId;
        ctxParentSnapshotId = ctx.parentSnapshotId;
        ctxTurnIndex = ctx.turnIndex;
      });

      // The id is reserved at turn start and made available to the handler.
      assert.ok(ctxSnapshotId, 'handler should receive a reserved snapshotId');
      // First turn of a fresh session has no parent.
      assert.strictEqual(ctxParentSnapshotId, undefined);
      assert.strictEqual(ctxTurnIndex, 0);

      // The snapshot persisted at turn end reuses the reserved id.
      assert.strictEqual(endTurnSnapshotId, ctxSnapshotId);
      const saved = await store.getSnapshot({ snapshotId: ctxSnapshotId! });
      assert.ok(saved);
      assert.strictEqual(saved?.snapshotId, ctxSnapshotId);
    });

    it('passes the prior snapshotId as parentSnapshotId on subsequent turns', async () => {
      const store = new InMemorySessionStore();
      const session = new Session({});

      async function* inputGen() {
        yield {
          message: { role: 'user' as const, content: [{ text: 'one' }] },
        };
        yield {
          message: { role: 'user' as const, content: [{ text: 'two' }] },
        };
      }

      const snapshotIds: string[] = [];
      const parentIds: Array<string | undefined> = [];

      const runner = new SessionRunner(session, inputGen(), { store });

      await runner.run(async (_input, ctx) => {
        snapshotIds.push(ctx.snapshotId);
        parentIds.push(ctx.parentSnapshotId);
      });

      assert.strictEqual(snapshotIds.length, 2);
      // First turn: no parent. Second turn: parent is the first turn's snapshot.
      assert.strictEqual(parentIds[0], undefined);
      assert.strictEqual(parentIds[1], snapshotIds[0]);
    });

    it('does not reserve a snapshotId when no store is configured', async () => {
      const session = new Session({});

      async function* inputGen() {
        yield {
          message: { role: 'user' as const, content: [{ text: 'hi' }] },
        };
      }

      let ctxSnapshotId: string | undefined = 'sentinel';
      const runner = new SessionRunner(session, inputGen());
      await runner.run(async (_input, ctx) => {
        ctxSnapshotId = ctx.snapshotId;
      });

      // Without a store there is nothing to reserve, so snapshotId is undefined.
      assert.strictEqual(ctxSnapshotId, undefined);
    });
  });

  describe('reserveSnapshotId', () => {
    const UUID =
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

    it('mints a plain UUID snapshotId', () => {
      const id = reserveSnapshotId();
      assert.match(id, UUID, `id should be a UUID: ${id}`);
    });

    it('produces unique ids on successive calls', () => {
      const a = reserveSnapshotId();
      const b = reserveSnapshotId();
      assert.notStrictEqual(a, b);
    });
  });

  describe('defineCustomAgent', () => {
    it('should set client stateManagement and abortable=false when no store is provided', () => {
      const registry = new Registry();
      const agent = defineCustomAgent(
        registry,
        { name: 'noStoreMetadataTest' },
        async () => ({ artifacts: [] })
      );
      assert.strictEqual(
        agent.__action.metadata?.agent?.stateManagement,
        'client'
      );
      assert.strictEqual(agent.__action.metadata?.agent?.abortable, false);
    });

    it('should set server stateManagement and abortable=true when store with onSnapshotStateChange is provided', () => {
      const registry = new Registry();
      const store = new InMemorySessionStore();
      const agent = defineCustomAgent(
        registry,
        { name: 'fullStoreMetadataTest', store },
        async () => ({ artifacts: [] })
      );
      assert.strictEqual(
        agent.__action.metadata?.agent?.stateManagement,
        'server'
      );
      assert.strictEqual(agent.__action.metadata?.agent?.abortable, true);
    });

    it('should reject init.state for server-managed agents (store is set)', async () => {
      const registry = new Registry();
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<{ foo: string }>(
        registry,
        { name: 'rejectInitStateTest', store },
        async (sess) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'done' }] },
          };
        }
      );

      // Pass init.state - this is API misuse for a server-managed agent and
      // must throw a FAILED_PRECONDITION error (rather than resolving with a
      // graceful finishReason 'failed' output).
      const session = flow.streamBidi({
        state: {
          custom: { foo: 'should-be-rejected' },
          messages: [{ role: 'user', content: [{ text: 'stale history' }] }],
          artifacts: [],
        },
      });
      session.send({
        message: { role: 'user', content: [{ text: 'hello' }] },
      });
      session.close();

      let thrown: any;
      try {
        for await (const _ of session.stream) {
        }
        await session.output;
      } catch (e: any) {
        thrown = e;
      }

      assert.ok(thrown, 'Expected the turn to throw an error');
      assert.strictEqual(thrown.status, 'FAILED_PRECONDITION');
      assert.ok(
        thrown.message.includes("Cannot send 'state' to agent"),
        `Expected FAILED_PRECONDITION message, got: ${thrown.message}`
      );
    });

    it('should use init.state for client-managed agents (no store)', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent<{ foo: string }>(
        registry,
        { name: 'useInitStateTest' },
        async (sess) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'done' }] },
          };
        }
      );

      // Pass init.state - it should be used because no store is set
      const session = flow.streamBidi({
        state: {
          custom: { foo: 'seeded' },
          messages: [{ role: 'user', content: [{ text: 'prior msg' }] }],
          artifacts: [],
        },
      });
      session.send({
        message: { role: 'user', content: [{ text: 'hello' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      // State should include the seeded state plus the new message
      assert.ok(output.state);
      assert.strictEqual((output.state!.custom as any).foo, 'seeded');
      // Messages: 1 from init.state + 1 from input
      assert.strictEqual(output.state!.messages!.length, 2);
      assert.strictEqual(
        output.state!.messages![0].content[0].text,
        'prior msg'
      );
      assert.strictEqual(output.state!.messages![1].content[0].text, 'hello');
    });

    it('should set server stateManagement and abortable=false when store lacks onSnapshotStateChange', () => {
      const registry = new Registry();
      const store: any = {
        getSnapshot: async () => undefined,
        saveSnapshot: async () => {},
        // no onSnapshotStateChange
      };
      const agent = defineCustomAgent(
        registry,
        { name: 'noAbortStoreMetadataTest', store },
        async () => ({ artifacts: [] })
      );
      assert.strictEqual(
        agent.__action.metadata?.agent?.stateManagement,
        'server'
      );
      assert.strictEqual(agent.__action.metadata?.agent?.abortable, false);
    });

    it('should register and execute agent', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        { name: 'testFlow' },
        async (sess, { sendChunk }) => {
          let receivedInput = false;
          await sess.run(async (input) => {
            receivedInput = true;
            assert.strictEqual(input.message?.role, 'user');
          });
          assert.ok(receivedInput);
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});

      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const output = await session.output;
      assert.strictEqual(output.message?.role, 'model');
      assert.strictEqual(output.message?.content[0].text, 'done');
    });

    it('should automatically stream artifacts added via Session.addArtifacts()', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        { name: 'testEventFlow' },
        async (sess, { sendChunk }) => {
          await sess.run(async (input) => {
            sess.session.addArtifacts([
              { name: 'testArt', parts: [{ text: 'testPart' }] },
            ]);
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const artChunks = chunks.filter((c) => !!c.artifact);
      assert.strictEqual(artChunks.length, 1);
      assert.strictEqual(artChunks[0].artifact?.name, 'testArt');
    });

    it('should stream artifactUpdated chunks when an artifact is replaced', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        { name: 'testArtifactUpdateFlow' },
        async (sess) => {
          await sess.run(async () => {
            sess.session.addArtifacts([{ name: 'a', parts: [{ text: 'v1' }] }]);
            sess.session.addArtifacts([{ name: 'a', parts: [{ text: 'v2' }] }]);
          });
          return {};
        }
      );

      const session = flow.streamBidi({});
      session.send({ message: { role: 'user', content: [{ text: 'go' }] } });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const artChunks = chunks.filter((c) => !!c.artifact);
      assert.strictEqual(artChunks.length, 2);
      assert.strictEqual(artChunks[0].artifact?.parts[0].text, 'v1');
      assert.strictEqual(artChunks[1].artifact?.parts[0].text, 'v2');
    });

    it('records the snapshotId and state on the turn span (server-managed)', async () => {
      spanExporter.exportedSpans = [];
      const store = new InMemorySessionStore<{ count: number }>();

      const flow = defineCustomAgent<{ count: number }>(
        new Registry(),
        { name: 'turnSpanServerTest', store },
        async (sess) => {
          await sess.run(async () => {
            sess.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'done' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;
      assert.ok(output.snapshotId);

      const turnSpan = spanExporter.exportedSpans.find(
        (s) => s.displayName === 'runTurn-1'
      );
      assert.ok(turnSpan, 'expected a runTurn-1 span to be exported');

      // The turn span carries the snapshotId this turn persisted under.
      assert.strictEqual(
        turnSpan.attributes['genkit:metadata:agent:snapshotId'],
        output.snapshotId
      );

      // The turn span's output is the session state this turn produced.
      const out = JSON.parse(turnSpan.attributes['genkit:output']);
      assert.deepStrictEqual(out.state.custom, { count: 1 });
    });

    it('records state on the turn span for a client-managed agent', async () => {
      spanExporter.exportedSpans = [];
      const registry = new Registry();

      const flow = defineCustomAgent<{ count: number }>(
        registry,
        { name: 'turnSpanClientTest' },
        async (sess) => {
          await sess.run(async () => {
            sess.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'done' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      await session.output;

      const turnSpan = spanExporter.exportedSpans.find(
        (s) => s.displayName === 'runTurn-1'
      );
      assert.ok(turnSpan, 'expected a runTurn-1 span to be exported');

      // The turn span's output carries the session state this turn produced,
      // for client-managed agents too.
      const out = JSON.parse(turnSpan.attributes['genkit:output']);
      assert.deepStrictEqual(out.state.custom, { count: 1 });
    });
  });

  describe('sessionId', () => {
    it('should generate sessionId for a fresh client-managed agent', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        { name: 'sessionIdFreshClient' },
        async (sess) => {
          await sess.run(async () => {});
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      // Client-managed agents return state, which should contain a sessionId
      assert.ok(output.state, 'output.state should be present');
      assert.ok(
        output.state!.sessionId,
        'sessionId should be generated for a fresh session'
      );
      // Should be a valid UUID
      assert.match(
        output.state!.sessionId!,
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
      );
    });

    it('should preserve sessionId across turns for client-managed agents', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        { name: 'sessionIdPreserveClient' },
        async (sess) => {
          await sess.run(async () => {});
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      // Turn 1: fresh session
      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session1.close();
      for await (const _ of session1.stream) {
      }
      const output1 = await session1.output;
      const firstSessionId = output1.state!.sessionId!;
      assert.ok(firstSessionId, 'First turn should have sessionId');

      // Turn 2: pass state back (client-managed)
      const session2 = flow.streamBidi({ state: output1.state });
      session2.send({
        message: { role: 'user' as const, content: [{ text: 'bye' }] },
      });
      session2.close();
      for await (const _ of session2.stream) {
      }
      const output2 = await session2.output;

      assert.strictEqual(
        output2.state!.sessionId,
        firstSessionId,
        'sessionId should be preserved across turns'
      );
    });

    it('should generate sessionId for a fresh server-managed agent and persist in snapshot', async () => {
      const registry = new Registry();
      const store = new InMemorySessionStore();

      const flow = defineCustomAgent(
        registry,
        { name: 'sessionIdServerManaged', store },
        async (sess) => {
          await sess.run(async () => {});
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      assert.ok(output.snapshotId, 'should have snapshotId');

      // Read snapshot and verify sessionId is persisted in the state
      const snapshot = await store.getSnapshot({
        snapshotId: output.snapshotId!,
      });
      assert.ok(snapshot, 'snapshot should exist');
      assert.ok(
        snapshot!.state.sessionId,
        'sessionId should be persisted in snapshot state'
      );
      assert.match(
        snapshot!.state.sessionId!,
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
      );
    });

    it('should preserve sessionId from snapshot for server-managed agents', async () => {
      const registry = new Registry();
      const store = new InMemorySessionStore();

      const flow = defineCustomAgent(
        registry,
        { name: 'sessionIdServerPreserve', store },
        async (sess) => {
          await sess.run(async () => {});
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      // Turn 1
      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session1.close();
      for await (const _ of session1.stream) {
      }
      const output1 = await session1.output;
      const firstSnapshotId = output1.snapshotId!;

      const snapshot1 = await store.getSnapshot({
        snapshotId: firstSnapshotId,
      });
      const firstSessionId = snapshot1!.state.sessionId!;
      assert.ok(firstSessionId, 'First turn should have sessionId');

      // Turn 2: resume from snapshot
      const session2 = flow.streamBidi({ snapshotId: firstSnapshotId });
      session2.send({
        message: { role: 'user' as const, content: [{ text: 'bye' }] },
      });
      session2.close();
      for await (const _ of session2.stream) {
      }
      const output2 = await session2.output;

      const snapshot2 = await store.getSnapshot({
        snapshotId: output2.snapshotId!,
      });
      assert.strictEqual(
        snapshot2!.state.sessionId,
        firstSessionId,
        'sessionId should be preserved across turns via snapshot'
      );
    });

    it('resumes a snapshot when both snapshotId and a matching sessionId are provided', async () => {
      const registry = new Registry();
      const store = new InMemorySessionStore();

      const flow = defineCustomAgent(
        registry,
        { name: 'snapshotAndMatchingSession', store },
        async (sess) => {
          await sess.run(async () => {});
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      // Turn 1
      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session1.close();
      for await (const _ of session1.stream) {
      }
      const output1 = await session1.output;
      const firstSnapshotId = output1.snapshotId!;
      const sessionId = output1.sessionId!;
      assert.ok(sessionId, 'First turn should have sessionId');

      // Turn 2: resume passing BOTH the snapshotId and its owning sessionId.
      const session2 = flow.streamBidi({
        snapshotId: firstSnapshotId,
        sessionId,
      });
      session2.send({
        message: { role: 'user' as const, content: [{ text: 'bye' }] },
      });
      session2.close();
      for await (const _ of session2.stream) {
      }
      const output2 = await session2.output;

      assert.notStrictEqual(
        output2.finishReason,
        'failed',
        `Expected a successful resume, got error: ${output2.error?.message}`
      );
      assert.strictEqual(
        output2.sessionId,
        sessionId,
        'sessionId should be preserved when resuming by snapshotId + sessionId'
      );
    });

    it('rejects when snapshotId belongs to a different session than the provided sessionId', async () => {
      const registry = new Registry();
      const store = new InMemorySessionStore();

      const flow = defineCustomAgent(
        registry,
        { name: 'snapshotSessionMismatch', store },
        async (sess) => {
          await sess.run(async () => {});
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      // Create a snapshot that belongs to session A.
      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session1.close();
      for await (const _ of session1.stream) {
      }
      const output1 = await session1.output;
      const snapshotId = output1.snapshotId!;

      // Resume that snapshot but claim it belongs to a different session. This
      // is API misuse and must throw (rather than resolving gracefully).
      const session2 = flow.streamBidi({
        snapshotId,
        sessionId: 'a-different-session-id',
      });
      session2.send({
        message: { role: 'user' as const, content: [{ text: 'bye' }] },
      });
      session2.close();

      let thrown: any;
      try {
        for await (const _ of session2.stream) {
        }
        await session2.output;
      } catch (e: any) {
        thrown = e;
      }

      assert.ok(thrown, 'Expected the turn to throw an error');
      assert.strictEqual(thrown.status, 'INVALID_ARGUMENT');
      assert.ok(
        thrown.message.includes('does not belong to session'),
        `Expected an ownership-mismatch error, got: ${thrown.message}`
      );
    });
  });

  describe('definePromptAgent', () => {
    it('should register and execute agent from prompt', async () => {
      const registry = new Registry();
      defineEchoModel(registry);
      definePrompt(registry, {
        name: 'agent',
        model: 'echoModel',
        config: { temperature: 1 },
        system: 'hello from template',
      });

      const flow = definePromptAgent(registry, {
        promptName: 'agent',
      });

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const output = await session.output;
      assert.strictEqual(output.message?.role, 'model');
    });

    it('should pass promptInput to the prompt template', async () => {
      const registry = new Registry();
      defineEchoModel(registry);
      definePrompt(registry, {
        name: 'personaAgentPrompt',
        model: 'echoModel',
        input: { schema: z.object({ persona: z.string() }) },
        system: 'You are a {{persona}}.',
      });

      const flow = definePromptAgent<unknown, z.ZodTypeAny>(registry, {
        promptName: 'personaAgentPrompt',
        promptInput: { persona: 'pirate' },
      });

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
        // drain
      }

      const output = await session.output;
      // The echo model echoes the rendered system message, which interpolated
      // the supplied promptInput.
      assert.ok(
        output.message?.content[0].text?.includes('You are a pirate.'),
        `expected rendered prompt to include the promptInput, got: ${output.message?.content[0].text}`
      );
    });

    it('should let agents in separate registries reuse one prompt definition with different promptInput', async () => {
      // The same prompt definition (config) reused to build two differently
      // customized agents. Each lives in its own registry because an agent is
      // keyed by its promptName.
      const promptConfig = {
        name: 'sharedPersonaPrompt',
        model: 'echoModel',
        input: { schema: z.object({ persona: z.string() }) },
        system: 'You are a {{persona}}.',
      };

      const buildAgent = (persona: string) => {
        const registry = new Registry();
        defineEchoModel(registry);
        definePrompt(registry, promptConfig);
        return definePromptAgent<unknown, z.ZodTypeAny>(registry, {
          promptName: 'sharedPersonaPrompt',
          promptInput: { persona },
        });
      };

      const runAgent = async (flow: ReturnType<typeof buildAgent>) => {
        const session = flow.streamBidi({});
        session.send({
          message: { role: 'user' as const, content: [{ text: 'hi' }] },
        });
        session.close();
        for await (const _ of session.stream) {
          // drain
        }
        return session.output;
      };

      const pirateOut = await runAgent(buildAgent('pirate'));
      const ninjaOut = await runAgent(buildAgent('ninja'));

      assert.ok(pirateOut.message?.content[0].text?.includes('pirate'));
      assert.ok(ninjaOut.message?.content[0].text?.includes('ninja'));
    });

    it('should detach asynchronously and continue execution in the background', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let resolvePromise: () => void = () => {};
      const releasePromise = new Promise<void>((resolve) => {
        resolvePromise = resolve;
      });

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'detachTest',
          store,
        },
        async (sess, { sendChunk }) => {
          await sess.run(async () => {
            await releasePromise;
          });
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
        detach: true,
      });

      const output = await session.output;
      const snapshotId = output.snapshotId;
      assert.ok(snapshotId);

      const snapPending = await store.getSnapshot({ snapshotId: snapshotId! });
      assert.strictEqual(snapPending?.status, 'pending');

      resolvePromise();
      session.close();

      const snapDone = await waitForSnapshotStatus(
        store,
        snapshotId!,
        'completed'
      );
      assert.strictEqual(snapDone.status, 'completed');
    });

    it('should abort a detached agent', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let aborted = false;

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'abortTest',
          store,
        },
        async (sess, { abortSignal }) => {
          if (abortSignal) {
            abortSignal.onabort = () => {
              aborted = true;
            };
          }
          await sess.run(async () => {
            await new Promise((resolve) => setTimeout(resolve, 5000));
          });
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
        detach: true,
      });

      const output = await session.output;
      const snapshotId = output.snapshotId;
      assert.ok(snapshotId);

      const previousStatus = await flow.abort(snapshotId!);

      assert.strictEqual(previousStatus, 'pending');
      const snapAborted = await store.getSnapshot({ snapshotId: snapshotId! });
      assert.strictEqual(snapAborted?.status, 'aborted');
      // AbortController.abort() fires onabort synchronously, so no delay needed.
      assert.strictEqual(aborted, true);
    });

    it('should stamp a heartbeat on the pending detached snapshot', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let resolvePromise: () => void = () => {};
      const releasePromise = new Promise<void>((resolve) => {
        resolvePromise = resolve;
      });

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'heartbeatStampTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {
            await releasePromise;
          });
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
        detach: true,
      });

      const output = await session.output;
      const snapshotId = output.snapshotId!;
      assert.ok(snapshotId);

      const snapPending = await store.getSnapshot({ snapshotId });
      assert.strictEqual(snapPending?.status, 'pending');
      assert.ok(
        snapPending?.heartbeatAt,
        'pending detached snapshot should carry a heartbeatAt'
      );

      resolvePromise();
      session.close();
      await waitForSnapshotStatus(store, snapshotId, 'completed');
    });

    it('reports a pending snapshot with a stale heartbeat as expired', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'heartbeatExpiredTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {});
          return { artifacts: [] };
        }
      );

      // Simulate an orphaned detached snapshot: pending, with a heartbeat that
      // is older than the expiry timeout (default 60s).
      const stale = new Date(Date.now() - 120_000).toISOString();
      const snapshotId = await store.saveSnapshot(undefined, () => ({
        createdAt: stale,
        updatedAt: stale,
        heartbeatAt: stale,
        status: 'pending',
        state: { sessionId: 'sess-expired', custom: { foo: 'bar' } },
      }));
      assert.ok(snapshotId);

      // The raw store still has it as pending (compute-on-read does not write
      // back), but getSnapshotData surfaces it as expired.
      const raw = await store.getSnapshot({ snapshotId: snapshotId! });
      assert.strictEqual(raw?.status, 'pending');

      const viaData = await flow.getSnapshotData({ snapshotId: snapshotId! });
      assert.strictEqual(viaData?.status, 'expired');
    });

    it('keeps a pending snapshot with a fresh heartbeat as pending', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'heartbeatFreshTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {});
          return { artifacts: [] };
        }
      );

      const now = new Date().toISOString();
      const snapshotId = await store.saveSnapshot(undefined, () => ({
        createdAt: now,
        updatedAt: now,
        heartbeatAt: now,
        status: 'pending',
        state: { sessionId: 'sess-fresh', custom: { foo: 'bar' } },
      }));
      assert.ok(snapshotId);

      const viaData = await flow.getSnapshotData({ snapshotId: snapshotId! });
      assert.strictEqual(viaData?.status, 'pending');
    });

    it('does not expire a pending snapshot that has no heartbeat yet', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'heartbeatNoneTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {});
          return { artifacts: [] };
        }
      );

      const old = new Date(Date.now() - 120_000).toISOString();
      const snapshotId = await store.saveSnapshot(undefined, () => ({
        createdAt: old,
        updatedAt: old,
        status: 'pending',
        state: { sessionId: 'sess-noheartbeat', custom: { foo: 'bar' } },
      }));
      assert.ok(snapshotId);

      const viaData = await flow.getSnapshotData({ snapshotId: snapshotId! });
      assert.strictEqual(viaData?.status, 'pending');
    });

    it('should not override terminal status when aborting an already-completed flow', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'abortDoneTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();
      const output = await session.output;
      assert.ok(output.snapshotId);

      // Snapshot should be 'done' now
      const snapBefore = await store.getSnapshot({
        snapshotId: output.snapshotId!,
      });
      assert.strictEqual(snapBefore?.status, 'completed');

      // Abort returns the previous status but does not override terminal states
      const previousStatus = await flow.abort(output.snapshotId!);
      assert.strictEqual(previousStatus, 'completed');

      // Snapshot should still be 'done' - the mutator skips terminal states
      const snapAfter = await store.getSnapshot({
        snapshotId: output.snapshotId!,
      });
      assert.strictEqual(snapAfter?.status, 'completed');
    });

    it('should return undefined when aborting a non-existent snapshot', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'abortMissingTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const previousStatus = await flow.abort('non-existent-id');
      assert.strictEqual(previousStatus, undefined);
    });

    it('should throw error when detach is requested without session store', async () => {
      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'noStoreTest',
        },
        async (sess) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
        detach: true,
      });

      try {
        await session.output;
        assert.fail('Should have thrown error');
      } catch (e: any) {
        assert.strictEqual(
          e.message,
          'FAILED_PRECONDITION: Detach is only supported when a session store is provided.'
        );
      }
    });

    it('should save failed snapshot if detached flow throws', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let resolvePromise: () => void = () => {};
      const releasePromise = new Promise<void>((resolve) => {
        resolvePromise = resolve;
      });

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'detachErrorTest',
          store,
        },
        async (sess, { sendChunk }) => {
          await sess.run(async () => {
            await releasePromise;
            throw new Error('intentional background failure');
          });
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
        detach: true,
      });

      const output = await session.output;
      const snapshotId = output.snapshotId;
      assert.ok(snapshotId);

      resolvePromise();
      session.close();

      const snapFailed = await waitForSnapshotStatus(
        store,
        snapshotId!,
        'failed'
      );
      assert.strictEqual(snapFailed.status, 'failed');
      assert.strictEqual(
        snapFailed.error?.message,
        'intentional background failure'
      );
    });

    it('should mark snapshot aborted even without subscription support', async () => {
      const baseStore = new InMemorySessionStore();
      const store = Object.assign(Object.create(baseStore), {
        onSnapshotStateChange: undefined,
        getSnapshot: baseStore.getSnapshot.bind(baseStore),
        saveSnapshot: baseStore.saveSnapshot.bind(baseStore),
      }) as InMemorySessionStore<any>;

      let resolveBlock: () => void = () => {};
      const blockPromise = new Promise<void>((resolve) => {
        resolveBlock = resolve;
      });

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'legacyStoreTest',
          store,
        },
        async (sess, { sendChunk }) => {
          await sess.run(async () => {
            await blockPromise; // Keep flow pending until abort is called
          });
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
        detach: true,
      });

      const output = await session.output;
      const snapshotId = output.snapshotId;

      // Snapshot should be 'pending' since the flow is still blocked
      const snapBefore = await store.getSnapshot({ snapshotId: snapshotId! });
      assert.strictEqual(snapBefore?.status, 'pending');

      await flow.abort(snapshotId!);

      const snapshot = await store.getSnapshot({ snapshotId: snapshotId! });
      assert.strictEqual(snapshot?.status, 'aborted');

      // Release the flow so it doesn't hang
      resolveBlock();
      session.close();
    });

    it('should fetch snapshot data via companion action', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'companionActionFlow',
          store,
        },
        async (sess) => {
          // Mutate session state so a snapshot is persisted on turn end.
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();
      const output = await session.output;

      assert.ok(output.snapshotId, 'should have a snapshotId');
      const snapData = await flow.getSnapshotData({
        snapshotId: output.snapshotId!,
      });
      assert.strictEqual(snapData?.snapshotId, output.snapshotId);
    });

    it('should chain parentId properly across session snapshots', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'lineageTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user' as const, content: [{ text: 'first' }] },
      });
      session1.close();
      const output1 = await session1.output;

      const session2 = flow.streamBidi({
        snapshotId: output1.snapshotId,
      });

      session2.send({
        message: { role: 'user' as const, content: [{ text: 'second' }] },
      });
      session2.close();
      const output2 = await session2.output;

      const snapshot2 = await store.getSnapshot({
        snapshotId: output2.snapshotId!,
      });
      assert.strictEqual(snapshot2?.parentId, output1.snapshotId);
    });

    it('should detach immediately when a detach input is queued', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let releasePromise: () => void = () => {};
      const blockPromise = new Promise<void>((resolve) => {
        releasePromise = resolve;
      });

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'immediateDetachTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {
            await blockPromise;
          });
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'heavy task' }] },
      });
      session.send({
        detach: true,
      });

      const output = await session.output;
      assert.ok(output.snapshotId);
      const snapshot = await store.getSnapshot({
        snapshotId: output.snapshotId!,
      });
      assert.strictEqual(snapshot?.status, 'pending');

      releasePromise();
      session.close();
    });

    it('should process messages even when detach is present in the same payload', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'mixedPayloadTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: {
          role: 'user' as const,
          content: [{ text: 'appended message' }],
        },
        detach: true,
      });

      const output = await session.output;
      assert.ok(output.snapshotId);

      const snapDone = await waitForSnapshotStatus(
        store,
        output.snapshotId!,
        'completed'
      );
      assert.ok(snapDone.state.messages);
      assert.strictEqual(snapDone.state.messages.length, 1);
      assert.strictEqual(
        snapDone.state.messages[0].content[0].text,
        'appended message'
      );

      session.close();
    });

    it('should accumulate message history across multiple turns in one invocation', async () => {
      const registry = new Registry();
      defineEchoModel(registry);
      definePrompt(registry, {
        name: 'multiTurnAccumPrompt',
        model: 'echoModel',
        config: { temperature: 1 },
        system: 'sys',
      });

      const flow = definePromptAgent(registry, {
        promptName: 'multiTurnAccumPrompt',
      });

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'turn1' }] },
      });
      session.send({
        message: { role: 'user' as const, content: [{ text: 'turn2' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      // Two turns must have completed.
      const turnEndChunks = chunks.filter((c) => c.turnEnd !== undefined);
      assert.strictEqual(turnEndChunks.length, 2);

      const output = await session.output;
      assert.strictEqual(output.message?.role, 'model');

      // The second-turn echo should contain the first model reply in its history,
      // proving the session history was passed to the second generate call.
      const turn2Text =
        output.message?.content.map((c) => c.text).join('') ?? '';
      assert.ok(
        turn2Text.includes('Echo:'),
        `Expected second turn to be an echo response, got: ${turn2Text}`
      );

      // Model chunks must have been emitted for both turns.
      const modelChunks = chunks.filter((c) => c.modelChunk !== undefined);
      assert.ok(
        modelChunks.length >= 2,
        'Expected model chunks from both turns'
      );
    });

    it('should successfully handle native tool interrupts and tool response resumption', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const store = new InMemorySessionStore<{}>();

      const pm = defineProgrammableModel(registry, undefined, 'interruptModel');

      const myInterrupt = interrupt({
        name: 'myInterrupt',
        description: 'Ask user',
        inputSchema: z.object({ query: z.string() }),
        outputSchema: z.object({ answer: z.string() }),
      });
      registry.registerAction('tool', myInterrupt);

      definePrompt(registry, {
        name: 'interruptPrompt',
        model: 'interruptModel',
        tools: ['myInterrupt'],
        config: { temperature: 1 },
      });

      const flow = definePromptAgent(registry, {
        promptName: 'interruptPrompt',
        store,
      });

      // Phase 1: User says hello, model responds with a toolRequest (interrupt)
      pm.handleResponse = async () => {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'myInterrupt',
                  input: { query: 'yes?' },
                  ref: '123',
                },
              },
            ],
          },
          finishReason: 'stop',
        };
      };

      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user', content: [{ text: 'hello' }] },
      });
      session1.close(); // IMPORTANT: close the stream so it doesn't hang!

      for await (const chunk of session1.stream) {
      }
      const output1 = await session1.output;

      assert.ok(output1.snapshotId);
      assert.ok(output1.message);
      assert.ok(output1.message.content[0].toolRequest);
      assert.strictEqual(
        output1.message.content[0].toolRequest.name,
        'myInterrupt'
      );

      // Phase 2: Resume with the tool response
      pm.handleResponse = async (req) => {
        // Assert that the resumed request contains the tool response!
        const lastMsg = req.messages[req.messages.length - 1];
        assert.strictEqual(lastMsg.role, 'tool');
        assert.strictEqual(
          (lastMsg.content[0] as any).toolResponse.output.answer,
          'yes indeed'
        );

        return {
          message: {
            role: 'model',
            content: [{ text: 'Task completed successfully!' }],
          },
          finishReason: 'stop',
        };
      };

      const session2 = flow.streamBidi({ snapshotId: output1.snapshotId });
      session2.send({
        resume: {
          respond: [
            {
              toolResponse: {
                name: 'myInterrupt',
                ref: '123',
                output: { answer: 'yes indeed' },
              },
            },
          ],
        },
      });
      session2.close(); // IMPORTANT: close the stream so it doesn't hang!

      for await (const chunk of session2.stream) {
      }
      const output2 = await session2.output;

      assert.strictEqual(output2.message?.role, 'model');
      assert.strictEqual(
        output2.message?.content[0].text,
        'Task completed successfully!'
      );
    });

    it('should handle resume.restart for tool re-execution with metadata', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const store = new InMemorySessionStore<{}>();

      const pm = defineProgrammableModel(registry, undefined, 'restartModel');

      // Track whether the tool was called and with what resumed metadata
      let toolCallCount = 0;
      let lastResumedMetadata: any = undefined;

      defineTool(
        registry,
        {
          name: 'dangerousTool',
          description: 'A tool that requires confirmation',
          inputSchema: z.object({ action: z.string() }),
          outputSchema: z.object({ result: z.string() }),
        },
        async (input, { resumed }) => {
          toolCallCount++;
          lastResumedMetadata = resumed;

          if (!resumed) {
            // First call - interrupt to ask for user confirmation
            throw new ToolInterruptError({ requiresConfirmation: true });
          }
          // Restarted with confirmation metadata
          return { result: `confirmed and executed ${input.action}` };
        }
      );

      definePrompt(registry, {
        name: 'restartPrompt',
        model: 'restartModel',
        tools: ['dangerousTool'],
        config: { temperature: 1 },
      });

      const flow = definePromptAgent(registry, {
        promptName: 'restartPrompt',
        store,
      });

      // Phase 1: Model requests the tool. The tool throws ToolInterruptError,
      // causing the generate action to return finishReason: 'interrupted'.
      pm.handleResponse = async () => {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'dangerousTool',
                  input: { action: 'delete files' },
                  ref: 'tr1',
                },
              },
            ],
          },
          finishReason: 'stop',
        };
      };

      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user', content: [{ text: 'please delete files' }] },
      });
      session1.close();

      for await (const chunk of session1.stream) {
      }
      const output1 = await session1.output;

      assert.ok(output1.snapshotId);
      assert.ok(output1.message);
      assert.ok(output1.message.content[0].toolRequest);
      assert.strictEqual(
        output1.message.content[0].toolRequest.name,
        'dangerousTool'
      );

      // Phase 2: Client resumes with restart - re-execute the tool with metadata
      toolCallCount = 0; // Reset counter

      pm.handleResponse = async (req) => {
        // After restart, the model should receive the tool response from re-execution
        const toolMsgs = req.messages.filter((m: any) => m.role === 'tool');
        assert.ok(
          toolMsgs.length > 0,
          'Model should receive a tool response message'
        );
        const lastToolMsg = toolMsgs[toolMsgs.length - 1];
        assert.strictEqual(
          (lastToolMsg.content[0] as any).toolResponse.output.result,
          'confirmed and executed delete files'
        );

        return {
          message: {
            role: 'model',
            content: [{ text: 'Files deleted successfully!' }],
          },
          finishReason: 'stop',
        };
      };

      const session2 = flow.streamBidi({ snapshotId: output1.snapshotId });
      session2.send({
        resume: {
          restart: [
            {
              toolRequest: {
                name: 'dangerousTool',
                input: { action: 'delete files' },
                ref: 'tr1',
              },
              metadata: { resumed: { approved: true } },
            },
          ],
        },
      });
      session2.close();

      for await (const chunk of session2.stream) {
      }
      const output2 = await session2.output;

      // Verify the tool was actually re-executed
      assert.strictEqual(
        toolCallCount,
        1,
        'Tool should be called once on restart'
      );
      assert.ok(lastResumedMetadata, 'Tool should receive resumed metadata');
      assert.strictEqual(lastResumedMetadata.approved, true);

      assert.strictEqual(output2.message?.role, 'model');
      assert.strictEqual(
        output2.message?.content[0].text,
        'Files deleted successfully!'
      );
    });

    it('should reject resume.restart with forged (modified) inputs', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const store = new InMemorySessionStore<{}>();

      const pm = defineProgrammableModel(
        registry,
        undefined,
        'forgedRestartModel'
      );

      defineTool(
        registry,
        {
          name: 'sensitiveTool',
          description: 'Tool with sensitive inputs',
          inputSchema: z.object({ target: z.string() }),
          outputSchema: z.object({ result: z.string() }),
        },
        async (input, { resumed }) => {
          if (!resumed) {
            throw new ToolInterruptError({ needsApproval: true });
          }
          return { result: `executed on ${input.target}` };
        }
      );

      definePrompt(registry, {
        name: 'forgedRestartPrompt',
        model: 'forgedRestartModel',
        tools: ['sensitiveTool'],
        config: { temperature: 1 },
      });

      const flow = definePromptAgent(registry, {
        promptName: 'forgedRestartPrompt',
        store,
      });

      // Phase 1: Model requests tool, tool interrupts
      pm.handleResponse = async () => ({
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'sensitiveTool',
                input: { target: 'safe-file.txt' },
                ref: 'ref1',
              },
            },
          ],
        },
        finishReason: 'stop',
      });

      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user', content: [{ text: 'do it' }] },
      });
      session1.close();
      for await (const _ of session1.stream) {
      }
      const output1 = await session1.output;
      assert.ok(output1.snapshotId);

      // Phase 2: Client forges restart with DIFFERENT input
      const session2 = flow.streamBidi({ snapshotId: output1.snapshotId });
      session2.send({
        resume: {
          restart: [
            {
              toolRequest: {
                name: 'sensitiveTool',
                input: { target: '/etc/passwd' }, // FORGED!
                ref: 'ref1',
              },
              metadata: { resumed: { approved: true } },
            },
          ],
        },
      });
      session2.close();

      // Forged inputs fail gracefully with finishReason 'failed' and the
      // original INVALID_ARGUMENT status preserved on the error.
      for await (const _ of session2.stream) {
      }
      const output2 = await session2.output;
      assert.strictEqual(output2.finishReason, 'failed');
      assert.ok(output2.error);
      assert.strictEqual(output2.error!.status, 'INVALID_ARGUMENT');
      assert.ok(
        output2.error!.message.includes('modified inputs'),
        `Expected modified inputs error, got: ${output2.error!.message}`
      );
    });

    it('should reject resume.respond referencing a non-existent tool', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const store = new InMemorySessionStore<{}>();

      const pm = defineProgrammableModel(
        registry,
        undefined,
        'fakeRespondModel'
      );

      const myInterrupt = interrupt({
        name: 'realInterrupt',
        description: 'A real interrupt',
        inputSchema: z.object({ q: z.string() }),
        outputSchema: z.object({ a: z.string() }),
      });
      registry.registerAction('tool', myInterrupt);

      definePrompt(registry, {
        name: 'fakeRespondPrompt',
        model: 'fakeRespondModel',
        tools: ['realInterrupt'],
        config: { temperature: 1 },
      });

      const flow = definePromptAgent(registry, {
        promptName: 'fakeRespondPrompt',
        store,
      });

      // Phase 1: Model requests the real interrupt tool
      pm.handleResponse = async () => ({
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'realInterrupt',
                input: { q: 'confirm?' },
                ref: 'r1',
              },
            },
          ],
        },
        finishReason: 'stop',
      });

      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session1.close();
      for await (const _ of session1.stream) {
      }
      const output1 = await session1.output;
      assert.ok(output1.snapshotId);

      // Phase 2: Client responds with a FAKE tool name/ref
      const session2 = flow.streamBidi({ snapshotId: output1.snapshotId });
      session2.send({
        resume: {
          respond: [
            {
              toolResponse: {
                name: 'fakeToolThatDoesNotExist',
                ref: 'fake-ref',
                output: { a: 'hacked' },
              },
            },
          ],
        },
      });
      session2.close();

      // Fails gracefully with finishReason 'failed' and INVALID_ARGUMENT.
      for await (const _ of session2.stream) {
      }
      const output2 = await session2.output;
      assert.strictEqual(output2.finishReason, 'failed');
      assert.ok(output2.error);
      assert.strictEqual(output2.error!.status, 'INVALID_ARGUMENT');
      assert.ok(
        output2.error!.message.includes('not found in session history'),
        `Expected not found error, got: ${output2.error!.message}`
      );
    });

    it('should reject resume.restart referencing a non-existent tool', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const store = new InMemorySessionStore<{}>();

      const pm = defineProgrammableModel(
        registry,
        undefined,
        'fakeRestartModel'
      );

      definePrompt(registry, {
        name: 'fakeRestartPrompt',
        model: 'fakeRestartModel',
        config: { temperature: 1 },
      });

      const flow = definePromptAgent(registry, {
        promptName: 'fakeRestartPrompt',
        store,
      });

      // Phase 1: Model returns a simple text response (no tools at all)
      pm.handleResponse = async () => ({
        message: {
          role: 'model',
          content: [{ text: 'hello' }],
        },
        finishReason: 'stop',
      });

      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session1.close();
      for await (const _ of session1.stream) {
      }
      const output1 = await session1.output;
      assert.ok(output1.snapshotId);

      // Phase 2: Client fabricates a restart for a tool that was never requested
      const session2 = flow.streamBidi({ snapshotId: output1.snapshotId });
      session2.send({
        resume: {
          restart: [
            {
              toolRequest: {
                name: 'inventedTool',
                input: { evil: true },
                ref: 'fake-ref',
              },
              metadata: { resumed: true },
            },
          ],
        },
      });
      session2.close();

      // Fails gracefully with finishReason 'failed' and INVALID_ARGUMENT.
      for await (const _ of session2.stream) {
      }
      const output2 = await session2.output;
      assert.strictEqual(output2.finishReason, 'failed');
      assert.ok(output2.error);
      assert.strictEqual(output2.error!.status, 'INVALID_ARGUMENT');
      assert.ok(
        output2.error!.message.includes('not found in session history'),
        `Expected not found error, got: ${output2.error!.message}`
      );
    });

    it('should process all pre-queued messages in the background after detaching', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let processedCount = 0;

      const flow = defineCustomAgent<{ foo: string }>(
        new Registry(),
        {
          name: 'sequentialBackgroundTest',
          store,
        },
        async (sess) => {
          await sess.run(async () => {
            processedCount++;
          });
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'task 1' }] },
      });
      session.send({
        message: { role: 'user' as const, content: [{ text: 'task 2' }] },
      });
      session.send({ detach: true });

      const output = await session.output;
      assert.ok(output.snapshotId);

      // Detach-only messages are not forwarded to the runner - 2 turns, not 3.
      const snapDone = await waitForSnapshotStatus(
        store,
        output.snapshotId!,
        'completed'
      );
      assert.strictEqual(snapDone.status, 'completed');
      assert.strictEqual(processedCount, 2);

      session.close();
    });
  });

  describe('clientTransform', () => {
    it('should transform state in AgentOutput for client-managed agents', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent<{
        publicField: string;
        secretField: string;
      }>(
        registry,
        {
          name: 'clientTransformTest',
          clientTransform: {
            state: (state) => ({
              custom: { publicField: (state.custom as any)?.publicField },
              // Strip messages and artifacts
            }),
          },
        },

        async (sess) => {
          sess.session.updateCustom(() => ({
            publicField: 'visible',
            secretField: 'top-secret',
          }));
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'done' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      assert.ok(output.state);
      assert.strictEqual((output.state!.custom as any).publicField, 'visible');
      assert.strictEqual((output.state!.custom as any).secretField, undefined);
      // Messages were stripped by the transform
      assert.strictEqual(output.state!.messages, undefined);
    });

    it('should return full state when no clientStateTransform is provided', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent<{
        publicField: string;
        secretField: string;
      }>(registry, { name: 'noTransformTest' }, async (sess) => {
        sess.session.updateCustom(() => ({
          publicField: 'visible',
          secretField: 'top-secret',
        }));
        await sess.run(async () => {});
        return {
          artifacts: [],
          message: { role: 'model', content: [{ text: 'done' }] },
        };
      });

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      assert.ok(output.state);
      assert.strictEqual((output.state!.custom as any).publicField, 'visible');
      assert.strictEqual(
        (output.state!.custom as any).secretField,
        'top-secret'
      );
      // Messages should be present
      assert.ok(output.state!.messages);
      assert.strictEqual(output.state!.messages!.length, 1);
    });

    it('should transform snapshot state in getSnapshotData for server-managed agents', async () => {
      const store = new InMemorySessionStore<{
        publicField: string;
        secretField: string;
      }>();

      const flow = defineCustomAgent<{
        publicField: string;
        secretField: string;
      }>(
        new Registry(),
        {
          name: 'snapshotTransformTest',
          store,
          clientTransform: {
            state: (state) => ({
              custom: { publicField: (state.custom as any)?.publicField },
            }),
          },
        },

        async (sess) => {
          sess.session.updateCustom(() => ({
            publicField: 'visible',
            secretField: 'top-secret',
          }));
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'done' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;
      assert.ok(output.snapshotId);

      // getSnapshotData should return transformed state
      const snapshot = await flow.getSnapshotData({
        snapshotId: output.snapshotId!,
      });
      assert.ok(snapshot);
      assert.strictEqual(
        (snapshot!.state.custom as any).publicField,
        'visible'
      );
      assert.strictEqual(
        (snapshot!.state.custom as any).secretField,
        undefined
      );
      // Messages were stripped
      assert.strictEqual(snapshot!.state.messages, undefined);

      // But the raw store should still have the full state
      const rawSnapshot = await store.getSnapshot({
        snapshotId: output.snapshotId!,
      });
      assert.ok(rawSnapshot);
      assert.strictEqual(rawSnapshot!.state.custom?.secretField, 'top-secret');
      assert.ok(rawSnapshot!.state.messages);
    });

    it('should transform snapshot state in getSnapshotDataAction for server-managed agents', async () => {
      const registry = new Registry();
      const store = new InMemorySessionStore<{
        publicField: string;
        secretField: string;
      }>();

      const flow = defineCustomAgent<{
        publicField: string;
        secretField: string;
      }>(
        registry,
        {
          name: 'snapshotActionTransformTest',
          store,
          clientTransform: {
            state: (state) => ({
              custom: { publicField: (state.custom as any)?.publicField },
            }),
          },
        },

        async (sess) => {
          sess.session.updateCustom(() => ({
            publicField: 'visible',
            secretField: 'top-secret',
          }));
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'done' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;
      assert.ok(output.snapshotId);

      // Invoke the companion action directly
      const actionResult = await flow.getSnapshotDataAction({
        snapshotId: output.snapshotId!,
      });
      assert.ok(actionResult);
      assert.strictEqual(
        (actionResult as any).state.custom.publicField,
        'visible'
      );
      assert.strictEqual(
        (actionResult as any).state.custom.secretField,
        undefined
      );
    });

    it('should transform state in detached output for client-managed agents', async () => {
      const store = new InMemorySessionStore<{
        publicField: string;
        secretField: string;
      }>();
      let resolvePromise: () => void = () => {};
      const releasePromise = new Promise<void>((resolve) => {
        resolvePromise = resolve;
      });

      // Client-managed (no store in config), but we need a store for detach;
      // use a server-managed config to test detach transform path
      const flow = defineCustomAgent<{
        publicField: string;
        secretField: string;
      }>(
        new Registry(),
        {
          name: 'detachTransformTest',
          store,
          clientTransform: {
            state: (state) => ({
              custom: { publicField: (state.custom as any)?.publicField },
            }),
          },
        },

        async (sess) => {
          sess.session.updateCustom(() => ({
            publicField: 'visible',
            secretField: 'top-secret',
          }));
          await sess.run(async () => {
            await releasePromise;
          });
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'done' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
        detach: true,
      });

      const output = await session.output;
      assert.ok(output.snapshotId);
      // Server-managed agents don't return state in output (state is undefined)
      // but the snapshot should have the transformed state
      const snapshot = await flow.getSnapshotData({
        snapshotId: output.snapshotId!,
      });
      assert.ok(snapshot);
      assert.strictEqual(
        (snapshot!.state.custom as any).publicField,
        'visible'
      );
      assert.strictEqual(
        (snapshot!.state.custom as any).secretField,
        undefined
      );

      resolvePromise();
      session.close();
    });

    it('should pass clientTransform through definePromptAgent', async () => {
      const registry = new Registry();
      defineEchoModel(registry);
      definePrompt(registry, {
        name: 'transformPromptAgent',
        model: 'echoModel',
        config: { temperature: 1 },
      });

      const flow = definePromptAgent<{ secret: string }>(registry, {
        promptName: 'transformPromptAgent',
        clientTransform: {
          state: (state) => ({
            // strip custom state entirely, keep messages
            messages: state.messages,
          }),
        },
      });

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      assert.ok(output.state);
      // Custom state should be stripped
      assert.strictEqual(output.state!.custom, undefined);
      // Messages should be present
      assert.ok(output.state!.messages);
      assert.ok(output.state!.messages!.length > 0);
    });

    it('should pass clientTransform through defineAgent', async () => {
      const registry = new Registry();
      defineEchoModel(registry);

      const flow = defineAgent<{ secret: string }>(registry, {
        name: 'transformDefineAgent',
        model: 'echoModel',
        config: { temperature: 1 },
        clientTransform: {
          state: (state) => ({
            // strip custom state entirely, keep messages
            messages: state.messages,
          }),
        },
      });

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      assert.ok(output.state);
      // Custom state should be stripped
      assert.strictEqual(output.state!.custom, undefined);
      // Messages should be present
      assert.ok(output.state!.messages);
      assert.ok(output.state!.messages!.length > 0);
    });

    it('should reshape stream chunks via clientTransform.chunk', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        {
          name: 'chunkReshapeTest',
          clientTransform: {
            // Redact any text content out of streamed model chunks.
            chunk: (chunk) => {
              if (chunk.modelChunk) {
                return {
                  ...chunk,
                  modelChunk: {
                    ...chunk.modelChunk,
                    content: chunk.modelChunk.content.map((p) =>
                      p.text ? { text: '[redacted]' } : p
                    ),
                  },
                };
              }
              return chunk;
            },
          },
        },
        async (sess, { sendChunk }) => {
          await sess.run(async () => {
            sendChunk({
              modelChunk: { role: 'model', content: [{ text: 'secret' }] },
            });
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'done' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const modelChunks = chunks.filter((c) => !!c.modelChunk);
      assert.ok(modelChunks.length > 0);
      // The text content was redacted by the chunk transform.
      assert.strictEqual(
        modelChunks[0].modelChunk?.content[0].text,
        '[redacted]'
      );
    });

    it('should drop stream chunks when clientTransform.chunk returns nullish', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        {
          name: 'chunkDropTest',
          clientTransform: {
            // Drop artifact chunks entirely, keep everything else.
            chunk: (chunk) => (chunk.artifact ? null : chunk),
          },
        },
        async (sess, { sendChunk }) => {
          await sess.run(async () => {
            sess.session.addArtifacts([
              { name: 'internalArt', parts: [{ text: 'secret' }] },
            ]);
            sendChunk({
              modelChunk: { role: 'model', content: [{ text: 'visible' }] },
            });
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'done' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      // Artifact chunks were dropped by the transform.
      assert.strictEqual(chunks.filter((c) => !!c.artifact).length, 0);
      // Model chunks were preserved.
      assert.ok(chunks.filter((c) => !!c.modelChunk).length > 0);
    });

    it('should pass clientTransform.chunk through defineAgent', async () => {
      const registry = new Registry();
      defineEchoModel(registry);

      const flow = defineAgent(registry, {
        name: 'chunkDefineAgentTest',
        model: 'echoModel',
        config: { temperature: 1 },
        clientTransform: {
          chunk: (chunk) => {
            if (chunk.modelChunk) {
              return {
                ...chunk,
                modelChunk: {
                  ...chunk.modelChunk,
                  content: chunk.modelChunk.content.map((p) =>
                    p.text ? { text: '[redacted]' } : p
                  ),
                },
              };
            }
            return chunk;
          },
        },
      });

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user', content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const modelChunks = chunks.filter((c) => !!c.modelChunk);
      assert.ok(modelChunks.length > 0);
      for (const c of modelChunks) {
        for (const p of c.modelChunk!.content) {
          if ('text' in p && p.text !== undefined) {
            assert.strictEqual(p.text, '[redacted]');
          }
        }
      }
    });
  });

  // =========================================================================
  // Prompt rendering across turns
  // =========================================================================

  describe('prompt rendering across turns', () => {
    /** Run a single invocation, collecting all model requests made during it. */
    async function runAgent(
      agent: ReturnType<typeof defineAgent>,
      pm: ProgrammableModel,
      opts: {
        init?: any;
        inputs: any[];
        modelResponses: any[];
      }
    ) {
      const modelRequests: any[] = [];
      let reqCounter = 0;

      pm.handleResponse = async (req) => {
        modelRequests.push(JSON.parse(JSON.stringify(req)));
        return opts.modelResponses[reqCounter++]!;
      };

      const session = agent.streamBidi(opts.init || {});
      for (const input of opts.inputs) {
        session.send(input);
      }
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const output = await session.output;
      return { output, chunks, modelRequests };
    }

    it('system-only: system appears in model request each turn, not in stored history', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const pm = defineProgrammableModel(registry);

      const agent = defineAgent(registry, {
        name: 'systemOnlyAgent',
        model: 'programmableModel',
        system: 'You are a helpful assistant.',
      });

      const { output, modelRequests } = await runAgent(agent, pm, {
        inputs: [
          { message: { role: 'user', content: [{ text: 'turn1' }] } },
          { message: { role: 'user', content: [{ text: 'turn2' }] } },
        ],
        modelResponses: [
          {
            message: { role: 'model', content: [{ text: 'reply1' }] },
            finishReason: 'stop',
          },
          {
            message: { role: 'model', content: [{ text: 'reply2' }] },
            finishReason: 'stop',
          },
        ],
      });

      // --- Model request assertions ---

      // Turn 1: model sees [system("You are a helpful assistant."), user("turn1")]
      const t1 = modelRequests[0].messages;
      assert.strictEqual(
        t1.length,
        2,
        'Turn 1: model should receive 2 messages'
      );
      assert.strictEqual(t1[0].role, 'system');
      assert.strictEqual(t1[0].content[0].text, 'You are a helpful assistant.');
      assert.strictEqual(t1[1].role, 'user');
      assert.strictEqual(t1[1].content[0].text, 'turn1');

      // Turn 2: model sees [system, user("turn1"), model("reply1"), user("turn2")]
      const t2 = modelRequests[1].messages;
      assert.strictEqual(
        t2.length,
        4,
        'Turn 2: model should receive 4 messages'
      );
      assert.strictEqual(t2[0].role, 'system');
      assert.strictEqual(t2[0].content[0].text, 'You are a helpful assistant.');
      assert.strictEqual(t2[1].role, 'user');
      assert.strictEqual(t2[1].content[0].text, 'turn1');
      assert.strictEqual(t2[2].role, 'model');
      assert.strictEqual(t2[2].content[0].text, 'reply1');
      assert.strictEqual(t2[3].role, 'user');
      assert.strictEqual(t2[3].content[0].text, 'turn2');

      // No duplicate system messages
      assert.strictEqual(t2.filter((m: any) => m.role === 'system').length, 1);

      // --- Stored messages assertions ---
      const storedMessages = output.state?.messages || [];
      assert.strictEqual(
        storedMessages.filter((m: any) => m.role === 'system').length,
        0,
        'Stored history should not contain system messages'
      );
      assert.strictEqual(storedMessages.length, 4);
    });

    it('system + user prompt: template user prompt appears each turn but does not accumulate', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const pm = defineProgrammableModel(registry);

      const agent = defineAgent(registry, {
        name: 'systemAndPromptAgent',
        model: 'programmableModel',
        system: 'You are a helpful assistant.',
        prompt: 'Always respond concisely.',
      });

      const { output, modelRequests } = await runAgent(agent, pm, {
        inputs: [
          { message: { role: 'user', content: [{ text: 'turn1' }] } },
          { message: { role: 'user', content: [{ text: 'turn2' }] } },
        ],
        modelResponses: [
          {
            message: { role: 'model', content: [{ text: 'reply1' }] },
            finishReason: 'stop',
          },
          {
            message: { role: 'model', content: [{ text: 'reply2' }] },
            finishReason: 'stop',
          },
        ],
      });

      // Turn 2: template user prompt should appear exactly once
      const templateMsgs = modelRequests[1].messages.filter(
        (m: any) =>
          m.role === 'user' &&
          m.content?.[0]?.text?.includes('Always respond concisely')
      );
      assert.strictEqual(templateMsgs.length, 1);

      // Stored history should NOT contain system or template user prompt
      const storedMessages = output.state?.messages || [];
      assert.strictEqual(
        storedMessages.filter((m: any) => m.role === 'system').length,
        0
      );
      assert.strictEqual(
        storedMessages.filter(
          (m: any) =>
            m.role === 'user' &&
            m.content?.[0]?.text?.includes('Always respond concisely')
        ).length,
        0
      );
      assert.strictEqual(storedMessages.length, 4);
    });

    it('cross-invocation: system + prompt do not duplicate when state is carried over', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const pm = defineProgrammableModel(registry);

      const agent = defineAgent(registry, {
        name: 'crossInvAgent',
        model: 'programmableModel',
        system: 'You are a helpful assistant.',
        prompt: 'Always respond concisely.',
      });

      // Invocation 1
      const result1 = await runAgent(agent, pm, {
        inputs: [{ message: { role: 'user', content: [{ text: 'first' }] } }],
        modelResponses: [
          {
            message: { role: 'model', content: [{ text: 'reply1' }] },
            finishReason: 'stop',
          },
        ],
      });

      // Invocation 2: seed with state from invocation 1
      const result2 = await runAgent(agent, pm, {
        init: { state: result1.output.state },
        inputs: [{ message: { role: 'user', content: [{ text: 'second' }] } }],
        modelResponses: [
          {
            message: { role: 'model', content: [{ text: 'reply2' }] },
            finishReason: 'stop',
          },
        ],
      });

      const req2msgs = result2.modelRequests[0].messages;
      assert.strictEqual(
        req2msgs.filter((m: any) => m.role === 'system').length,
        1
      );
      assert.strictEqual(
        req2msgs.filter(
          (m: any) =>
            m.role === 'user' &&
            m.content?.[0]?.text?.includes('Always respond concisely')
        ).length,
        1
      );

      // Stored messages should be clean
      const storedMessages = result2.output.state?.messages || [];
      assert.strictEqual(storedMessages.length, 4);
      assert.strictEqual(
        storedMessages.filter((m: any) => m.role === 'system').length,
        0
      );
    });

    it('message ordering: [system, ...history, user_prompt_from_template]', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const pm = defineProgrammableModel(registry);

      const agent = defineAgent(registry, {
        name: 'orderingAgent',
        model: 'programmableModel',
        system: 'Be helpful.',
        prompt: 'Be concise.',
      });

      const { modelRequests } = await runAgent(agent, pm, {
        inputs: [
          { message: { role: 'user', content: [{ text: 'q1' }] } },
          { message: { role: 'user', content: [{ text: 'q2' }] } },
        ],
        modelResponses: [
          {
            message: { role: 'model', content: [{ text: 'a1' }] },
            finishReason: 'stop',
          },
          {
            message: { role: 'model', content: [{ text: 'a2' }] },
            finishReason: 'stop',
          },
        ],
      });

      // Turn 2: render places history between system and user prompt
      const req2msgs = modelRequests[1].messages;
      const roles = req2msgs.map((m: any) => m.role);
      // Expected: [system, user(q1), model(a1), user(q2), user(Be concise.)]
      assert.deepStrictEqual(roles, [
        'system',
        'user',
        'model',
        'user',
        'user',
      ]);
      // Preamble messages are tagged agentPreamble; history messages are
      // clean (the internal _genkit_history tag is stripped before the model
      // sees them).
      assert.ok(
        req2msgs[0].metadata?.agentPreamble,
        'system is preamble-tagged'
      );
      assert.strictEqual(
        req2msgs[1].metadata?.agentPreamble,
        undefined,
        'q1 has no preamble tag'
      );
      assert.strictEqual(
        req2msgs[1].metadata?._genkit_history,
        undefined,
        'q1 has no history tag (stripped)'
      );
      assert.strictEqual(
        req2msgs[2].metadata?._genkit_history,
        undefined,
        'a1 has no history tag (stripped)'
      );
      assert.strictEqual(
        req2msgs[3].metadata?._genkit_history,
        undefined,
        'q2 has no history tag (stripped)'
      );
      assert.ok(
        req2msgs[4].metadata?.agentPreamble,
        'Be concise is preamble-tagged'
      );
    });

    it('dotprompt {{history}}: history is inserted where the template specifies', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const pm = defineProgrammableModel(registry);

      // Define a prompt with a dotprompt messages template that uses {{history}}
      definePrompt(registry, {
        name: 'historyTemplatePrompt',
        model: 'programmableModel',
        system: 'You are a helpful assistant.',
        messages: `{{role "user"}}Here is the conversation so far:
{{history}}
Now respond to the latest message.`,
      });

      const agent = definePromptAgent(registry, {
        promptName: 'historyTemplatePrompt',
      });

      const { output, modelRequests } = await runAgent(agent, pm, {
        inputs: [
          { message: { role: 'user', content: [{ text: 'hello' }] } },
          { message: { role: 'user', content: [{ text: 'how are you' }] } },
        ],
        modelResponses: [
          {
            message: { role: 'model', content: [{ text: 'hi there' }] },
            finishReason: 'stop',
          },
          {
            message: { role: 'model', content: [{ text: 'doing well' }] },
            finishReason: 'stop',
          },
        ],
      });

      // --- Turn 1 model request assertions ---
      // Model sees: [system, user(template-before), user(hello), model(template-after)]
      const t1 = modelRequests[0].messages;
      assert.strictEqual(t1.length, 4, 'Turn 1: 4 messages');

      assert.strictEqual(t1[0].role, 'system');
      assert.strictEqual(t1[0].content[0].text, 'You are a helpful assistant.');
      assert.ok(t1[0].metadata?.agentPreamble, 'T1: system is preamble');

      assert.strictEqual(t1[1].role, 'user');
      assert.ok(
        t1[1].content[0].text.includes('Here is the conversation so far'),
        'T1: template text before {{history}}'
      );
      assert.ok(
        t1[1].metadata?.agentPreamble,
        'T1: template-before is preamble'
      );

      assert.strictEqual(t1[2].role, 'user');
      assert.strictEqual(t1[2].content[0].text, 'hello');
      assert.strictEqual(
        t1[2].metadata?.agentPreamble,
        undefined,
        'T1: hello is not preamble'
      );
      assert.strictEqual(
        t1[2].metadata?._genkit_history,
        undefined,
        'T1: hello has no internal tag'
      );

      assert.strictEqual(t1[3].role, 'model');
      assert.ok(
        t1[3].content[0].text.includes('Now respond to the latest message'),
        'T1: template text after {{history}}'
      );
      assert.ok(
        t1[3].metadata?.agentPreamble,
        'T1: template-after is preamble'
      );

      // --- Turn 2 model request assertions ---
      // Model sees: [system, user(template-before), user(hello), model(hi there),
      //              user(how are you), model(template-after)]
      const t2 = modelRequests[1].messages;
      assert.strictEqual(t2.length, 6, 'Turn 2: 6 messages');

      assert.strictEqual(t2[0].role, 'system');
      assert.ok(t2[0].metadata?.agentPreamble, 'T2: system is preamble');

      assert.strictEqual(t2[1].role, 'user');
      assert.ok(
        t2[1].metadata?.agentPreamble,
        'T2: template-before is preamble'
      );

      // History messages are embedded between template parts, clean of internal tags
      assert.strictEqual(t2[2].role, 'user');
      assert.strictEqual(t2[2].content[0].text, 'hello');
      assert.strictEqual(
        t2[2].metadata?.agentPreamble,
        undefined,
        'T2: hello not preamble'
      );
      assert.strictEqual(
        t2[2].metadata?._genkit_history,
        undefined,
        'T2: hello no internal tag'
      );

      assert.strictEqual(t2[3].role, 'model');
      assert.strictEqual(t2[3].content[0].text, 'hi there');
      assert.strictEqual(
        t2[3].metadata?.agentPreamble,
        undefined,
        'T2: hi there not preamble'
      );
      assert.strictEqual(
        t2[3].metadata?._genkit_history,
        undefined,
        'T2: hi there no internal tag'
      );

      assert.strictEqual(t2[4].role, 'user');
      assert.strictEqual(t2[4].content[0].text, 'how are you');
      assert.strictEqual(
        t2[4].metadata?.agentPreamble,
        undefined,
        'T2: how are you not preamble'
      );

      assert.strictEqual(t2[5].role, 'model');
      assert.ok(
        t2[5].content[0].text.includes('Now respond to the latest message'),
        'T2: template-after text'
      );
      assert.ok(
        t2[5].metadata?.agentPreamble,
        'T2: template-after is preamble'
      );

      // --- Stored messages should be clean (no system, no template wrapper) ---
      const storedMessages = output.state?.messages || [];
      assert.strictEqual(
        storedMessages.filter((m: any) => m.role === 'system').length,
        0,
        'No system in stored history'
      );
      // Should have the 4 conversation messages
      assert.strictEqual(storedMessages.length, 4);
      assert.strictEqual(storedMessages[0].content[0].text, 'hello');
      assert.strictEqual(storedMessages[1].content[0].text, 'hi there');
      assert.strictEqual(storedMessages[2].content[0].text, 'how are you');
      assert.strictEqual(storedMessages[3].content[0].text, 'doing well');
    });
  });

  describe('finishReason', () => {
    it('reports the explicit finishReason from a custom agent turn', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        { name: 'frCustom' },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'interrupted' as const };
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const turnEndChunk = chunks.find((c) => !!c.turnEnd);
      assert.strictEqual(turnEndChunk?.turnEnd?.finishReason, 'interrupted');

      const output = await session.output;
      assert.strictEqual(output.finishReason, 'interrupted');
    });

    it('prefers an explicit final finishReason on the AgentResult', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        { name: 'frCustomFinal' },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'interrupted' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'done' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }

      const output = await session.output;
      assert.strictEqual(output.finishReason, 'stop');
    });

    it('omits finishReason when the turn does not report one', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent(
        registry,
        { name: 'frCustomNone' },
        async (sess) => {
          await sess.run(async () => {
            // no finishReason returned
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const turnEndChunk = chunks.find((c) => !!c.turnEnd);
      assert.strictEqual(turnEndChunk?.turnEnd?.finishReason, undefined);

      const output = await session.output;
      assert.strictEqual(output.finishReason, undefined);
    });

    it('persists the finishReason in the session snapshot', async () => {
      const store = new InMemorySessionStore<{}>();

      const flow = defineCustomAgent<{}>(
        new Registry(),
        { name: 'frPersist', store },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'interrupted' as const };
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }

      const output = await session.output;
      assert.ok(output.snapshotId);
      const snapshot = await store.getSnapshot({
        snapshotId: output.snapshotId!,
      });
      assert.strictEqual(snapshot?.finishReason, 'interrupted');
    });

    it('reports failed as the finishReason when a turn throws', async () => {
      const store = new InMemorySessionStore<{}>();

      const flow = defineCustomAgent<{}>(
        new Registry(),
        { name: 'frFailed', store },
        async (sess) => {
          await sess.run(async () => {
            throw new Error('boom');
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      // The agent no longer throws on an in-band turn failure - it resolves
      // gracefully with finishReason 'failed' and a structured error.
      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }
      const output = await session.output;

      assert.strictEqual(output.finishReason, 'failed');
      assert.ok(output.error);
      assert.strictEqual(output.error!.status, 'INTERNAL');
      assert.ok(output.error!.message.includes('boom'));

      const turnEndChunk = chunks.find((c) => !!c.turnEnd);
      assert.strictEqual(turnEndChunk?.turnEnd?.finishReason, 'failed');

      // The failed turn's snapshot records the failure.
      const turnEndSnapshotId = turnEndChunk?.turnEnd?.snapshotId;
      assert.ok(turnEndSnapshotId);
      const snapshot = await store.getSnapshot({
        snapshotId: turnEndSnapshotId!,
      });
      assert.strictEqual(snapshot?.finishReason, 'failed');
      assert.strictEqual(snapshot?.status, 'failed');

      // First-turn failure: no prior successful turn ran, so the last-good
      // state is the seed the client already has. No redundant recovery
      // snapshot is written and snapshotId is left unset.
      assert.strictEqual(output.snapshotId, undefined);
    });

    it('does not validate a never-set custom state against a required-field stateSchema across turns', async () => {
      // Regression: a fresh session must NOT seed custom state with `{}`.
      // If it did, `{}` would be persisted into the first snapshot and then
      // fail validation on the next resume when the stateSchema has required
      // fields - even though the user never set any custom state.
      const store = new InMemorySessionStore<{ required: string }>();

      const flow = defineCustomAgent<{ required: string }>(
        new Registry(),
        {
          name: 'requiredStateAgent',
          store,
          // `required` is a required field, so `{}` would be invalid.
          stateSchema: z.object({ required: z.string() }),
        },
        async (sess) => {
          // The handler never sets custom state.
          await sess.run(async () => {
            return { finishReason: 'stop' as const };
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      // Turn 1: fresh session (no custom state set). The persisted snapshot
      // holds `custom: undefined` rather than `{}`.
      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user' as const, content: [{ text: 'one' }] },
      });
      session1.close();
      for await (const _ of session1.stream) {
      }
      const output1 = await session1.output;
      assert.strictEqual(
        output1.finishReason,
        'stop',
        `turn1 error: ${JSON.stringify(output1.error)}`
      );
      assert.ok(output1.snapshotId);

      // Turn 2: resume from the turn-1 snapshot. This loads + validates the
      // persisted custom state. It must NOT fail validation, since the
      // never-set custom state stays `undefined` rather than `{}`.
      const session2 = flow.streamBidi({ snapshotId: output1.snapshotId });
      session2.send({
        message: { role: 'user' as const, content: [{ text: 'two' }] },
      });
      session2.close();
      for await (const _ of session2.stream) {
      }
      const output2 = await session2.output;
      assert.strictEqual(
        output2.finishReason,
        'stop',
        `turn2 error: ${JSON.stringify(output2.error)}`
      );
      assert.strictEqual(output2.error, undefined);
    });

    it('does not leak the raw thrown error into error.details', async () => {
      const store = new InMemorySessionStore<{}>();

      // Throw a value with a circular reference and no `detail`/`details`
      // field. The old behavior fell back to placing the whole error object in
      // `details`, which both leaked internals and could break JSON.stringify
      // (here: a circular structure) when persisting the failed snapshot.
      const circular: any = new Error('boom');
      circular.self = circular;

      const flow = defineCustomAgent<{}>(
        new Registry(),
        { name: 'frNoLeak', store },
        async (sess) => {
          await sess.run(async () => {
            throw circular;
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }
      const output = await session.output;

      assert.strictEqual(output.finishReason, 'failed');
      assert.ok(output.error);
      assert.strictEqual(output.error!.status, 'INTERNAL');
      assert.ok(output.error!.message.includes('boom'));
      // details must NOT fall back to the raw (circular) error object.
      assert.strictEqual(output.error!.details, undefined);

      // The failed snapshot persisted without throwing on the circular error.
      const turnEndChunk = chunks.find((c) => !!c.turnEnd);
      const snapshot = await store.getSnapshot({
        snapshotId: turnEndChunk!.turnEnd!.snapshotId!,
      });
      assert.strictEqual(snapshot?.status, 'failed');
      assert.strictEqual(snapshot?.error?.details, undefined);
    });

    it('does not fail the turn when a stream emit throws (closed stream)', async () => {
      // Simulate a client that has gone away mid-turn: arg.sendChunk throws.
      // The synchronous custom/artifact emits fire from inside the user's
      // handler (updateCustom / addArtifacts); a throw there must NOT escape
      // and turn an otherwise-successful turn into a 'failed' one.
      const registry = new Registry();

      let sendChunkCalls = 0;
      const flow = defineCustomAgent<{ count: number }>(
        registry,
        {
          name: 'frStreamThrows',
          clientTransform: {
            chunk: () => {
              // Throwing here is equivalent to arg.sendChunk throwing on a
              // closed stream - emitChunk must absorb it.
              sendChunkCalls++;
              throw new Error('stream closed');
            },
          },
        },
        async (sess) => {
          await sess.run(async () => {
            // These both emit synchronously via the guarded emitter.
            sess.session.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            sess.session.addArtifacts([{ name: 'a', parts: [{ text: 'v1' }] }]);
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'done' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      // The turn succeeded despite every emit throwing.
      assert.strictEqual(output.finishReason, 'stop');
      assert.strictEqual(output.error, undefined);
      assert.ok(sendChunkCalls > 0, 'emit should have been attempted');
      // The state mutations still applied.
      assert.strictEqual((output.state!.custom as any).count, 1);
    });

    it('client-managed: preserves prior-turn state when a later turn fails', async () => {
      let turn = 0;
      const flow = defineCustomAgent<{ count: number }>(
        new Registry(),
        { name: 'frClientPreserve' },
        async (sess) => {
          await sess.run(async () => {
            turn++;
            sess.session.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            if (turn === 2) {
              throw new Error('second turn boom');
            }
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'one' }] },
      });
      session.send({
        message: { role: 'user' as const, content: [{ text: 'two' }] },
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      assert.strictEqual(output.finishReason, 'failed');
      assert.ok(output.error);
      assert.ok(output.error!.message.includes('second turn boom'));
      // The returned state is the last-good state (after turn 1 succeeded),
      // not the partial mutations from the failed second turn.
      assert.ok(output.state);
      assert.strictEqual((output.state!.custom as any).count, 1);
      // The first user message is preserved in the recovered history.
      assert.ok(output.state!.messages);
      assert.strictEqual(output.state!.messages![0].content[0].text, 'one');
    });

    it('server-managed: failure returns the last-good (done) snapshot, not the failed one', async () => {
      const store = new InMemorySessionStore<{ count: number }>();
      let turn = 0;

      // Every turn is persisted. On failure the output points at the prior
      // successful turn's `done` snapshot (the last-good state), not the failed
      // turn's snapshot. No extra recovery snapshot is written - the last good
      // turn is already persisted.
      const flow = defineCustomAgent<{ count: number }>(
        new Registry(),
        { name: 'frServerRecoveryDefault', store },
        async (sess) => {
          await sess.run(async () => {
            turn++;
            sess.session.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            if (turn === 2) {
              throw new Error('default cb boom');
            }
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'one' }] },
      });
      session.send({
        message: { role: 'user' as const, content: [{ text: 'two' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }
      const output = await session.output;

      assert.strictEqual(output.finishReason, 'failed');
      assert.ok(output.snapshotId);

      // Collect the per-turn snapshot ids: turn 1 (done) then turn 2 (failed).
      const turnEndChunks = chunks.filter((c) => !!c.turnEnd);
      const successfulTurnSnapshotId = turnEndChunks[0]?.turnEnd?.snapshotId;
      const failedTurnSnapshotId =
        turnEndChunks[turnEndChunks.length - 1]?.turnEnd?.snapshotId;
      assert.ok(successfulTurnSnapshotId);
      assert.ok(failedTurnSnapshotId);
      assert.notStrictEqual(successfulTurnSnapshotId, failedTurnSnapshotId);

      // The failed turn's snapshot records the partial (failed-turn) state and
      // is persisted (inspectable) but is NOT resumable.
      const failedSnap = await store.getSnapshot({
        snapshotId: failedTurnSnapshotId!,
      });
      assert.strictEqual(failedSnap?.status, 'failed');
      assert.strictEqual((failedSnap!.state.custom as any).count, 2);

      // The returned snapshotId is the last-good (turn 1) `done` snapshot - no
      // separate recovery snapshot is written.
      assert.strictEqual(output.snapshotId, successfulTurnSnapshotId);
      const lastGood = await store.getSnapshot({
        snapshotId: output.snapshotId!,
      });
      assert.strictEqual(lastGood?.status, 'completed');
      assert.strictEqual((lastGood!.state.custom as any).count, 1);

      // The raw store still returns the failed leaf for a sessionId lookup -
      // resumability is enforced by the agent, not the store.
      const bySession = await store.getSnapshot({
        sessionId: lastGood!.state.sessionId!,
      });
      assert.strictEqual(bySession?.snapshotId, failedTurnSnapshotId);

      // But resuming by sessionId walks back over the failed leaf to the
      // last-good `done` snapshot: the next turn continues from count 1 (not the
      // failed turn's partial count 2) and chains from the last-good snapshot.
      const sessionId = lastGood!.state.sessionId!;
      const resumeSession = flow.streamBidi({ sessionId });
      resumeSession.send({
        message: { role: 'user' as const, content: [{ text: 'three' }] },
      });
      resumeSession.close();
      for await (const _ of resumeSession.stream) {
      }
      const resumeOutput = await resumeSession.output;
      assert.strictEqual(resumeOutput.finishReason, undefined);
      assert.ok(resumeOutput.snapshotId);
      const resumed = await store.getSnapshot({
        snapshotId: resumeOutput.snapshotId!,
      });
      // count 1 (last-good) + 1 (this turn) = 2, and it chains from the
      // last-good snapshot, not the failed leaf.
      assert.strictEqual((resumed!.state.custom as any).count, 2);
      assert.strictEqual(resumed?.parentId, successfulTurnSnapshotId);
    });

    it('server-managed: resuming a non-done (failed) snapshot is rejected', async () => {
      const store = new InMemorySessionStore<{ count: number }>();

      const flow = defineCustomAgent<{ count: number }>(
        new Registry(),
        { name: 'frResumeNonDone', store },
        async (sess) => {
          await sess.run(async () => {
            sess.session.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            throw new Error('boom');
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      // First turn fails, persisting a `failed` snapshot.
      const session1 = flow.streamBidi({});
      session1.send({
        message: { role: 'user' as const, content: [{ text: 'one' }] },
      });
      session1.close();
      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session1.stream) {
        chunks.push(chunk);
      }
      await session1.output;

      const failedTurnSnapshotId = chunks.filter((c) => !!c.turnEnd).pop()
        ?.turnEnd?.snapshotId;
      assert.ok(failedTurnSnapshotId);

      // Resuming that failed snapshot by snapshotId is rejected.
      const session2 = flow.streamBidi({ snapshotId: failedTurnSnapshotId });
      session2.send({
        message: { role: 'user' as const, content: [{ text: 'two' }] },
      });
      session2.close();
      for await (const _ of session2.stream) {
      }
      const output2 = await session2.output;

      assert.strictEqual(output2.finishReason, 'failed');
      assert.ok(output2.error);
      assert.strictEqual(output2.error!.status, 'INVALID_ARGUMENT');
      assert.ok(
        output2.error!.message.includes('not resumable'),
        `Expected not-resumable error, got: ${output2.error!.message}`
      );
    });

    it('server-managed: no redundant recovery snapshot when the first turn fails', async () => {
      const store = new InMemorySessionStore<{ count: number }>();

      // The very first turn fails - there is no prior successful turn, so the
      // last-good state is the seed the client already has. Writing a recovery
      // snapshot would be a redundant no-diff write, so none is created and
      // snapshotId is left unset.
      const flow = defineCustomAgent<{ count: number }>(
        new Registry(),
        { name: 'frServerRecoveryFirstTurn', store },
        async (sess) => {
          await sess.run(async () => {
            sess.session.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            throw new Error('first turn boom');
          });
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'one' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }
      const output = await session.output;

      assert.strictEqual(output.finishReason, 'failed');
      // No recovery snapshot is returned for a first-turn failure.
      assert.strictEqual(output.snapshotId, undefined);

      // The only snapshot in the store is the failed turn's own snapshot - no
      // separate 'done' seed recovery snapshot was written.
      const turnEndChunks = chunks.filter((c) => !!c.turnEnd);
      const failedTurnSnapshotId =
        turnEndChunks[turnEndChunks.length - 1]?.turnEnd?.snapshotId;
      assert.ok(failedTurnSnapshotId);
      const failedSnap = await store.getSnapshot({
        snapshotId: failedTurnSnapshotId!,
      });
      assert.strictEqual(failedSnap?.status, 'failed');
    });

    it('surfaces the generate finishReason from a prompt agent', async () => {
      const registry = new Registry();
      registry.apiStability = 'beta';
      const pm = defineProgrammableModel(registry, undefined, 'frModel');

      definePrompt(registry, {
        name: 'frPrompt',
        model: 'frModel',
        config: { temperature: 1 },
      });

      const flow = definePromptAgent(registry, {
        promptName: 'frPrompt',
      });

      pm.handleResponse = async () => ({
        message: { role: 'model', content: [{ text: 'hello' }] },
        finishReason: 'stop',
      });

      const session = flow.streamBidi({});
      session.send({
        message: { role: 'user' as const, content: [{ text: 'hi' }] },
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const turnEndChunk = chunks.find((c) => !!c.turnEnd);
      assert.strictEqual(turnEndChunk?.turnEnd?.finishReason, 'stop');

      const output = await session.output;
      assert.strictEqual(output.finishReason, 'stop');
    });
  });

  // -------------------------------------------------------------------------
  // AgentAPI surface (`chat`, `loadChat`, `getSnapshot`, `abort`) - the same
  // ergonomic interface returned by `remoteAgent` on the client, but driven
  // in-process over the agent action.
  // -------------------------------------------------------------------------
  describe('AgentAPI (server-side chat)', () => {
    it('streams text and resolves a response (client-managed)', async () => {
      const agent = defineCustomAgent(
        new Registry(),
        { name: 'apiStream' },
        async (sess, { sendChunk }) => {
          await sess.run(async () => {
            sendChunk({
              modelChunk: { role: 'model', content: [{ text: 'Hello ' }] },
            });
            sendChunk({
              modelChunk: { role: 'model', content: [{ text: 'world' }] },
            });
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'Hello world' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const chat = agent.chat();
      const turn = chat.sendStream('hi');

      const seen: string[] = [];

      for await (const chunk of turn.stream) {
        if (chunk.text) seen.push(chunk.accumulatedText);
      }
      assert.deepEqual(seen, ['Hello ', 'Hello world']);

      const res = await turn.response;
      assert.strictEqual(res.text, 'Hello world');
      assert.strictEqual(res.finishReason, 'stop');
    });

    it('carries state across multi-turn (client-managed)', async () => {
      const agent = defineCustomAgent<{ count: number }>(
        new Registry(),
        { name: 'apiState' },
        async (sess) => {
          await sess.run(async () => {
            sess.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'ok' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const chat = agent.chat();
      const res1 = await chat.send('one');
      assert.deepEqual(res1.state, { count: 1 });
      assert.deepEqual(chat.state, { count: 1 });
      const res2 = await chat.send('two');
      assert.deepEqual(res2.state, { count: 2 });

      assert.deepEqual(chat.state, { count: 2 });
    });

    it('carries snapshotId across multi-turn (server-managed)', async () => {
      const store = new InMemorySessionStore<{}>();
      const agent = defineCustomAgent<{}>(
        new Registry(),
        { name: 'apiSnap', store },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'ok' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const chat = agent.chat();
      await chat.send('one');
      const firstSnapshotId = chat.snapshotId;
      assert.ok(firstSnapshotId);
      await chat.send('two');
      assert.ok(chat.snapshotId);
      assert.notStrictEqual(chat.snapshotId, firstSnapshotId);
    });

    it('exposes a stable sessionId across multi-turn (server-managed)', async () => {
      const store = new InMemorySessionStore<{}>();
      const agent = defineCustomAgent<{}>(
        new Registry(),
        { name: 'apiSession', store },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'ok' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const chat = agent.chat();
      // No sessionId before the first turn (a fresh, unsent chat).
      assert.strictEqual(chat.sessionId, undefined);

      const res1 = await chat.send('one');
      const sessionId = chat.sessionId;
      assert.ok(sessionId, 'sessionId should be set after the first turn');
      // res.sessionId mirrors chat.sessionId.
      assert.strictEqual(res1.sessionId, sessionId);

      // The sessionId is stable across subsequent turns (the snapshotId is not).
      const res2 = await chat.send('two');
      assert.strictEqual(chat.sessionId, sessionId);
      assert.strictEqual(res2.sessionId, sessionId);
    });

    it('keeps chat.state/res.state live across a non-streaming send() (server-managed)', async () => {
      // Server-managed agents return only a snapshotId on the wire (no custom
      // `state`); custom state reaches the client via streamed customPatch
      // chunks. send() must drain the stream so chat.state stays live after a
      // non-streaming turn, and res.state must mirror it.
      const store = new InMemorySessionStore<{ count: number }>();
      const agent = defineCustomAgent<{ count: number }>(
        new Registry(),
        { name: 'apiCounter', store },
        async (sess) => {
          await sess.run(async () => {
            sess.updateCustom((c) => ({ count: (c?.count ?? 0) + 1 }));
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'ok' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const chat = agent.chat();
      const res1 = await chat.send('inc');
      assert.deepEqual(chat.state, { count: 1 });
      // res.state mirrors chat.state even though the wire output omits `state`.
      assert.deepEqual(res1.state, { count: 1 });

      // A second non-streaming send() must refresh chat.state (not go stale).
      const res2 = await chat.send('inc');
      assert.deepEqual(chat.state, { count: 2 });
      assert.deepEqual(res2.state, { count: 2 });
    });

    it('exposes interrupts and builds resume parts', async () => {
      const agent = defineCustomAgent(
        new Registry(),
        { name: 'apiInterrupt' },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'interrupted' as const };
          });
          return {
            message: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    name: 'userApproval',
                    ref: 'r1',
                    input: { action: 'transfer' },
                  },
                  metadata: { interrupt: true },
                },
              ],
            },
            finishReason: 'interrupted' as const,
          };
        }
      );

      const chat = agent.chat();
      const res = await chat.send('Transfer $500');
      assert.strictEqual(res.finishReason, 'interrupted');

      assert.strictEqual(res.interrupts.length, 1);
      const approval = res.interrupts[0];
      assert.strictEqual(approval.name, 'userApproval');
      assert.deepEqual(approval.input, { action: 'transfer' });

      assert.deepEqual(approval.respond({ approved: true }), {
        toolResponse: {
          name: 'userApproval',
          ref: 'r1',
          output: { approved: true },
        },
      });
      assert.deepEqual(approval.restart(), {
        toolRequest: {
          name: 'userApproval',
          ref: 'r1',
          input: { action: 'transfer' },
        },
      });
    });

    it('throws AgentError on a failed turn', async () => {
      const store = new InMemorySessionStore<{}>();
      const agent = defineCustomAgent<{}>(
        new Registry(),
        { name: 'apiFailed', store },
        async (sess) => {
          await sess.run(async () => {
            throw new Error('boom');
          });
          return { message: { role: 'model', content: [{ text: 'ok' }] } };
        }
      );

      const chat = agent.chat();
      await assert.rejects(
        () => chat.send('hi'),
        (err: unknown) => {
          assert.ok(err instanceof AgentError);
          assert.strictEqual(err.status, 'INTERNAL');
          assert.ok(err.message.includes('boom'));
          return true;
        }
      );
    });

    it('loadChat() restores history from a snapshot (server-managed)', async () => {
      const store = new InMemorySessionStore<{}>();
      const agent = defineCustomAgent<{}>(
        new Registry(),
        { name: 'apiLoad', store },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'reply' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const chat = agent.chat();
      await chat.send('earlier');
      const snapshotId = chat.snapshotId!;
      assert.ok(snapshotId);

      const restored = await agent.loadChat({ snapshotId });
      assert.strictEqual(restored.snapshotId, snapshotId);
      assert.ok(restored.messages.length >= 1);
      assert.strictEqual(restored.messages[0].content[0].text, 'earlier');
    });

    it('getSnapshot reads a snapshot (server-managed)', async () => {
      const store = new InMemorySessionStore<{}>();
      const agent = defineCustomAgent<{}>(
        new Registry(),
        { name: 'apiGetSnap', store },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'ok' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const chat = agent.chat();
      await chat.send('hi');
      const snapshotId = chat.snapshotId!;
      const snap = await agent.getSnapshot(snapshotId);

      assert.strictEqual(snap?.snapshotId, snapshotId);
    });

    it('detach submits a background task that completes', async () => {
      const store = new InMemorySessionStore<{}>();
      const agent = defineCustomAgent<{}>(
        new Registry(),
        { name: 'apiDetach', store },
        async (sess) => {
          await sess.run(async () => {
            return { finishReason: 'stop' as const };
          });
          return {
            message: { role: 'model', content: [{ text: 'ok' }] },
            finishReason: 'stop' as const,
          };
        }
      );

      const chat = agent.chat();
      const task = await chat.detach('long job');
      assert.ok(task.snapshotId);
      const snap = await task.wait({ intervalMs: 1 });
      assert.ok(snap.status);
    });
  });
});
