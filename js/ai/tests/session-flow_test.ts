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
import * as assert from 'assert';
import { describe, it } from 'node:test';

import { z } from '@genkit-ai/core';
import { definePrompt } from '../src/prompt.js';
import {
  AgentStreamChunk,
  SessionRunner,
  defineAgent,
  defineCustomAgent,
  definePromptAgent,
} from '../src/session-flow.js';
import {
  InMemorySessionStore,
  Session,
  type SessionSnapshot,
} from '../src/session.js';
import { interrupt } from '../src/tool.js';
import {
  defineEchoModel,
  defineProgrammableModel,
  type ProgrammableModel,
} from './helpers.js';

initNodeFeatures();

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
    store.getSnapshot(snapshotId).then((snap) => {
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

  describe('InMemorySessionStore', () => {
    it('should save and get snapshots', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      const snapshot = {
        snapshotId: 'snap-123',
        createdAt: new Date().toISOString(),
        event: 'turnEnd' as const,
        state: { custom: { foo: 'bar' } },
      };
      await store.saveSnapshot(snapshot);

      const got = await store.getSnapshot('snap-123');
      assert.deepStrictEqual(got, snapshot);
    });

    it('should return undefined for missing snapshot', async () => {
      const store = new InMemorySessionStore();
      const got = await store.getSnapshot('missing');
      assert.strictEqual(got, undefined);
    });

    it('should deep copy on save and get', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      const state = { foo: 'bar' };
      const snapshot = {
        snapshotId: 'snap-123',
        createdAt: new Date().toISOString(),
        event: 'turnEnd' as const,
        state: { custom: state },
      };
      await store.saveSnapshot(snapshot);

      // Mutate local state
      state.foo = 'baz';

      const got = await store.getSnapshot('snap-123');
      assert.strictEqual(got?.state.custom?.foo, 'bar');
    });
  });

  describe('SessionRunner', () => {
    it('should loop over inputs and call handler', async () => {
      const session = new Session({});
      const inputs = [
        { messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }] },
        { messages: [{ role: 'user' as const, content: [{ text: 'bye' }] }] },
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
        { messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }] },
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

      const saved = await store.getSnapshot(turnSnapshotId!);
      assert.ok(saved);
      assert.strictEqual(saved?.snapshotId, turnSnapshotId);
    });

    it('should respect snapshot callback', async () => {
      const store = new InMemorySessionStore();
      const session = new Session({});
      const inputs = [
        { messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }] },
      ];

      async function* inputGen() {
        for (const input of inputs) {
          yield input;
        }
      }

      const runner = new SessionRunner(session, inputGen(), {
        store,
        snapshotCallback: () => false, // Never snapshot
      });

      await runner.run(async () => {});

      // Verify the store is empty (callback suppressed all snapshots).
      const onEndTurnSnapshotId = await new Promise<string | undefined>(
        (resolve) => {
          const r = new SessionRunner(session, inputGen(), {
            store,
            onEndTurn: resolve,
          });
          r.run(async () => {}).catch(() => {});
        }
      );
      // The callback-suppressed runner should have produced no entries.
      const keys = Array.from((store as any).snapshots.keys()) as string[];
      // Only the snapshot from the second (non-callback) runner should exist.
      assert.ok(keys.every((k) => k === onEndTurnSnapshotId));
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
      assert.strictEqual(agent.__action.metadata?.stateManagement, 'client');
      assert.strictEqual(agent.__action.metadata?.abortable, false);
    });

    it('should set server stateManagement and abortable=true when store with onSnapshotStateChange is provided', () => {
      const registry = new Registry();
      const store = new InMemorySessionStore();
      const agent = defineCustomAgent(
        registry,
        { name: 'fullStoreMetadataTest', store },
        async () => ({ artifacts: [] })
      );
      assert.strictEqual(agent.__action.metadata?.stateManagement, 'server');
      assert.strictEqual(agent.__action.metadata?.abortable, true);
    });

    it('should ignore init.state for server-managed agents (store is set)', async () => {
      const registry = new Registry();
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<unknown, { foo: string }>(
        registry,
        { name: 'ignoreInitStateTest', store },
        async (sess) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'done' }] },
          };
        }
      );

      // Pass init.state with messages — it should be ignored because store is set
      const session = flow.streamBidi({
        state: {
          custom: { foo: 'should-be-ignored' },
          messages: [{ role: 'user', content: [{ text: 'stale history' }] }],
          artifacts: [],
        },
      });
      session.send({
        messages: [{ role: 'user', content: [{ text: 'hello' }] }],
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;

      // Verify state was NOT seeded from init.state
      const snapshot = await store.getSnapshot(output.snapshotId!);
      assert.ok(snapshot);
      // Only the message sent via input should be present, not the stale history
      assert.strictEqual(snapshot!.state.messages!.length, 1);
      assert.strictEqual(snapshot!.state.messages![0].content[0].text, 'hello');
      // Custom state should be empty default, not the init.state value
      assert.deepStrictEqual(snapshot!.state.custom, {});
    });

    it('should use init.state for client-managed agents (no store)', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent<unknown, { foo: string }>(
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

      // Pass init.state — it should be used because no store is set
      const session = flow.streamBidi({
        state: {
          custom: { foo: 'seeded' },
          messages: [{ role: 'user', content: [{ text: 'prior msg' }] }],
          artifacts: [],
        },
      });
      session.send({
        messages: [{ role: 'user', content: [{ text: 'hello' }] }],
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
      assert.strictEqual(agent.__action.metadata?.stateManagement, 'server');
      assert.strictEqual(agent.__action.metadata?.abortable, false);
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
            assert.strictEqual(input.messages?.[0].role, 'user');
          });
          assert.ok(receivedInput);
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const session = flow.streamBidi({});

      session.send({
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
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
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
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
      session.send({ messages: [{ role: 'user', content: [{ text: 'go' }] }] });
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
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
      });
      session.close();

      const chunks: AgentStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const output = await session.output;
      assert.strictEqual(output.message?.role, 'model');
    });

    it('should detach asynchronously and continue execution in the background', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let resolvePromise: () => void = () => {};
      const releasePromise = new Promise<void>((resolve) => {
        resolvePromise = resolve;
      });

      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
        detach: true,
      });

      const output = await session.output;
      const snapshotId = output.snapshotId;
      assert.ok(snapshotId);

      const snapPending = await store.getSnapshot(snapshotId!);
      assert.strictEqual(snapPending?.status, 'pending');

      resolvePromise();
      session.close();

      const snapDone = await waitForSnapshotStatus(store, snapshotId!, 'done');
      assert.strictEqual(snapDone.status, 'done');
    });

    it('should abort a detached agent', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let aborted = false;

      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
        detach: true,
      });

      const output = await session.output;
      const snapshotId = output.snapshotId;
      assert.ok(snapshotId);

      const previousStatus = await flow.abort(snapshotId!);

      assert.strictEqual(previousStatus, 'pending');
      const snapAborted = await store.getSnapshot(snapshotId!);
      assert.strictEqual(snapAborted?.status, 'aborted');
      // AbortController.abort() fires onabort synchronously, so no delay needed.
      assert.strictEqual(aborted, true);
    });

    it('should return "done" when aborting an already-completed flow', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
      });
      session.close();
      const output = await session.output;
      assert.ok(output.snapshotId);

      // Snapshot should be 'done' now
      const snapBefore = await store.getSnapshot(output.snapshotId!);
      assert.strictEqual(snapBefore?.status, 'done');

      const previousStatus = await flow.abort(output.snapshotId!);
      assert.strictEqual(previousStatus, 'done');

      const snapAfter = await store.getSnapshot(output.snapshotId!);
      assert.strictEqual(snapAfter?.status, 'aborted');
    });

    it('should return undefined when aborting a non-existent snapshot', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();

      const flow = defineCustomAgent<unknown, { foo: string }>(
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
      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
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

      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
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
      }) as InMemorySessionStore;
      const flow = defineCustomAgent<unknown, { foo: string }>(
        new Registry(),
        {
          name: 'legacyStoreTest',
          store,
        },
        async (sess, { sendChunk }) => {
          await sess.run(async () => {});
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
        detach: true,
      });

      const output = await session.output;
      const snapshotId = output.snapshotId;

      await flow.abort(snapshotId!);

      const snapshot = await store.getSnapshot(snapshotId!);
      assert.strictEqual(snapshot?.status, 'aborted');
    });

    it('should fetch snapshot data via companion action', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      const flow = defineCustomAgent<unknown, { foo: string }>(
        new Registry(),
        {
          name: 'companionActionFlow',
          store,
        },
        async (sess) => {
          return {
            artifacts: [],
            message: { role: 'model', content: [{ text: 'hi' }] },
          };
        }
      );

      const session = flow.streamBidi({});
      session.send({
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
      });
      session.close();
      const output = await session.output;

      const snapData = await flow.getSnapshotData(output.snapshotId!);
      assert.strictEqual(snapData?.snapshotId, output.snapshotId);
    });

    it('should chain parentId properly across session snapshots', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [{ role: 'user' as const, content: [{ text: 'first' }] }],
      });
      session1.close();
      const output1 = await session1.output;

      const session2 = flow.streamBidi({
        snapshotId: output1.snapshotId,
      });

      session2.send({
        messages: [{ role: 'user' as const, content: [{ text: 'second' }] }],
      });
      session2.close();
      const output2 = await session2.output;

      const snapshot2 = await store.getSnapshot(output2.snapshotId!);
      assert.strictEqual(snapshot2?.parentId, output1.snapshotId);
    });

    it('should detach immediately when a detach input is queued', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let releasePromise: () => void = () => {};
      const blockPromise = new Promise<void>((resolve) => {
        releasePromise = resolve;
      });

      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [
          { role: 'user' as const, content: [{ text: 'heavy task' }] },
        ],
      });
      session.send({
        detach: true,
      });

      const output = await session.output;
      assert.ok(output.snapshotId);
      const snapshot = await store.getSnapshot(output.snapshotId!);
      assert.strictEqual(snapshot?.status, 'pending');

      releasePromise();
      session.close();
    });

    it('should process messages even when detach is present in the same payload', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [
          { role: 'user' as const, content: [{ text: 'appended message' }] },
        ],
        detach: true,
      });

      const output = await session.output;
      assert.ok(output.snapshotId);

      const snapDone = await waitForSnapshotStatus(
        store,
        output.snapshotId!,
        'done'
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
        messages: [{ role: 'user' as const, content: [{ text: 'turn1' }] }],
      });
      session.send({
        messages: [{ role: 'user' as const, content: [{ text: 'turn2' }] }],
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
        messages: [{ role: 'user', content: [{ text: 'hello' }] }],
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
        messages: [
          {
            role: 'tool',
            content: [
              {
                toolResponse: {
                  name: 'myInterrupt',
                  ref: '123',
                  output: { answer: 'yes indeed' },
                },
              },
            ],
          },
        ],
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

    it('should process all pre-queued messages in the background after detaching', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let processedCount = 0;

      const flow = defineCustomAgent<unknown, { foo: string }>(
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
        messages: [{ role: 'user' as const, content: [{ text: 'task 1' }] }],
      });
      session.send({
        messages: [{ role: 'user' as const, content: [{ text: 'task 2' }] }],
      });
      session.send({ detach: true });

      const output = await session.output;
      assert.ok(output.snapshotId);

      // Detach-only messages are not forwarded to the runner — 2 turns, not 3.
      const snapDone = await waitForSnapshotStatus(
        store,
        output.snapshotId!,
        'done'
      );
      assert.strictEqual(snapDone.status, 'done');
      assert.strictEqual(processedCount, 2);

      session.close();
    });
  });

  describe('clientStateTransform', () => {
    it('should transform state in AgentOutput for client-managed agents', async () => {
      const registry = new Registry();

      const flow = defineCustomAgent<
        unknown,
        { publicField: string; secretField: string }
      >(
        registry,
        {
          name: 'clientTransformTest',
          clientStateTransform: (state) => ({
            custom: { publicField: (state.custom as any)?.publicField },
            // Strip messages and artifacts
          }),
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
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
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

      const flow = defineCustomAgent<
        unknown,
        { publicField: string; secretField: string }
      >(registry, { name: 'noTransformTest' }, async (sess) => {
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
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
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

      const flow = defineCustomAgent<
        unknown,
        { publicField: string; secretField: string }
      >(
        new Registry(),
        {
          name: 'snapshotTransformTest',
          store,
          clientStateTransform: (state) => ({
            custom: { publicField: (state.custom as any)?.publicField },
          }),
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
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;
      assert.ok(output.snapshotId);

      // getSnapshotData should return transformed state
      const snapshot = await flow.getSnapshotData(output.snapshotId!);
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
      const rawSnapshot = await store.getSnapshot(output.snapshotId!);
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

      const flow = defineCustomAgent<
        unknown,
        { publicField: string; secretField: string }
      >(
        registry,
        {
          name: 'snapshotActionTransformTest',
          store,
          clientStateTransform: (state) => ({
            custom: { publicField: (state.custom as any)?.publicField },
          }),
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
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      });
      session.close();

      for await (const _ of session.stream) {
      }
      const output = await session.output;
      assert.ok(output.snapshotId);

      // Invoke the companion action directly
      const actionResult = await flow.getSnapshotDataAction(output.snapshotId!);
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
      const flow = defineCustomAgent<
        unknown,
        { publicField: string; secretField: string }
      >(
        new Registry(),
        {
          name: 'detachTransformTest',
          store,
          clientStateTransform: (state) => ({
            custom: { publicField: (state.custom as any)?.publicField },
          }),
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
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
        detach: true,
      });

      const output = await session.output;
      assert.ok(output.snapshotId);
      // Server-managed agents don't return state in output (state is undefined)
      // but the snapshot should have the transformed state
      const snapshot = await flow.getSnapshotData(output.snapshotId!);
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

    it('should pass clientStateTransform through definePromptAgent', async () => {
      const registry = new Registry();
      defineEchoModel(registry);
      definePrompt(registry, {
        name: 'transformPromptAgent',
        model: 'echoModel',
        config: { temperature: 1 },
      });

      const flow = definePromptAgent<{ secret: string }>(registry, {
        promptName: 'transformPromptAgent',
        clientStateTransform: (state) => ({
          // strip custom state entirely, keep messages
          messages: state.messages,
        }),
      });

      const session = flow.streamBidi({});
      session.send({
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
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

    it('should pass clientStateTransform through defineAgent', async () => {
      const registry = new Registry();
      defineEchoModel(registry);

      const flow = defineAgent<{ secret: string }>(registry, {
        name: 'transformDefineAgent',
        model: 'echoModel',
        config: { temperature: 1 },
        clientStateTransform: (state) => ({
          // strip custom state entirely, keep messages
          messages: state.messages,
        }),
      });

      const session = flow.streamBidi({});
      session.send({
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
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
          { messages: [{ role: 'user', content: [{ text: 'turn1' }] }] },
          { messages: [{ role: 'user', content: [{ text: 'turn2' }] }] },
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
          { messages: [{ role: 'user', content: [{ text: 'turn1' }] }] },
          { messages: [{ role: 'user', content: [{ text: 'turn2' }] }] },
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
        inputs: [
          { messages: [{ role: 'user', content: [{ text: 'first' }] }] },
        ],
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
        inputs: [
          { messages: [{ role: 'user', content: [{ text: 'second' }] }] },
        ],
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
          { messages: [{ role: 'user', content: [{ text: 'q1' }] }] },
          { messages: [{ role: 'user', content: [{ text: 'q2' }] }] },
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
          { messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
          { messages: [{ role: 'user', content: [{ text: 'how are you' }] }] },
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
});
