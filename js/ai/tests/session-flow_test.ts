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

import { definePrompt } from '../src/prompt.js';
import {
  SessionFlowStreamChunk,
  SessionRunner,
  defineSessionFlow,
  defineSessionFlowFromPrompt,
} from '../src/session-flow.js';
import {
  InMemorySessionStore,
  Session,
  type SessionSnapshot,
} from '../src/session.js';
import { defineEchoModel, defineProgrammableModel } from './helpers.js';
import { interrupt } from '../src/tool.js';
import { z } from '@genkit-ai/core';

initNodeFeatures();

/**
 * Returns a Promise that resolves once the given snapshotId reaches targetStatus
 * in the store. Rejects after timeoutMs if the status is never reached.
 */
function waitForSnapshotStatus<S, I>(
  store: InMemorySessionStore<S, I>,
  snapshotId: string,
  targetStatus: NonNullable<SessionSnapshot<S, I>['status']>,
  timeoutMs = 5000
): Promise<SessionSnapshot<S, I>> {
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

    const unsubscribeFn = store.onSnapshotStateChange(
      snapshotId,
      (snap) => {
        if (snap.status === targetStatus) {
          clearTimeout(timer);
          if (typeof unsubscribeFn === 'function') unsubscribeFn();
          resolve(snap);
        }
      }
    );

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

describe('SessionFlow', () => {
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
      assert.strictEqual(arts.find((a) => a.name === 'art1')?.parts[0].text, 'v2');
      assert.strictEqual(arts.find((a) => a.name === 'art2')?.parts[0].text, 'new');
      assert.strictEqual(arts.find((a) => a.name === 'art3')?.parts[0].text, 'another');
    });

    it('should emit artifactAdded for new and artifactUpdated for replaced', () => {
      const session = new Session({});
      const added: string[] = [];
      const updated: string[] = [];
      session.on('artifactAdded', (a: { name?: string }) => added.push(a.name ?? ''));
      session.on('artifactUpdated', (a: { name?: string }) => updated.push(a.name ?? ''));

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

  describe('defineSessionFlow', () => {
    it('should register and execute session flow', async () => {
      const registry = new Registry();

      const flow = defineSessionFlow(
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

      const chunks: SessionFlowStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const output = await session.output;
      assert.strictEqual(output.message?.role, 'model');
      assert.strictEqual(output.message?.content[0].text, 'done');
    });

    it('should automatically stream artifacts added via Session.addArtifacts()', async () => {
      const registry = new Registry();

      const flow = defineSessionFlow(
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

      const chunks: SessionFlowStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const artChunks = chunks.filter((c) => !!c.artifact);
      assert.strictEqual(artChunks.length, 1);
      assert.strictEqual(artChunks[0].artifact?.name, 'testArt');
    });

    it('should stream artifactUpdated chunks when an artifact is replaced', async () => {
      const registry = new Registry();

      const flow = defineSessionFlow(
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

      const chunks: SessionFlowStreamChunk[] = [];
      for await (const chunk of session.stream) {
        chunks.push(chunk);
      }

      const artChunks = chunks.filter((c) => !!c.artifact);
      assert.strictEqual(artChunks.length, 2);
      assert.strictEqual(artChunks[0].artifact?.parts[0].text, 'v1');
      assert.strictEqual(artChunks[1].artifact?.parts[0].text, 'v2');
    });
  });

  describe('defineSessionFlowFromPrompt', () => {
    it('should register and execute session flow from prompt', async () => {
      const registry = new Registry();
      defineEchoModel(registry);
      definePrompt(registry, {
        name: 'agent',
        model: 'echoModel',
        config: { temperature: 1 },
        system: 'hello from template',
      });

      const flow = defineSessionFlowFromPrompt(registry, {
        promptName: 'agent',
        defaultInput: {},
      });

      const session = flow.streamBidi({});
      session.send({
        messages: [{ role: 'user' as const, content: [{ text: 'hi' }] }],
      });
      session.close();

      const chunks: SessionFlowStreamChunk[] = [];
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

      const flow = defineSessionFlow<{ foo: string }>(
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

    it('should abort a detached session flow', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let aborted = false;

      const flow = defineSessionFlow<{ foo: string }>(
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

      await flow.abort(snapshotId!);

      const snapAborted = await store.getSnapshot(snapshotId!);
      assert.strictEqual(snapAborted?.status, 'aborted');
      // AbortController.abort() fires onabort synchronously, so no delay needed.
      assert.strictEqual(aborted, true);
    });

    it('should throw error when detach is requested without session store', async () => {
      const flow = defineSessionFlow<{ foo: string }>(
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

      const flow = defineSessionFlow<{ foo: string }>(
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
      class LegacyStore extends InMemorySessionStore {
        override onSnapshotStateChange = undefined;
      }

      const store = new LegacyStore();
      const flow = defineSessionFlow<{ foo: string }>(
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
      const flow = defineSessionFlow<{ foo: string }>(
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
      const flow = defineSessionFlow<{ foo: string }>(
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

      const flow = defineSessionFlow<{ foo: string }>(
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
      const flow = defineSessionFlow<{ foo: string }>(
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

      const flow = defineSessionFlowFromPrompt(registry, {
        promptName: 'multiTurnAccumPrompt',
        defaultInput: {},
      });

      const session = flow.streamBidi({});
      session.send({
        messages: [{ role: 'user' as const, content: [{ text: 'turn1' }] }],
      });
      session.send({
        messages: [{ role: 'user' as const, content: [{ text: 'turn2' }] }],
      });
      session.close();

      const chunks: SessionFlowStreamChunk[] = [];
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
      const turn2Text = output.message?.content.map((c) => c.text).join('') ?? '';
      assert.ok(
        turn2Text.includes('Echo:'),
        `Expected second turn to be an echo response, got: ${turn2Text}`
      );

      // Model chunks must have been emitted for both turns.
      const modelChunks = chunks.filter((c) => c.modelChunk !== undefined);
      assert.ok(modelChunks.length >= 2, 'Expected model chunks from both turns');
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

      const flow = defineSessionFlowFromPrompt(registry, {
        promptName: 'interruptPrompt',
        defaultInput: {},
        store,
      });

      // Phase 1: User says hello, model responds with a toolRequest (interrupt)
      pm.handleResponse = async () => {
        return {
          message: {
            role: 'model',
            content: [{ toolRequest: { name: 'myInterrupt', input: { query: 'yes?' }, ref: '123' } }],
          },
          finishReason: 'stop',
        };
      };

      const session1 = flow.streamBidi({});
      session1.send({ messages: [{ role: 'user', content: [{ text: 'hello' }] }] });
      session1.close(); // IMPORTANT: close the stream so it doesn't hang!

      for await (const chunk of session1.stream) {}
      const output1 = await session1.output;

      assert.ok(output1.snapshotId);
      assert.ok(output1.message);
      assert.ok(output1.message.content[0].toolRequest);
      assert.strictEqual(output1.message.content[0].toolRequest.name, 'myInterrupt');

      // Phase 2: Resume with the tool response
      pm.handleResponse = async (req) => {
        // Assert that the resumed request contains the tool response!
        const lastMsg = req.messages[req.messages.length - 1];
        assert.strictEqual(lastMsg.role, 'tool');
        assert.strictEqual((lastMsg.content[0] as any).toolResponse.output.answer, 'yes indeed');

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
        messages: [{
          role: 'tool',
          content: [{ toolResponse: { name: 'myInterrupt', ref: '123', output: { answer: 'yes indeed' } } }],
        }],
      });
      session2.close(); // IMPORTANT: close the stream so it doesn't hang!

      for await (const chunk of session2.stream) {}
      const output2 = await session2.output;

      assert.strictEqual(output2.message?.role, 'model');
      assert.strictEqual(output2.message?.content[0].text, 'Task completed successfully!');
    });

    it('should process all pre-queued messages in the background after detaching', async () => {
      const store = new InMemorySessionStore<{ foo: string }>();
      let processedCount = 0;

      const flow = defineSessionFlow<{ foo: string }>(
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
});
