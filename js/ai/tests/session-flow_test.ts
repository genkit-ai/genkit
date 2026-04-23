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
import { InMemorySessionStore, Session } from '../src/session.js';
import { defineEchoModel } from './helpers.js';

initNodeFeatures();

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

      // Wait, how do we verify it didn't snapshot?
      // We can check if onEndTurn was called with undefined snapshotId.
      // Or we can check the store is empty.
      const keys = Array.from((store as any).snapshots.keys());
      assert.strictEqual(keys.length, 0);
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

      let snapDone: any | undefined;
      for (let i = 0; i < 20; i++) {
        snapDone = await store.getSnapshot(snapshotId!);
        if (snapDone?.status === 'done') {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 50));
      }

      assert.strictEqual(snapDone?.status, 'done');
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

      // wait for AbortSignal to fire in event loop
      await new Promise((resolve) => setTimeout(resolve, 100));
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

      let snapFailed: SessionSnapshot<any, any> | undefined;
      for (let i = 0; i < 20; i++) {
        snapFailed = await store.getSnapshot(snapshotId!);
        if (snapFailed?.status === 'failed') {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 50));
      }

      assert.strictEqual(snapFailed?.status, 'failed');
      assert.strictEqual(
        snapFailed?.error?.message,
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

      await new Promise((resolve) => setTimeout(resolve, 100));
      const snap = await store.getSnapshot(output.snapshotId!);
      assert.ok(snap?.state.messages);
      assert.strictEqual(snap?.state.messages.length, 1);
      assert.strictEqual(
        snap?.state.messages[0].content[0].text,
        'appended message'
      );

      session.close();
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

      await new Promise((resolve) => setTimeout(resolve, 200));
      assert.strictEqual(processedCount, 3);

      session.close();
    });
  });
});
