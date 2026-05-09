/**
 * Copyright 2025 Google LLC
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

import { afterEach, beforeEach, describe, it } from '@jest/globals';
import * as assert from 'assert';
import { deleteApp, initializeApp, type App } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { FirestoreStreamManager } from '../src/stream-manager/firestore';

describe('FirestoreStreamManager', () => {
  let app: App;
  let streamManager: FirestoreStreamManager;

  beforeEach(() => {
    process.env.FIRESTORE_EMULATOR_HOST = '127.0.0.1:8080';
    app = initializeApp({
      projectId: 'genkit-test',
    });
    streamManager = new FirestoreStreamManager({
      firebaseApp: app,
      collection: 'genkit-streams',
    });
  });

  afterEach(async () => {
    const db = getFirestore(app);
    const collections = await db.listCollections();
    for (const collection of collections) {
      await db.recursiveDelete(collection);
    }
    await deleteApp(app);
  });

  it('should open a stream and write chunks', async () => {
    const streamId = 'test-stream-1';
    const stream = await streamManager.open(streamId);

    await stream.write({ foo: 'bar' });
    await stream.write({ bar: 'baz' });

    const db = getFirestore(app);
    const snapshot = await db.collection('genkit-streams').doc(streamId).get();
    const data = snapshot.data();
    assert.ok(data);

    assert.strictEqual(data.stream.length, 2);
    assert.strictEqual(data.stream[0].type, 'chunk');
    assert.deepStrictEqual(data.stream[0].chunk, { foo: 'bar' });
    assert.ok(data.stream[0].uuid);
    assert.strictEqual(data.stream[1].type, 'chunk');
    assert.deepStrictEqual(data.stream[1].chunk, { bar: 'baz' });
    assert.ok(data.stream[1].uuid);
    assert.ok(data.createdAt);
    assert.ok(data.updatedAt);
  });

  it('should preserve duplicate chunks', async () => {
    const streamId = 'test-stream-dupes';
    const stream = await streamManager.open(streamId);

    await stream.write({ foo: 'bar' });
    await stream.write({ foo: 'bar' });

    const db = getFirestore(app);
    const snapshot = await db.collection('genkit-streams').doc(streamId).get();
    const data = snapshot.data();
    assert.ok(data);

    assert.strictEqual(data.stream.length, 2);
    assert.deepStrictEqual(data.stream[0].chunk, { foo: 'bar' });
    assert.deepStrictEqual(data.stream[1].chunk, { foo: 'bar' });
    assert.notStrictEqual(data.stream[0].uuid, data.stream[1].uuid);
  });

  it('should open a stream and write done', async () => {
    const streamId = 'test-stream-2';
    const stream = await streamManager.open(streamId);

    await stream.done({ result: 'success' });

    const db = getFirestore(app);
    const snapshot = await db.collection('genkit-streams').doc(streamId).get();
    const data = snapshot.data();
    assert.ok(data);

    assert.deepStrictEqual(data.stream, [
      { type: 'done', output: { result: 'success' } },
    ]);
    assert.ok(data.createdAt);
    assert.ok(data.updatedAt);
  });

  it('should open a stream and write error', async () => {
    const streamId = 'test-stream-3';
    const stream = await streamManager.open(streamId);

    await stream.error(new Error('test error'));

    const db = getFirestore(app);
    const snapshot = await db.collection('genkit-streams').doc(streamId).get();
    const data = snapshot.data();
    assert.ok(data);

    assert.strictEqual(data.stream.length, 1);
    assert.strictEqual(data.stream[0].type, 'error');
    assert.strictEqual(data.stream[0].err.message, 'test error');
    assert.ok(data.createdAt);
    assert.ok(data.updatedAt);
  });

  it('should subscribe to a stream', (done) => {
    const streamId = 'test-stream-4';
    const chunks: any[] = [];
    let finalOutput: any;

    const db = getFirestore(app);
    const streamDoc = db.collection('genkit-streams').doc(streamId);
    streamDoc
      .set({
        stream: [
          { type: 'chunk', chunk: { foo: 'bar' } },
          { type: 'chunk', chunk: { bar: 'baz' } },
          { type: 'done', output: { result: 'success' } },
        ],
      })
      .then(() => {
        streamManager.subscribe(streamId, {
          onChunk: (chunk) => {
            chunks.push(chunk);
          },
          onDone: (output) => {
            finalOutput = output;
            assert.deepStrictEqual(chunks, [{ foo: 'bar' }, { bar: 'baz' }]);
            assert.deepStrictEqual(finalOutput, { result: 'success' });
            done();
          },
          onError(error) {
            console.log(error);
          },
        });
      });
  });

  it('should throw an error when subscribing to a non-existent stream', async () => {
    await assert.rejects(
      streamManager.subscribe('non-existent-stream', {
        onChunk: () => {},
        onDone: () => {},
        onError: () => {},
      }),
      (err: any) => {
        assert.strictEqual(err.name, 'StreamNotFoundError');
        return true;
      }
    );
  });

  it('should time out when no chunks are received', (done) => {
    const streamId = 'test-stream-5';
    streamManager = new FirestoreStreamManager({
      firebaseApp: app,
      collection: 'genkit-streams',
      timeout: 100,
    });
    streamManager.open(streamId).then(() => {
      streamManager.subscribe(streamId, {
        onChunk: () => {
          assert.fail('should not have received a chunk');
        },
        onDone: () => {
          assert.fail('should not have received done');
        },
        onError: (err) => {
          assert.strictEqual(err.status, 'DEADLINE_EXCEEDED');
          done();
        },
      });
    });
  });
});
