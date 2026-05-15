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
import { getDatabase } from 'firebase-admin/database';
import { RtdbStreamManager } from '../src/stream-manager/rtdb';

describe('RtdbStreamManager', () => {
  let app: App;
  let streamManager: RtdbStreamManager;

  beforeEach(() => {
    app = initializeApp({
      databaseURL: 'http://127.0.0.1:9000?ns=genkit-test',
    });
    streamManager = new RtdbStreamManager({
      firebaseApp: app,
    });
  });

  afterEach(async () => {
    await deleteApp(app);
  });

  it('should open a stream and write chunks', async () => {
    const streamId = 'test-stream-1';
    const stream = await streamManager.open(streamId);

    await stream.write({ foo: 'bar' });
    await stream.write({ bar: 'baz' });

    const db = getDatabase(app);
    const snapshot = await db.ref(`genkit-streams/${streamId}`).get();
    const data = snapshot.val();
    const values = Object.values(data.stream);

    assert.deepStrictEqual(values, [
      { type: 'chunk', chunk: { foo: 'bar' } },
      { type: 'chunk', chunk: { bar: 'baz' } },
    ]);
    assert.ok(data.metadata.createdAt);
    assert.ok(data.metadata.updatedAt);
  });

  it('should open a stream and write done', async () => {
    const streamId = 'test-stream-2';
    const stream = await streamManager.open(streamId);

    await stream.done({ result: 'success' });

    const db = getDatabase(app);
    const snapshot = await db.ref(`genkit-streams/${streamId}`).get();
    const data = snapshot.val();
    const values = Object.values(data.stream);

    assert.deepStrictEqual(values, [
      { type: 'done', output: { result: 'success' } },
    ]);
    assert.ok(data.metadata.createdAt);
    assert.ok(data.metadata.updatedAt);
  });

  it('should open a stream and write error', async () => {
    const streamId = 'test-stream-3';
    const stream = await streamManager.open(streamId);

    await stream.error(new Error('test error'));

    const db = getDatabase(app);
    const snapshot = await db.ref(`genkit-streams/${streamId}`).get();
    const data = snapshot.val();
    const values = Object.values(data.stream) as any[];

    assert.strictEqual(values.length, 1);
    assert.strictEqual(values[0].type, 'error');
    assert.strictEqual(values[0].err.message, 'test error');
    assert.ok(data.metadata.createdAt);
    assert.ok(data.metadata.updatedAt);
  });

  it('should subscribe to a stream', (done) => {
    const streamId = 'test-stream-4';
    const chunks: any[] = [];
    let finalOutput: any;

    const db = getDatabase(app);
    const streamRef = db.ref(`genkit-streams/${streamId}`);
    streamRef.set({ metadata: { createdAt: Date.now() } }).then(async () => {
      await streamManager.subscribe(streamId, {
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

      const streamDataRef = streamRef.child('stream');
      streamDataRef.push({ type: 'chunk', chunk: { foo: 'bar' } });
      streamDataRef.push({ type: 'chunk', chunk: { bar: 'baz' } });
      streamDataRef.push({ type: 'done', output: { result: 'success' } });
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
    streamManager = new RtdbStreamManager({
      firebaseApp: app,
      refPrefix: 'genkit-streams',
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
