/**
 * Copyright 2024 Google LLC
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

import { LogRecordData } from '@genkit-ai/tools-common';
import * as assert from 'assert';
import * as fs from 'fs';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { LocalFileLogStore } from '../src/file-log-store';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('LocalFileLogStore', () => {
  const testRoot = path.join(__dirname, '.test_log_store');
  const indexRoot = path.join(testRoot, 'idx');
  const storeRoot = path.join(testRoot, 'logs');
  let logStore: LocalFileLogStore;

  beforeEach(async () => {
    if (fs.existsSync(testRoot)) {
      fs.rmSync(testRoot, { recursive: true, force: true });
    }
    logStore = new LocalFileLogStore({ storeRoot, indexRoot });
    await logStore.init();
  });

  afterEach(() => {
    if (fs.existsSync(testRoot)) {
      fs.rmSync(testRoot, { recursive: true, force: true });
    }
  });

  it('should store logs and retrieve them via traceId lookup', async () => {
    const mockLogs: LogRecordData[] = [
      {
        logId: 'log-1',
        traceId: 'trace-1',
        spanId: 'span-1',
        timestamp: 1000,
        severityText: 'INFO',
        body: 'First message',
      },
      {
        logId: 'log-2',
        traceId: 'trace-2',
        spanId: 'span-2',
        timestamp: 2000,
        severityText: 'ERROR',
        body: 'Second message',
      },
      {
        logId: 'log-3',
        traceId: 'trace-1',
        spanId: 'span-3',
        timestamp: 3000,
        severityText: 'DEBUG',
        body: 'Third message',
      },
    ];

    await logStore.save(mockLogs);

    const resultTrace1 = await logStore.list({ traceId: 'trace-1' });
    const resultTrace2 = await logStore.list({ traceId: 'trace-2' });

    // Returns newest first based on the LogIndex implementation
    assert.strictEqual(resultTrace1.logs.length, 2);
    assert.strictEqual(resultTrace1.logs[0].logId, 'log-3');
    assert.strictEqual(resultTrace1.logs[1].logId, 'log-1');

    assert.strictEqual(resultTrace2.logs.length, 1);
    assert.strictEqual(resultTrace2.logs[0].logId, 'log-2');
  });

  it('should retrieve logs by spanId', async () => {
    const mockLogs: LogRecordData[] = [
      {
        logId: 'log-1',
        traceId: 'trace-1',
        spanId: 'span-x',
        timestamp: 1000,
        body: 'hello',
      },
      {
        logId: 'log-2',
        traceId: 'trace-1',
        spanId: 'span-y',
        timestamp: 2000,
        body: 'world',
      },
    ];

    await logStore.save(mockLogs);

    const result = await logStore.list({ spanId: 'span-x' });
    assert.strictEqual(result.logs.length, 1);
    assert.strictEqual(result.logs[0].logId, 'log-1');
  });

  it('should populate logId if none belongs', async () => {
    const mockLogs: LogRecordData[] = [
      {
        traceId: 'trace-missing-id',
        spanId: 'span-missing-id',
        timestamp: 1234,
        body: 'generated ID test',
        logId: '',
      },
    ];

    await logStore.save(mockLogs);

    const result = await logStore.list({ traceId: 'trace-missing-id' });
    assert.strictEqual(result.logs.length, 1);
    assert.ok(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
        result.logs[0].logId
      )
    );
  });
});
