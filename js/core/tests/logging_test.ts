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
import { afterEach, beforeEach, describe, it } from 'node:test';
import { logger } from '../src/logging.js';

describe('logging consolidated api', () => {
  let loggedArgs: any[][] = [];
  let originalLogger: any;

  beforeEach(() => {
    loggedArgs = [];
    originalLogger = logger.defaultLogger;

    // Mock the underlying logger
    logger.init({
      level: 'info',
      info(...args: any[]) {
        loggedArgs.push(['info', ...args]);
      },
      debug(...args: any[]) {
        loggedArgs.push(['debug', ...args]);
      },
      warn(...args: any[]) {
        loggedArgs.push(['warn', ...args]);
      },
      error(...args: any[]) {
        loggedArgs.push(['error', ...args]);
      },
    });
  });

  afterEach(() => {
    logger.init(originalLogger);
  });

  it('should support message only', () => {
    logger.info('Database connection established');
    assert.strictEqual(loggedArgs.length, 1);
    assert.deepStrictEqual(loggedArgs[0], [
      'info',
      'Database connection established',
      {},
    ]);
  });

  it('should support message + metadata', () => {
    logger.info('User completed checkout', {
      userId: 'usr_98234',
      cartTotal: 129.99,
    });
    assert.strictEqual(loggedArgs.length, 1);
    assert.deepStrictEqual(loggedArgs[0], [
      'info',
      'User completed checkout',
      { userId: 'usr_98234', cartTotal: 129.99 },
    ]);
  });

  it('should support message + error', () => {
    const error = new Error('fail');
    logger.info('External service call failed', error);
    assert.strictEqual(loggedArgs.length, 1);
    assert.strictEqual(loggedArgs[0][0], 'info');
    assert.strictEqual(loggedArgs[0][1], 'External service call failed');
    const metadata = loggedArgs[0][2];
    assert.strictEqual(metadata['exception.type'], 'Error');
    assert.strictEqual(metadata['exception.message'], 'fail');
    assert.ok(metadata['exception.stacktrace']);
  });

  it('should support message + metadata + error', () => {
    const error = new Error('fail');
    logger.info(
      'Failed to update inventory',
      { sku: 'SKU-492', qty: 10 },
      error
    );
    assert.strictEqual(loggedArgs.length, 1);
    assert.strictEqual(loggedArgs[0][0], 'info');
    assert.strictEqual(loggedArgs[0][1], 'Failed to update inventory');
    const metadata = loggedArgs[0][2];
    assert.strictEqual(metadata.sku, 'SKU-492');
    assert.strictEqual(metadata.qty, 10);
    assert.strictEqual(metadata['exception.type'], 'Error');
    assert.strictEqual(metadata['exception.message'], 'fail');
    assert.ok(metadata['exception.stacktrace']);
  });

  it('should support single log payload object', () => {
    const error = new Error('sync fail');
    logger.info({
      message: 'Cache synchronization complete',
      metadata: { keysSynchronized: 450, elapsedMs: 120 },
      error: error,
    });
    assert.strictEqual(loggedArgs.length, 1);
    assert.strictEqual(loggedArgs[0][0], 'info');
    assert.strictEqual(loggedArgs[0][1], 'Cache synchronization complete');
    const metadata = loggedArgs[0][2];
    assert.strictEqual(metadata.keysSynchronized, 450);
    assert.strictEqual(metadata.elapsedMs, 120);
    assert.strictEqual(metadata['exception.type'], 'Error');
    assert.strictEqual(metadata['exception.message'], 'sync fail');
    assert.ok(metadata['exception.stacktrace']);
  });

  it('should support formatting string with args', () => {
    logger.info('Attempting retry %d of %d in %dms', 1, 3, 1000);
    assert.strictEqual(loggedArgs.length, 1);
    assert.deepStrictEqual(loggedArgs[0], [
      'info',
      'Attempting retry %d of %d in %dms',
      1,
      3,
      1000,
    ]);
  });

  it('should support warn with message + metadata + error', () => {
    const error = new Error('fail');
    logger.warn('Failed structured', { sku: 'SKU-492' }, error);
    assert.strictEqual(loggedArgs.length, 1);
    assert.strictEqual(loggedArgs[0][0], 'warn');
    assert.strictEqual(loggedArgs[0][1], 'Failed structured');
    const metadata = loggedArgs[0][2];
    assert.strictEqual(metadata.sku, 'SKU-492');
    assert.strictEqual(metadata['exception.type'], 'Error');
    assert.strictEqual(metadata['exception.message'], 'fail');
    assert.ok(metadata['exception.stacktrace']);
  });

  it('should support passing error object directly as message', () => {
    const error = new Error('direct fail');
    logger.warn(error);
    assert.strictEqual(loggedArgs.length, 1);
    assert.strictEqual(loggedArgs[0][0], 'warn');
    assert.strictEqual(loggedArgs[0][1], 'direct fail');
    const metadata = loggedArgs[0][2];
    assert.strictEqual(metadata['exception.type'], 'Error');
    assert.strictEqual(metadata['exception.message'], 'direct fail');
    assert.ok(metadata['exception.stacktrace']);
  });
});
