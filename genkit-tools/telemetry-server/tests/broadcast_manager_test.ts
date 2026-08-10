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

import type { SpanData } from '@genkit-ai/tools-common';
import * as assert from 'assert';
import { EventEmitter } from 'events';
import type { Response } from 'express';
import { beforeEach, describe, it } from 'node:test';
import { BroadcastManager, type SpanEvent } from '../src/broadcast-manager';

/**
 * Creates a mock Response object for testing.
 */
function createMockResponse(): Response & {
  writtenData: string[];
  ended: boolean;
  emit: (event: string) => boolean;
} {
  const emitter = new EventEmitter();
  const mock = {
    writtenData: [] as string[],
    ended: false,
    write(data: string) {
      if (mock.ended) {
        throw new Error('Cannot write to ended response');
      }
      mock.writtenData.push(data);
      return true;
    },
    end() {
      mock.ended = true;
    },
    on(event: string, handler: (...args: any[]) => void) {
      emitter.on(event, handler);
      return mock;
    },
    emit(event: string) {
      return emitter.emit(event);
    },
  };
  return mock as any;
}

function createSpanEvent(
  traceId: string,
  type: 'span_start' | 'span_end' = 'span_start'
): SpanEvent {
  return {
    type,
    traceId,
    span: {
      traceId,
      spanId: 'test-span',
      displayName: 'Test Span',
      startTime: 1000,
      endTime: type === 'span_end' ? 2000 : 0,
      instrumentationLibrary: { name: 'genkit' },
      spanKind: 'INTERNAL',
      attributes: {},
      status: { code: 0 },
    } as SpanData,
  };
}

describe('BroadcastManager', () => {
  let manager: BroadcastManager;

  beforeEach(() => {
    manager = new BroadcastManager();
  });

  it('subscribes connections and broadcasts events to them', () => {
    const response1 = createMockResponse();
    const response2 = createMockResponse();

    manager.subscribe('trace-1', response1);
    manager.subscribe('trace-1', response2);

    const event = createSpanEvent('trace-1');
    manager.broadcast('trace-1', event);

    const expectedData = `data: ${JSON.stringify(event)}\n\n`;
    assert.strictEqual(response1.writtenData[0], expectedData);
    assert.strictEqual(response2.writtenData[0], expectedData);
  });

  it('auto-unsubscribes when connection closes', () => {
    const response = createMockResponse();
    manager.subscribe('trace-1', response);

    assert.strictEqual(manager.getConnectionCount('trace-1'), 1);

    response.emit('close');

    assert.strictEqual(manager.getConnectionCount('trace-1'), 0);
  });

  it('close() ends all connections for a traceId', () => {
    const response1 = createMockResponse();
    const response2 = createMockResponse();

    manager.subscribe('trace-1', response1);
    manager.subscribe('trace-1', response2);

    manager.close('trace-1');

    assert.strictEqual(response1.ended, true);
    assert.strictEqual(response2.ended, true);
    assert.strictEqual(manager.hasConnections('trace-1'), false);
  });
});
