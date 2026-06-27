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
import { describe, it } from 'node:test';
import { spanBroadcastEvents } from '../src/index';

function span(spanId: string, startTime: number, endTime: number): SpanData {
  return {
    spanId,
    traceId: 't',
    startTime,
    endTime,
    attributes: {},
    displayName: spanId,
    spanKind: 'INTERNAL',
    sameProcessAsParentSpan: { value: true },
    status: { code: 0 },
    instrumentationLibrary: { name: 'test' },
    timeEvents: { timeEvent: [] },
  } as unknown as SpanData;
}

describe('spanBroadcastEvents', () => {
  it('emits span_start then span_end for a completed span', () => {
    const s = span('a', 100, 200);
    const events = spanBroadcastEvents('t', [s], { a: s });
    assert.deepStrictEqual(
      events.map((e) => e.type),
      ['span_start', 'span_end']
    );
  });

  it('emits only span_start for an in-progress span', () => {
    const s = span('a', 100, 0);
    const events = spanBroadcastEvents('t', [s], { a: s });
    assert.deepStrictEqual(
      events.map((e) => e.type),
      ['span_start']
    );
  });

  it('suppresses a stale start for a span the store already has as ended', () => {
    // The merged store has the span ended (a prior end save won the merge); a
    // late in-progress start arrives for the same span. It must NOT re-open it.
    const incoming = span('a', 100, 0); // start payload, no endTime
    const merged = { a: span('a', 100, 200) }; // already ended on disk
    const events = spanBroadcastEvents('t', [incoming], merged);
    assert.deepStrictEqual(events, []);
  });

  it('still emits span_start for a genuinely new in-progress span', () => {
    // Merged reflects the just-saved in-progress span (endTime 0): not stale.
    const incoming = span('a', 100, 0);
    const merged = { a: span('a', 100, 0) };
    const events = spanBroadcastEvents('t', [incoming], merged);
    assert.deepStrictEqual(
      events.map((e) => e.type),
      ['span_start']
    );
  });

  it('orders events chronologically, start before end on ties', () => {
    const parent = span('parent', 100, 400);
    const child = span('child', 150, 150);
    const events = spanBroadcastEvents('t', [parent, child], {
      parent,
      child,
    });
    assert.deepStrictEqual(
      events.map((e) => `${e.type}:${e.span.spanId}`),
      [
        'span_start:parent',
        'span_start:child',
        'span_end:child',
        'span_end:parent',
      ]
    );
  });
});
