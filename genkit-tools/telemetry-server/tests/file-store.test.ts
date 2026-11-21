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

import type { TraceData, TraceQueryFilter } from '@genkit-ai/tools-common';
import fs from 'fs';
import getPort from 'get-port';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { Index } from '../src/file-trace-store.js';
import {
  LocalFileLogStore,
  LocalFileTraceStore,
  startTelemetryServer,
  stopTelemetryApi,
} from '../src/index.js';
import { sleep, span } from './utils.js';

const TRACE_ID = '1234';
const TRACE_ID_1 = '1234';
const TRACE_ID_2 = '2345';
const TRACE_ID_3 = '3456';
const SPAN_A = 'abc';
const SPAN_B = 'bcd';
const SPAN_C = 'cde';

describe('local-file-store', () => {
  let port: number;
  let storeRoot: string;
  let indexRoot: string;
  let url: string;

  beforeEach(async () => {
    port = await getPort();
    url = `http://localhost:${port}`;
    storeRoot = path.resolve(
      os.tmpdir(),
      `./telemetry-server-api-test-${Date.now()}/traces`
    );
    indexRoot = path.resolve(
      os.tmpdir(),
      `./telemetry-server-api-test-${Date.now()}/traces_idx`
    );

    await startTelemetryServer({
      port,
      traceStore: new LocalFileTraceStore({
        storeRoot,
        indexRoot,
      }),
      logStore: new LocalFileLogStore({
        storeRoot,
        indexRoot,
      }),
    });
  });

  afterEach(async () => {
    await stopTelemetryApi();
  });

  it('writes traces', async () => {
    const traceData = {
      traceId: TRACE_ID,
      displayName: 'trace',
      spans: { [SPAN_A]: span(TRACE_ID, SPAN_A, 100, 100) },
    } as TraceData;
    const res = await fetch(`${url}/api/traces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(traceData),
    });
    expect(res.status).toBe(200);

    await assertTraceData(TRACE_ID, traceData);
  });

  it('uppends spans to the trace', async () => {
    const spanA = span(TRACE_ID, SPAN_A, 100, 100);
    const spanB = span(TRACE_ID, SPAN_B, 200, 200);
    const res1 = await fetch(`${url}/api/traces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        traceId: TRACE_ID,
        spans: { [SPAN_A]: spanA },
      } as TraceData),
    });
    expect(res1.status).toBe(200);
    const res2 = await fetch(`${url}/api/traces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        traceId: TRACE_ID,
        spans: { [SPAN_B]: spanB },
      } as TraceData),
    });
    expect(res2.status).toBe(200);

    await assertTraceData(TRACE_ID, {
      traceId: TRACE_ID,
      spans: {
        [SPAN_A]: spanA,
        [SPAN_B]: spanB,
      },
    });
  });

  it('prevents overwriting completed span with incomplete span (race condition)', async () => {
    const spanA = span(TRACE_ID, SPAN_A, 100, 200);
    const spanA_incomplete = span(TRACE_ID, SPAN_A, 100, 100);
    delete (spanA_incomplete as any).endTime;

    // Save complete span first
    await fetch(`${url}/api/traces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        traceId: TRACE_ID,
        spans: { [SPAN_A]: spanA },
      } as TraceData),
    });

    // Save incomplete span second (stale start event)
    await fetch(`${url}/api/traces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        traceId: TRACE_ID,
        spans: { [SPAN_A]: spanA_incomplete },
      } as TraceData),
    });

    // Verify trace is still complete
    await assertTraceData(TRACE_ID, {
      traceId: TRACE_ID,
      spans: {
        [SPAN_A]: spanA,
      },
    });
  });

  it('updated final trace data', async () => {
    const spanA = span(TRACE_ID, SPAN_A, 100, 100);
    const spanB = span(TRACE_ID, SPAN_B, 200, 200);
    const res1 = await fetch(`${url}/api/traces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        traceId: TRACE_ID,
        spans: { [SPAN_A]: spanA },
      }),
    });
    expect(res1.status).toBe(200);

    await assertTraceData(TRACE_ID, {
      traceId: TRACE_ID,
      spans: {
        [SPAN_A]: spanA,
      },
    } as TraceData);

    const res2 = await fetch(`${url}/api/traces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        traceId: TRACE_ID,
        displayName: 'final display name',
        startTime: 111,
        endTime: 999,
        spans: { [SPAN_B]: spanB },
      }),
    });
    expect(res2.status).toBe(200);

    await assertTraceData(TRACE_ID, {
      traceId: TRACE_ID,
      displayName: 'final display name',
      startTime: 111,
      endTime: 999,
      spans: {
        [SPAN_A]: spanA,
        [SPAN_B]: spanB,
      },
    });
  });

  it('lists trace data', async () => {
    const wantTraces = [] as TraceData[];
    for (let i = 0; i < 3; i++) {
      const spanId = `abc_${i}`;
      const traceId = TRACE_ID + `_${i}`;
      const spanData = span(traceId, spanId, 100 + i, 100 + i);
      const trace = {
        traceId: traceId,
        displayName: 'trace',
        spans: { [spanId]: spanData },
      };
      wantTraces.push(trace);

      const res = await fetch(`${url}/api/traces`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trace),
      });
      expect(res.status).toBe(200);
      await sleep(1); // sleep a little to ensure consistent chronological ordering
    }

    const getResp = await fetch(`${url}/api/traces`);
    expect(getResp.status).toBe(200);
    const tracesResponse = await getResp.json();
    expect(tracesResponse.traces).toEqual(wantTraces.reverse());
  });

  it('lists filtered trace data', async () => {
    const wantTraces = [] as TraceData[];
    for (let i = 0; i < 3; i++) {
      const spanId = `abc_${i}`;
      const traceId = TRACE_ID + `_${i}`;
      const spanData = span(traceId, spanId, 100 + i, 100 + i);
      if (i % 2 == 0) {
        spanData.attributes['genkit:type'] = 'banana';
      }
      const trace = {
        traceId: traceId,
        displayName: 'trace',
        spans: { [spanId]: spanData },
      };
      if (i % 2 != 0) {
        wantTraces.push(trace);
      }

      const res = await fetch(`${url}/api/traces`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trace),
      });
      expect(res.status).toBe(200);
      await sleep(1); // sleep a little to ensure consistent chronological ordering
    }

    const filter = { neq: { type: 'banana' } } as TraceQueryFilter;
    const getResp = await fetch(
      `${url}/api/traces?filter=${encodeURI(JSON.stringify(filter))}`
    );
    expect(getResp.status).toBe(200);
    const tracesResponse = await getResp.json();
    expect(tracesResponse.traces).toEqual(wantTraces.reverse());
  });

  it('lists trace data with pagination', async () => {
    const wantTraces = [] as TraceData[];
    for (let i = 0; i < 20; i++) {
      const spanId = `abc_${i}`;
      const traceId = TRACE_ID + `_${i}`;
      const spanData = span(traceId, spanId, 100 + i, 100 + i);
      const trace = {
        traceId: traceId,
        displayName: 'trace',
        spans: { [spanId]: spanData },
      };
      wantTraces.push(trace);

      const res = await fetch(`${url}/api/traces`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trace),
      });
      expect(res.status).toBe(200);
      await sleep(1); // sleep a little to ensure consistent chronological ordering
    }
    const pageSize = 4;

    const respP1 = await fetch(`${url}/api/traces?limit=${pageSize}`);
    expect(respP1.status).toBe(200);
    const tracesResponseP1 = await respP1.json();
    expect(tracesResponseP1.traces).toEqual(
      wantTraces.slice(wantTraces.length - pageSize).reverse()
    );

    // continue to page 2
    const respP2 = await fetch(
      `${url}/api/traces?limit=${pageSize}&continuationToken=${tracesResponseP1.continuationToken}`
    );
    expect(respP2.status).toBe(200);
    const tracesResponseP2 = await respP2.json();
    expect(tracesResponseP2.traces).toEqual(
      wantTraces
        .slice(wantTraces.length - pageSize * 2, wantTraces.length - pageSize)
        .reverse()
    );
  });

  async function assertTraceData(traceId: string, traceData: TraceData) {
    const getResp = await fetch(`${url}/api/traces/${traceId}`);
    expect(getResp.status).toBe(200);
    expect(await getResp.json()).toEqual(traceData);
  }
});

describe('local-file-store path traversal', () => {
  let storeRoot: string;
  let indexRoot: string;
  let store: LocalFileTraceStore;

  beforeEach(() => {
    const base = path.resolve(
      os.tmpdir(),
      `./telemetry-server-traversal-test-${Date.now()}-${Math.floor(
        Math.random() * 1000
      )}`
    );
    storeRoot = path.resolve(base, 'traces');
    indexRoot = path.resolve(base, 'traces_idx');
    store = new LocalFileTraceStore({ storeRoot, indexRoot });
  });

  afterEach(() => {
    fs.rmSync(storeRoot, { recursive: true, force: true });
    fs.rmSync(indexRoot, { recursive: true, force: true });
  });

  const maliciousIds = [
    '../../pwned',
    '../../../etc/passwd',
    '..\\..\\pwned',
    '/etc/passwd',
    'foo/bar',
    'foo\\bar',
    '.',
    '..',
    'has\0null',
    '',
  ];

  const trace = {
    traceId: 'x',
    displayName: 'trace',
    spans: { [SPAN_A]: span('x', SPAN_A, 100, 100) },
  } as TraceData;

  for (const id of maliciousIds) {
    it(`rejects save with malicious id ${JSON.stringify(id)}`, async () => {
      await expect(store.save(id, trace)).rejects.toThrow(/Invalid trace id/);
    });

    it(`rejects load with malicious id ${JSON.stringify(id)}`, async () => {
      await expect(store.load(id)).rejects.toThrow(/Invalid trace id/);
    });
  }

  it('does not write outside of storeRoot on traversal save', async () => {
    const outside = path.resolve(storeRoot, '..', 'pwned');
    await expect(store.save('../pwned', trace)).rejects.toThrow(
      /Invalid trace id/
    );
    expect(fs.existsSync(outside)).toBe(false);
  });

  it('does not read outside of storeRoot on traversal load', async () => {
    // Plant a file just outside storeRoot and confirm it cannot be read.
    const secretPath = path.resolve(storeRoot, '..', 'secret');
    fs.writeFileSync(secretPath, JSON.stringify({ hello: 'world' }));
    try {
      await expect(store.load('../secret')).rejects.toThrow(/Invalid trace id/);
    } finally {
      fs.rmSync(secretPath, { force: true });
    }
  });

  it('still allows valid trace ids', async () => {
    const validTrace = {
      traceId: 'valid_1234',
      displayName: 'trace',
      spans: { [SPAN_A]: span('valid_1234', SPAN_A, 100, 100) },
    } as TraceData;
    await store.save('valid_1234', validTrace);
    const loaded = await store.load('valid_1234');
    expect(loaded).toBeTruthy();
    expect(loaded?.traceId).toBe('valid_1234');
    // The store writes trace files under `<storeRoot>/.genkit/traces`.
    expect(
      fs.existsSync(path.resolve(storeRoot, '.genkit/traces', 'valid_1234'))
    ).toBeTruthy();
  });
});

describe('index', () => {
  let indexRoot: string;
  let index: Index;

  beforeEach(async () => {
    indexRoot = path.resolve(
      os.tmpdir(),
      `./telemetry-server-api-test-${Date.now()}-${Math.floor(Math.random() * 1000)}/traces_idx`
    );

    index = new Index(indexRoot);
  });

  afterEach(() => {
    fs.rmSync(indexRoot, { recursive: true, force: true });
  });

  it('should index and search spans', () => {
    const spanA = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanA.displayName = 'spanA';
    spanA.startTime = 1234;
    spanA.endTime = 2345;

    const spanB = span(TRACE_ID_2, SPAN_B, 200, 200);
    spanB.displayName = 'spanB';
    spanB.startTime = 2345;
    spanB.endTime = 3456;

    index.add({
      traceId: TRACE_ID_1,
      spans: {
        [SPAN_A]: spanA,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_2,
      spans: {
        [SPAN_B]: spanB,
      },
    } as TraceData);

    expect(index.search({ limit: 5 }).data).toEqual([
      {
        id: TRACE_ID_2,
        type: 'flow',
        name: 'spanB',
        start: 2345,
        end: 3456,
        status: 0,
      },
      {
        id: TRACE_ID_1,
        type: 'flow',
        name: 'spanA',
        start: 1234,
        end: 2345,
        status: 0,
      },
    ]);
  });

  it('should apply search filters', () => {
    const spanA = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanA.displayName = 'flowA';
    spanA.attributes['genkit:type'] = 'banana';

    const spanB = span(TRACE_ID_2, SPAN_B, 200, 200);
    spanB.displayName = 'flowA';

    const spanC = span(TRACE_ID_3, SPAN_C, 200, 200);
    spanC.displayName = 'flowB';

    index.add({
      traceId: TRACE_ID_1,
      spans: {
        [SPAN_A]: spanA,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_2,
      spans: {
        [SPAN_B]: spanB,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_3,
      spans: {
        [SPAN_C]: spanC,
      },
    } as TraceData);

    expect(
      index.search({
        limit: 5,
        filter: {
          eq: { name: 'flowA' },
        },
      }).data
    ).toEqual([
      {
        id: TRACE_ID_2,
        type: 'flow',
        name: 'flowA',
        start: 1,
        end: 2,
        status: 0,
      },
      {
        id: TRACE_ID_1,
        type: 'banana',
        name: 'flowA',
        start: 1,
        end: 2,
        status: 0,
      },
    ]);

    expect(
      index.search({
        limit: 5,
        filter: {
          eq: { name: 'flowA', type: 'banana' },
        },
      }).data
    ).toEqual([
      {
        id: TRACE_ID_1,
        type: 'banana',
        name: 'flowA',
        start: 1,
        end: 2,
        status: 0,
      },
    ]);

    expect(
      index.search({
        limit: 5,
        filter: {
          eq: { name: 'flowA' },
          neq: { type: 'banana' },
        },
      }).data
    ).toEqual([
      {
        id: TRACE_ID_2,
        type: 'flow',
        name: 'flowA',
        start: 1,
        end: 2,
        status: 0,
      },
    ]);
  });

  it('should index and filter by sessionId', () => {
    const spanA = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanA.displayName = 'agentA';
    spanA.attributes['genkit:metadata:agent:sessionId'] = 'session-1';

    const spanB = span(TRACE_ID_2, SPAN_B, 200, 200);
    spanB.displayName = 'agentA';
    spanB.attributes['session.id'] = 'session-2';

    const spanC = span(TRACE_ID_3, SPAN_C, 200, 200);
    spanC.displayName = 'flowB';

    index.add({
      traceId: TRACE_ID_1,
      spans: { [SPAN_A]: spanA },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_2,
      spans: { [SPAN_B]: spanB },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_3,
      spans: { [SPAN_C]: spanC },
    } as TraceData);

    expect(
      index.search({
        limit: 5,
        filter: {
          eq: { sessionId: 'session-1' },
        },
      }).data
    ).toEqual([
      {
        id: TRACE_ID_1,
        type: 'flow',
        name: 'agentA',
        start: 1,
        end: 2,
        status: 0,
        sessionId: 'session-1',
      },
    ]);

    expect(
      index.search({
        limit: 5,
        filter: {
          eq: { sessionId: 'session-2' },
        },
      }).data
    ).toEqual([
      {
        id: TRACE_ID_2,
        type: 'flow',
        name: 'agentA',
        start: 1,
        end: 2,
        status: 0,
        sessionId: 'session-2',
      },
    ]);
  });

  it('can filter out unknown types', () => {
    const spanA = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanA.displayName = 'flowA';
    spanA.attributes['genkit:type'] = 'banana';

    const spanB = span(TRACE_ID_2, SPAN_B, 200, 200);
    spanB.displayName = 'flowB';
    spanB.attributes['genkit:type'] = undefined;

    const spanC = span(TRACE_ID_3, SPAN_C, 200, 200);
    spanC.displayName = 'flowC';
    spanC.attributes['genkit:type'] = undefined;

    index.add({
      traceId: TRACE_ID_1,
      spans: {
        [SPAN_A]: spanA,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_2,
      spans: {
        [SPAN_B]: spanB,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_3,
      spans: {
        [SPAN_C]: spanC,
      },
    } as TraceData);

    expect(
      index.search({
        limit: 5,
        filter: {
          neq: { type: 'UNKNOWN' },
        },
      }).data
    ).toEqual([
      {
        id: TRACE_ID_1,
        type: 'banana',
        name: 'flowA',
        start: 1,
        end: 2,
        status: 0,
      },
    ]);
  });

  it('should support array filters (IN/NOT IN)', () => {
    const spanA = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanA.displayName = 'flowA';
    spanA.attributes['genkit:type'] = 'banana';

    const spanB = span(TRACE_ID_2, SPAN_B, 200, 200);
    spanB.displayName = 'flowB';
    spanB.attributes['genkit:type'] = undefined;

    const spanC = span(TRACE_ID_3, SPAN_C, 200, 200);
    spanC.displayName = 'flowC';
    spanC.attributes['genkit:type'] = undefined;

    index.add({
      traceId: TRACE_ID_1,
      spans: {
        [SPAN_A]: spanA,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_2,
      spans: {
        [SPAN_B]: spanB,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_3,
      spans: {
        [SPAN_C]: spanC,
      },
    } as TraceData);

    // Test array filters (IN)
    expect(
      index.search({
        limit: 5,
        filter: {
          eq: { name: ['flowA', 'flowC'] },
        },
      }).data
    ).toEqual([
      {
        id: TRACE_ID_3,
        type: 'UNKNOWN',
        name: 'flowC',
        start: 1,
        end: 2,
        status: 0,
      },
      {
        id: TRACE_ID_1,
        type: 'banana',
        name: 'flowA',
        start: 1,
        end: 2,
        status: 0,
      },
    ]);

    // Test array filters (NOT IN)
    expect(
      index.search({
        limit: 5,
        filter: {
          neq: { name: ['flowB', 'flowC'] },
        },
      }).data
    ).toEqual([
      {
        id: TRACE_ID_1,
        type: 'banana',
        name: 'flowA',
        start: 1,
        end: 2,
        status: 0,
      },
    ]);
  });

  it('should support contains filters (case-insensitive)', () => {
    const spanA = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanA.displayName = 'Flow with Banana';
    spanA.attributes['genkit:type'] = 'FruitFlow';

    const spanB = span(TRACE_ID_2, SPAN_B, 200, 200);
    spanB.displayName = 'Another Apple Flow';
    spanB.attributes['genkit:type'] = 'fruitflow';

    const spanC = span(TRACE_ID_3, SPAN_C, 200, 200);
    spanC.displayName = 'Cherry Flow';
    spanC.attributes['genkit:type'] = 'Other';

    index.add({
      traceId: TRACE_ID_1,
      spans: { [SPAN_A]: spanA },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_2,
      spans: { [SPAN_B]: spanB },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_3,
      spans: { [SPAN_C]: spanC },
    } as TraceData);

    // Test basic contains (case-insensitive)
    expect(
      index
        .search({
          limit: 5,
          filter: {
            contains: { name: 'apple' },
          },
        })
        .data.map((d) => d.id)
    ).toEqual([TRACE_ID_2]);

    // Test contains with array (OR behavior)
    expect(
      index
        .search({
          limit: 5,
          filter: {
            contains: { name: ['BANANA', 'cherry'] },
          },
        })
        .data.map((d) => d.id)
    ).toEqual([TRACE_ID_3, TRACE_ID_1]);

    // Test contains on type
    expect(
      index
        .search({
          limit: 5,
          filter: {
            contains: { type: 'FRUIT' },
          },
        })
        .data.map((d) => d.id)
    ).toEqual([TRACE_ID_2, TRACE_ID_1]);
  });

  it('should support mixed type array filters (number and string)', () => {
    const spanA = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanA.displayName = 'flowA';
    spanA.attributes['genkit:type'] = 'banana';

    const spanB = span(TRACE_ID_2, SPAN_B, 200, 200);
    spanB.displayName = 'flowB';
    spanB.attributes['genkit:type'] = undefined;
    // Set status to undefined so it indexes as 'UNKNOWN'
    spanB.status = undefined;

    const spanC = span(TRACE_ID_3, SPAN_C, 200, 200);
    spanC.displayName = 'flowC';
    spanC.attributes['genkit:type'] = 'flow';
    spanC.status = { code: 1 }; // Status 1

    index.add({
      traceId: TRACE_ID_1,
      spans: {
        [SPAN_A]: spanA,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_2,
      spans: {
        [SPAN_B]: spanB,
      },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_3,
      spans: {
        [SPAN_C]: spanC,
      },
    } as TraceData);

    expect(
      index
        .search({
          limit: 5,
          filter: {
            eq: { status: [1, 'UNKNOWN'] },
          },
        })
        .data.map((d) => ({ id: d.id, status: d.status }))
    ).toEqual([
      { id: TRACE_ID_3, status: 1 },
      { id: TRACE_ID_2, status: 'UNKNOWN' },
    ]);
  });

  it('should support numeric comparison filters (gt/gte/lt/lte)', () => {
    // Traces with different start times
    const span1 = span('t1', 's1', 10, 10);
    span1.startTime = 100;
    span1.endTime = 200;

    const span2 = span('t2', 's2', 10, 10);
    span2.startTime = 200;
    span2.endTime = 300;

    const span3 = span('t3', 's3', 10, 10);
    span3.startTime = 300;
    span3.endTime = 400;

    const span4 = span('t4', 's4', 10, 10);
    span4.startTime = 400;
    span4.endTime = 500;

    index.add({ traceId: 't1', spans: { s1: span1 } } as TraceData);
    index.add({ traceId: 't2', spans: { s2: span2 } } as TraceData);
    index.add({ traceId: 't3', spans: { s3: span3 } } as TraceData);
    index.add({ traceId: 't4', spans: { s4: span4 } } as TraceData);

    // gt: start > 200 -> t3 (300), t4 (400)
    expect(
      index
        .search({
          limit: 10,
          filter: { gt: { start: 200 } },
        })
        .data.map((d) => d.id)
    ).toEqual(['t4', 't3']);

    // gte: start >= 200 -> t2 (200), t3 (300), t4 (400)
    expect(
      index
        .search({
          limit: 10,
          filter: { gte: { start: 200 } },
        })
        .data.map((d) => d.id)
    ).toEqual(['t4', 't3', 't2']);

    // lt: start < 300 -> t1 (100), t2 (200)
    expect(
      index
        .search({
          limit: 10,
          filter: { lt: { start: 300 } },
        })
        .data.map((d) => d.id)
    ).toEqual(['t2', 't1']);

    // lte: start <= 300 -> t1 (100), t2 (200), t3 (300)
    expect(
      index
        .search({
          limit: 10,
          filter: { lte: { start: 300 } },
        })
        .data.map((d) => d.id)
    ).toEqual(['t3', 't2', 't1']);

    // Combined: start > 100 AND start < 400 -> t2 (200), t3 (300)
    expect(
      index
        .search({
          limit: 10,
          filter: { gt: { start: 100 }, lt: { start: 400 } },
        })
        .data.map((d) => d.id)
    ).toEqual(['t3', 't2']);
  });

  it('should paginate search', () => {
    for (let i = 0; i < 20; i++) {
      const traceId = 'trace_' + i;
      const spanId = 'span_' + i;
      const s = span(traceId, spanId, 100 + i, 200 + i);
      s.displayName = 'spanA';
      s.startTime = 1234 + i;
      s.endTime = 2345 + i;

      index.add({
        traceId,
        spans: {
          [spanId]: s,
        },
      } as TraceData);
    }

    const result1 = index.search({ limit: 3 });
    expect(result1.data.map((d) => d.id)).toEqual([
      'trace_19',
      'trace_18',
      'trace_17',
    ]);
    expect(result1.pageLastIndex).toBe(3);

    const result2 = index.search({
      limit: 3,
      startFromIndex: result1.pageLastIndex,
    });
    expect(result2.data.map((d) => d.id)).toEqual([
      'trace_16',
      'trace_15',
      'trace_14',
    ]);
    expect(result2.pageLastIndex).toBe(6);

    // check edge conditions

    const result3 = index.search({ limit: 3, startFromIndex: 17 });
    expect(result3.data.map((d) => d.id)).toEqual([
      'trace_2',
      'trace_1',
      'trace_0',
    ]);
    expect(result3.pageLastIndex).toBeUndefined();

    const result4 = index.search({ limit: 10, startFromIndex: 18 });
    expect(result4.data.map((d) => d.id)).toEqual(['trace_1', 'trace_0']);
    expect(result4.pageLastIndex).toBeUndefined();
  });

  it('should deduplicate when root span is posted twice (start then end)', () => {
    // Root span posted at start (no endTime)
    const spanStart = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanStart.displayName = 'rootSpan';
    spanStart.startTime = 1000;
    delete (spanStart as any).endTime;

    // Same root span posted at end (has endTime)
    const spanEnd = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanEnd.displayName = 'rootSpan';
    spanEnd.startTime = 1000;
    spanEnd.endTime = 2000;

    index.add({
      traceId: TRACE_ID_1,
      spans: { [SPAN_A]: spanStart },
    } as TraceData);

    index.add({
      traceId: TRACE_ID_1,
      spans: { [SPAN_A]: spanEnd },
    } as TraceData);

    const result = index.search({ limit: 10 });

    expect(result.data).toHaveLength(1);
    expect(result.data[0].id).toBe(TRACE_ID_1);
    expect(result.data[0].start).toBe(1000);
    expect(result.data[0].end).toBe(2000);
  });

  it('should return empty array for empty index', () => {
    const result = index.search({ limit: 10 });
    expect(result.data).toEqual([]);
    expect(result.pageLastIndex).toBeUndefined();
  });

  it('should apply pagination correctly after deduplication', () => {
    // 5 unique traces, each posted twice (start then end)
    const traces = [
      { traceId: TRACE_ID_1, start: 5000, end: undefined },
      { traceId: TRACE_ID_2, start: 2000, end: undefined },
      { traceId: TRACE_ID_3, start: 3000, end: undefined },
      { traceId: 'trace_4', start: 4000, end: undefined },
      { traceId: 'trace_5', start: 1000, end: undefined },
      { traceId: TRACE_ID_1, start: 5000, end: 5500 },
      { traceId: TRACE_ID_2, start: 2000, end: 2500 },
      { traceId: TRACE_ID_3, start: 3000, end: 3500 },
      { traceId: 'trace_4', start: 4000, end: 4500 },
      { traceId: 'trace_5', start: 1000, end: 1500 },
    ];

    for (const t of traces) {
      const s = span(t.traceId, 'span', 100, 100);
      s.startTime = t.start;
      if (t.end) s.endTime = t.end;
      else delete (s as any).endTime;
      index.add({ traceId: t.traceId, spans: { span: s } } as TraceData);
    }

    // After dedup: TRACE_ID_1 (5000), trace_4 (4000), TRACE_ID_3 (3000), TRACE_ID_2 (2000), trace_5 (1000)
    const result = index.search({ limit: 2 });
    expect(result.data).toHaveLength(2);
    expect(result.data[0].id).toBe(TRACE_ID_1);
    expect(result.data[0].start).toBe(5000);
    expect(result.data[0].end).toBe(5500); // has endTime from second post
    expect(result.data[1].id).toBe('trace_4');
    expect(result.pageLastIndex).toBe(2);

    // Page 2 - get remaining 3
    const result2 = index.search({ limit: 3, startFromIndex: 2 });
    expect(result2.data).toHaveLength(3);
    expect(result2.data[0].id).toBe(TRACE_ID_3);
    expect(result2.data[1].id).toBe(TRACE_ID_2);
    expect(result2.data[2].id).toBe('trace_5');
  });

  it('should sort entries by start time descending', () => {
    const traces = [
      { traceId: TRACE_ID_1, start: 100 },
      { traceId: TRACE_ID_2, start: 300 },
      { traceId: TRACE_ID_3, start: 200 },
    ];

    for (const t of traces) {
      const s = span(t.traceId, 'span', 100, 100);
      s.startTime = t.start;
      index.add({ traceId: t.traceId, spans: { span: s } } as TraceData);
    }

    const result = index.search({ limit: 10 });

    // Should be sorted by start time descending: 300, 200, 100
    expect(result.data).toHaveLength(3);
    expect(result.data[0].id).toBe(TRACE_ID_2); // start: 300
    expect(result.data[1].id).toBe(TRACE_ID_3); // start: 200
    expect(result.data[2].id).toBe(TRACE_ID_1); // start: 100
  });

  it('should handle spans with only startTime (in-progress spans)', () => {
    // Span with only startTime, no endTime (simulates span_start event)
    const inProgressSpan = span(TRACE_ID_1, SPAN_A, 100, 100);
    inProgressSpan.startTime = 1000;
    inProgressSpan.endTime = 0; // In-progress span has no endTime

    index.add({
      traceId: TRACE_ID_1,
      spans: { [SPAN_A]: inProgressSpan },
    } as TraceData);

    const result = index.search({ limit: 10 });

    expect(result.data).toHaveLength(1);
    expect(result.data[0].id).toBe(TRACE_ID_1);
    expect(result.data[0].start).toBe(1000);
    // end should not be set for in-progress spans
    expect(result.data[0].end).toBeUndefined();
  });

  it('should deduplicate when span updates from start to end', () => {
    // First: span_start event (no endTime)
    const spanStart = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanStart.startTime = 1000;
    spanStart.endTime = 0;

    index.add({
      traceId: TRACE_ID_1,
      spans: { [SPAN_A]: spanStart },
    } as TraceData);

    // Second: span_end event (with endTime) - same trace, updated
    const spanEnd = span(TRACE_ID_1, SPAN_A, 100, 100);
    spanEnd.startTime = 1000;
    spanEnd.endTime = 2000;

    index.add({
      traceId: TRACE_ID_1,
      spans: { [SPAN_A]: spanEnd },
    } as TraceData);

    const result = index.search({ limit: 10 });

    // Should only have one entry
    expect(result.data).toHaveLength(1);
    expect(result.data[0].id).toBe(TRACE_ID_1);
    expect(result.data[0].start).toBe(1000);
    // Should have the end time from the completed span
    expect(result.data[0].end).toBe(2000);
  });

  it('should handle mix of in-progress and completed spans', () => {
    // Completed span
    const completedSpan = span(TRACE_ID_1, SPAN_A, 100, 100);
    completedSpan.startTime = 1000;
    completedSpan.endTime = 2000;

    // In-progress span (started later)
    const inProgressSpan = span(TRACE_ID_2, SPAN_B, 100, 100);
    inProgressSpan.startTime = 3000;
    inProgressSpan.endTime = 0;

    index.add({
      traceId: TRACE_ID_1,
      spans: { [SPAN_A]: completedSpan },
    } as TraceData);
    index.add({
      traceId: TRACE_ID_2,
      spans: { [SPAN_B]: inProgressSpan },
    } as TraceData);

    const result = index.search({ limit: 10 });

    // Both should be returned, sorted by start time descending
    expect(result.data).toHaveLength(2);
    expect(result.data[0].id).toBe(TRACE_ID_2); // start: 3000 (in-progress)
    expect(result.data[0].end).toBeUndefined();
    expect(result.data[1].id).toBe(TRACE_ID_1); // start: 1000 (completed)
    expect(result.data[1].end).toBe(2000);
  });
});

describe('otlp-endpoint', () => {
  let port: number;
  let storeRoot: string;
  let indexRoot: string;
  let url: string;

  beforeEach(async () => {
    port = await getPort();
    url = `http://localhost:${port}`;
    storeRoot = path.resolve(
      os.tmpdir(),
      `./telemetry-server-api-test-${Date.now()}/traces`
    );
    indexRoot = path.resolve(
      os.tmpdir(),
      `./telemetry-server-api-test-${Date.now()}/traces_idx`
    );

    await startTelemetryServer({
      port,
      traceStore: new LocalFileTraceStore({
        storeRoot,
        indexRoot,
      }),
      logStore: new LocalFileLogStore({
        storeRoot,
        indexRoot,
      }),
    });
  });

  afterEach(async () => {
    await stopTelemetryApi();
  });

  it('saves a single trace', async () => {
    const traceId = 'childTraceId';
    const otlpPayload = {
      resourceSpans: [
        {
          resource: {
            attributes: [
              { key: 'service.name', value: { stringValue: 'test' } },
            ],
          },
          scopeSpans: [
            {
              scope: { name: 'test-scope' },
              spans: [
                {
                  traceId,
                  spanId: 'childSpanId1',
                  name: 'span1',
                  startTimeUnixNano: '1000000',
                  endTimeUnixNano: '2000000',
                  kind: 1,
                  attributes: [],
                },
              ],
            },
          ],
        },
      ],
    };

    const res = await fetch(`${url}/api/otlp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(otlpPayload),
    });
    expect(res.status).toBe(200);

    const getResp = await fetch(`${url}/api/traces/${traceId}`);
    expect(getResp.status).toBe(200);
    const trace = await getResp.json();
    expect(trace.traceId).toBe(traceId);
    expect(Object.keys(trace.spans)).toHaveLength(1);
    const span = Object.values(trace.spans)[0] as any;
    expect(span.traceId).toBe(traceId);
    expect(span.spanId).toBe('childSpanId1');
  });

  it('saves a trace with multiple spans', async () => {
    const traceId = 'childTraceId';
    const otlpPayload = {
      resourceSpans: [
        {
          resource: {
            attributes: [
              { key: 'service.name', value: { stringValue: 'test' } },
            ],
          },
          scopeSpans: [
            {
              scope: { name: 'test-scope' },
              spans: [
                {
                  traceId,
                  spanId: 'childSpanId1',
                  name: 'span1',
                  startTimeUnixNano: '1000000',
                  endTimeUnixNano: '2000000',
                  kind: 1,
                  attributes: [],
                },
                {
                  traceId,
                  spanId: 'childSpanId2',
                  name: 'span2',
                  startTimeUnixNano: '3000000',
                  endTimeUnixNano: '4000000',
                  kind: 1,
                  attributes: [],
                },
              ],
            },
          ],
        },
      ],
    };

    const res = await fetch(`${url}/api/otlp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(otlpPayload),
    });
    expect(res.status).toBe(200);

    const getResp = await fetch(`${url}/api/traces/${traceId}`);
    expect(getResp.status).toBe(200);
    const trace = await getResp.json();
    expect(trace.traceId).toBe(traceId);
    expect(Object.keys(trace.spans)).toHaveLength(2);
    const span1 = trace.spans['childSpanId1'];
    expect(span1.traceId).toBe(traceId);
    const span2 = trace.spans['childSpanId2'];
    expect(span2.traceId).toBe(traceId);
  });

  it('handles errors', async () => {
    const res = await fetch(`${url}/api/otlp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: 'invalid json',
    });
    expect(res.status).toBe(500);
  });
});

describe('otlp-endpoint (with parent)', () => {
  let port: number;
  let storeRoot: string;
  let indexRoot: string;
  let url: string;

  beforeEach(async () => {
    port = await getPort();
    url = `http://localhost:${port}`;
    storeRoot = path.resolve(
      os.tmpdir(),
      `./telemetry-server-api-test-${Date.now()}/traces`
    );
    indexRoot = path.resolve(
      os.tmpdir(),
      `./telemetry-server-api-test-${Date.now()}/traces_idx`
    );

    await startTelemetryServer({
      port,
      traceStore: new LocalFileTraceStore({
        storeRoot,
        indexRoot,
      }),
      logStore: new LocalFileLogStore({
        storeRoot,
        indexRoot,
      }),
    });
  });

  afterEach(async () => {
    await stopTelemetryApi();
  });

  it('saves a single trace', async () => {
    const parentTraceId = 'parentTraceId';
    const parentSpanId = 'parentSpanId';
    const otlpPayload = {
      resourceSpans: [
        {
          resource: {
            attributes: [
              { key: 'service.name', value: { stringValue: 'test' } },
            ],
          },
          scopeSpans: [
            {
              scope: { name: 'test-scope' },
              spans: [
                {
                  traceId: 'childTraceId',
                  spanId: 'childSpanId1',
                  name: 'span1',
                  startTimeUnixNano: '1000000',
                  endTimeUnixNano: '2000000',
                  kind: 1,
                  attributes: [],
                },
              ],
            },
          ],
        },
      ],
    };

    const res = await fetch(
      `${url}/api/otlp/${parentTraceId}/${parentSpanId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(otlpPayload),
      }
    );
    expect(res.status).toBe(200);

    const getResp = await fetch(`${url}/api/traces/${parentTraceId}`);
    expect(getResp.status).toBe(200);
    const trace = await getResp.json();
    expect(trace.traceId).toBe(parentTraceId);
    expect(Object.keys(trace.spans)).toHaveLength(1);
    const span = Object.values(trace.spans)[0] as any;
    expect(span.traceId).toBe(parentTraceId);
    expect(span.parentSpanId).toBe(parentSpanId);
    expect(span.spanId).toBe('childSpanId1');
    expect(span.attributes['genkit:otlp-traceId']).toBe('childTraceId');
  });

  it('saves a trace with multiple spans', async () => {
    const parentTraceId = 'parentTraceId';
    const parentSpanId = 'parentSpanId';
    const otlpPayload = {
      resourceSpans: [
        {
          resource: {
            attributes: [
              { key: 'service.name', value: { stringValue: 'test' } },
            ],
          },
          scopeSpans: [
            {
              scope: { name: 'test-scope' },
              spans: [
                {
                  traceId: 'childTraceId', // this will be overwritten
                  spanId: 'childSpanId1',
                  name: 'span1',
                  startTimeUnixNano: '1000000',
                  endTimeUnixNano: '2000000',
                  kind: 1,
                  attributes: [],
                },
                {
                  traceId: 'childTraceId', // this will be overwritten
                  spanId: 'childSpanId2',
                  name: 'span2',
                  startTimeUnixNano: '3000000',
                  endTimeUnixNano: '4000000',
                  kind: 1,
                  attributes: [],
                },
              ],
            },
          ],
        },
      ],
    };

    const res = await fetch(
      `${url}/api/otlp/${parentTraceId}/${parentSpanId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(otlpPayload),
      }
    );
    expect(res.status).toBe(200);

    const getResp = await fetch(`${url}/api/traces/${parentTraceId}`);
    expect(getResp.status).toBe(200);
    const trace = await getResp.json();
    expect(trace.traceId).toBe(parentTraceId);
    expect(Object.keys(trace.spans)).toHaveLength(2);
    const span1 = trace.spans['childSpanId1'];
    expect(span1.traceId).toBe(parentTraceId);
    expect(span1.parentSpanId).toBe(parentSpanId);
    const span2 = trace.spans['childSpanId2'];
    expect(span2.traceId).toBe(parentTraceId);
    expect(span2.parentSpanId).toBe(parentSpanId);
  });

  it('saves multiple batches of traces', async () => {
    const parentTraceId = 'parentTraceId';
    const parentSpanId = 'parentSpanId';
    const otlpPayload1 = {
      resourceSpans: [
        {
          resource: {
            attributes: [
              { key: 'service.name', value: { stringValue: 'test' } },
            ],
          },
          scopeSpans: [
            {
              scope: { name: 'test-scope' },
              spans: [
                {
                  traceId: 'childTraceId',
                  spanId: 'childSpanId1',
                  name: 'span1',
                  startTimeUnixNano: '1000000',
                  endTimeUnixNano: '2000000',
                  kind: 1,
                  attributes: [],
                },
              ],
            },
          ],
        },
      ],
    };
    const otlpPayload2 = {
      resourceSpans: [
        {
          resource: {
            attributes: [
              { key: 'service.name', value: { stringValue: 'test' } },
            ],
          },
          scopeSpans: [
            {
              scope: { name: 'test-scope' },
              spans: [
                {
                  traceId: 'childTraceId',
                  spanId: 'childSpanId2',
                  name: 'span2',
                  startTimeUnixNano: '3000000',
                  endTimeUnixNano: '4000000',
                  kind: 1,
                  attributes: [],
                },
              ],
            },
          ],
        },
      ],
    };

    const res1 = await fetch(
      `${url}/api/otlp/${parentTraceId}/${parentSpanId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(otlpPayload1),
      }
    );
    expect(res1.status).toBe(200);

    const res2 = await fetch(
      `${url}/api/otlp/${parentTraceId}/${parentSpanId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(otlpPayload2),
      }
    );
    expect(res2.status).toBe(200);

    const getResp = await fetch(`${url}/api/traces/${parentTraceId}`);
    expect(getResp.status).toBe(200);
    const trace = await getResp.json();
    expect(trace.traceId).toBe(parentTraceId);
    expect(Object.keys(trace.spans)).toHaveLength(2);
    expect(trace.spans['childSpanId1']).toBeTruthy();
    expect(trace.spans['childSpanId2']).toBeTruthy();
  });

  it('handles errors', async () => {
    const parentTraceId = 'parentTraceId';
    const parentSpanId = 'parentSpanId';
    const res = await fetch(
      `${url}/api/otlp/${parentTraceId}/${parentSpanId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'invalid json',
      }
    );
    expect(res.status).toBe(500);
  });
});
