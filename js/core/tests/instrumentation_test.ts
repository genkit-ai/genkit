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

import { trace, TraceFlags, type Span as ApiSpan } from '@opentelemetry/api';
import * as assert from 'assert';
import * as http from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { logger } from '../src/logging.js';
import { initNodeFeatures } from '../src/node.js';
import {
  configureInstrumentation,
  flushTracing,
  resetInstrumentation,
  runInNewSpan,
  setTelemetryServerUrl,
  type GenkitLogRecord,
  type GenkitSpanContext,
  type Instrumentation,
  type InstrumentationNext,
  type InstrumentationSpanInfo,
  type LogRecordingInstrumentation,
} from '../src/tracing.js';
import { sleep } from './utils.js';

initNodeFeatures();

/** A minimal provider that records calls and mints deterministic ids. */
class FakeInstrumentation
  implements Instrumentation, LogRecordingInstrumentation
{
  spans: InstrumentationSpanInfo[] = [];
  logs: GenkitLogRecord[] = [];

  constructor(
    private readonly traceId = '',
    private readonly spanId = ''
  ) {}

  async runInNewSpan<T>(
    info: InstrumentationSpanInfo,
    next: InstrumentationNext<T>
  ): Promise<T> {
    this.spans.push(info);
    const ctx: GenkitSpanContext = {
      traceId: this.traceId,
      spanId: this.spanId,
      setMetadata() {},
    };
    // A wrapped-context span is enough; callers only read spanContext().
    const span = {
      spanContext: () => ({
        traceId: this.traceId || '0'.repeat(32),
        spanId: this.spanId || '0'.repeat(16),
        traceFlags: 1,
      }),
    } as ApiSpan;
    return next(span, ctx);
  }

  recordLog(record: GenkitLogRecord): void {
    this.logs.push(record);
  }
}

describe('instrumentation abstraction', () => {
  beforeEach(() => {
    delete process.env.GENKIT_TELEMETRY_SERVER;
    setTelemetryServerUrl(''); // clear leaked module-level url between tests
    resetInstrumentation();
  });

  afterEach(() => {
    resetInstrumentation();
  });

  it('routes runInNewSpan through the configured provider', async () => {
    const fake = new FakeInstrumentation('a'.repeat(32), 'b'.repeat(16));
    configureInstrumentation(fake);

    const out = await runInNewSpan(
      { metadata: { name: 'root' }, labels: { 'genkit:type': 'flow' } },
      async (metadata) => `ran ${metadata.name}`
    );

    assert.equal(out, 'ran root');
    assert.equal(fake.spans.length, 1);
    assert.equal(fake.spans[0].metadata.name, 'root');
    // Dispatcher builds the genkit path before handing off to the provider.
    assert.equal(fake.spans[0].metadata.path, '/{root,t:flow}');
  });

  it('exposes composite span ids to the wrapped fn', async () => {
    const fake = new FakeInstrumentation('c'.repeat(32), 'd'.repeat(16));
    configureInstrumentation(fake);

    let seen = { traceId: '', spanId: '' };
    await runInNewSpan({ metadata: { name: 'ids' } }, async (_m, span) => {
      seen = {
        traceId: span.spanContext().traceId,
        spanId: span.spanContext().spanId,
      };
    });

    assert.equal(seen.traceId, 'c'.repeat(32));
    assert.equal(seen.spanId, 'd'.repeat(16));
  });

  it('prepends Direct instrumentation when a telemetry server is set', async () => {
    const base = new FakeInstrumentation('e'.repeat(32), 'f'.repeat(16));
    configureInstrumentation(base);
    // Direct is created from this URL; its POST is fire-and-forget so the
    // (unreachable) server here does not affect the result.
    setTelemetryServerUrl('http://127.0.0.1:0');

    // Direct mints real ids and is first in the chain, so its ids win over the
    // configured provider's.
    let traceId = '';
    await runInNewSpan(
      { metadata: { name: 'dev' }, labels: { 'genkit:type': 'flow' } },
      async (_m, span) => {
        traceId = span.spanContext().traceId;
      }
    );

    assert.notEqual(traceId, 'e'.repeat(32));
    assert.match(traceId, /^[0-9a-f]{32}$/);
    // The configured provider still runs (composition, not replacement).
    assert.equal(base.spans.length, 1);
  });

  it('resolves first non-empty id across the chain', async () => {
    // Empty-id provider first; a real-id provider is only reachable if we
    // compose. Here we verify the composite skips empty ids.
    const empty = new FakeInstrumentation('', '');
    configureInstrumentation(empty);

    let traceId = 'unset';
    await runInNewSpan({ metadata: { name: 'blank' } }, async (_m, span) => {
      traceId = span.spanContext().traceId;
    });

    // Single empty provider => composite id is '' (surfaced as all-zeros on the
    // wrapped OTel span).
    assert.equal(traceId, '0'.repeat(32));
  });

  it('fans out logs to providers with recordLog', async () => {
    const fake = new FakeInstrumentation('1'.repeat(32), '2'.repeat(16));
    configureInstrumentation(fake);

    logger.info('hello world');

    assert.equal(fake.logs.length, 1);
    assert.equal(fake.logs[0].severity, 'info');
    assert.equal(fake.logs[0].body, 'hello world');
  });

  it('fans out callback span attribute writes via setMetadata', async () => {
    const written: Record<string, unknown>[] = [];
    const recording: Instrumentation = {
      runInNewSpan(info, next) {
        const ctx: GenkitSpanContext = {
          traceId: '3'.repeat(32),
          spanId: '4'.repeat(16),
          setMetadata: (values) => written.push(values),
        };
        const span = trace.wrapSpanContext({
          traceId: ctx.traceId,
          spanId: ctx.spanId,
          traceFlags: TraceFlags.SAMPLED,
        });
        return next(span, ctx);
      },
    };
    configureInstrumentation(recording);

    await runInNewSpan({ metadata: { name: 'attrs' } }, async (_m, span) => {
      span.setAttribute('a', 1).setAttributes({ b: 'two' });
    });

    assert.deepStrictEqual(written, [{ a: 1 }, { b: 'two' }]);
  });
});

describe('DirectTelemetryInstrumentation realtime export', () => {
  let server: http.Server;
  let url: string;
  const posted: any[] = [];
  const prevRealtime = process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;
  // Simulates a slow telemetry server; a post is only recorded once it
  // "lands", right before the response is sent.
  let responseDelayMs = 0;

  beforeEach(async () => {
    posted.length = 0;
    responseDelayMs = 0;
    server = http.createServer((req, res) => {
      let body = '';
      req.on('data', (c) => (body += c));
      req.on('end', async () => {
        if (responseDelayMs) await sleep(responseDelayMs);
        if (req.url === '/api/traces') posted.push(JSON.parse(body));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end('{}');
      });
    });
    await new Promise<void>((resolve) =>
      server.listen(0, '127.0.0.1', resolve)
    );
    const addr = server.address() as AddressInfo;
    url = `http://127.0.0.1:${addr.port}`;
    setTelemetryServerUrl(url);
    process.env.GENKIT_ENABLE_REALTIME_TELEMETRY = 'true';
    resetInstrumentation();
  });

  afterEach(async () => {
    if (prevRealtime === undefined) {
      delete process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;
    } else {
      process.env.GENKIT_ENABLE_REALTIME_TELEMETRY = prevRealtime;
    }
    setTelemetryServerUrl('');
    resetInstrumentation();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  // Waits for the fire-and-forget POSTs to land for the given span.
  async function postsFor(spanId: string) {
    for (let i = 0; i < 50; i++) {
      const spans = posted
        .flatMap((t) => Object.values(t.spans ?? {}))
        .filter((s: any) => s.spanId === spanId);
      if (spans.length >= 2) return spans as any[];
      await sleep(10);
    }
    return posted
      .flatMap((t) => Object.values(t.spans ?? {}))
      .filter((s: any) => s.spanId === spanId) as any[];
  }

  it('exports a pending span (endTime 0) on start, then a final span', async () => {
    let spanId = '';
    let pendingSeen: any[] = [];

    await runInNewSpan(
      {
        metadata: { name: 'realtime', input: { q: 'hello' } },
        labels: { 'genkit:type': 'flow' },
      },
      async (_m, span) => {
        spanId = span.spanContext().spanId;
        // Let the fire-and-forget start POST land while we're still running.
        for (let i = 0; i < 50 && pendingSeen.length === 0; i++) {
          await sleep(10);
          pendingSeen = posted
            .flatMap((t) => Object.values(t.spans ?? {}))
            .filter((s: any) => s.spanId === spanId);
        }
      }
    );

    // While running, the span was exported with endTime 0 (pending)...
    assert.equal(pendingSeen.length, 1, 'expected an in-progress export');
    assert.equal(pendingSeen[0].endTime, 0);
    // ...and it already carries the input provided when the span was opened.
    assert.equal(
      pendingSeen[0].attributes['genkit:input'],
      JSON.stringify({ q: 'hello' })
    );

    // After completion, a final export carries a real endTime.
    const all = await postsFor(spanId);
    const final = all.find((s: any) => s.endTime > 0);
    assert.ok(final, 'expected a final export with a non-zero endTime');
    assert.ok(final.endTime >= final.startTime);
  });

  it('does not export on start when realtime is disabled', async () => {
    delete process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;
    resetInstrumentation();

    let spanId = '';
    let midRun: any[] = [];
    await runInNewSpan(
      { metadata: { name: 'norealtime' }, labels: { 'genkit:type': 'flow' } },
      async (_m, span) => {
        spanId = span.spanContext().spanId;
        await sleep(30);
        midRun = posted
          .flatMap((t) => Object.values(t.spans ?? {}))
          .filter((s: any) => s.spanId === spanId);
      }
    );

    assert.equal(midRun.length, 0, 'should not export before completion');
    const all = await postsFor(spanId);
    assert.ok(
      all.some((s: any) => s.endTime > 0),
      'still exports once on completion'
    );
  });

  it('flushTracing waits for in-flight posts', async () => {
    delete process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;
    responseDelayMs = 200;

    let spanId = '';
    await runInNewSpan(
      { metadata: { name: 'flushme' }, labels: { 'genkit:type': 'flow' } },
      async (_m, span) => {
        spanId = span.spanContext().spanId;
      }
    );
    await flushTracing();

    // No polling: the final root span must already be saved.
    const spans = posted
      .flatMap((t) => Object.values(t.spans ?? {}))
      .filter((s: any) => s.spanId === spanId) as any[];
    assert.equal(spans.length, 1);
    assert.ok(spans[0].endTime > 0);
  });

  it('starts a new trace for an explicit isRoot span', async () => {
    delete process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;

    let outer = { traceId: '', spanId: '' };
    let inner = { traceId: '', spanId: '' };
    await runInNewSpan(
      { metadata: { name: 'outer' }, labels: { 'genkit:type': 'flow' } },
      async (_m, outerSpan) => {
        outer = outerSpan.spanContext();
        await runInNewSpan(
          {
            metadata: { name: 'inner', isRoot: true },
            labels: { 'genkit:type': 'flow' },
          },
          async (_m2, innerSpan) => {
            inner = innerSpan.spanContext();
          }
        );
      }
    );
    await flushTracing();

    assert.notEqual(inner.traceId, outer.traceId);
    const exported = posted
      .flatMap((t) => Object.values(t.spans ?? {}))
      .find((s: any) => s.spanId === inner.spanId) as any;
    assert.ok(exported, 'expected the inner span to be exported');
    assert.equal(exported.traceId, inner.traceId);
    assert.equal(exported.parentSpanId, undefined);
  });

  it('exports attributes written on the callback span', async () => {
    delete process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;

    let spanId = '';
    await runInNewSpan(
      { metadata: { name: 'attrs' }, labels: { 'genkit:type': 'flow' } },
      async (_m, span) => {
        spanId = span.spanContext().spanId;
        span.setAttribute('custom:count', 3);
        span.setAttributes({ 'custom:flag': true, 'genkit:name': 'spoofed' });
      }
    );
    await flushTracing();

    const exported = posted
      .flatMap((t) => Object.values(t.spans ?? {}))
      .find((s: any) => s.spanId === spanId) as any;
    assert.equal(exported.attributes['custom:count'], 3);
    assert.equal(exported.attributes['custom:flag'], true);
    // Genkit's own attributes are not overridable.
    assert.equal(exported.attributes['genkit:name'], 'attrs');
  });

  // Runs a failing span and returns the exported exception event attributes.
  async function exportedExceptionFor(err: Error) {
    delete process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;
    let spanId = '';
    await assert.rejects(
      runInNewSpan(
        { metadata: { name: 'boom' }, labels: { 'genkit:type': 'flow' } },
        async (_m, span) => {
          spanId = span.spanContext().spanId;
          throw err;
        }
      )
    );
    await flushTracing();
    const exported = posted
      .flatMap((t) => Object.values(t.spans ?? {}))
      .find((s: any) => s.spanId === spanId) as any;
    return exported.timeEvents.timeEvent.map(
      (e: any) => e.annotation.attributes
    );
  }

  it('omits an empty exception.message, like OTel recordException', async () => {
    // ToolInterruptError shape: empty message. The Dev UI relies on the missing
    // message to title the span "Interrupted".
    const err = Object.assign(new Error(), { name: 'ToolInterruptError' });
    const [attrs] = await exportedExceptionFor(err);
    assert.equal(attrs['exception.type'], 'ToolInterruptError');
    assert.ok(!('exception.message' in attrs));
    assert.ok(attrs['exception.stacktrace']);
  });

  it('prefers error code over name for exception.type', async () => {
    const err = Object.assign(new Error('nope'), { code: 404 });
    const [attrs] = await exportedExceptionFor(err);
    assert.equal(attrs['exception.type'], '404');
    assert.equal(attrs['exception.message'], 'nope');
  });

  it('records name and message for a plain error', async () => {
    const [attrs] = await exportedExceptionFor(new TypeError('bad input'));
    assert.equal(attrs['exception.type'], 'TypeError');
    assert.equal(attrs['exception.message'], 'bad input');
  });
});
