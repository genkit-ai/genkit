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

import { SpanKind, TraceFlags, type Span } from '@opentelemetry/api';
import {
  AlwaysOffSampler,
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
} from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import type {
  GenkitSpanContext,
  InstrumentationSpanInfo,
} from 'genkit/tracing';
import { beforeEach, describe, it } from 'node:test';
import { GenAiInstrumentation } from '../src/gen-ai-instrumentation.js';
import { RecordingLogger, getTracer } from './utils.js';

const exporter = new InMemorySpanExporter();
const tracer = getTracer(exporter);

function info(overrides: Partial<InstrumentationSpanInfo> = {}) {
  return {
    metadata: { name: 'test', ...(overrides.metadata ?? {}) },
    labels: overrides.labels ?? {},
    ...(overrides.links ? { links: overrides.links } : {}),
  } as InstrumentationSpanInfo;
}

const modelInfo = info({
  metadata: {
    name: 'googleai/gemini-flash-latest',
    input: { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
  },
  labels: { 'genkit:metadata:subtype': 'model' },
});

const modelResponse = {
  finishReason: 'stop',
  message: { role: 'model', content: [{ text: 'hello' }] },
};

const toolInfo = info({
  metadata: { name: 'getWeather', input: { city: 'SF' } },
  labels: { 'genkit:metadata:subtype': 'tool' },
});

async function runAndGetSpan(
  inst: GenAiInstrumentation,
  spanInfo: InstrumentationSpanInfo,
  result: unknown
): Promise<ReadableSpan> {
  await inst.runInNewSpan(spanInfo, async (_span: Span, _ctx) => result);
  const spans = exporter.getFinishedSpans();
  return spans[spans.length - 1];
}

describe('GenAiInstrumentation', () => {
  beforeEach(() => exporter.reset());

  it('creates a chat span for model actions', async () => {
    const inst = new GenAiInstrumentation({ tracer });
    const span = await runAndGetSpan(
      inst,
      info({
        metadata: {
          name: 'googleai/gemini-flash-latest',
          input: {
            messages: [{ role: 'user', content: [{ text: 'hi' }] }],
            config: { temperature: 0.5, maxOutputTokens: 100 },
          },
        },
        labels: { 'genkit:metadata:subtype': 'model' },
      }),
      { finishReason: 'stop', usage: { inputTokens: 3, outputTokens: 7 } }
    );

    assert.strictEqual(span.name, 'chat gemini-flash-latest');
    assert.strictEqual(span.kind, SpanKind.CLIENT);
    assert.strictEqual(span.attributes['gen_ai.operation.name'], 'chat');
    assert.strictEqual(
      span.attributes['gen_ai.request.model'],
      'gemini-flash-latest'
    );
    assert.strictEqual(span.attributes['gen_ai.provider.name'], 'gcp.gemini');
    assert.strictEqual(span.attributes['gen_ai.request.temperature'], 0.5);
    assert.strictEqual(span.attributes['gen_ai.request.max_tokens'], 100);
    assert.deepStrictEqual(span.attributes['gen_ai.response.finish_reasons'], [
      'stop',
    ]);
    assert.strictEqual(span.attributes['gen_ai.usage.input_tokens'], 3);
    assert.strictEqual(span.attributes['gen_ai.usage.output_tokens'], 7);
  });

  it('reports tool_calls finish reason when a tool request is present', async () => {
    const inst = new GenAiInstrumentation({ tracer });
    const span = await runAndGetSpan(
      inst,
      info({
        metadata: { name: 'googleai/gemini-flash-latest' },
        labels: { 'genkit:metadata:subtype': 'model' },
      }),
      {
        finishReason: 'stop',
        message: {
          role: 'model',
          content: [{ toolRequest: { name: 'getWeather', input: {} } }],
        },
      }
    );
    assert.deepStrictEqual(span.attributes['gen_ai.response.finish_reasons'], [
      'tool_calls',
    ]);
  });

  it('emits execute_tool spans only when enabled', async () => {
    const off = new GenAiInstrumentation({ tracer });
    const genericSpan = await runAndGetSpan(off, toolInfo, 'ok');
    assert.strictEqual(genericSpan.name, 'getWeather');
    assert.strictEqual(genericSpan.attributes['genkit.action.type'], 'tool');

    exporter.reset();
    const on = new GenAiInstrumentation({ tracer, emitToolSpans: true });
    const toolSpan = await runAndGetSpan(on, toolInfo, 'ok');
    assert.strictEqual(toolSpan.name, 'execute_tool getWeather');
    assert.strictEqual(
      toolSpan.attributes['gen_ai.operation.name'],
      'execute_tool'
    );
    assert.strictEqual(toolSpan.attributes['gen_ai.tool.name'], 'getWeather');
  });

  it('tags generic spans with the genkit action type', async () => {
    const inst = new GenAiInstrumentation({ tracer });
    const span = await runAndGetSpan(
      inst,
      info({
        metadata: { name: 'myFlow' },
        labels: { 'genkit:metadata:subtype': 'flow' },
      }),
      'ok'
    );
    assert.strictEqual(span.name, 'myFlow');
    assert.strictEqual(span.kind, SpanKind.INTERNAL);
    assert.strictEqual(span.attributes['genkit.action.type'], 'flow');
  });

  it('falls back to genkit:type when there is no subtype', async () => {
    // Directly-wrapped spans (generate, dotprompt, ...) carry only genkit:type.
    const inst = new GenAiInstrumentation({ tracer });
    const span = await runAndGetSpan(
      inst,
      info({
        metadata: { name: 'generate' },
        labels: { 'genkit:type': 'util' },
      }),
      'ok'
    );
    assert.strictEqual(span.name, 'generate');
    assert.strictEqual(span.attributes['genkit.action.type'], 'util');
  });

  it('captures content on the span in SPAN_ONLY mode', async () => {
    const inst = new GenAiInstrumentation({
      tracer,
      contentCapturingMode: 'SPAN_ONLY',
    });
    const span = await runAndGetSpan(
      inst,
      info({
        metadata: {
          name: 'googleai/gemini-flash-latest',
          input: {
            messages: [
              { role: 'system', content: [{ text: 'be nice' }] },
              { role: 'user', content: [{ text: 'hi' }] },
            ],
          },
        },
        labels: { 'genkit:metadata:subtype': 'model' },
      }),
      {
        finishReason: 'stop',
        message: { role: 'model', content: [{ text: 'hello' }] },
      }
    );
    const inputMessages = JSON.parse(
      span.attributes['gen_ai.input.messages'] as string
    );
    assert.strictEqual(inputMessages[0].role, 'user');
    const systemInstructions = JSON.parse(
      span.attributes['gen_ai.system_instructions'] as string
    );
    assert.strictEqual(systemInstructions[0].content, 'be nice');
    const outputMessages = JSON.parse(
      span.attributes['gen_ai.output.messages'] as string
    );
    assert.strictEqual(outputMessages[0].role, 'assistant');
    assert.strictEqual(outputMessages[0].finish_reason, 'stop');
  });

  it('reads message and finish reason from legacy candidates[0]', async () => {
    const inst = new GenAiInstrumentation({
      tracer,
      contentCapturingMode: 'SPAN_ONLY',
    });
    // Legacy plugin shape: no top-level message/finishReason, only candidates.
    const span = await runAndGetSpan(
      inst,
      info({
        metadata: {
          name: 'googleai/gemini-flash-latest',
          input: { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
        },
        labels: { 'genkit:metadata:subtype': 'model' },
      }),
      {
        usage: { inputTokens: 3, outputTokens: 7 },
        candidates: [
          {
            index: 0,
            finishReason: 'stop',
            message: { role: 'model', content: [{ text: 'hello there' }] },
          },
        ],
      }
    );
    assert.deepStrictEqual(span.attributes['gen_ai.response.finish_reasons'], [
      'stop',
    ]);
    const outputMessages = JSON.parse(
      span.attributes['gen_ai.output.messages'] as string
    );
    assert.strictEqual(outputMessages[0].role, 'assistant');
    assert.strictEqual(outputMessages[0].parts[0].content, 'hello there');
    assert.strictEqual(outputMessages[0].finish_reason, 'stop');
  });

  it('does not capture content by default (NO_CONTENT)', async () => {
    const inst = new GenAiInstrumentation({ tracer });
    const span = await runAndGetSpan(
      inst,
      info({
        metadata: {
          name: 'googleai/gemini-flash-latest',
          input: { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
        },
        labels: { 'genkit:metadata:subtype': 'model' },
      }),
      { finishReason: 'stop' }
    );
    assert.strictEqual(span.attributes['gen_ai.input.messages'], undefined);
  });

  it('emits content only on the event in EVENT_ONLY mode', async () => {
    const logger = new RecordingLogger();
    const inst = new GenAiInstrumentation({
      tracer,
      logger,
      contentCapturingMode: 'EVENT_ONLY',
    });
    const span = await runAndGetSpan(inst, modelInfo, modelResponse);
    assert.strictEqual(span.attributes['gen_ai.input.messages'], undefined);
    assert.strictEqual(span.attributes['gen_ai.output.messages'], undefined);

    assert.strictEqual(logger.records.length, 1);
    const [record] = logger.records;
    assert.strictEqual(
      record.eventName,
      'gen_ai.client.inference.operation.details'
    );
    const input = JSON.parse(
      record.attributes?.['gen_ai.input.messages'] as string
    );
    assert.strictEqual(input[0].parts[0].content, 'hi');
    const output = JSON.parse(
      record.attributes?.['gen_ai.output.messages'] as string
    );
    assert.strictEqual(output[0].parts[0].content, 'hello');
  });

  it('emits content on both span and event in SPAN_AND_EVENT mode', async () => {
    const logger = new RecordingLogger();
    const inst = new GenAiInstrumentation({
      tracer,
      logger,
      contentCapturingMode: 'SPAN_AND_EVENT',
    });
    const span = await runAndGetSpan(inst, modelInfo, modelResponse);
    assert.strictEqual(logger.records.length, 1);
    const eventAttrs = logger.records[0].attributes ?? {};
    for (const key of ['gen_ai.input.messages', 'gen_ai.output.messages']) {
      assert.ok(span.attributes[key]);
      assert.strictEqual(span.attributes[key], eventAttrs[key]);
    }
  });

  it('does not emit events in SPAN_ONLY or NO_CONTENT mode', async () => {
    for (const mode of ['SPAN_ONLY', 'NO_CONTENT'] as const) {
      const logger = new RecordingLogger();
      const inst = new GenAiInstrumentation({
        tracer,
        logger,
        contentCapturingMode: mode,
      });
      await runAndGetSpan(inst, modelInfo, modelResponse);
      assert.strictEqual(logger.records.length, 0, mode);
    }
  });

  it('records tool call arguments and result on the span in SPAN_ONLY mode', async () => {
    const inst = new GenAiInstrumentation({
      tracer,
      emitToolSpans: true,
      contentCapturingMode: 'SPAN_ONLY',
    });
    const span = await runAndGetSpan(inst, toolInfo, { temp: 20 });
    assert.strictEqual(
      span.attributes['gen_ai.tool.call.arguments'],
      '{"city":"SF"}'
    );
    assert.strictEqual(
      span.attributes['gen_ai.tool.call.result'],
      '{"temp":20}'
    );
  });

  it('records tool call content only on the event in EVENT_ONLY mode', async () => {
    const logger = new RecordingLogger();
    const inst = new GenAiInstrumentation({
      tracer,
      logger,
      emitToolSpans: true,
      contentCapturingMode: 'EVENT_ONLY',
    });
    const span = await runAndGetSpan(inst, toolInfo, { temp: 20 });
    assert.strictEqual(
      span.attributes['gen_ai.tool.call.arguments'],
      undefined
    );
    assert.strictEqual(logger.records.length, 1);
    assert.deepStrictEqual(logger.records[0].attributes, {
      'gen_ai.tool.call.arguments': '{"city":"SF"}',
      'gen_ai.tool.call.result': '{"temp":20}',
    });
  });

  it('does not record tool call content by default', async () => {
    const inst = new GenAiInstrumentation({ tracer, emitToolSpans: true });
    const span = await runAndGetSpan(inst, toolInfo, { temp: 20 });
    assert.strictEqual(
      span.attributes['gen_ai.tool.call.arguments'],
      undefined
    );
    assert.strictEqual(span.attributes['gen_ai.tool.call.result'], undefined);
  });

  it('does not fail the action on unserializable content', async () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const logger = new RecordingLogger();
    const inst = new GenAiInstrumentation({
      tracer,
      logger,
      emitToolSpans: true,
      contentCapturingMode: 'SPAN_AND_EVENT',
    });
    const span = await runAndGetSpan(inst, toolInfo, circular);
    assert.match(
      span.attributes['gen_ai.tool.call.result'] as string,
      /^Unable to encode/
    );
    assert.match(
      logger.records[0].attributes?.['gen_ai.tool.call.result'] as string,
      /^Unable to encode/
    );
  });

  it('forwards span links and labels', async () => {
    const link = {
      context: {
        traceId: 'a'.repeat(32),
        spanId: 'b'.repeat(16),
        traceFlags: TraceFlags.SAMPLED,
      },
    };
    const inst = new GenAiInstrumentation({ tracer, emitToolSpans: true });
    const labels = {
      'genkit:metadata:subtype': 'tool',
      team: 'search',
      // A label must not clobber a GenAI attribute.
      'gen_ai.tool.name': 'spoofed',
    };
    const span = await runAndGetSpan(
      inst,
      info({ metadata: { name: 'getWeather' }, labels, links: [link] }),
      'ok'
    );
    assert.strictEqual(span.attributes['team'], 'search');
    assert.strictEqual(span.attributes['gen_ai.tool.name'], 'getWeather');
    assert.strictEqual(span.links.length, 1);
    assert.strictEqual(span.links[0].context.traceId, link.context.traceId);
  });

  it('captures raw action IO only when enabled', async () => {
    const inst = new GenAiInstrumentation({ tracer, captureActionIO: true });
    const span = await runAndGetSpan(
      inst,
      info({
        metadata: {
          name: 'googleai/gemini-flash-latest',
          input: { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
        },
        labels: { 'genkit:metadata:subtype': 'model' },
      }),
      { finishReason: 'stop' }
    );
    assert.ok(span.attributes['genkit.input']);
    assert.ok(span.attributes['genkit.output']);
  });

  it('records error type and status on failure', async () => {
    const inst = new GenAiInstrumentation({ tracer });
    await assert.rejects(
      inst.runInNewSpan(
        info({
          metadata: { name: 'googleai/gemini-flash-latest' },
          labels: { 'genkit:metadata:subtype': 'model' },
        }),
        async () => {
          throw new TypeError('boom');
        }
      )
    );
    const span = exporter.getFinishedSpans().at(-1)!;
    assert.strictEqual(span.attributes['error.type'], 'TypeError');
    assert.strictEqual(span.status.code, 2 /* ERROR */);
  });

  it('warns once when no SDK is recording', async () => {
    // No injected tracer and no registered SDK -> the API returns a
    // non-recording span, so telemetry would be silently dropped.
    const inst = new GenAiInstrumentation();
    const warnings: unknown[] = [];
    const originalWarn = console.warn;
    console.warn = (...args: unknown[]) => {
      warnings.push(args[0]);
    };
    try {
      const modelInfo = info({
        metadata: { name: 'googleai/gemini-flash-latest' },
        labels: { 'genkit:metadata:subtype': 'model' },
      });
      await inst.runInNewSpan(modelInfo, async () => ({
        finishReason: 'stop',
      }));
      // A second call must not warn again.
      await inst.runInNewSpan(modelInfo, async () => ({
        finishReason: 'stop',
      }));
    } finally {
      console.warn = originalWarn;
    }
    const notRecording = warnings.filter(
      (w) => typeof w === 'string' && w.includes('no OpenTelemetry SDK')
    );
    assert.strictEqual(notRecording.length, 1);
  });

  it('does not warn when the SDK is registered but samples the span out', async () => {
    const sampledOut = new BasicTracerProvider({
      sampler: new AlwaysOffSampler(),
    }).getTracer('test');
    const inst = new GenAiInstrumentation({ tracer: sampledOut });
    const warnings: unknown[] = [];
    const originalWarn = console.warn;
    console.warn = (...args: unknown[]) => {
      warnings.push(args[0]);
    };
    try {
      await inst.runInNewSpan(
        info({
          metadata: { name: 'googleai/gemini-flash-latest' },
          labels: { 'genkit:metadata:subtype': 'model' },
        }),
        async () => ({ finishReason: 'stop' })
      );
    } finally {
      console.warn = originalWarn;
    }
    const notRecording = warnings.filter(
      (w) => typeof w === 'string' && w.includes('no OpenTelemetry SDK')
    );
    assert.strictEqual(notRecording.length, 0);
  });

  it('exposes trace/span ids and setMetadata through the context', async () => {
    const inst = new GenAiInstrumentation({ tracer });
    let captured: GenkitSpanContext | undefined;
    await inst.runInNewSpan(
      info({
        metadata: { name: 'myFlow' },
        labels: { 'genkit:metadata:subtype': 'flow' },
      }),
      async (_span, ctx) => {
        captured = ctx;
        ctx.setMetadata({ custom: 'value' });
        return 'ok';
      }
    );
    assert.match(captured!.traceId, /^[0-9a-f]{32}$/);
    assert.match(captured!.spanId, /^[0-9a-f]{16}$/);
    const span = exporter.getFinishedSpans().at(-1)!;
    assert.strictEqual(span.attributes['genkit:metadata:custom'], 'value');
  });

  it('setMetadata keeps primitives, skips nullish, and never throws', async () => {
    const inst = new GenAiInstrumentation({ tracer });
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    await inst.runInNewSpan(
      info({
        metadata: { name: 'myFlow' },
        labels: { 'genkit:metadata:subtype': 'flow' },
      }),
      async (_span, ctx) => {
        ctx.setMetadata({
          count: 3,
          flag: true,
          obj: { a: 1 },
          missing: undefined,
          empty: null,
          big: BigInt(1),
          circular,
        });
        return 'ok';
      }
    );
    const attrs = exporter.getFinishedSpans().at(-1)!.attributes;
    assert.strictEqual(attrs['genkit:metadata:count'], 3);
    assert.strictEqual(attrs['genkit:metadata:flag'], true);
    assert.strictEqual(attrs['genkit:metadata:obj'], '{"a":1}');
    assert.ok(!('genkit:metadata:missing' in attrs));
    assert.ok(!('genkit:metadata:empty' in attrs));
    assert.match(attrs['genkit:metadata:big'] as string, /^Unable to encode/);
    assert.match(
      attrs['genkit:metadata:circular'] as string,
      /^Unable to encode/
    );
  });
});
