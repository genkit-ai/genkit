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

import { SpanKind, type Span } from '@opentelemetry/api';
import {
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
import { getTracer } from './utils.js';

const exporter = new InMemorySpanExporter();
const tracer = getTracer(exporter);

function info(overrides: Partial<InstrumentationSpanInfo> = {}) {
  return {
    metadata: { name: 'test', ...(overrides.metadata ?? {}) },
    labels: overrides.labels ?? {},
  } as InstrumentationSpanInfo;
}

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
    const toolInfo = info({
      metadata: { name: 'getWeather' },
      labels: { 'genkit:metadata:subtype': 'tool' },
    });

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

  it('captures content on the span in span mode', async () => {
    const inst = new GenAiInstrumentation({
      tracer,
      captureContent: true,
      contentMode: 'span',
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

  it('does not capture content by default', async () => {
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
});
