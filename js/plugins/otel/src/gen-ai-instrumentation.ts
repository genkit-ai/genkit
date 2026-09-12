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

import {
  metrics,
  SpanKind,
  SpanStatusCode,
  trace,
  type Attributes,
  type Meter,
  type Span,
  type Tracer,
} from '@opentelemetry/api';
import { logs } from '@opentelemetry/api-logs';
import type {
  GenerateRequest,
  GenerateResponseData,
  MessageData,
} from 'genkit/model';
import type {
  GenkitSpanContext,
  Instrumentation,
  InstrumentationSpanInfo,
} from 'genkit/tracing';
import {
  captureContentEnvVar,
  deriveOutputType,
  deriveProviderName,
  GenAiAttr,
  GenAiOperation,
  genAiOperationDetailsEvent,
  GenkitAttr,
  mapFinishReason,
  splitModelName,
} from './genai/gen-ai-attributes.js';
import {
  isToolRequestPart,
  mapOutputMessage,
  normalizeMessages,
} from './genai/gen-ai-message-mapping.js';
import { GenAiMetrics } from './genai/gen-ai-metrics.js';

/** The continuation passed by the dispatcher; returns the raw action result. */
type Next<T> = (span: Span, ctx: GenkitSpanContext) => Promise<T>;

/** Where captured prompt/response content is recorded. */
export type GenAiContentMode =
  /**
   * Emit a single `gen_ai.client.inference.operation.details` event carrying
   * the content, correlated to the span via context. Keeps large bodies off
   * the span. This is the default when content capture is enabled.
   */
  | 'event'
  /** Attach content directly to the span as `gen_ai.*` JSON-string attributes. */
  | 'span';

/** Options for {@link GenAiInstrumentation}. */
export interface GenAiInstrumentationOptions {
  /**
   * Whether to capture spec-shaped GenAI message content on model spans, i.e.
   * the `gen_ai.*.messages` attributes / operation.details event.
   *
   * Content may contain PII, so it is off by default. Also enabled when the
   * env var `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` is set.
   */
  captureContent?: boolean;

  /** Where captured content is recorded (event vs span attributes). */
  contentMode?: GenAiContentMode;

  /**
   * Whether to capture raw Genkit action input/output as `genkit.input` /
   * `genkit.output` JSON attributes on every span (model, tool, flow, etc.).
   *
   * Independent of {@link captureContent}: it records the raw Genkit payloads
   * rather than the spec-shaped `gen_ai.*` content. May contain PII, off by
   * default.
   */
  captureActionIO?: boolean;

  /** Whether to emit `execute_tool` spans for tool actions. Off by default. */
  emitToolSpans?: boolean;

  /**
   * Whether to emit the spec's GenAI client metrics (token usage, operation
   * duration) for model operations. On by default; low cardinality and cheap.
   */
  emitMetrics?: boolean;

  /** Instrumentation scope name for the tracer/logger/meter. */
  scopeName?: string;

  /** Optional explicit tracer (escape hatch). */
  tracer?: Tracer;

  /** Optional explicit meter (escape hatch). */
  meter?: Meter;
}

function captureContentFromEnv(): boolean {
  return process.env[captureContentEnvVar]?.toLowerCase() === 'true';
}

/** The label the dispatcher stores the Genkit action type under. */
const SUBTYPE_LABEL = 'genkit:metadata:subtype';

/**
 * An {@link Instrumentation} that emits OpenTelemetry telemetry following the
 * [OTel GenAI semantic conventions][spec].
 *
 * The application owns SDK setup: configure a TracerProvider / MeterProvider /
 * LoggerProvider (e.g. via `@opentelemetry/sdk-node`) before constructing
 * Genkit. When no provider is configured, `@opentelemetry/api` returns non-
 * recording spans / no-op instruments and this provider is effectively inert.
 *
 * Wire it up with `configureInstrumentation(new GenAiInstrumentation())` from
 * `genkit/tracing`. It composes with the built-in dev instrumentation, which
 * feeds the Developer UI on a separate pipeline.
 *
 * [spec]: https://github.com/open-telemetry/semantic-conventions-genai
 */
export class GenAiInstrumentation implements Instrumentation {
  private readonly captureContent: boolean;
  private readonly contentMode: GenAiContentMode;
  private readonly captureActionIO: boolean;
  private readonly emitToolSpans: boolean;
  private readonly emitMetrics: boolean;
  private readonly scopeName: string;
  private readonly injectedTracer?: Tracer;
  private readonly injectedMeter?: Meter;

  private cachedTracer?: Tracer;
  private cachedMetrics?: GenAiMetrics;

  constructor(options: GenAiInstrumentationOptions = {}) {
    this.captureContent = options.captureContent ?? captureContentFromEnv();
    this.contentMode = options.contentMode ?? 'event';
    this.captureActionIO = options.captureActionIO ?? false;
    this.emitToolSpans = options.emitToolSpans ?? false;
    this.emitMetrics = options.emitMetrics ?? true;
    this.scopeName = options.scopeName ?? 'genkit-genai';
    this.injectedTracer = options.tracer;
    this.injectedMeter = options.meter;
  }

  private get tracer(): Tracer {
    return (this.cachedTracer ??=
      this.injectedTracer ?? trace.getTracer(this.scopeName));
  }

  private get metrics(): GenAiMetrics {
    return (this.cachedMetrics ??= new GenAiMetrics(
      this.injectedMeter ?? metrics.getMeter(this.scopeName)
    ));
  }

  async runInNewSpan<T>(
    info: InstrumentationSpanInfo,
    next: Next<T>
  ): Promise<T> {
    const subtype = info.labels?.[SUBTYPE_LABEL];
    switch (subtype) {
      case 'model':
        return this.runModelSpan(info, next);
      case 'tool':
        if (this.emitToolSpans) return this.runToolSpan(info, next);
        return this.runGenericSpan(info, next, subtype);
      default:
        return this.runGenericSpan(info, next, subtype);
    }
  }

  private async runModelSpan<T>(
    info: InstrumentationSpanInfo,
    next: Next<T>
  ): Promise<T> {
    const { model, prefix } = splitModelName(info.metadata.name);
    const provider = deriveProviderName(prefix);
    const request = asGenerateRequest(info.metadata.input);

    const attrs: Attributes = {
      [GenAiAttr.operationName]: GenAiOperation.chat,
      [GenAiAttr.requestModel]: model,
      ...(provider ? { [GenAiAttr.providerName]: provider } : {}),
    };
    if (request) this.addRequestConfigAttributes(attrs, request);

    // Base metric attributes shared by both histograms: low cardinality only.
    const metricAttrs: Attributes = {
      [GenAiAttr.operationName]: GenAiOperation.chat,
      [GenAiAttr.requestModel]: model,
      ...(provider ? { [GenAiAttr.providerName]: provider } : {}),
    };
    const startTime = performance.now();

    return this.tracer.startActiveSpan(
      `${GenAiOperation.chat} ${model}`,
      { kind: SpanKind.CLIENT, attributes: attrs },
      async (span) => {
        try {
          const output = await next(span, spanContextOf(span));
          const response = asGenerateResponse(output);
          if (response) this.addResponseAttributes(span, response, false);
          if (this.captureContent) this.recordContent(span, request, response);
          this.maybeCaptureActionIO(span, info.metadata.input, output);
          if (this.emitMetrics) {
            this.recordModelMetrics(startTime, metricAttrs, response);
          }
          return output;
        } catch (e) {
          this.recordError(span, e);
          if (this.emitMetrics) {
            this.recordModelMetrics(
              startTime,
              metricAttrs,
              undefined,
              errorTypeOf(e)
            );
          }
          throw e;
        } finally {
          span.end();
        }
      }
    );
  }

  private async runToolSpan<T>(
    info: InstrumentationSpanInfo,
    next: Next<T>
  ): Promise<T> {
    const attrs: Attributes = {
      [GenAiAttr.operationName]: GenAiOperation.executeTool,
      [GenAiAttr.toolName]: info.metadata.name,
      [GenAiAttr.toolType]: 'function',
    };
    return this.tracer.startActiveSpan(
      `${GenAiOperation.executeTool} ${info.metadata.name}`,
      { kind: SpanKind.INTERNAL, attributes: attrs },
      async (span) => {
        try {
          const output = await next(span, spanContextOf(span));
          this.maybeCaptureActionIO(span, info.metadata.input, output);
          return output;
        } catch (e) {
          this.recordError(span, e);
          throw e;
        } finally {
          span.end();
        }
      }
    );
  }

  private async runGenericSpan<T>(
    info: InstrumentationSpanInfo,
    next: Next<T>,
    subtype: string | undefined
  ): Promise<T> {
    const attrs: Attributes = subtype
      ? { [GenkitAttr.actionType]: subtype }
      : {};
    return this.tracer.startActiveSpan(
      info.metadata.name,
      { kind: SpanKind.INTERNAL, attributes: attrs },
      async (span) => {
        try {
          const output = await next(span, spanContextOf(span));
          this.maybeCaptureActionIO(span, info.metadata.input, output);
          return output;
        } catch (e) {
          this.recordError(span, e);
          throw e;
        } finally {
          span.end();
        }
      }
    );
  }

  /** Records the token-usage and operation-duration metrics for a model call. */
  private recordModelMetrics(
    startTime: number,
    baseAttrs: Attributes,
    response?: GenerateResponseData,
    errorType?: string
  ): void {
    const usage = response?.usage;
    if (usage) {
      this.metrics.recordTokenUsage(
        baseAttrs,
        usage.inputTokens,
        usage.outputTokens
      );
    }
    const seconds = (performance.now() - startTime) / 1000;
    this.metrics.recordDuration(seconds, {
      ...baseAttrs,
      ...(errorType ? { [GenAiAttr.errorType]: errorType } : {}),
    });
  }

  /**
   * Records raw Genkit input/output on `span` as `genkit.*` JSON attributes
   * when {@link captureActionIO} is enabled. Kept out of the reserved
   * `gen_ai.*` namespace so GenAI-aware backends don't misrender it.
   */
  private maybeCaptureActionIO(
    span: Span,
    input: unknown,
    output: unknown
  ): void {
    if (!this.captureActionIO) return;
    this.setJsonAttribute(span, GenkitAttr.input, input);
    this.setJsonAttribute(span, GenkitAttr.output, output);
  }

  private addRequestConfigAttributes(
    attrs: Attributes,
    request: GenerateRequest
  ): void {
    const config = (request.config ?? {}) as Record<string, unknown>;
    const num = (v: unknown): number | undefined =>
      typeof v === 'number' ? v : undefined;

    const temperature = num(config.temperature);
    if (temperature != null) attrs[GenAiAttr.requestTemperature] = temperature;
    const topP = num(config.topP);
    if (topP != null) attrs[GenAiAttr.requestTopP] = topP;
    const topK = num(config.topK);
    if (topK != null) attrs[GenAiAttr.requestTopK] = topK;
    const maxTokens = num(config.maxOutputTokens);
    if (maxTokens != null) attrs[GenAiAttr.requestMaxTokens] = maxTokens;
    if (Array.isArray(config.stopSequences) && config.stopSequences.length) {
      attrs[GenAiAttr.requestStopSequences] = config.stopSequences.map(String);
    }
    const frequencyPenalty = num(config.frequencyPenalty);
    if (frequencyPenalty != null) {
      attrs[GenAiAttr.requestFrequencyPenalty] = frequencyPenalty;
    }
    const presencePenalty = num(config.presencePenalty);
    if (presencePenalty != null) {
      attrs[GenAiAttr.requestPresencePenalty] = presencePenalty;
    }
    const seed = num(config.seed);
    if (seed != null) attrs[GenAiAttr.requestSeed] = seed;
    const choiceCount = num(config.candidateCount);
    if (choiceCount != null && choiceCount !== 1) {
      attrs[GenAiAttr.requestChoiceCount] = choiceCount;
    }

    const output = request.output;
    if (output) {
      const outputType = deriveOutputType(output.format, output.contentType);
      if (outputType) attrs[GenAiAttr.outputType] = outputType;
    }
  }

  private addResponseAttributes(
    span: Span,
    response: GenerateResponseData,
    failed: boolean
  ): void {
    const finishReasons = this.resolveFinishReasons(response, failed);
    if (finishReasons.length) {
      span.setAttribute(GenAiAttr.responseFinishReasons, finishReasons);
    }
    const usage = response.usage;
    if (usage) {
      if (usage.inputTokens != null) {
        span.setAttribute(GenAiAttr.usageInputTokens, usage.inputTokens);
      }
      if (usage.outputTokens != null) {
        span.setAttribute(GenAiAttr.usageOutputTokens, usage.outputTokens);
      }
      if (usage.thoughtsTokens != null) {
        span.setAttribute(
          GenAiAttr.usageReasoningOutputTokens,
          usage.thoughtsTokens
        );
      }
      if (usage.cachedContentTokens != null) {
        span.setAttribute(
          GenAiAttr.usageCacheReadInputTokens,
          usage.cachedContentTokens
        );
      }
    }
  }

  private resolveFinishReasons(
    response: GenerateResponseData,
    failed: boolean
  ): string[] {
    const content = response.message?.content ?? [];
    if (content.some(isToolRequestPart)) {
      // Following the OpenAI GenAI profile: a turn ending in tool calls is the
      // more informative signal for consumers.
      return ['tool_calls'];
    }
    return [mapFinishReason(response.finishReason, failed)];
  }

  private recordContent(
    span: Span,
    request: GenerateRequest | undefined,
    response: GenerateResponseData | undefined
  ): void {
    const inputMessages = request
      ? normalizeMessages(request.messages)
      : undefined;
    const outputMessages: Record<string, unknown>[] = [];
    if (response?.message) {
      const reason = this.resolveFinishReasons(response, false)[0];
      outputMessages.push(
        mapOutputMessage(response.message as MessageData, reason)
      );
    }

    if (this.contentMode === 'span') {
      if (inputMessages) {
        this.setJsonAttribute(
          span,
          GenAiAttr.inputMessages,
          inputMessages.messages
        );
        if (inputMessages.systemInstructions.length) {
          this.setJsonAttribute(
            span,
            GenAiAttr.systemInstructions,
            inputMessages.systemInstructions
          );
        }
      }
      if (outputMessages.length) {
        this.setJsonAttribute(span, GenAiAttr.outputMessages, outputMessages);
      }
      return;
    }

    // Event mode (default): emit a single operation.details event correlated
    // to the span via the active context.
    const eventAttrs: Attributes = {};
    if (inputMessages) {
      eventAttrs[GenAiAttr.inputMessages] = JSON.stringify(
        inputMessages.messages
      );
      if (inputMessages.systemInstructions.length) {
        eventAttrs[GenAiAttr.systemInstructions] = JSON.stringify(
          inputMessages.systemInstructions
        );
      }
    }
    if (outputMessages.length) {
      eventAttrs[GenAiAttr.outputMessages] = JSON.stringify(outputMessages);
    }
    logs.getLogger(this.scopeName).emit({
      eventName: genAiOperationDetailsEvent,
      attributes: eventAttrs,
    });
  }

  private recordError(span: Span, e: unknown): void {
    const message = e instanceof Error ? e.message : String(e);
    span.setStatus({ code: SpanStatusCode.ERROR, message });
    span.setAttribute(GenAiAttr.errorType, errorTypeOf(e));
    if (e instanceof Error) span.recordException(e);
  }

  private setJsonAttribute(span: Span, key: string, value: unknown): void {
    if (value == null) return;
    let encoded: string;
    try {
      encoded = JSON.stringify(value);
    } catch (e) {
      encoded = `Unable to encode: ${e}`;
    }
    span.setAttribute(key, encoded);
  }
}

/** Reports the error type for the `error.type` attribute. */
function errorTypeOf(e: unknown): string {
  if (e instanceof Error) return e.name;
  return typeof e;
}

/** A backend-independent span context derived from the OTel span. */
function spanContextOf(span: Span): GenkitSpanContext {
  return {
    get traceId() {
      return span.spanContext().traceId;
    },
    get spanId() {
      return span.spanContext().spanId;
    },
    setMetadata(values: Record<string, unknown>) {
      for (const [k, v] of Object.entries(values)) {
        span.setAttribute(
          `genkit:metadata:${k}`,
          typeof v === 'string' ? v : JSON.stringify(v)
        );
      }
    },
  };
}

/** Duck-types the span input as a GenerateRequest. Defensive, no zod parse. */
function asGenerateRequest(input: unknown): GenerateRequest | undefined {
  if (
    input &&
    typeof input === 'object' &&
    Array.isArray((input as { messages?: unknown }).messages)
  ) {
    return input as GenerateRequest;
  }
  return undefined;
}

/** Duck-types the action result as a GenerateResponseData. */
function asGenerateResponse(output: unknown): GenerateResponseData | undefined {
  if (
    output &&
    typeof output === 'object' &&
    ('message' in output ||
      'finishReason' in output ||
      'usage' in output ||
      'candidates' in output)
  ) {
    return output as GenerateResponseData;
  }
  return undefined;
}
