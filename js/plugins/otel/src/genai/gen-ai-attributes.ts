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

/**
 * Pure helpers for mapping Genkit data to OpenTelemetry GenAI semantic
 * conventions. Deliberately free of any OpenTelemetry imports so the mapping
 * logic can be unit tested in isolation.
 *
 * See the spec:
 * https://github.com/open-telemetry/semantic-conventions-genai
 */

/**
 * Canonical `gen_ai.*` attribute names used by this instrumentation.
 *
 * Grouped as a namespace of constants so call sites read like
 * `GenAiAttr.requestModel` rather than repeating string literals.
 */
export const GenAiAttr = {
  operationName: 'gen_ai.operation.name',
  providerName: 'gen_ai.provider.name',

  requestModel: 'gen_ai.request.model',
  requestTemperature: 'gen_ai.request.temperature',
  requestTopP: 'gen_ai.request.top_p',
  requestTopK: 'gen_ai.request.top_k',
  requestMaxTokens: 'gen_ai.request.max_tokens',
  requestStopSequences: 'gen_ai.request.stop_sequences',
  requestFrequencyPenalty: 'gen_ai.request.frequency_penalty',
  requestPresencePenalty: 'gen_ai.request.presence_penalty',
  requestSeed: 'gen_ai.request.seed',
  requestChoiceCount: 'gen_ai.request.choice.count',

  outputType: 'gen_ai.output.type',

  responseFinishReasons: 'gen_ai.response.finish_reasons',

  /** Distinguishes token-usage measurements: `input` vs `output`. */
  tokenType: 'gen_ai.token.type',

  usageInputTokens: 'gen_ai.usage.input_tokens',
  usageOutputTokens: 'gen_ai.usage.output_tokens',
  usageReasoningOutputTokens: 'gen_ai.usage.reasoning.output_tokens',
  usageCacheReadInputTokens: 'gen_ai.usage.cache_read.input_tokens',

  toolName: 'gen_ai.tool.name',
  toolType: 'gen_ai.tool.type',

  // Content attributes (opt-in; may contain PII).
  inputMessages: 'gen_ai.input.messages',
  outputMessages: 'gen_ai.output.messages',
  systemInstructions: 'gen_ai.system_instructions',

  errorType: 'error.type',
} as const;

/**
 * Non-reserved `genkit.*` attributes. Kept out of the `gen_ai.*` namespace so
 * GenAI-aware backends (e.g. Jaeger's GenAI view) never try to render raw
 * Genkit payloads as spec message content.
 */
export const GenkitAttr = {
  /**
   * Keeps the span tree connected across Genkit action types that have no
   * GenAI mapping (flow, util, etc.).
   */
  actionType: 'genkit.action.type',

  /** Raw Genkit action input/output as JSON strings (opt-in; may contain PII). */
  input: 'genkit.input',
  output: 'genkit.output',
} as const;

/** Well-known values for `gen_ai.operation.name`. */
export const GenAiOperation = {
  chat: 'chat',
  executeTool: 'execute_tool',
} as const;

/** Canonical `gen_ai.*` metric instrument names. */
export const GenAiMetric = {
  tokenUsage: 'gen_ai.client.token.usage',
  operationDuration: 'gen_ai.client.operation.duration',
} as const;

/**
 * The OTel GenAI semantic-conventions version this instrumentation targets.
 * Recorded so future readers know which shape the mapping was written against.
 */
export const genAiSemConvVersion = '1.38.0';

/**
 * The dedicated event that carries prompt/response content independently of
 * the span, per the spec.
 */
export const genAiOperationDetailsEvent =
  'gen_ai.client.inference.operation.details';

/** The spec's canonical opt-in env var for capturing message content. */
export const captureContentEnvVar =
  'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT';

/**
 * Splits a fully qualified Genkit model name into `(prefix, model)`.
 *
 * `googleai/gemini-flash-latest` -> `{ prefix: 'googleai', model: 'gemini-flash-latest' }`.
 * A name without a `/` yields an undefined prefix and the name as the model.
 */
export function splitModelName(name: string): {
  prefix?: string;
  model: string;
} {
  const i = name.indexOf('/');
  if (i < 0) return { model: name };
  return { prefix: name.substring(0, i), model: name.substring(i + 1) };
}

/**
 * Derives `gen_ai.provider.name` from a Genkit model-name prefix.
 *
 * Maps known Genkit plugin prefixes to the spec's well-known provider names,
 * and passes unknown prefixes through lowercased so custom plugins still get a
 * discriminator. Returns undefined when there is no prefix.
 */
export function deriveProviderName(prefix?: string): string | undefined {
  if (!prefix) return undefined;
  switch (prefix.toLowerCase()) {
    case 'googleai':
    case 'google-genai':
    case 'google_genai':
      // Gemini API (AI Studio), distinct from Vertex AI.
      return 'gcp.gemini';
    case 'vertexai':
    case 'vertex-ai':
    case 'vertex_ai':
      return 'gcp.vertex_ai';
    case 'openai':
      return 'openai';
    case 'anthropic':
      return 'anthropic';
    default:
      return prefix.toLowerCase();
  }
}

/**
 * Maps a Genkit finish reason string to the GenAI `finish_reasons` value.
 *
 * `failed` selects the fallback for ambiguous reasons (`other`/`unknown`):
 * `error` when the span failed, otherwise `stop`.
 */
export function mapFinishReason(
  genkitReason: string | undefined,
  failed: boolean
): string {
  switch (genkitReason) {
    case 'stop':
      return 'stop';
    case 'length':
      return 'length';
    case 'blocked':
      return 'content_filter';
    case 'interrupted':
      // No exact spec value; treat an interrupted turn as a normal stop.
      return 'stop';
    case 'other':
    case 'unknown':
    default:
      return failed ? 'error' : 'stop';
  }
}

/**
 * Derives `gen_ai.output.type` from an output format / content type.
 *
 * Returns `json` when JSON output was requested, `text` when a text format was
 * requested, otherwise undefined (omit the attribute).
 */
export function deriveOutputType(
  format?: string,
  contentType?: string
): string | undefined {
  const f = format?.toLowerCase();
  const ct = contentType?.toLowerCase();
  if (f === 'json' || (ct && ct.includes('json'))) return 'json';
  if (f === 'text' || (ct && ct.startsWith('text/'))) return 'text';
  return undefined;
}
