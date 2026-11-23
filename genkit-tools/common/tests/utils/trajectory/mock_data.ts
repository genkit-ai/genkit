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

import type {
  GenerationUsage,
  MessageData,
  SpanData,
  ToolDefinition,
} from '../../../src/types/index.js';
import { makeSpan, makeTrace } from '../trace.js';

export { makeSpan, makeTrace };

type SpanOverrides = Partial<Omit<SpanData, 'attributes'>> & {
  attributes?: Record<string, unknown>;
};

function stringifyIfDefined(val: unknown): Record<string, string> {
  return val !== undefined ? { value: JSON.stringify(val) } : {};
}

/** Options for {@link makeAgentSpan}. */
export interface MakeAgentSpanOptions extends SpanOverrides {
  name?: string;
  sessionId?: string;
  input?: unknown;
  output?: unknown;
}

/** Builds an `/agent/<name>` SpanData with serialized `genkit:input` / `genkit:output` attributes. */
export function makeAgentSpan(options: MakeAgentSpanOptions = {}): SpanData {
  const {
    name = 'weatherAgent',
    sessionId,
    input,
    output,
    attributes,
    ...rest
  } = options;
  const cleanName = name.replace(/^\/agent\//, '');
  const inAttr = stringifyIfDefined(input);
  const outAttr = stringifyIfDefined(output);

  return makeSpan({
    spanId: 'root-agent',
    startTime: 100,
    endTime: 500,
    ...rest,
    displayName:
      rest.displayName ??
      (name.startsWith('/agent/') ? name : `/agent/${cleanName}`),
    attributes: {
      'genkit:type': 'action',
      'genkit:metadata:subtype': 'agent',
      'genkit:name': cleanName,
      ...(sessionId ? { 'genkit:metadata:agent:sessionId': sessionId } : {}),
      ...(inAttr.value !== undefined ? { 'genkit:input': inAttr.value } : {}),
      ...(outAttr.value !== undefined
        ? { 'genkit:output': outAttr.value }
        : {}),
      ...attributes,
    },
  });
}

/** Options for {@link makeTurnSpan}. */
export interface MakeTurnSpanOptions extends SpanOverrides {
  turn?: number;
  input?: unknown;
  output?: unknown;
}

/** Builds a `runTurn-<N>` flowStep SpanData with serialized `genkit:input` / `genkit:output` attributes. */
export function makeTurnSpan(options: MakeTurnSpanOptions = {}): SpanData {
  const { turn = 1, input, output, attributes, ...rest } = options;
  const inAttr = stringifyIfDefined(input);
  const outAttr = stringifyIfDefined(output);

  return makeSpan({
    spanId: `turn-${turn}`,
    parentSpanId: 'root-agent',
    startTime: 105,
    endTime: 495,
    ...rest,
    displayName: rest.displayName ?? `runTurn-${turn}`,
    attributes: {
      'genkit:type': 'flowStep',
      ...(inAttr.value !== undefined ? { 'genkit:input': inAttr.value } : {}),
      ...(outAttr.value !== undefined
        ? { 'genkit:output': outAttr.value }
        : {}),
      ...attributes,
    },
  });
}

/** Options for {@link makeModelSpan}. */
export interface MakeModelSpanOptions extends SpanOverrides {
  model?: string;
  messages?: Array<
    | MessageData
    | { role: string; content: unknown; metadata?: Record<string, unknown> }
  >;
  tools?: Partial<ToolDefinition>[];
  input?: unknown;
  outputMessage?: MessageData;
  finishReason?: string;
  usage?: GenerationUsage;
  outputError?: { message: string; details?: { stack?: string } };
  candidates?: unknown[];
  output?: unknown;
}

/** Builds a `/model/<model>` SpanData hiding `genkit:input` / `genkit:output` JSON serialization. */
export function makeModelSpan(options: MakeModelSpanOptions = {}): SpanData {
  const {
    model = 'googleai/gemini-2.5-pro',
    messages,
    tools,
    input,
    outputMessage,
    finishReason,
    usage,
    outputError,
    candidates,
    output,
    attributes,
    ...rest
  } = options;
  const cleanModel = model.replace(/^\/model\//, '');

  const resolvedInput =
    input !== undefined
      ? input
      : messages !== undefined || tools !== undefined
        ? {
            ...(messages !== undefined ? { messages } : {}),
            ...(tools !== undefined ? { tools } : {}),
          }
        : undefined;

  const hasStructuredOutput =
    outputMessage !== undefined ||
    finishReason !== undefined ||
    usage !== undefined ||
    outputError !== undefined ||
    candidates !== undefined;
  const resolvedOutput =
    output !== undefined
      ? output
      : hasStructuredOutput
        ? {
            ...(outputMessage !== undefined ? { message: outputMessage } : {}),
            ...(finishReason !== undefined
              ? { finishReason }
              : outputMessage !== undefined
                ? { finishReason: 'stop' }
                : {}),
            ...(usage !== undefined ? { usage } : {}),
            ...(outputError !== undefined ? { error: outputError } : {}),
            ...(candidates !== undefined ? { candidates } : {}),
          }
        : undefined;

  const inAttr = stringifyIfDefined(resolvedInput);
  const outAttr = stringifyIfDefined(resolvedOutput);

  return makeSpan({
    spanId: 'model-1',
    parentSpanId: 'turn-1',
    startTime: 110,
    endTime: 250,
    ...rest,
    displayName:
      rest.displayName ??
      (model.startsWith('/model/') ? model : `/model/${cleanModel}`),
    attributes: {
      'genkit:type': 'action',
      'genkit:metadata:subtype': 'model',
      'genkit:name': cleanModel,
      ...(inAttr.value !== undefined ? { 'genkit:input': inAttr.value } : {}),
      ...(outAttr.value !== undefined
        ? { 'genkit:output': outAttr.value }
        : {}),
      ...attributes,
    },
  });
}

/** Options for {@link makeToolSpan}. */
export interface MakeToolSpanOptions extends SpanOverrides {
  name?: string;
  input?: unknown;
  output?: unknown;
  interrupt?: unknown;
  resumed?: unknown;
}

/** Builds a `/tool/<name>` or `/tool.v2/<name>` SpanData hiding JSON serialization. */
export function makeToolSpan(options: MakeToolSpanOptions = {}): SpanData {
  const {
    name = 'getWeather',
    input,
    output,
    interrupt,
    resumed,
    attributes,
    ...rest
  } = options;
  const isV2 = name.startsWith('/tool.v2/');
  const cleanName = name.replace(/^\/tool(?:\.v2)?\//, '');
  const inAttr = stringifyIfDefined(input);
  const outAttr = stringifyIfDefined(output);

  return makeSpan({
    spanId: 'tool-1',
    parentSpanId: 'turn-1',
    startTime: 255,
    endTime: 320,
    ...rest,
    displayName:
      rest.displayName ??
      (/^\/tool(?:\.v2)?\//.test(name) ? name : `/tool/${cleanName}`),
    attributes: {
      'genkit:type': 'action',
      'genkit:metadata:subtype': isV2 ? 'tool.v2' : 'tool',
      'genkit:name': cleanName,
      ...(interrupt !== undefined
        ? {
            'genkit:metadata:interrupt':
              typeof interrupt === 'string'
                ? interrupt
                : JSON.stringify(interrupt),
          }
        : {}),
      ...(resumed !== undefined
        ? {
            'genkit:metadata:resumed':
              typeof resumed === 'boolean' || typeof resumed === 'string'
                ? resumed
                : JSON.stringify(resumed),
          }
        : {}),
      ...(inAttr.value !== undefined ? { 'genkit:input': inAttr.value } : {}),
      ...(outAttr.value !== undefined
        ? { 'genkit:output': outAttr.value }
        : {}),
      ...attributes,
    },
  });
}
