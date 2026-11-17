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
  TraceData,
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

const WEATHER_PREAMBLE_MESSAGE: MessageData = {
  role: 'system',
  content: [{ text: 'You are a helpful weather assistant.' }],
  metadata: { agentPreamble: true },
};

const WEATHER_USER_MESSAGE: MessageData = {
  role: 'user',
  content: [{ text: 'What is the weather in Seattle?' }],
};

/** Canonical root agent span (`/agent/weatherAgent`). */
export const MOCK_SPAN_AGENT: SpanData = makeAgentSpan({
  spanId: 'root-agent',
  name: 'weatherAgent',
  sessionId: 'sess-123',
  startTime: 100,
  endTime: 500,
});

/** Canonical first agent turn span (`runTurn-1`). */
export const MOCK_SPAN_TURN_1: SpanData = makeTurnSpan({
  spanId: 'turn-1',
  turn: 1,
  parentSpanId: 'root-agent',
  startTime: 105,
  endTime: 495,
});

/** Canonical model span emitting reasoning + `getWeather` `toolRequest`. */
export const MOCK_SPAN_MODEL_TOOL_REQ: SpanData = makeModelSpan({
  spanId: 'model-1',
  parentSpanId: 'turn-1',
  model: 'googleai/gemini-2.5-pro',
  startTime: 110,
  endTime: 250,
  messages: [WEATHER_PREAMBLE_MESSAGE, WEATHER_USER_MESSAGE],
  tools: [{ name: 'getWeather', description: 'Gets weather for a city' }],
  outputMessage: {
    role: 'model',
    content: [
      { reasoning: 'Need to call getWeather for Seattle.' },
      {
        toolRequest: {
          ref: 'call-1',
          name: 'getWeather',
          input: { city: 'Seattle' },
        },
      },
    ],
  },
  finishReason: 'stop',
  usage: { inputTokens: 20, outputTokens: 15, totalTokens: 35 },
});

/** Canonical `getWeather` tool execution span paired with `call-1`. */
export const MOCK_SPAN_TOOL: SpanData = makeToolSpan({
  spanId: 'tool-1',
  parentSpanId: 'turn-1',
  name: 'getWeather',
  callId: 'call-1',
  startTime: 255,
  endTime: 320,
  input: { city: 'Seattle' },
  output: { tempF: 62, condition: 'Rainy' },
});

/** Canonical follow-up model span summarizing the `getWeather` `toolResponse`. */
export const MOCK_SPAN_MODEL_REPLY: SpanData = makeModelSpan({
  spanId: 'model-2',
  parentSpanId: 'turn-1',
  model: 'googleai/gemini-2.5-pro',
  startTime: 330,
  endTime: 480,
  messages: [
    WEATHER_PREAMBLE_MESSAGE,
    WEATHER_USER_MESSAGE,
    {
      role: 'model',
      content: [
        {
          toolRequest: {
            ref: 'call-1',
            name: 'getWeather',
            input: { city: 'Seattle' },
          },
        },
      ],
    },
    {
      role: 'tool',
      content: [
        {
          toolResponse: {
            ref: 'call-1',
            name: 'getWeather',
            output: { tempF: 62, condition: 'Rainy' },
          },
        },
      ],
    },
  ],
  outputMessage: {
    role: 'model',
    content: [{ text: 'It is 62°F and rainy in Seattle.' }],
  },
  finishReason: 'stop',
  usage: { inputTokens: 45, outputTokens: 12, totalTokens: 57 },
});

/** Canonical single-turn agent + tool call + final reply trace (`trace-1`). */
export const MOCK_AGENT_TOOL_TRACE: TraceData = makeTrace(
  [
    MOCK_SPAN_AGENT,
    MOCK_SPAN_TURN_1,
    MOCK_SPAN_MODEL_TOOL_REQ,
    MOCK_SPAN_TOOL,
    MOCK_SPAN_MODEL_REPLY,
  ],
  { traceId: 'trace-1', startTime: 100, endTime: 500 }
);

/** Canonical parent agent delegating to a nested `/agent/researcher` subagent (`trace-sub`). */
export const MOCK_SUBAGENT_TRACE: TraceData = makeTrace(
  [
    makeAgentSpan({
      spanId: 'root-agent',
      name: 'codingAgent',
      sessionId: 'sess-outer',
      startTime: 100,
      endTime: 900,
    }),
    makeModelSpan({
      spanId: 'model-1',
      parentSpanId: 'root-agent',
      model: 'gemini-pro',
      startTime: 130,
      endTime: 200,
      messages: [{ role: 'user', content: [{ text: 'Research OATR spec' }] }],
      outputMessage: {
        role: 'model',
        content: [
          {
            toolRequest: {
              ref: 'sub-call-1',
              name: 'delegate_to_researcher',
              input: { topic: 'OATR' },
            },
          },
        ],
      },
    }),
    makeToolSpan({
      spanId: 'delegate-tool',
      parentSpanId: 'root-agent',
      name: 'delegate_to_researcher',
      callId: 'sub-call-1',
      startTime: 210,
      endTime: 600,
      input: { topic: 'OATR' },
      output: { summary: 'OATR is a flat bundle' },
    }),
    makeAgentSpan({
      spanId: 'sub-agent',
      parentSpanId: 'delegate-tool',
      name: 'researcher',
      sessionId: 'sess-sub',
      startTime: 215,
      endTime: 590,
    }),
    makeModelSpan({
      spanId: 'sub-model-1',
      parentSpanId: 'sub-agent',
      model: 'gemini-flash',
      startTime: 220,
      endTime: 580,
      messages: [{ role: 'user', content: [{ text: 'Topic: OATR' }] }],
      outputMessage: {
        role: 'model',
        content: [{ text: 'OATR defines TrajectoryBundle.' }],
      },
    }),
    makeModelSpan({
      spanId: 'model-2',
      parentSpanId: 'root-agent',
      model: 'gemini-pro',
      startTime: 610,
      endTime: 850,
      messages: [
        { role: 'user', content: [{ text: 'Research OATR spec' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                ref: 'sub-call-1',
                name: 'delegate_to_researcher',
                input: { topic: 'OATR' },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                ref: 'sub-call-1',
                name: 'delegate_to_researcher',
                output: { summary: 'OATR is a flat bundle' },
              },
            },
          ],
        },
      ],
      outputMessage: {
        role: 'model',
        content: [{ text: 'Here is the OATR summary.' }],
      },
    }),
  ],
  { traceId: 'trace-sub', startTime: 100, endTime: 900 }
);
