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

import type {
  GenerationUsage,
  MessageData,
  ToolDefinition,
} from '../../src/types/model.js';
import type { SpanData, TraceData } from '../../src/types/trace.js';

export const TRACE_ID = '7c273c22b219d077c6731a10d46b7d40';
export const BASE_FLOW_SPAN_ID = '22b219d077c67';
export const WRAPPER_ACTION_SPAN_ID = 'eec307f9311c5617';

type SpanOverrides = Partial<Omit<SpanData, 'attributes'>> & {
  attributes?: Record<string, unknown>;
};

function stringifyIfDefined(val: unknown): Record<string, string> {
  return val !== undefined ? { value: JSON.stringify(val) } : {};
}

/** Builds a minimal valid SpanData for unit tests with optional field overrides. */
export function makeSpan(overrides: Partial<SpanData> = {}): SpanData {
  const spanId = overrides.spanId ?? 'span-1';
  return {
    traceId: 'trace-1',
    spanId,
    startTime: 100,
    endTime: 200,
    attributes: {},
    instrumentationLibrary: { name: 'genkit-tracer', version: 'v1' },
    spanKind: 'INTERNAL',
    status: { code: 0 },
    ...overrides,
    displayName: overrides.displayName ?? spanId,
  };
}

/**
 * Builds a TraceData record auto-keyed by each span's `spanId`.
 * Infers `startTime` and `endTime` bounds from `spans` when not explicitly overridden.
 */
export function makeTrace(
  spans: SpanData[],
  overrides: Partial<TraceData> = {}
): TraceData {
  const traceId = overrides.traceId ?? spans[0]?.traceId ?? 'trace-1';
  const keyedSpans: Record<string, SpanData> = {};
  let minStart = Number.POSITIVE_INFINITY;
  let maxEnd = 0;

  for (const span of spans) {
    keyedSpans[span.spanId] = overrides.traceId
      ? { ...span, traceId }
      : { ...span };
    if (span.startTime < minStart) minStart = span.startTime;
    if (span.endTime > maxEnd) maxEnd = span.endTime;
  }

  return {
    traceId,
    ...(spans.length > 0
      ? {
          startTime: Number.isFinite(minStart) ? minStart : 0,
          endTime: maxEnd,
        }
      : {}),
    ...overrides,
    spans: overrides.spans ? { ...keyedSpans, ...overrides.spans } : keyedSpans,
  };
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
  callId?: string;
  input?: unknown;
  output?: unknown;
  interrupt?: unknown;
  resumed?: unknown;
}

/** Builds a `/tool/<name>` SpanData hiding `genkit:input` / `genkit:output` JSON serialization. */
export function makeToolSpan(options: MakeToolSpanOptions = {}): SpanData {
  const {
    name = 'getWeather',
    callId,
    input,
    output,
    interrupt,
    resumed,
    attributes,
    ...rest
  } = options;
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
      'genkit:metadata:subtype': 'tool',
      'genkit:name': cleanName,
      ...(callId ? { 'genkit:metadata:tool:callId': callId } : {}),
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

/** Canonical internal Dev UI wrapper span (`genkit:metadata:genkit-dev-internal`). */
export const MOCK_SPAN_DEV_WRAPPER: SpanData = makeSpan({
  spanId: 'dev-wrapper',
  displayName: 'dev-run-action-wrapper',
  startTime: 95,
  endTime: 505,
  attributes: {
    'genkit:name': 'dev-run-action-wrapper',
    'genkit:isRoot': true,
    'genkit:metadata:genkit-dev-internal': true,
    'genkit:state': 'success',
  },
});

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

/** Canonical HITL tool interrupt baseline trace (`trace-hitl`). */
export const MOCK_HITL_INTERRUPT_TRACE: TraceData = makeTrace(
  [
    makeAgentSpan({
      spanId: 'root-agent',
      name: 'bankAgent',
      sessionId: 'sess-hitl',
      startTime: 100,
      endTime: 300,
    }),
    makeTurnSpan({
      spanId: 'turn-1',
      turn: 1,
      parentSpanId: 'root-agent',
      startTime: 105,
      endTime: 295,
    }),
    makeModelSpan({
      spanId: 'model-1',
      parentSpanId: 'turn-1',
      model: 'gemini',
      startTime: 110,
      endTime: 200,
      messages: [
        { role: 'user', content: [{ text: 'Transfer $500 to Alice' }] },
      ],
      outputMessage: {
        role: 'model',
        content: [
          {
            toolRequest: {
              ref: 'tx-1',
              name: 'transferFunds',
              input: { to: 'Alice', amount: 500 },
            },
          },
        ],
      },
    }),
    makeToolSpan({
      spanId: 'tool-1',
      parentSpanId: 'turn-1',
      name: 'transferFunds',
      callId: 'tx-1',
      startTime: 205,
      endTime: 250,
      input: { to: 'Alice', amount: 500 },
      interrupt: { confirmRequired: true },
      status: { code: 2, message: 'ToolInterruptError' },
    }),
  ],
  { traceId: 'trace-hitl', startTime: 100, endTime: 300 }
);

/** Helper class to manage traces in tests */
export class MockTrace {
  private BASE_FLOW_SPAN: SpanData = {
    spanId: BASE_FLOW_SPAN_ID,
    traceId: TRACE_ID,
    startTime: 1714059149481,
    endTime: 1714059149484.9653,
    attributes: {
      'genkit:type': 'flow',
      'genkit:name': 'multiSteps',
      'genkit:isRoot': true,
      'genkit:path': '/dev-run-action-wrapper/multiSteps/multiSteps',
      'genkit:metadata:flow:execution': '0',
      'genkit:metadata:flow:name': 'multiSteps',
      'genkit:metadata:flow:id': 'c68dd3b2-6bb4-44ef-907d-cef5720d6b6b',
      'genkit:metadata:flow:dispatchType': 'start',
      'genkit:metadata:flow:state': 'done',
      'genkit:input': '"Douglas Adams"',
      'genkit:output': '42',
      'genkit:state': 'success',
    },
    displayName: 'multiSteps',
    links: [],
    instrumentationLibrary: {
      name: 'genkit-tracer',
      version: 'v1',
    },
    spanKind: 'INTERNAL',
    sameProcessAsParentSpan: {
      value: true,
    },
    status: {
      code: 0,
    },
    timeEvents: {
      timeEvent: [],
    },
  };
  private WRAPPER_ACTION_SPAN: SpanData = {
    spanId: WRAPPER_ACTION_SPAN_ID,
    traceId: '7c273c22b219d077c6731a10d46b7d40',
    parentSpanId: BASE_FLOW_SPAN_ID,
    startTime: 1714059149480,
    endTime: 1714059149485.578,
    attributes: {
      'genkit:type': 'action',
      'genkit:name': 'multiSteps',
      'genkit:path': '/dev-run-action-wrapper/multiSteps',
      'genkit:input': '{"start":{"input":"Douglas Adams"}}',
      'genkit:metadata:flow:wrapperAction': 'true',
      'genkit:output':
        '{"flowId":"c68dd3b2-6bb4-44ef-907d-cef5720d6b6b","name":"multiSteps","startTime":1714059149480,"input":"Douglas Adams","cache":{"step1":{"value":"Hello, Douglas Adams! step 1"},"step2":{"value":"Hello, Douglas Adams! step 1 Faf "},"step3-array":{"value":["Hello, Douglas Adams! step 1 Faf ","Hello, Douglas Adams! step 1 Faf "]},"step4-num":{"value":"Hello, Douglas Adams! step 1 Faf -()-Hello, Douglas Adams! step 1 Faf "}},"eventsTriggered":{},"blockedOnStep":null,"executions":[{"startTime":1714059149481,"traceIds":["7c273c22b219d077c6731a10d46b7d40"]}],"operation":{"name":"c68dd3b2-6bb4-44ef-907d-cef5720d6b6b","done":true,"result":{"response":42}},"traceContext":"{\\"traceId\\":\\"7c273c22b219d077c6731a10d46b7d40\\",\\"spanId\\":\\"30506fdf0beccc15\\",\\"traceFlags\\":1}"}',
      'genkit:state': 'success',
    },
    displayName: 'multiSteps',
    links: [],
    instrumentationLibrary: {
      name: 'genkit-tracer',
      version: 'v1',
    },
    spanKind: 'INTERNAL',
    sameProcessAsParentSpan: {
      value: true,
    },
    status: {
      code: 0,
    },
    timeEvents: {
      timeEvent: [],
    },
  };

  private BASE_SPAN: SpanData = {
    spanId: 'dfe3b782071ea308',
    traceId: '7c273c22b219d077c6731a10d46b7d40',
    parentSpanId: WRAPPER_ACTION_SPAN_ID,
    startTime: 1714059149481,
    endTime: 1714059149481.1533,
    attributes: {},
    displayName: 'baseStep',
    links: [],
    instrumentationLibrary: {
      name: 'genkit-tracer',
      version: 'v1',
    },
    spanKind: 'INTERNAL',
    sameProcessAsParentSpan: {
      value: true,
    },
    status: {
      code: 0,
    },
    timeEvents: {
      timeEvent: [],
    },
  };

  private BASE_MOCK_TRACE: TraceData = {
    traceId: '7c273c22b219d077c6731a10d46b7d40',
    displayName: 'dev-run-action-wrapper',
    startTime: 1714059149480,
    endTime: 1714059149486.6926,
    spans: {
      '30506fdf0beccc15': {
        spanId: '30506fdf0beccc15',
        traceId: '7c273c22b219d077c6731a10d46b7d40',
        startTime: 1714059149480,
        endTime: 1714059149486.6926,
        attributes: {
          'genkit:name': 'dev-run-action-wrapper',
          'genkit:isRoot': true,
          'genkit:path': '/dev-run-action-wrapper',
          'genkit:metadata:genkit-dev-internal': 'true',
          'genkit:state': 'success',
        },
        displayName: 'dev-run-action-wrapper',
        links: [],
        instrumentationLibrary: {
          name: 'genkit-tracer',
          version: 'v1',
        },
        spanKind: 'INTERNAL',
        sameProcessAsParentSpan: {
          value: true,
        },
        status: {
          code: 0,
        },
        timeEvents: {
          timeEvent: [],
        },
      },
    },
  };

  private trace: TraceData;

  getTrace(): TraceData {
    return this.trace;
  }

  addSpan(params: {
    stepName: string;
    spanType: 'flowStep' | 'action';
    retrieverConfig?: {
      text: string[];
      query: string;
    };
    input?: any;
    output?: any;
  }): MockTrace {
    const { stepName, spanType, retrieverConfig, input, output } = {
      ...params,
    };
    const flowInput = input ?? 'Douglas Adams';
    const flowOutput = output ?? 42;

    if (spanType === 'flowStep') {
      const flowStep = { ...this.BASE_SPAN };
      flowStep.displayName = stepName;
      flowStep.attributes = {
        'genkit:type': 'flowStep',
        'genkit:name': stepName,
        'genkit:path': `/dev-run-action-wrapper/multiSteps/multiSteps/${stepName}`,
        'genkit:metadata:flow:stepType': 'run',
        'genkit:metadata:flow:stepName': stepName,
        'genkit:metadata:flow:resolvedStepName': stepName,
        'genkit:metadata:flow:state': 'run',
        'genkit:input': JSON.stringify(flowInput),
        'genkit:output': JSON.stringify(flowOutput),
        'genkit:state': 'success',
      };
      this.trace.spans[`flowStep-${stepName}`] = flowStep;
    } else {
      // spanType === "action"
      const actionStep = { ...this.BASE_SPAN };
      actionStep.displayName = stepName;
      actionStep.attributes = {
        'genkit:type': 'action',
        'genkit:name': `${stepName}`,
        'genkit:path': `/dev-run-action-wrapper/multiSteps/multiSteps/${stepName}`,
        'genkit:input': JSON.stringify(flowInput),
        'genkit:output': JSON.stringify(flowOutput),
        'genkit:state': 'success',
      };
      if (!!retrieverConfig) {
        const retrieverInput = {
          query: { content: [{ text: retrieverConfig.query }] },
          options: { k: 3 },
        };
        const retrieverResponse = {
          documents: retrieverConfig.text.map((t) => {
            return { content: [{ text: t }] };
          }),
        };
        actionStep.attributes['genkit:metadata:subtype'] = 'retriever';
        actionStep.attributes['genkit:input'] = JSON.stringify(retrieverInput);
        actionStep.attributes['genkit:output'] =
          JSON.stringify(retrieverResponse);
      }
      this.trace.spans[`action-${stepName}`] = actionStep;
    }
    return this;
  }

  constructor(
    traceInput?: any,
    traceOutput?: any,
    baseFlowState: 'done' | 'error' = 'done'
  ) {
    const flowInput = traceInput ?? 'Douglas Adams';
    const flowOutput = traceOutput ?? 42;
    const baseFlowSpan = { ...this.BASE_FLOW_SPAN };
    baseFlowSpan.attributes['genkit:input'] = JSON.stringify(flowInput);
    baseFlowSpan.attributes['genkit:output'] = JSON.stringify(flowOutput);

    const wrapperActionSpan = { ...this.WRAPPER_ACTION_SPAN };
    wrapperActionSpan.attributes['genkit:input'] = JSON.stringify({
      start: { input: flowInput },
    });
    wrapperActionSpan.attributes['genkit:output'] = JSON.stringify({
      flowId: 'c68dd3b2-6bb4-44ef-907d-cef5720d6b6b',
      name: 'multiSteps',
      startTime: 1714059149480,
      input: JSON.stringify(flowInput),
      eventsTriggered: {},
      blockedOnStep: null,
      executions: [
        {
          startTime: 1714059149481,
          traceIds: ['7c273c22b219d077c6731a10d46b7d40'],
        },
      ],
      operation: {
        name: 'c68dd3b2-6bb4-44ef-907d-cef5720d6b6b',
        done: true,
        result: {
          response: JSON.stringify(flowOutput),
        },
      },
      traceContext: JSON.stringify({
        traceId: '7c273c22b219d077c6731a10d46b7d40',
        spanId: '30506fdf0beccc15',
        traceFlags: 1,
      }),
    });

    this.trace = { ...this.BASE_MOCK_TRACE };
    this.trace.spans['baseFlowSpan'] = baseFlowSpan;
    this.trace.spans['wrapperActionSpan'] = wrapperActionSpan;

    return this;
  }
}
