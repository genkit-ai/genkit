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

import type { SpanData, TraceData } from '../../src/types/trace.js';

export const TRACE_ID = '7c273c22b219d077c6731a10d46b7d40';
export const BASE_FLOW_SPAN_ID = '22b219d077c67';
export const WRAPPER_ACTION_SPAN_ID = 'eec307f9311c5617';

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
