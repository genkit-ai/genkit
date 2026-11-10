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

import { describe, expect, it } from '@jest/globals';
import { TrajectoryBundleSchema, type TraceData } from '../../src/types';
import {
  extractSessionIdFromTrace,
  extractTrajectoryFromTrace,
} from '../../src/utils';
import { makeSpan } from './trace';

function buildSingleTurnWeatherTrace(): TraceData {
  const rootSpan = makeSpan({
    spanId: 'root-agent',
    displayName: '/agent/weatherAgent',
    startTime: 100,
    endTime: 500,
    attributes: {
      'genkit:type': 'action',
      'genkit:metadata:subtype': 'agent',
      'genkit:metadata:agent:sessionId': 'sess-123',
    },
  });

  const turnSpan = makeSpan({
    spanId: 'turn-1',
    parentSpanId: 'root-agent',
    displayName: 'runTurn-1',
    startTime: 105,
    endTime: 495,
    attributes: {
      'genkit:type': 'flowStep',
    },
  });

  const modelSpan1 = makeSpan({
    spanId: 'model-1',
    parentSpanId: 'turn-1',
    displayName: '/model/googleai/gemini-2.5-pro',
    startTime: 110,
    endTime: 250,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:name': 'googleai/gemini-2.5-pro',
      'genkit:input': JSON.stringify({
        messages: [
          {
            role: 'system',
            content: [{ text: 'You are a helpful weather assistant.' }],
            metadata: { agentPreamble: true },
          },
          {
            role: 'user',
            content: [{ text: 'What is the weather in Seattle?' }],
          },
        ],
        tools: [
          {
            name: 'getWeather',
            description: 'Gets weather for a city',
          },
        ],
      }),
      'genkit:output': JSON.stringify({
        message: {
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
        usage: {
          inputTokens: 20,
          outputTokens: 15,
          totalTokens: 35,
        },
      }),
    },
  });

  const toolSpan1 = makeSpan({
    spanId: 'tool-1',
    parentSpanId: 'turn-1',
    displayName: '/tool/getWeather',
    startTime: 255,
    endTime: 320,
    attributes: {
      'genkit:metadata:subtype': 'tool',
      'genkit:name': 'getWeather',
      'genkit:input': JSON.stringify({ city: 'Seattle' }),
      'genkit:output': JSON.stringify({ tempF: 62, condition: 'Rainy' }),
    },
  });

  const modelSpan2 = makeSpan({
    spanId: 'model-2',
    parentSpanId: 'turn-1',
    displayName: '/model/googleai/gemini-2.5-pro',
    startTime: 330,
    endTime: 480,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:name': 'googleai/gemini-2.5-pro',
      'genkit:input': JSON.stringify({
        messages: [
          {
            role: 'system',
            content: [{ text: 'You are a helpful weather assistant.' }],
            metadata: { agentPreamble: true },
          },
          {
            role: 'user',
            content: [{ text: 'What is the weather in Seattle?' }],
          },
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
      }),
      'genkit:output': JSON.stringify({
        message: {
          role: 'model',
          content: [{ text: 'It is 62°F and rainy in Seattle.' }],
        },
        finishReason: 'stop',
        usage: {
          inputTokens: 45,
          outputTokens: 12,
          totalTokens: 57,
        },
      }),
    },
  });

  return {
    traceId: 'trace-1',
    startTime: 100,
    endTime: 500,
    spans: {
      'root-agent': rootSpan,
      'turn-1': turnSpan,
      'model-1': modelSpan1,
      'tool-1': toolSpan1,
      'model-2': modelSpan2,
    },
  };
}

function buildParallelToolTrace(): TraceData {
  const rootSpan = makeSpan({
    spanId: 'root',
    displayName: '/agent/parallelAgent',
    startTime: 100,
    endTime: 400,
    attributes: { 'genkit:metadata:subtype': 'agent' },
  });

  const modelSpan = makeSpan({
    spanId: 'model-1',
    parentSpanId: 'root',
    displayName: '/model/gemini',
    startTime: 110,
    endTime: 200,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({
        messages: [{ role: 'user', content: [{ text: 'Compare A and B' }] }],
      }),
      'genkit:output': JSON.stringify({
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                ref: 'ref-b',
                name: 'readFile',
                input: { path: 'b.txt' },
              },
            },
            {
              toolRequest: {
                ref: 'ref-a',
                name: 'readFile',
                input: { path: 'a.txt' },
              },
            },
            {
              toolRequest: {
                name: 'search',
                input: { q: 'first' },
              },
            },
            {
              toolRequest: {
                name: 'search',
                input: { q: 'second' },
              },
            },
          ],
        },
        finishReason: 'stop',
      }),
    },
  });

  // Tool span for 'ref-a' starts earlier than 'ref-b', but explicit callId matches accurately
  const toolReadA = makeSpan({
    spanId: 'tool-read-a',
    parentSpanId: 'root',
    displayName: '/tool/readFile',
    startTime: 205,
    endTime: 230,
    attributes: {
      'genkit:metadata:subtype': 'tool',
      'genkit:metadata:tool:callId': 'ref-a',
      'genkit:output': JSON.stringify('content-A'),
    },
  });

  const toolReadB = makeSpan({
    spanId: 'tool-read-b',
    parentSpanId: 'root',
    displayName: '/tool/readFile',
    startTime: 210,
    endTime: 240,
    attributes: {
      'genkit:metadata:subtype': 'tool',
      'genkit:metadata:tool:callId': 'ref-b',
      'genkit:output': JSON.stringify('content-B'),
    },
  });

  // Fallback order matching for 'search' without callId
  const toolSearch1 = makeSpan({
    spanId: 'tool-search-1',
    parentSpanId: 'root',
    displayName: '/tool/search',
    startTime: 215,
    endTime: 250,
    attributes: {
      'genkit:metadata:subtype': 'tool',
      'genkit:output': JSON.stringify(['res-1']),
    },
  });

  const toolSearch2 = makeSpan({
    spanId: 'tool-search-2',
    parentSpanId: 'root',
    displayName: '/tool/search',
    startTime: 220,
    endTime: 260,
    attributes: {
      'genkit:metadata:subtype': 'tool',
      'genkit:output': JSON.stringify(['res-2']),
    },
  });

  return {
    traceId: 'trace-parallel',
    spans: {
      root: rootSpan,
      'model-1': modelSpan,
      'tool-read-a': toolReadA,
      'tool-read-b': toolReadB,
      'tool-search-1': toolSearch1,
      'tool-search-2': toolSearch2,
    },
  };
}

function buildConflictingRefTrace(): TraceData {
  const root = makeSpan({
    spanId: 'root',
    displayName: '/agent/parallelAgent',
    startTime: 100,
    endTime: 400,
    attributes: { 'genkit:metadata:subtype': 'agent' },
  });
  const m1 = makeSpan({
    spanId: 'm1',
    parentSpanId: 'root',
    displayName: '/model/gemini',
    startTime: 110,
    endTime: 150,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({
        messages: [{ role: 'user', content: [{ text: 'Transfer twice' }] }],
      }),
      'genkit:output': JSON.stringify({
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                ref: 'tx-1',
                name: 'transfer',
                input: { amount: 10 },
              },
              metadata: { interrupt: true },
            },
            {
              toolRequest: {
                ref: 'tx-2',
                name: 'transfer',
                input: { amount: 20 },
              },
              metadata: { interrupt: true },
            },
          ],
        },
        finishReason: 'interrupted',
      }),
    },
  });
  // Only tx-2 is resumed (via restart) and executed; tx-1 must NOT steal tx-2's decision, span, or response
  const resumedTurn = makeSpan({
    spanId: 'turn-2',
    parentSpanId: 'root',
    displayName: 'runTurn-2',
    startTime: 200,
    endTime: 390,
    attributes: {
      'genkit:input': JSON.stringify({
        resume: {
          restart: [
            {
              toolRequest: {
                ref: 'tx-2',
                name: 'transfer',
                input: { amount: 20 },
              },
            },
          ],
        },
      }),
    },
  });
  const restartedTx2 = makeSpan({
    spanId: 'tool-tx-2',
    parentSpanId: 'turn-2',
    displayName: '/tool/transfer',
    startTime: 210,
    endTime: 240,
    attributes: {
      'genkit:metadata:subtype': 'tool',
      'genkit:metadata:resumed': true,
      'genkit:metadata:tool:callId': 'tx-2',
      'genkit:output': JSON.stringify({ ok: 20 }),
    },
  });
  const m2 = makeSpan({
    spanId: 'm2',
    parentSpanId: 'turn-2',
    displayName: '/model/gemini',
    startTime: 250,
    endTime: 300,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({
        messages: [
          { role: 'user', content: [{ text: 'Transfer twice' }] },
          {
            role: 'tool',
            content: [
              {
                toolResponse: {
                  ref: 'tx-2',
                  name: 'transfer',
                  output: { ok: 20 },
                },
              },
            ],
          },
        ],
      }),
      'genkit:output': JSON.stringify({
        message: { role: 'model', content: [{ text: 'Transferred 20.' }] },
        finishReason: 'stop',
      }),
    },
  });

  return {
    traceId: 'trace-ref-conflict',
    spans: {
      root,
      m1,
      'turn-2': resumedTurn,
      'tool-tx-2': restartedTx2,
      m2,
    },
  };
}

function buildSubagentAndExclusionTrace(): TraceData {
  return {
    traceId: 'trace-sub',
    startTime: 100,
    endTime: 900,
    spans: {
      root: makeSpan({
        traceId: 'trace-sub',
        spanId: 'root',
        displayName: '/agent/codingAgent',
        startTime: 100,
        endTime: 900,
        attributes: { 'genkit:metadata:subtype': 'agent' },
      }),
      guardrail: makeSpan({
        traceId: 'trace-sub',
        spanId: 'guardrail',
        parentSpanId: 'root',
        displayName: 'safetyClassifier',
        startTime: 105,
        endTime: 120,
        attributes: {
          'genkit:type': 'flowStep',
          'genkit:metadata:trajectory:exclude': 'true',
        },
      }),
      guardrailModel: makeSpan({
        traceId: 'trace-sub',
        spanId: 'guardrailModel',
        parentSpanId: 'guardrail',
        displayName: '/model/gemini-flash',
        startTime: 106,
        endTime: 119,
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:output': JSON.stringify({
            message: { role: 'model', content: [{ text: 'SAFE' }] },
          }),
        },
      }),
      promptRender: makeSpan({
        traceId: 'trace-sub',
        spanId: 'promptRender',
        parentSpanId: 'root',
        displayName: 'renderPrompt',
        startTime: 121,
        endTime: 125,
        attributes: {
          'genkit:metadata:subtype': 'promptTemplate',
        },
      }),
      rootModel1: makeSpan({
        traceId: 'trace-sub',
        spanId: 'rootModel1',
        parentSpanId: 'root',
        displayName: '/model/gemini-pro',
        startTime: 130,
        endTime: 200,
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:input': JSON.stringify({
            messages: [
              {
                role: 'user',
                content: [{ text: 'Research OATR spec' }],
              },
            ],
          }),
          'genkit:output': JSON.stringify({
            message: {
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
            finishReason: 'stop',
          }),
        },
      }),
      delegateTool: makeSpan({
        traceId: 'trace-sub',
        spanId: 'delegateTool',
        parentSpanId: 'root',
        displayName: '/tool/delegate_to_researcher',
        startTime: 210,
        endTime: 600,
        attributes: {
          'genkit:metadata:subtype': 'tool',
          'genkit:input': JSON.stringify({ topic: 'OATR' }),
          'genkit:output': JSON.stringify({
            summary: 'OATR is a flat bundle',
          }),
        },
      }),
      subAgent: makeSpan({
        traceId: 'trace-sub',
        spanId: 'subAgent',
        parentSpanId: 'delegateTool',
        displayName: '/agent/researcher',
        startTime: 215,
        endTime: 590,
        attributes: {
          'genkit:metadata:subtype': 'agent',
        },
      }),
      subModel1: makeSpan({
        traceId: 'trace-sub',
        spanId: 'subModel1',
        parentSpanId: 'subAgent',
        displayName: '/model/gemini-flash',
        startTime: 220,
        endTime: 580,
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:input': JSON.stringify({
            messages: [{ role: 'user', content: [{ text: 'Topic: OATR' }] }],
          }),
          'genkit:output': JSON.stringify({
            message: {
              role: 'model',
              content: [{ text: 'OATR defines TrajectoryBundle.' }],
            },
            finishReason: 'stop',
          }),
        },
      }),
      rootModel2: makeSpan({
        traceId: 'trace-sub',
        spanId: 'rootModel2',
        parentSpanId: 'root',
        displayName: '/model/gemini-pro',
        startTime: 610,
        endTime: 850,
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:input': JSON.stringify({
            messages: [
              {
                role: 'user',
                content: [{ text: 'Research OATR spec' }],
              },
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
          }),
          'genkit:output': JSON.stringify({
            message: {
              role: 'model',
              content: [{ text: 'Here is the OATR summary.' }],
            },
            finishReason: 'stop',
          }),
        },
      }),
    },
  };
}

function buildDelegateAndSkewedFallbackTrace(): TraceData {
  const root = makeSpan({
    spanId: 'root',
    displayName: '/agent/coordinator',
    startTime: 100,
    endTime: 400,
    attributes: { 'genkit:metadata:subtype': 'agent' },
  });
  const turn1 = makeSpan({
    spanId: 'turn-1',
    parentSpanId: 'root',
    displayName: 'runTurn-1',
    startTime: 100,
    endTime: 390,
  });
  const model1 = makeSpan({
    spanId: 'm1',
    parentSpanId: 'turn-1',
    displayName: '/model/gemini',
    startTime: 120,
    endTime: 160,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({
        messages: [{ role: 'user', content: [{ text: 'Research & lookup' }] }],
      }),
      'genkit:output': JSON.stringify({
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'researcher',
                input: { topic: 'mars' },
              },
            },
            {
              toolRequest: {
                name: 'lookupCache',
                input: { key: 'k1' },
              },
            },
          ],
        },
        finishReason: 'stop',
      }),
    },
  });
  const delegateToolSpan = makeSpan({
    spanId: 'tool-delegate',
    parentSpanId: 'turn-1',
    displayName: '/tool/delegate_to_researcher',
    startTime: 165,
    endTime: 240,
    attributes: {
      'genkit:metadata:subtype': 'tool',
      'genkit:output': JSON.stringify({ summary: 'Mars facts' }),
    },
  });
  const intermediateStep = makeSpan({
    spanId: 'inner-wrapper',
    parentSpanId: 'tool-delegate',
    displayName: 'invokeSubagent',
    startTime: 166,
    endTime: 235,
  });
  const subagentSpan = makeSpan({
    spanId: 'sub-researcher',
    parentSpanId: 'inner-wrapper',
    displayName: '/agent/researcher',
    startTime: 168,
    endTime: 230,
    attributes: { 'genkit:metadata:subtype': 'agent' },
  });
  // Skewed startTime earlier than m1.startTime so timestamp window misses and turnIndex fallback matches;
  // also omits genkit:output so ref-less toolResponse in m2 history supplies the output.
  const skewedToolSpan = makeSpan({
    spanId: 'tool-skewed',
    parentSpanId: 'turn-1',
    displayName: '/tool/lookupCache',
    startTime: 115,
    endTime: 175,
    attributes: { 'genkit:metadata:subtype': 'tool' },
  });
  const model2 = makeSpan({
    spanId: 'm2',
    parentSpanId: 'turn-1',
    displayName: '/model/gemini',
    startTime: 250,
    endTime: 300,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({
        messages: [
          { role: 'user', content: [{ text: 'Research & lookup' }] },
          {
            role: 'tool',
            content: [
              {
                toolResponse: {
                  name: 'lookupCache',
                  output: { hit: true },
                },
              },
            ],
          },
        ],
      }),
      'genkit:output': JSON.stringify({
        message: { role: 'model', content: [{ text: 'All done.' }] },
        finishReason: 'stop',
      }),
    },
  });

  return {
    traceId: 'trace-delegate-fallback',
    spans: {
      root,
      'turn-1': turn1,
      m1: model1,
      'tool-delegate': delegateToolSpan,
      'inner-wrapper': intermediateStep,
      'sub-researcher': subagentSpan,
      'tool-skewed': skewedToolSpan,
      m2: model2,
    },
  };
}

function buildWrappedTrace(): TraceData {
  return {
    traceId: 'trace-wrapped',
    spans: {
      // Insert deeper subagent first to verify shallowest span wins
      subAgent: makeSpan({
        traceId: 'trace-wrapped',
        spanId: 'subAgent',
        parentSpanId: 'outerAgent',
        displayName: '/agent/sub',
        startTime: 120,
        endTime: 180,
        attributes: {
          'genkit:metadata:subtype': 'agent',
          'genkit:metadata:agent:sessionId': 'sub-session',
        },
      }),
      wrapper: makeSpan({
        traceId: 'trace-wrapped',
        spanId: 'wrapper',
        displayName: 'dev-run-action-wrapper',
        startTime: 100,
        endTime: 200,
        attributes: {
          'genkit:metadata:genkit-dev-internal': true,
        },
      }),
      outerAgent: makeSpan({
        traceId: 'trace-wrapped',
        spanId: 'outerAgent',
        parentSpanId: 'wrapper',
        displayName: '/agent/main',
        startTime: 105,
        endTime: 195,
        attributes: {
          'genkit:metadata:subtype': 'agent',
          'genkit:metadata:agent:sessionId': 'outer-session',
          'genkit:input': JSON.stringify({
            messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
          }),
        },
      }),
    },
  };
}

function buildErrorAndPendingTrace(): TraceData {
  return {
    traceId: 'trace-err',
    spans: {
      root: makeSpan({
        traceId: 'trace-err',
        spanId: 'root',
        displayName: '/agent/opsAgent',
        startTime: 100,
        endTime: 500,
        attributes: { 'genkit:metadata:subtype': 'agent' },
      }),
      m1: makeSpan({
        traceId: 'trace-err',
        spanId: 'm1',
        parentSpanId: 'root',
        displayName: '/model/gemini',
        startTime: 110,
        endTime: 200,
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:input': JSON.stringify({
            messages: [
              {
                role: 'user',
                content: [{ text: 'Old turn prompt' }],
                metadata: { _genkit_history: true },
              },
              {
                role: 'model',
                content: [{ text: 'Old turn reply' }],
                metadata: { _genkit_history: true },
              },
              {
                role: 'user',
                content: [{ text: 'Deploy build' }],
              },
            ],
          }),
          'genkit:output': JSON.stringify({
            message: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    ref: 'dep-1',
                    name: 'deploy',
                    input: { env: 'staging' },
                  },
                },
                {
                  toolRequest: {
                    ref: 'chk-1',
                    name: 'healthCheck',
                    input: {},
                  },
                },
              ],
            },
            finishReason: 'tool_calls',
          }),
        },
      }),
      failedTool: makeSpan({
        traceId: 'trace-err',
        spanId: 'failedTool',
        parentSpanId: 'root',
        displayName: '/tool/deploy',
        startTime: 205,
        endTime: 240,
        status: { code: 2, message: 'Permission denied' },
        attributes: {
          'genkit:metadata:subtype': 'tool',
          'genkit:metadata:tool:callId': 'dep-1',
        },
      }),
      pendingTool: makeSpan({
        traceId: 'trace-err',
        spanId: 'pendingTool',
        parentSpanId: 'root',
        displayName: '/tool/healthCheck',
        startTime: 210,
        endTime: 0,
        attributes: {
          'genkit:metadata:subtype': 'tool',
          'genkit:metadata:tool:callId': 'chk-1',
        },
      }),
      failedModel: makeSpan({
        traceId: 'trace-err',
        spanId: 'failedModel',
        parentSpanId: 'root',
        displayName: '/model/gemini',
        startTime: 300,
        endTime: 350,
        status: { code: 2, message: 'Quota exceeded' },
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:input': JSON.stringify({ messages: [] }),
        },
      }),
    },
  };
}

function buildUnresumedInterruptTrace(): TraceData {
  return {
    traceId: 'trace-int-unresumed',
    startTime: 100,
    endTime: 300,
    spans: {
      root: makeSpan({
        traceId: 'trace-int-unresumed',
        spanId: 'root',
        displayName: '/agent/bankAgent',
        startTime: 100,
        endTime: 300,
        attributes: {
          'genkit:metadata:subtype': 'agent',
          'genkit:metadata:agent:sessionId': 'sess-int',
        },
      }),
      model1: makeSpan({
        traceId: 'trace-int-unresumed',
        spanId: 'model1',
        parentSpanId: 'root',
        displayName: '/model/gemini',
        startTime: 110,
        endTime: 200,
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:input': JSON.stringify({
            messages: [
              { role: 'user', content: [{ text: 'Transfer $500 to Alice' }] },
            ],
          }),
          'genkit:output': JSON.stringify({
            message: {
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
            finishReason: 'stop',
          }),
        },
      }),
      tool1: makeSpan({
        traceId: 'trace-int-unresumed',
        spanId: 'tool1',
        parentSpanId: 'root',
        displayName: '/tool/transferFunds',
        startTime: 205,
        endTime: 250,
        status: { code: 2, message: 'ToolInterruptError' },
        attributes: {
          'genkit:metadata:subtype': 'tool',
          'genkit:metadata:interrupt': JSON.stringify({
            confirmRequired: true,
          }),
          'genkit:input': JSON.stringify({ to: 'Alice', amount: 500 }),
        },
      }),
    },
  };
}

function buildInTraceResumedInterruptTrace(): TraceData {
  return {
    traceId: 'trace-int-resumed',
    startTime: 100,
    endTime: 650,
    spans: {
      root: makeSpan({
        traceId: 'trace-int-resumed',
        spanId: 'root',
        displayName: '/agent/travelAgent',
        startTime: 100,
        endTime: 650,
        attributes: {
          'genkit:metadata:subtype': 'agent',
        },
      }),
      turn1: makeSpan({
        traceId: 'trace-int-resumed',
        spanId: 'turn1',
        parentSpanId: 'root',
        displayName: 'runTurn-1',
        startTime: 105,
        endTime: 300,
        attributes: { 'genkit:type': 'flowStep' },
      }),
      model1: makeSpan({
        traceId: 'trace-int-resumed',
        spanId: 'model1',
        parentSpanId: 'turn1',
        displayName: '/model/gemini',
        startTime: 110,
        endTime: 200,
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:input': JSON.stringify({
            messages: [
              {
                role: 'user',
                content: [{ text: 'Pay $500 and book flight' }],
              },
            ],
          }),
          'genkit:output': JSON.stringify({
            message: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    ref: 'tx-1',
                    name: 'transferFunds',
                    input: { amount: 500 },
                  },
                },
                {
                  toolRequest: {
                    ref: 'flt-1',
                    name: 'bookFlight',
                    input: { dest: 'SEA' },
                  },
                },
              ],
            },
            finishReason: 'stop',
          }),
        },
      }),
      txTool1: makeSpan({
        traceId: 'trace-int-resumed',
        spanId: 'txTool1',
        parentSpanId: 'turn1',
        displayName: '/tool/transferFunds',
        startTime: 205,
        endTime: 230,
        status: { code: 2, message: 'ToolInterruptError' },
        attributes: {
          'genkit:metadata:subtype': 'tool',
          'genkit:metadata:tool:callId': 'tx-1',
          'genkit:metadata:interrupt': JSON.stringify({ confirm: true }),
        },
      }),
      fltTool1: makeSpan({
        traceId: 'trace-int-resumed',
        spanId: 'fltTool1',
        parentSpanId: 'turn1',
        displayName: '/tool/bookFlight',
        startTime: 210,
        endTime: 240,
        status: { code: 2, message: 'ToolInterruptError' },
        attributes: {
          'genkit:metadata:subtype': 'tool',
          'genkit:metadata:tool:callId': 'flt-1',
          'genkit:metadata:interrupt': JSON.stringify({ seatNeeded: true }),
        },
      }),
      turn2: makeSpan({
        traceId: 'trace-int-resumed',
        spanId: 'turn2',
        parentSpanId: 'root',
        displayName: 'runTurn-2',
        startTime: 350,
        endTime: 640,
        attributes: {
          'genkit:type': 'flowStep',
          'genkit:input': JSON.stringify({
            resume: {
              respond: [
                {
                  toolResponse: {
                    ref: 'tx-1',
                    name: 'transferFunds',
                    output: { approved: true, receipt: 'r-1' },
                  },
                },
              ],
              restart: [
                {
                  toolRequest: {
                    ref: 'flt-1',
                    name: 'bookFlight',
                    input: { dest: 'SEA' },
                  },
                  metadata: { resumed: { seat: '12A' } },
                },
              ],
            },
          }),
        },
      }),
      fltToolRestarted: makeSpan({
        traceId: 'trace-int-resumed',
        spanId: 'fltToolRestarted',
        parentSpanId: 'turn2',
        displayName: '/tool/bookFlight',
        startTime: 360,
        endTime: 410,
        attributes: {
          'genkit:metadata:subtype': 'tool',
          'genkit:metadata:resumed': '{"seat":"12A"}',
          'genkit:output': JSON.stringify({ pnr: 'XYZ987', seat: '12A' }),
        },
      }),
      model2: makeSpan({
        traceId: 'trace-int-resumed',
        spanId: 'model2',
        parentSpanId: 'turn2',
        displayName: '/model/gemini',
        startTime: 420,
        endTime: 600,
        attributes: {
          'genkit:metadata:subtype': 'model',
          'genkit:input': JSON.stringify({
            messages: [
              {
                role: 'user',
                content: [{ text: 'Pay $500 and book flight' }],
              },
              {
                role: 'model',
                content: [
                  {
                    toolRequest: {
                      ref: 'tx-1',
                      name: 'transferFunds',
                      input: { amount: 500 },
                    },
                    metadata: { resolvedInterrupt: { confirm: true } },
                  },
                  {
                    toolRequest: {
                      ref: 'flt-1',
                      name: 'bookFlight',
                      input: { dest: 'SEA' },
                    },
                    metadata: { resumed: { seat: '12A' } },
                  },
                ],
              },
              {
                role: 'tool',
                content: [
                  {
                    toolResponse: {
                      ref: 'tx-1',
                      name: 'transferFunds',
                      output: { approved: true, receipt: 'r-1' },
                    },
                    metadata: { interruptResponse: true },
                  },
                  {
                    toolResponse: {
                      ref: 'flt-1',
                      name: 'bookFlight',
                      output: { pnr: 'XYZ987', seat: '12A' },
                    },
                  },
                ],
              },
            ],
          }),
          'genkit:output': JSON.stringify({
            message: {
              role: 'model',
              content: [{ text: 'Paid $500 (r-1) and booked flight XYZ987.' }],
            },
            finishReason: 'stop',
          }),
        },
      }),
    },
  };
}

function buildAncestorHitlTrace(): TraceData {
  const root = makeSpan({
    spanId: 'root',
    displayName: '/agent/hitlAgent',
    startTime: 100,
    endTime: 500,
    attributes: { 'genkit:metadata:subtype': 'agent' },
  });
  const turn1 = makeSpan({
    spanId: 'turn-1',
    parentSpanId: 'root',
    displayName: 'runTurn-1',
    startTime: 105,
    endTime: 250,
    attributes: {
      'genkit:output': JSON.stringify({
        finishReason: 'interrupted',
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                ref: 'r-approve',
                name: 'approveAction',
                input: { id: 1 },
              },
              metadata: { interrupt: { prompt: 'Approve?' } },
            },
          ],
        },
      }),
    },
  });
  // Model span has plain toolRequests without interrupt metadata and no tool execution spans in turn-1
  const model1 = makeSpan({
    spanId: 'm1',
    parentSpanId: 'turn-1',
    displayName: '/model/gemini',
    startTime: 110,
    endTime: 200,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({
        messages: [{ role: 'user', content: [{ text: 'Run guarded ops' }] }],
      }),
      'genkit:output': JSON.stringify({
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                ref: 'r-approve',
                name: 'approveAction',
                input: { id: 1 },
              },
            },
            {
              toolRequest: {
                ref: 'r-retry',
                name: 'retryOp',
                input: { id: 2 },
              },
            },
          ],
        },
        finishReason: 'stop',
      }),
    },
  });

  const turn2 = makeSpan({
    spanId: 'turn-2',
    parentSpanId: 'root',
    displayName: 'runTurn-2',
    startTime: 300,
    endTime: 490,
    attributes: {
      'genkit:input': JSON.stringify({
        resume: {
          respond: [
            {
              toolResponse: {
                ref: 'r-approve',
                name: 'approveAction',
                output: { confirmed: true },
              },
            },
          ],
          restart: [
            {
              toolRequest: {
                ref: 'r-retry',
                name: 'retryOp',
                input: { id: 2, force: true },
              },
            },
          ],
        },
      }),
    },
  });
  const restartedSpan = makeSpan({
    spanId: 'tool-restarted',
    parentSpanId: 'turn-2',
    displayName: '/tool/retryOp',
    startTime: 310,
    endTime: 350,
    attributes: {
      'genkit:metadata:subtype': 'tool',
      'genkit:metadata:resumed': true,
      'genkit:output': JSON.stringify({ retried: 'ok' }),
    },
  });
  const model2 = makeSpan({
    spanId: 'm2',
    parentSpanId: 'turn-2',
    displayName: '/model/gemini',
    startTime: 360,
    endTime: 450,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({
        messages: [{ role: 'user', content: [{ text: 'Run guarded ops' }] }],
      }),
      'genkit:output': JSON.stringify({
        message: { role: 'model', content: [{ text: 'Completed ops.' }] },
        finishReason: 'stop',
      }),
    },
  });

  return {
    traceId: 'trace-ancestor-hitl',
    spans: {
      root,
      'turn-1': turn1,
      m1: model1,
      'turn-2': turn2,
      'tool-restarted': restartedSpan,
      m2: model2,
    },
  };
}

function buildStandaloneModelTrace(): TraceData {
  const rootModel = makeSpan({
    spanId: 'm-root',
    displayName: '/model/googleai/gemini-2.5-flash',
    startTime: 100,
    endTime: 200,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({
        messages: [
          {
            role: 'system',
            content: [{ text: 'Be concise.' }],
          },
          {
            role: 'user',
            content: 'Write an essay',
          },
        ],
        tools: [
          {
            name: 'calc',
            key: '/tool/calc',
            description: 'Calculator',
            inputSchema: { type: 'object' },
            outputSchema: { type: 'number' },
            metadata: { category: 'math' },
          },
        ],
      }),
      'genkit:output': JSON.stringify({
        candidates: [
          {
            message: {
              role: 'model',
              content: [{ text: 'Truncated output...' }],
            },
            finishReason: 'length',
            usage: {
              inputTokens: 10,
              outputTokens: 50,
              custom: { thoughtsTokens: 7 },
            },
          },
        ],
      }),
    },
  });

  const childAborted = makeSpan({
    spanId: 'm-aborted',
    parentSpanId: 'm-root',
    displayName: '/model/googleai/gemini-2.5-flash',
    startTime: 210,
    endTime: 250,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({ messages: [] }),
      'genkit:output': JSON.stringify({
        message: { role: 'model', content: [{ text: 'Cancelled' }] },
        finishReason: 'aborted',
        usage: {
          inputTokens: 5,
          custom: { thoughtsTokens: 3 },
        },
      }),
    },
  });

  const childStructuredError = makeSpan({
    spanId: 'm-err',
    parentSpanId: 'm-root',
    displayName: '/model/googleai/gemini-2.5-flash',
    startTime: 260,
    endTime: 290,
    attributes: {
      'genkit:metadata:subtype': 'model',
      'genkit:input': JSON.stringify({ messages: [] }),
      'genkit:output': JSON.stringify({
        error: {
          message: 'Quota exceeded',
          details: { stack: 'Error: Quota exceeded\n    at callModel' },
        },
      }),
    },
  });

  return {
    traceId: 'trace-standalone',
    spans: {
      'm-root': rootModel,
      'm-aborted': childAborted,
      'm-err': childStructuredError,
    },
  };
}

describe('trajectory utils', () => {
  describe('single-turn agent trajectory', () => {
    it('conforms to TrajectoryBundleSchema', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(TrajectoryBundleSchema.parse(bundle)).toEqual(bundle);
    });

    it('sets rootTrajectoryId to the traceId', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.rootTrajectoryId).toBe('trace-1');
    });

    it('extracts sessionId onto the bundle', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.sessionId).toBe('sess-123');
    });

    it('extracts cleaned agentName onto the root trajectory', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.trajectories['trace-1'].agentName).toBe('weatherAgent');
    });

    it('extracts systemInstructions from preamble system message', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.trajectories['trace-1'].systemInstructions).toBe(
        'You are a helpful weather assistant.'
      );
    });

    it('extracts declared tool definitions', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.trajectories['trace-1'].tools).toEqual([
        expect.objectContaining({ name: 'getWeather' }),
      ]);
    });

    it('aggregates token usage across model spans', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.trajectories['trace-1'].usage).toEqual({
        inputTokens: 65,
        outputTokens: 27,
        totalTokens: 92,
      });
    });

    it('extracts user prompt as the first step', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.trajectories['trace-1'].steps[0]).toMatchObject({
        stepNumber: 1,
        source: 'user',
        content: [{ text: 'What is the weather in Seattle?' }],
      });
    });

    it('extracts agent reasoning, tools stopReason, and extended endTime in the second step', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.trajectories['trace-1'].steps[1]).toMatchObject({
        stepNumber: 2,
        source: 'agent',
        model: 'googleai/gemini-2.5-pro',
        content: [{ reasoning: 'Need to call getWeather for Seattle.' }],
        stopReason: 'tools',
        endTime: 320,
      });
    });

    it('pairs tool call execution details onto the second step action', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.trajectories['trace-1'].steps[1].actions).toEqual([
        expect.objectContaining({
          actionId: 'call-1',
          spanId: 'tool-1',
          name: 'getWeather',
          type: 'tool',
          status: 'success',
          input: { city: 'Seattle' },
          output: { tempF: 62, condition: 'Rainy' },
          durationMs: 65,
        }),
      ]);
    });

    it('extracts final agent response as the completed third step', () => {
      const bundle = extractTrajectoryFromTrace(buildSingleTurnWeatherTrace());

      expect(bundle.trajectories['trace-1'].steps[2]).toMatchObject({
        stepNumber: 3,
        source: 'agent',
        stopReason: 'completed',
        content: [{ text: 'It is 62°F and rainy in Seattle.' }],
      });
    });
  });

  describe('parallel tool call pairing', () => {
    it('pairs out-of-order parallel tool spans by explicit genkit:metadata:tool:callId', () => {
      const bundle = extractTrajectoryFromTrace(buildParallelToolTrace());
      const agentStep = bundle.trajectories['trace-parallel'].steps[1];

      expect(agentStep.actions.slice(0, 2)).toMatchObject([
        {
          actionId: 'ref-b',
          spanId: 'tool-read-b',
          output: 'content-B',
        },
        {
          actionId: 'ref-a',
          spanId: 'tool-read-a',
          output: 'content-A',
        },
      ]);
    });

    it('falls back to chronological span order when callId is absent', () => {
      const bundle = extractTrajectoryFromTrace(buildParallelToolTrace());
      const agentStep = bundle.trajectories['trace-parallel'].steps[1];

      expect(agentStep.actions.slice(2, 4)).toMatchObject([
        {
          spanId: 'tool-search-1',
          output: ['res-1'],
        },
        {
          spanId: 'tool-search-2',
          output: ['res-2'],
        },
      ]);
    });

    it('conforms to TrajectoryBundleSchema when parallel calls have conflicting explicit refs', () => {
      const bundle = extractTrajectoryFromTrace(buildConflictingRefTrace());

      expect(() => TrajectoryBundleSchema.parse(bundle)).not.toThrow();
    });

    it('keeps unresumed parallel call untouched when only a sibling call to the same tool is resumed', () => {
      const bundle = extractTrajectoryFromTrace(buildConflictingRefTrace());
      const actions =
        bundle.trajectories['trace-ref-conflict'].steps[1].actions;

      expect(actions[0]).toEqual({
        actionId: 'tx-1',
        name: 'transfer',
        type: 'interrupt',
        status: 'interrupted',
        input: { amount: 10 },
      });
    });

    it('pairs resumed parallel call accurately by explicit ref when sibling call is unresumed', () => {
      const bundle = extractTrajectoryFromTrace(buildConflictingRefTrace());
      const actions =
        bundle.trajectories['trace-ref-conflict'].steps[1].actions;

      expect(actions[1]).toMatchObject({
        actionId: 'tx-2',
        name: 'transfer',
        type: 'interrupt',
        status: 'interrupted',
        input: { amount: 20 },
        output: { ok: 20 },
        resume: { strategy: 'restart', payload: { amount: 20 } },
      });
    });
  });

  describe('subagent trajectories and span exclusion', () => {
    it('extracts both root and child subagent trajectories into flat trajectories map', () => {
      const bundle = extractTrajectoryFromTrace(
        buildSubagentAndExclusionTrace()
      );

      expect(Object.keys(bundle.trajectories)).toEqual([
        'subAgent',
        'trace-sub',
      ]);
    });

    it('excludes spans marked with genkit:metadata:trajectory:exclude and non-model helper spans from steps', () => {
      const bundle = extractTrajectoryFromTrace(
        buildSubagentAndExclusionTrace()
      );

      expect(bundle.trajectories['trace-sub'].steps).toHaveLength(3);
    });

    it('links parent delegation action to child subtrajectoryId', () => {
      const bundle = extractTrajectoryFromTrace(
        buildSubagentAndExclusionTrace()
      );

      expect(bundle.trajectories['trace-sub'].steps[1].actions).toEqual([
        expect.objectContaining({
          actionId: 'sub-call-1',
          name: 'delegate_to_researcher',
          type: 'subagent',
          status: 'success',
          subtrajectoryId: 'subAgent',
        }),
      ]);
    });

    it('populates parentTrajectoryId, agentName, and steps on extracted subagent trajectory', () => {
      const bundle = extractTrajectoryFromTrace(
        buildSubagentAndExclusionTrace()
      );

      expect(bundle.trajectories['subAgent']).toMatchObject({
        parentTrajectoryId: 'trace-sub',
        agentName: 'researcher',
        steps: [
          expect.objectContaining({ content: [{ text: 'Topic: OATR' }] }),
          expect.objectContaining({
            content: [{ text: 'OATR defines TrajectoryBundle.' }],
          }),
        ],
      });
    });

    it('conforms to TrajectoryBundleSchema when matching delegate_to_ subagents and skewed-clock fallbacks', () => {
      const bundle = extractTrajectoryFromTrace(
        buildDelegateAndSkewedFallbackTrace()
      );

      expect(() => TrajectoryBundleSchema.parse(bundle)).not.toThrow();
    });

    it('matches tool request to delegate_to_ tool span and finds subagent across intermediate wrapper spans', () => {
      const bundle = extractTrajectoryFromTrace(
        buildDelegateAndSkewedFallbackTrace()
      );
      const agentStep = bundle.trajectories['trace-delegate-fallback'].steps[1];

      expect(agentStep.actions[0]).toMatchObject({
        spanId: 'tool-delegate',
        type: 'subagent',
        subtrajectoryId: 'sub-researcher',
        output: { summary: 'Mars facts' },
      });
    });

    it('pairs skewed-timestamp tool span via turnIndex fallback and populates output from ref-less toolResponse', () => {
      const bundle = extractTrajectoryFromTrace(
        buildDelegateAndSkewedFallbackTrace()
      );
      const agentStep = bundle.trajectories['trace-delegate-fallback'].steps[1];

      expect(agentStep.actions[1]).toMatchObject({
        spanId: 'tool-skewed',
        type: 'tool',
        output: { hit: true },
      });
    });
  });

  describe('wrapped traces and sessionId resolution', () => {
    it('prefers outer agent sessionId over nested subagent sessionId in extractSessionIdFromTrace', () => {
      expect(extractSessionIdFromTrace(buildWrappedTrace())).toBe(
        'outer-session'
      );
    });

    it('populates outer agent sessionId onto extracted TrajectoryBundle', () => {
      const bundle = extractTrajectoryFromTrace(buildWrappedTrace());

      expect(bundle.sessionId).toBe('outer-session');
    });

    it('unwraps dev-run-action-wrapper to resolve root agentName', () => {
      const bundle = extractTrajectoryFromTrace(buildWrappedTrace());

      expect(bundle.trajectories['trace-wrapped'].agentName).toBe('main');
    });

    it('extracts fallback user step from unwrapped root agent input when no model span exists', () => {
      const bundle = extractTrajectoryFromTrace(buildWrappedTrace());

      expect(bundle.trajectories['trace-wrapped'].steps).toEqual([
        expect.objectContaining({
          source: 'user',
          content: [{ text: 'Hello' }],
        }),
      ]);
    });
  });

  describe('history filtering, error states, and pending actions', () => {
    it('skips messages marked with _genkit_history metadata when extracting user steps', () => {
      const bundle = extractTrajectoryFromTrace(buildErrorAndPendingTrace());

      expect(bundle.trajectories['trace-err'].steps[0]).toMatchObject({
        source: 'user',
        content: [{ text: 'Deploy build' }],
      });
    });

    it('marks failed tool actions with error status and span status message', () => {
      const bundle = extractTrajectoryFromTrace(buildErrorAndPendingTrace());

      expect(
        bundle.trajectories['trace-err'].steps[1].actions[0]
      ).toMatchObject({
        actionId: 'dep-1',
        status: 'error',
        error: 'Permission denied',
      });
    });

    it('marks unfinished tool spans (endTime 0) as pending', () => {
      const bundle = extractTrajectoryFromTrace(buildErrorAndPendingTrace());

      expect(
        bundle.trajectories['trace-err'].steps[1].actions[1]
      ).toMatchObject({
        actionId: 'chk-1',
        status: 'pending',
      });
    });

    it('marks failed model spans with errored stopReason and error message', () => {
      const bundle = extractTrajectoryFromTrace(buildErrorAndPendingTrace());

      expect(bundle.trajectories['trace-err'].steps[2]).toMatchObject({
        stopReason: 'errored',
        error: { message: 'Quota exceeded' },
      });
    });
  });

  describe('Human-in-the-Loop (HITL) interrupts', () => {
    describe('unresumed ToolInterruptError', () => {
      it('conforms to TrajectoryBundleSchema', () => {
        const bundle = extractTrajectoryFromTrace(
          buildUnresumedInterruptTrace()
        );

        expect(() => TrajectoryBundleSchema.parse(bundle)).not.toThrow();
      });

      it('marks step stopReason as interrupted', () => {
        const bundle = extractTrajectoryFromTrace(
          buildUnresumedInterruptTrace()
        );

        expect(
          bundle.trajectories['trace-int-unresumed'].steps[1].stopReason
        ).toBe('interrupted');
      });

      it('marks unresumed tool action with interrupt type and interrupted status', () => {
        const bundle = extractTrajectoryFromTrace(
          buildUnresumedInterruptTrace()
        );

        expect(
          bundle.trajectories['trace-int-unresumed'].steps[1].actions[0]
        ).toMatchObject({
          actionId: 'tx-1',
          name: 'transferFunds',
          type: 'interrupt',
          status: 'interrupted',
          input: { to: 'Alice', amount: 500 },
        });
      });

      it('leaves resume undefined on unresumed interrupted action', () => {
        const bundle = extractTrajectoryFromTrace(
          buildUnresumedInterruptTrace()
        );

        expect(
          bundle.trajectories['trace-int-unresumed'].steps[1].actions[0].resume
        ).toBeUndefined();
      });
    });

    describe('in-trace HITL resume decisions', () => {
      it('conforms to TrajectoryBundleSchema', () => {
        const bundle = extractTrajectoryFromTrace(
          buildInTraceResumedInterruptTrace()
        );

        expect(() => TrajectoryBundleSchema.parse(bundle)).not.toThrow();
      });

      it('keeps interrupted step stopReason as interrupted after in-trace resume', () => {
        const bundle = extractTrajectoryFromTrace(
          buildInTraceResumedInterruptTrace()
        );

        expect(
          bundle.trajectories['trace-int-resumed'].steps[1].stopReason
        ).toBe('interrupted');
      });

      it('pairs respond-strategy HITL resume payload and output onto interrupted action', () => {
        const bundle = extractTrajectoryFromTrace(
          buildInTraceResumedInterruptTrace()
        );

        expect(
          bundle.trajectories['trace-int-resumed'].steps[1].actions[0]
        ).toMatchObject({
          actionId: 'tx-1',
          type: 'interrupt',
          status: 'interrupted',
          output: { approved: true, receipt: 'r-1' },
          resume: {
            strategy: 'respond',
            payload: { approved: true, receipt: 'r-1' },
          },
        });
      });

      it('pairs restart-strategy HITL resume metadata and restarted tool output onto interrupted action', () => {
        const bundle = extractTrajectoryFromTrace(
          buildInTraceResumedInterruptTrace()
        );

        expect(
          bundle.trajectories['trace-int-resumed'].steps[1].actions[1]
        ).toMatchObject({
          actionId: 'flt-1',
          type: 'interrupt',
          status: 'interrupted',
          output: { pnr: 'XYZ987', seat: '12A' },
          resume: {
            strategy: 'restart',
            payload: { seat: '12A' },
          },
        });
      });

      it('advances turnIndex on completed step following in-trace resume turn', () => {
        const bundle = extractTrajectoryFromTrace(
          buildInTraceResumedInterruptTrace()
        );

        expect(bundle.trajectories['trace-int-resumed'].steps[2]).toMatchObject(
          {
            turnIndex: 1,
            stopReason: 'completed',
            content: [{ text: 'Paid $500 (r-1) and booked flight XYZ987.' }],
          }
        );
      });
    });

    describe('ancestor turn interrupt metadata and input.resume fallbacks', () => {
      it('conforms to TrajectoryBundleSchema', () => {
        const bundle = extractTrajectoryFromTrace(buildAncestorHitlTrace());

        expect(() => TrajectoryBundleSchema.parse(bundle)).not.toThrow();
      });

      it('marks step stopReason as interrupted from ancestor turn finishReason', () => {
        const bundle = extractTrajectoryFromTrace(buildAncestorHitlTrace());

        expect(
          bundle.trajectories['trace-ancestor-hitl'].steps[1].stopReason
        ).toBe('interrupted');
      });

      it('detects respond-strategy interrupt from ancestor turn output metadata and input.resume', () => {
        const bundle = extractTrajectoryFromTrace(buildAncestorHitlTrace());

        expect(
          bundle.trajectories['trace-ancestor-hitl'].steps[1].actions[0]
        ).toMatchObject({
          actionId: 'r-approve',
          type: 'interrupt',
          status: 'interrupted',
          output: { confirmed: true },
          resume: { strategy: 'respond', payload: { confirmed: true } },
        });
      });

      it('detects restart-strategy interrupt from ancestor turn finishReason and input.resume', () => {
        const bundle = extractTrajectoryFromTrace(buildAncestorHitlTrace());

        expect(
          bundle.trajectories['trace-ancestor-hitl'].steps[1].actions[1]
        ).toMatchObject({
          actionId: 'r-retry',
          type: 'interrupt',
          status: 'interrupted',
          output: { retried: 'ok' },
          resume: { strategy: 'restart', payload: { id: 2, force: true } },
        });
      });
    });
  });

  describe('empty, pruned, and model-less fallback traces', () => {
    it('falls back to "root" trajectory ID with empty steps when traceId and spans are empty', () => {
      const emptyBundle = extractTrajectoryFromTrace({
        traceId: '',
        spans: {},
      });

      expect(TrajectoryBundleSchema.parse(emptyBundle)).toMatchObject({
        rootTrajectoryId: 'root',
        trajectories: {
          root: expect.objectContaining({ steps: [] }),
        },
      });
    });

    it('preserves sessionId while producing empty steps when all spans are excluded', () => {
      const prunedWithSession = extractTrajectoryFromTrace({
        traceId: 'trace-pruned',
        startTime: 10,
        endTime: 20,
        spans: {
          excludedRoot: makeSpan({
            spanId: 'excludedRoot',
            attributes: {
              'genkit:metadata:trajectory:exclude': 'true',
              'session.id': 'sess-pruned',
            },
          }),
        },
      });

      expect(prunedWithSession).toMatchObject({
        sessionId: 'sess-pruned',
        trajectories: {
          'trace-pruned': expect.objectContaining({ steps: [] }),
        },
      });
    });

    it('extracts fallback user and system steps from scope inputs when no model spans exist', () => {
      const fallbackBundle = extractTrajectoryFromTrace({
        traceId: 'trace-fallback',
        spans: {
          agentRoot: makeSpan({
            spanId: 'agentRoot',
            displayName: '/agent/earlyFailAgent',
            startTime: 100,
            endTime: 150,
            attributes: {
              'genkit:metadata:subtype': 'agent',
              'genkit:input': JSON.stringify({
                message: {
                  role: 'user',
                  content: [{ text: 'Hello before model call' }],
                },
              }),
            },
          }),
          turn1: makeSpan({
            spanId: 'turn1',
            parentSpanId: 'agentRoot',
            displayName: 'runTurn-1',
            startTime: 110,
            endTime: 140,
            attributes: {
              'genkit:input': JSON.stringify({
                messages: [
                  {
                    role: 'system',
                    content: [{ text: 'System notice' }],
                  },
                ],
              }),
            },
          }),
        },
      });

      expect(
        TrajectoryBundleSchema.parse(fallbackBundle).trajectories[
          'trace-fallback'
        ].steps
      ).toMatchObject([
        {
          source: 'user',
          content: [{ text: 'Hello before model call' }],
        },
        {
          source: 'system',
          content: [{ text: 'System notice' }],
        },
      ]);
    });
  });

  describe('standalone model traces, stop reasons, and structured errors', () => {
    it('conforms to TrajectoryBundleSchema', () => {
      const bundle = extractTrajectoryFromTrace(buildStandaloneModelTrace());

      expect(() => TrajectoryBundleSchema.parse(bundle)).not.toThrow();
    });

    it('leaves agentName undefined when root span is a standalone model', () => {
      const bundle = extractTrajectoryFromTrace(buildStandaloneModelTrace());

      expect(bundle.trajectories['trace-standalone'].agentName).toBeUndefined();
    });

    it('extracts systemInstructions from non-preamble system message on standalone model', () => {
      const bundle = extractTrajectoryFromTrace(buildStandaloneModelTrace());

      expect(bundle.trajectories['trace-standalone'].systemInstructions).toBe(
        'Be concise.'
      );
    });

    it('normalizes string user message content into text part', () => {
      const bundle = extractTrajectoryFromTrace(buildStandaloneModelTrace());

      expect(bundle.trajectories['trace-standalone'].steps[0]).toMatchObject({
        source: 'user',
        content: [{ text: 'Write an essay' }],
      });
    });

    it('maps length finishReason from candidates output to max_tokens stopReason', () => {
      const bundle = extractTrajectoryFromTrace(buildStandaloneModelTrace());

      expect(bundle.trajectories['trace-standalone'].steps[1].stopReason).toBe(
        'max_tokens'
      );
    });

    it('maps aborted finishReason to cancelled stopReason', () => {
      const bundle = extractTrajectoryFromTrace(buildStandaloneModelTrace());

      expect(bundle.trajectories['trace-standalone'].steps[2].stopReason).toBe(
        'cancelled'
      );
    });

    it('aggregates custom token usage metrics across model spans', () => {
      const bundle = extractTrajectoryFromTrace(buildStandaloneModelTrace());

      expect(bundle.trajectories['trace-standalone'].usage).toEqual({
        inputTokens: 15,
        outputTokens: 50,
        custom: { thoughtsTokens: 10 },
      });
    });

    it('extracts structured output error message and stack trace onto errored model step', () => {
      const bundle = extractTrajectoryFromTrace(buildStandaloneModelTrace());

      expect(bundle.trajectories['trace-standalone'].steps[3]).toMatchObject({
        stopReason: 'errored',
        error: {
          message: 'Quota exceeded',
          stack: 'Error: Quota exceeded\n    at callModel',
        },
      });
    });
  });
});
