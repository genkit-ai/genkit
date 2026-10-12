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
import type { SpanData, TraceData } from '../../src/types';
import {
  extractTrajectoryFromTrace,
  mergeSessionTrajectories,
} from '../../src/utils/trajectory';

function makeSpan(overrides: Partial<SpanData> & { spanId: string }): SpanData {
  return {
    traceId: overrides.traceId || 'trace-1',
    spanId: overrides.spanId,
    parentSpanId: overrides.parentSpanId,
    startTime: overrides.startTime ?? 100,
    endTime: overrides.endTime ?? 200,
    displayName: overrides.displayName || overrides.spanId,
    attributes: overrides.attributes || {},
    instrumentationLibrary: { name: 'genkit-tracer', version: 'v1' },
    spanKind: 'INTERNAL',
    status: overrides.status || { code: 0 },
  };
}

describe('trajectory utils', () => {
  it('extracts single-turn agent trajectory with tool call and reasoning', () => {
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

    const trace: TraceData = {
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

    const bundle = extractTrajectoryFromTrace(trace);
    expect(bundle.rootTrajectoryId).toBe('trace-1');
    expect(bundle.sessionId).toBe('sess-123');

    const root = bundle.trajectories['trace-1'];
    expect(root).toBeDefined();
    expect(root.agentName).toBe('weatherAgent');
    expect(root.systemInstructions).toBe(
      'You are a helpful weather assistant.'
    );
    expect(root.tools).toHaveLength(1);
    expect(root.tools[0].name).toBe('getWeather');
    expect(root.usage).toEqual({
      inputTokens: 65,
      outputTokens: 27,
      totalTokens: 92,
    });

    expect(root.steps).toHaveLength(3);

    // Step 1: User prompt
    expect(root.steps[0].stepNumber).toBe(1);
    expect(root.steps[0].source).toBe('user');
    expect(root.steps[0].content).toEqual([
      { text: 'What is the weather in Seattle?' },
    ]);

    // Step 2: Agent reasoning + tool call
    expect(root.steps[1].stepNumber).toBe(2);
    expect(root.steps[1].source).toBe('agent');
    expect(root.steps[1].model).toBe('googleai/gemini-2.5-pro');
    expect(root.steps[1].content).toEqual([
      { reasoning: 'Need to call getWeather for Seattle.' },
    ]);
    expect(root.steps[1].stopReason).toBe('tools');
    expect(root.steps[1].endTime).toBe(320); // Extended to include paired toolSpan1 endTime
    expect(root.steps[1].actions).toHaveLength(1);
    expect(root.steps[1].actions[0]).toMatchObject({
      actionId: 'call-1',
      spanId: 'tool-1',
      name: 'getWeather',
      type: 'tool',
      status: 'success',
      input: { city: 'Seattle' },
      output: { tempF: 62, condition: 'Rainy' },
      durationMs: 65,
    });

    // Step 3: Final Agent response
    expect(root.steps[2].stepNumber).toBe(3);
    expect(root.steps[2].source).toBe('agent');
    expect(root.steps[2].stopReason).toBe('completed');
    expect(root.steps[2].content).toEqual([
      { text: 'It is 62°F and rainy in Seattle.' },
    ]);
  });

  it('pairs parallel tool calls using genkit:metadata:tool:callId and order fallback', () => {
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

    const trace: TraceData = {
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

    const bundle = extractTrajectoryFromTrace(trace);
    const agentStep = bundle.trajectories['trace-parallel'].steps[1];
    expect(agentStep.actions).toHaveLength(4);

    expect(agentStep.actions[0].actionId).toBe('ref-b');
    expect(agentStep.actions[0].spanId).toBe('tool-read-b');
    expect(agentStep.actions[0].output).toBe('content-B');

    expect(agentStep.actions[1].actionId).toBe('ref-a');
    expect(agentStep.actions[1].spanId).toBe('tool-read-a');
    expect(agentStep.actions[1].output).toBe('content-A');

    expect(agentStep.actions[2].spanId).toBe('tool-search-1');
    expect(agentStep.actions[2].output).toEqual(['res-1']);

    expect(agentStep.actions[3].spanId).toBe('tool-search-2');
    expect(agentStep.actions[3].output).toEqual(['res-2']);
  });

  it('handles ToolInterruptError and AgentInput.resume across turns', () => {
    // Trace 1: Tool triggers interrupt
    const trace1: TraceData = {
      traceId: 'trace-int-1',
      startTime: 100,
      endTime: 300,
      spans: {
        root1: makeSpan({
          traceId: 'trace-int-1',
          spanId: 'root1',
          displayName: '/agent/bankAgent',
          startTime: 100,
          endTime: 300,
          attributes: {
            'genkit:metadata:subtype': 'agent',
            'genkit:metadata:agent:sessionId': 'sess-int',
          },
        }),
        model1: makeSpan({
          traceId: 'trace-int-1',
          spanId: 'model1',
          parentSpanId: 'root1',
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
          traceId: 'trace-int-1',
          spanId: 'tool1',
          parentSpanId: 'root1',
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

    // Trace 2: User resumes with respond
    const trace2: TraceData = {
      traceId: 'trace-int-2',
      startTime: 400,
      endTime: 600,
      spans: {
        root2: makeSpan({
          traceId: 'trace-int-2',
          spanId: 'root2',
          displayName: '/agent/bankAgent',
          startTime: 400,
          endTime: 600,
          attributes: {
            'genkit:metadata:subtype': 'agent',
            'genkit:metadata:agent:sessionId': 'sess-int',
            'genkit:input': JSON.stringify({
              resume: {
                respond: [
                  {
                    toolResponse: {
                      ref: 'tx-1',
                      name: 'transferFunds',
                      output: { approved: true, txId: 'receipt-99' },
                    },
                  },
                ],
              },
            }),
          },
        }),
        model2: makeSpan({
          traceId: 'trace-int-2',
          spanId: 'model2',
          parentSpanId: 'root2',
          displayName: '/model/gemini',
          startTime: 420,
          endTime: 580,
          attributes: {
            'genkit:metadata:subtype': 'model',
            'genkit:input': JSON.stringify({
              messages: [
                { role: 'user', content: [{ text: 'Transfer $500 to Alice' }] },
                {
                  role: 'model',
                  content: [
                    {
                      toolRequest: {
                        ref: 'tx-1',
                        name: 'transferFunds',
                        input: { to: 'Alice', amount: 500 },
                      },
                      metadata: { resolvedInterrupt: { confirmRequired: true } },
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
                        output: { approved: true, txId: 'receipt-99' },
                      },
                      metadata: { interruptResponse: true },
                    },
                  ],
                },
              ],
            }),
            'genkit:output': JSON.stringify({
              message: {
                role: 'model',
                content: [
                  { text: 'Transfer of $500 to Alice completed (receipt-99).' },
                ],
              },
              finishReason: 'stop',
            }),
          },
        }),
      },
    };

    const singleBundle = extractTrajectoryFromTrace(trace1);
    const step2Single = singleBundle.trajectories['trace-int-1'].steps[1];
    expect(step2Single.stopReason).toBe('interrupted');
    expect(step2Single.actions[0].type).toBe('interrupt');
    expect(step2Single.actions[0].status).toBe('interrupted');

    const merged = mergeSessionTrajectories([trace2, trace1]); // Out-of-order input to test sorting
    expect(merged.sessionId).toBe('sess-int');
    const root = merged.trajectories['trace-int-1'];
    expect(root.steps).toHaveLength(3);

    const interruptedStep = root.steps[1];
    expect(interruptedStep.actions[0].type).toBe('interrupt');
    expect(interruptedStep.actions[0].status).toBe('interrupted');
    expect(interruptedStep.actions[0].resume).toEqual({
      strategy: 'respond',
      payload: { approved: true, txId: 'receipt-99' },
    });
    expect(interruptedStep.actions[0].output).toEqual({
      approved: true,
      txId: 'receipt-99',
    });

    const finalStep = root.steps[2];
    expect(finalStep.turnIndex).toBe(1);
    expect(finalStep.traceId).toBe('trace-int-2');
    expect(finalStep.content).toEqual([
      { text: 'Transfer of $500 to Alice completed (receipt-99).' },
    ]);
  });

  it('extracts nested subagent trajectories and excludes trajectory:exclude subtrees', () => {
    const trace: TraceData = {
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
            'genkit:output': JSON.stringify({ summary: 'OATR is a flat bundle' }),
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
              messages: [
                { role: 'user', content: [{ text: 'Topic: OATR' }] },
              ],
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

    const bundle = extractTrajectoryFromTrace(trace);
    expect(Object.keys(bundle.trajectories)).toEqual(['subAgent', 'trace-sub']);

    const root = bundle.trajectories['trace-sub'];
    // Guardrail model was excluded, so root only has 3 steps (user, model1+subagent, model2)
    expect(root.steps).toHaveLength(3);
    expect(root.steps[1].actions).toHaveLength(1);
    expect(root.steps[1].actions[0]).toMatchObject({
      actionId: 'sub-call-1',
      name: 'delegate_to_researcher',
      type: 'subagent',
      status: 'success',
      subtrajectoryId: 'subAgent',
    });

    const sub = bundle.trajectories['subAgent'];
    expect(sub.parentTrajectoryId).toBe('trace-sub');
    expect(sub.agentName).toBe('researcher');
    expect(sub.steps).toHaveLength(2);
    expect(sub.steps[0].content).toEqual([{ text: 'Topic: OATR' }]);
    expect(sub.steps[1].content).toEqual([
      { text: 'OATR defines TrajectoryBundle.' },
    ]);
  });

  it('deduplicates _genkit_history messages across multi-turn sessions', () => {
    const trace1: TraceData = {
      traceId: 't-turn-1',
      startTime: 100,
      endTime: 200,
      spans: {
        m1: makeSpan({
          traceId: 't-turn-1',
          spanId: 'm1',
          displayName: '/model/gemini',
          startTime: 100,
          endTime: 200,
          attributes: {
            'genkit:metadata:subtype': 'model',
            'genkit:metadata:agent:sessionId': 's-multi',
            'genkit:input': JSON.stringify({
              messages: [{ role: 'user', content: [{ text: 'First turn' }] }],
            }),
            'genkit:output': JSON.stringify({
              message: { role: 'model', content: [{ text: 'First reply' }] },
              finishReason: 'stop',
            }),
          },
        }),
      },
    };

    const trace2: TraceData = {
      traceId: 't-turn-2',
      startTime: 300,
      endTime: 400,
      spans: {
        m2: makeSpan({
          traceId: 't-turn-2',
          spanId: 'm2',
          displayName: '/model/gemini',
          startTime: 300,
          endTime: 400,
          attributes: {
            'genkit:metadata:subtype': 'model',
            'genkit:metadata:agent:sessionId': 's-multi',
            'genkit:input': JSON.stringify({
              messages: [
                {
                  role: 'user',
                  content: [{ text: 'First turn' }],
                  metadata: { _genkit_history: true },
                },
                {
                  role: 'model',
                  content: [{ text: 'First reply' }],
                  metadata: { _genkit_history: true },
                },
                {
                  role: 'user',
                  content: [{ text: 'Second turn' }],
                },
              ],
            }),
            'genkit:output': JSON.stringify({
              message: { role: 'model', content: [{ text: 'Second reply' }] },
              finishReason: 'stop',
            }),
          },
        }),
      },
    };

    const merged = mergeSessionTrajectories([trace1, trace2]);
    const root = merged.trajectories['t-turn-1'];
    expect(root.steps).toHaveLength(4);
    expect(root.steps.map((s) => [s.turnIndex, s.source, s.content[0].text])).toEqual([
      [0, 'user', 'First turn'],
      [0, 'agent', 'First reply'],
      [1, 'user', 'Second turn'],
      [1, 'agent', 'Second reply'],
    ]);
  });
});
