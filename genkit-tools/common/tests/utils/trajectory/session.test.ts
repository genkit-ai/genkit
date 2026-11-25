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

import { describe, expect, it } from 'vitest';
import { TrajectoryBundleSchema } from '../../../src/types/index.js';
import { extractTrajectoryFromTraces } from '../../../src/utils/index.js';
import {
  makeAgentSpan,
  makeModelSpan,
  makeSpan,
  makeToolSpan,
  makeTrace,
  makeTurnSpan,
} from './mock_data.js';

describe('extractTrajectoryFromTraces', () => {
  it('returns an empty bundle when given no traces', () => {
    expect(extractTrajectoryFromTraces([])).toEqual({
      rootTrajectoryId: '',
      trajectories: {},
    });
  });

  it('merges multi-trace session traces chronologically with sequential stepNumber, turnIndex, usage, and tools', () => {
    const trace1 = makeTrace(
      [
        makeAgentSpan({
          spanId: 'agent-1',
          name: 'weatherAgent',
          sessionId: 'sess-multi',
          startTime: 100,
          endTime: 300,
        }),
        makeTurnSpan({
          spanId: 'turn-1',
          parentSpanId: 'agent-1',
          turn: 1,
          startTime: 105,
          endTime: 295,
        }),
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: 'turn-1',
          startTime: 110,
          endTime: 280,
          messages: [
            { role: 'system', content: [{ text: 'Be concise.' }] },
            { role: 'user', content: [{ text: 'Hi' }] },
          ],
          tools: [{ name: 'getWeather', description: 'Gets weather' }],
          outputMessage: {
            role: 'model',
            content: [{ text: 'Hello!' }],
          },
          usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
        }),
      ],
      { traceId: 'trace-1', startTime: 100, endTime: 300 }
    );

    const trace2 = makeTrace(
      [
        makeAgentSpan({
          spanId: 'agent-2',
          name: 'weatherAgent',
          sessionId: 'sess-multi',
          startTime: 400,
          endTime: 700,
        }),
        makeTurnSpan({
          spanId: 'turn-1',
          parentSpanId: 'agent-2',
          turn: 1,
          startTime: 405,
          endTime: 695,
        }),
        makeModelSpan({
          spanId: 'm2',
          parentSpanId: 'turn-1',
          startTime: 410,
          endTime: 680,
          messages: [
            {
              role: 'user',
              content: [{ text: 'Hi' }],
              metadata: { _genkit_history: true },
            },
            {
              role: 'model',
              content: [{ text: 'Hello!' }],
              metadata: { _genkit_history: true },
            },
            { role: 'user', content: [{ text: 'Weather in NYC?' }] },
          ],
          tools: [
            { name: 'getWeather', description: 'Gets weather' },
            { name: 'getTime', description: 'Gets local time' },
          ],
          outputMessage: {
            role: 'model',
            content: [{ text: 'Sunny in NYC.' }],
          },
          usage: { inputTokens: 25, outputTokens: 8, totalTokens: 33 },
        }),
      ],
      { traceId: 'trace-2', startTime: 400, endTime: 700 }
    );

    // Intentionally pass out of chronological order
    const bundle = extractTrajectoryFromTraces([trace2, trace1]);

    expect(TrajectoryBundleSchema.parse(bundle)).toMatchObject({
      rootTrajectoryId: 'trace-1',
      sessionId: 'sess-multi',
      trajectories: {
        'trace-1': {
          trajectoryId: 'trace-1',
          agentName: 'weatherAgent',
          status: 'success',
          finishReason: 'stop',
          systemInstructions: 'Be concise.',
          startTime: 100,
          endTime: 700,
          tools: [
            { name: 'getWeather', description: 'Gets weather' },
            { name: 'getTime', description: 'Gets local time' },
          ],
          usage: { inputTokens: 35, outputTokens: 13, totalTokens: 48 },
          steps: [
            {
              stepNumber: 1,
              traceId: 'trace-1',
              turnIndex: 0,
              source: 'user',
              content: [{ text: 'Hi' }],
            },
            {
              stepNumber: 2,
              traceId: 'trace-1',
              turnIndex: 0,
              source: 'agent',
              content: [{ text: 'Hello!' }],
            },
            {
              stepNumber: 3,
              traceId: 'trace-2',
              turnIndex: 1,
              source: 'user',
              content: [{ text: 'Weather in NYC?' }],
            },
            {
              stepNumber: 4,
              traceId: 'trace-2',
              turnIndex: 1,
              source: 'agent',
              content: [{ text: 'Sunny in NYC.' }],
            },
          ],
        },
      },
    });
  });

  it('re-points child subagent parentTrajectoryId to the merged rootTrajectoryId', () => {
    const trace1 = makeTrace(
      [
        makeAgentSpan({
          spanId: 'agent-1',
          name: 'coordinator',
          sessionId: 'sess-sub',
          startTime: 100,
          endTime: 200,
        }),
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: 'agent-1',
          startTime: 110,
          endTime: 190,
          messages: [{ role: 'user', content: [{ text: 'Start' }] }],
          outputMessage: { role: 'model', content: [{ text: 'Ready.' }] },
        }),
      ],
      { traceId: 'trace-1', startTime: 100, endTime: 200 }
    );

    const trace2 = makeTrace(
      [
        makeAgentSpan({
          spanId: 'agent-2',
          name: 'coordinator',
          sessionId: 'sess-sub',
          startTime: 300,
          endTime: 600,
        }),
        makeModelSpan({
          spanId: 'm2',
          parentSpanId: 'agent-2',
          startTime: 310,
          endTime: 360,
          messages: [{ role: 'user', content: [{ text: 'Delegate' }] }],
          outputMessage: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  ref: 'sub-call',
                  name: 'delegate_to_helper',
                  input: {},
                },
              },
            ],
          },
        }),
        makeToolSpan({
          spanId: 'delegate-tool',
          parentSpanId: 'agent-2',
          name: 'delegate_to_helper',
          startTime: 370,
          endTime: 550,
          output: { ok: true },
        }),
        makeAgentSpan({
          spanId: 'sub-helper',
          parentSpanId: 'delegate-tool',
          name: 'helper',
          startTime: 380,
          endTime: 540,
        }),
        makeModelSpan({
          spanId: 'sub-m1',
          parentSpanId: 'sub-helper',
          startTime: 390,
          endTime: 530,
          messages: [{ role: 'user', content: [{ text: 'Help' }] }],
          outputMessage: { role: 'model', content: [{ text: 'Done.' }] },
        }),
      ],
      { traceId: 'trace-2', startTime: 300, endTime: 600 }
    );

    const bundle = extractTrajectoryFromTraces([trace1, trace2]);

    expect(TrajectoryBundleSchema.parse(bundle)).toMatchObject({
      rootTrajectoryId: 'trace-1',
      trajectories: {
        'trace-1': {
          steps: [
            { stepNumber: 1, traceId: 'trace-1' },
            { stepNumber: 2, traceId: 'trace-1' },
            { stepNumber: 3, traceId: 'trace-2' },
            {
              stepNumber: 4,
              traceId: 'trace-2',
              actions: [
                {
                  actionId: 'sub-call',
                  type: 'subagent',
                  subtrajectoryId: 'sub-helper',
                },
              ],
            },
          ],
        },
        'sub-helper': {
          parentTrajectoryId: 'trace-1',
          agentName: 'helper',
        },
      },
    });
  });

  it('pairs cross-trace respond HITL resume decisions onto earlier interrupted actions', () => {
    const interruptTrace = makeTrace(
      [
        makeAgentSpan({
          spanId: 'agent-1',
          name: 'bankingAgent',
          sessionId: 'sess-hitl-cross',
          startTime: 100,
          endTime: 300,
          output: { finishReason: 'interrupted' },
        }),
        makeTurnSpan({
          spanId: 'turn-1',
          parentSpanId: 'agent-1',
          turn: 1,
          startTime: 105,
          endTime: 295,
        }),
        makeSpan({
          spanId: 'gen-1',
          parentSpanId: 'turn-1',
          displayName: 'generate',
          startTime: 108,
          endTime: 290,
          attributes: {
            'genkit:type': 'util',
            'genkit:name': 'generate',
            'genkit:output': JSON.stringify({
              finishReason: 'interrupted',
              message: {
                role: 'model',
                content: [
                  {
                    toolRequest: {
                      name: 'userApproval',
                      ref: 'call_373433',
                      input: { action: 'Transfer Money' },
                    },
                    metadata: { interrupt: true },
                  },
                ],
              },
            }),
          },
        }),
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: 'gen-1',
          startTime: 110,
          endTime: 250,
          messages: [{ role: 'user', content: [{ text: 'Transfer $10,000' }] }],
          outputMessage: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'userApproval',
                  ref: 'call_373433',
                  input: { action: 'Transfer Money' },
                },
              },
            ],
          },
        }),
        makeToolSpan({
          spanId: 'tool-1',
          parentSpanId: 'gen-1',
          name: 'userApproval',
          startTime: 255,
          endTime: 260,
          input: { action: 'Transfer Money' },
          status: { code: 2, message: '' },
          attributes: {
            'genkit:state': 'error',
            'genkit:isFailureSource': true,
          },
        }),
      ],
      { traceId: 'trace-int', startTime: 100, endTime: 300 }
    );

    const resumeTrace = makeTrace(
      [
        makeAgentSpan({
          spanId: 'agent-2',
          name: 'bankingAgent',
          sessionId: 'sess-hitl-cross',
          startTime: 500,
          endTime: 700,
          input: {
            resume: {
              respond: [
                {
                  toolResponse: {
                    name: 'userApproval',
                    ref: 'call_373433',
                    output: { approved: false },
                  },
                },
              ],
            },
          },
          output: { finishReason: 'stop' },
        }),
        makeTurnSpan({
          spanId: 'turn-1-r',
          parentSpanId: 'agent-2',
          turn: 1,
          startTime: 505,
          endTime: 695,
          input: {
            resume: {
              respond: [
                {
                  toolResponse: {
                    name: 'userApproval',
                    ref: 'call_373433',
                    output: { approved: false },
                  },
                },
              ],
            },
          },
        }),
        makeModelSpan({
          spanId: 'm2',
          parentSpanId: 'turn-1-r',
          startTime: 510,
          endTime: 680,
          messages: [
            { role: 'user', content: [{ text: 'Transfer $10,000' }] },
            {
              role: 'tool',
              content: [
                {
                  toolResponse: {
                    name: 'userApproval',
                    ref: 'call_373433',
                    output: { approved: false },
                  },
                },
              ],
              metadata: { resumed: true },
            },
          ],
          outputMessage: {
            role: 'model',
            content: [{ text: 'Transfer cancelled.' }],
          },
        }),
      ],
      { traceId: 'trace-res', startTime: 500, endTime: 700 }
    );

    const bundle = extractTrajectoryFromTraces([interruptTrace, resumeTrace]);

    expect(TrajectoryBundleSchema.parse(bundle)).toMatchObject({
      rootTrajectoryId: 'trace-int',
      sessionId: 'sess-hitl-cross',
      trajectories: {
        'trace-int': {
          status: 'success',
          finishReason: 'stop',
          steps: [
            { traceId: 'trace-int', source: 'user' },
            {
              traceId: 'trace-int',
              finishReason: 'interrupted',
              actions: [
                {
                  actionId: 'call_373433',
                  name: 'userApproval',
                  status: 'interrupted',
                  interrupt: true,
                  output: { approved: false },
                  resume: {
                    strategy: 'respond',
                    payload: { approved: false },
                  },
                },
              ],
            },
            {
              traceId: 'trace-res',
              finishReason: 'stop',
              content: [{ text: 'Transfer cancelled.' }],
            },
          ],
        },
      },
    });
  });

  it('pairs cross-trace restart HITL resume decisions and restarted tool span outputs', () => {
    const interruptTrace = makeTrace(
      [
        makeAgentSpan({
          spanId: 'agent-1',
          name: 'travelAgent',
          sessionId: 'sess-restart',
          startTime: 100,
          endTime: 300,
          output: { finishReason: 'interrupted' },
        }),
        makeTurnSpan({
          spanId: 'turn-1',
          parentSpanId: 'agent-1',
          turn: 1,
          startTime: 105,
          endTime: 295,
        }),
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: 'turn-1',
          startTime: 110,
          endTime: 200,
          messages: [{ role: 'user', content: [{ text: 'Book flight' }] }],
          outputMessage: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  ref: 'flt-1',
                  name: 'bookFlight',
                  input: { dest: 'SEA' },
                },
              },
            ],
          },
        }),
        makeToolSpan({
          spanId: 'flt-span-1',
          parentSpanId: 'turn-1',
          name: 'bookFlight',
          startTime: 210,
          endTime: 240,
          interrupt: { seatNeeded: true },
          status: { code: 2, message: 'ToolInterruptError' },
        }),
      ],
      { traceId: 'trace-int-r', startTime: 100, endTime: 300 }
    );

    const resumeTrace = makeTrace(
      [
        makeAgentSpan({
          spanId: 'agent-2',
          name: 'travelAgent',
          sessionId: 'sess-restart',
          startTime: 400,
          endTime: 700,
          input: {
            resume: {
              restart: [
                {
                  toolRequest: {
                    ref: 'flt-1',
                    name: 'bookFlight',
                    input: { dest: 'SEA' },
                  },
                  metadata: { resumed: { seat: '14C' } },
                },
              ],
            },
          },
          output: { finishReason: 'stop' },
        }),
        makeTurnSpan({
          spanId: 'turn-1-r',
          parentSpanId: 'agent-2',
          turn: 1,
          startTime: 405,
          endTime: 695,
        }),
        makeToolSpan({
          spanId: 'flt-span-2',
          parentSpanId: 'turn-1-r',
          name: 'bookFlight',
          startTime: 410,
          endTime: 500,
          resumed: { seat: '14C' },
          attributes: { 'genkit:metadata:tool:callId': 'flt-1' },
          output: { pnr: 'ABC123', seat: '14C' },
        }),
        makeModelSpan({
          spanId: 'm2',
          parentSpanId: 'turn-1-r',
          startTime: 510,
          endTime: 650,
          messages: [
            {
              role: 'user',
              content: [{ text: 'Book flight' }],
              metadata: { _genkit_history: true },
            },
          ],
          outputMessage: {
            role: 'model',
            content: [{ text: 'Booked seat 14C.' }],
          },
        }),
      ],
      { traceId: 'trace-res-r', startTime: 400, endTime: 700 }
    );

    const bundle = extractTrajectoryFromTraces([interruptTrace, resumeTrace]);

    expect(TrajectoryBundleSchema.parse(bundle)).toMatchObject({
      rootTrajectoryId: 'trace-int-r',
      trajectories: {
        'trace-int-r': {
          status: 'success',
          finishReason: 'stop',
          steps: [
            {},
            {
              actions: [
                {
                  actionId: 'flt-1',
                  name: 'bookFlight',
                  status: 'interrupted',
                  interrupt: { seatNeeded: true },
                  output: { pnr: 'ABC123', seat: '14C' },
                  resume: { strategy: 'restart', payload: { seat: '14C' } },
                },
              ],
            },
            {
              traceId: 'trace-res-r',
              content: [{ text: 'Booked seat 14C.' }],
            },
          ],
        },
      },
    });
  });
});
