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
import {
  TrajectoryBundleSchema,
  type TraceData,
} from '../../../src/types/index.js';
import { extractTrajectoryFromTrace } from '../../../src/utils/index.js';
import {
  MOCK_AGENT_TOOL_TRACE,
  MOCK_HITL_INTERRUPT_TRACE,
  MOCK_SPAN_AGENT,
  MOCK_SPAN_MODEL_TOOL_REQ,
  MOCK_SPAN_TOOL,
  MOCK_SUBAGENT_TRACE,
  makeAgentSpan,
  makeModelSpan,
  makeSpan,
  makeToolSpan,
  makeTrace,
  makeTurnSpan,
} from './mock_data.js';

describe('extractTrajectoryFromTrace', () => {
  it('projects single-turn agent trace into schema-valid steps, actions, and usage', () => {
    const bundle = extractTrajectoryFromTrace(MOCK_AGENT_TOOL_TRACE);

    expect(TrajectoryBundleSchema.parse(bundle)).toEqual(bundle);
    expect(bundle.rootTrajectoryId).toBe('trace-1');
    expect(bundle.sessionId).toBe('sess-123');

    const root = bundle.trajectories['trace-1'];
    expect(root.agentName).toBe('weatherAgent');
    expect(root.status).toBe('success');
    expect(root.finishReason).toBe('stop');
    expect(root.systemInstructions).toBe(
      'You are a helpful weather assistant.'
    );
    expect(root.tools).toEqual([
      expect.objectContaining({ name: 'getWeather' }),
    ]);
    expect(root.usage).toEqual({
      inputTokens: 65,
      outputTokens: 27,
      totalTokens: 92,
    });
    expect(root.steps).toMatchObject([
      {
        stepId: 'turn-1-1',
        stepNumber: 1,
        source: 'user',
        content: [{ text: 'What is the weather in Seattle?' }],
      },
      {
        stepId: 'model-1-1',
        stepNumber: 2,
        source: 'agent',
        model: 'googleai/gemini-2.5-pro',
        content: [{ reasoning: 'Need to call getWeather for Seattle.' }],
        finishReason: 'stop',
        endTime: 320,
        actions: [
          {
            actionId: 'call-1',
            spanId: 'tool-1',
            name: 'getWeather',
            type: 'tool',
            status: 'success',
            input: { city: 'Seattle' },
            output: { tempF: 62, condition: 'Rainy' },
          },
        ],
      },
      {
        stepId: 'model-2-1',
        stepNumber: 3,
        source: 'agent',
        finishReason: 'stop',
        content: [{ text: 'It is 62°F and rainy in Seattle.' }],
      },
    ]);
  });

  it('prunes trajectory:exclude and prompt subtrees', () => {
    const trace: TraceData = {
      ...MOCK_AGENT_TOOL_TRACE,
      spans: {
        ...MOCK_AGENT_TOOL_TRACE.spans,
        guardrail: makeSpan({
          spanId: 'guardrail',
          parentSpanId: 'root-agent',
          startTime: 102,
          attributes: { 'genkit:metadata:trajectory:exclude': 'true' },
        }),
        guardrailModel: makeModelSpan({
          spanId: 'guardrail-model',
          parentSpanId: 'guardrail',
          startTime: 103,
          outputMessage: { role: 'model', content: [{ text: 'SAFE' }] },
        }),
        promptRender: makeSpan({
          spanId: 'prompt-render',
          parentSpanId: 'root-agent',
          startTime: 104,
          attributes: { 'genkit:metadata:subtype': 'promptTemplate' },
        }),
      },
    };

    const root = extractTrajectoryFromTrace(trace).trajectories['trace-1'];
    expect(root.agentName).toBe('weatherAgent');
    expect(root.steps).toHaveLength(3);
  });

  it('pairs parallel tool calls chronologically and leaves missing spans in-progress', () => {
    const trace: TraceData = {
      ...MOCK_AGENT_TOOL_TRACE,
      spans: {
        'root-agent': MOCK_SPAN_AGENT,
        'turn-1': MOCK_AGENT_TOOL_TRACE.spans['turn-1'],
        'model-1': makeModelSpan({
          spanId: 'model-1',
          parentSpanId: 'turn-1',
          startTime: 110,
          endTime: 200,
          messages: [
            { role: 'user', content: [{ text: 'Compare Seattle and NYC' }] },
          ],
          outputMessage: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  ref: 'ref-nyc',
                  name: 'getWeather',
                  input: { city: 'NYC' },
                },
              },
              {
                toolRequest: {
                  ref: 'ref-sea',
                  name: 'getWeather',
                  input: { city: 'Seattle' },
                },
              },
              {
                toolRequest: {
                  ref: 'ref-missing',
                  name: 'getWeather',
                  input: { city: 'Boston' },
                },
              },
            ],
          },
          finishReason: 'tool_calls',
        }),
        'tool-nyc': makeToolSpan({
          spanId: 'tool-nyc',
          parentSpanId: 'turn-1',
          name: 'getWeather',
          startTime: 205,
          endTime: 280,
          output: { tempF: 75 },
        }),
        'tool-sea': makeToolSpan({
          spanId: 'tool-sea',
          parentSpanId: 'turn-1',
          name: 'getWeather',
          startTime: 210,
          endTime: 260,
          output: { tempF: 62 },
        }),
      },
    };

    const actions =
      extractTrajectoryFromTrace(trace).trajectories['trace-1'].steps[1]
        .actions;
    expect(
      actions.map((a) => [a.actionId, a.spanId, a.status, a.output])
    ).toEqual([
      ['ref-nyc', 'tool-nyc', 'success', { tempF: 75 }],
      ['ref-sea', 'tool-sea', 'success', { tempF: 62 }],
      ['ref-missing', undefined, 'in-progress', undefined],
    ]);
  });

  it('skips _genkit_history messages and surfaces error and pending states', () => {
    const trace = makeTrace(
      [
        MOCK_SPAN_AGENT,
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: 'root-agent',
          startTime: 110,
          endTime: 200,
          messages: [
            {
              role: 'user',
              content: [{ text: 'Old turn' }],
              metadata: { _genkit_history: true },
            },
            { role: 'user', content: [{ text: 'Deploy build' }] },
          ],
          outputMessage: {
            role: 'model',
            content: [
              { toolRequest: { ref: 'dep-1', name: 'deploy', input: {} } },
              { toolRequest: { ref: 'chk-1', name: 'healthCheck', input: {} } },
            ],
          },
          finishReason: 'tool_calls',
        }),
        {
          ...MOCK_SPAN_TOOL,
          spanId: 'failed-tool',
          parentSpanId: 'root-agent',
          displayName: '/tool/deploy',
          status: { code: 2, message: 'Permission denied' },
          attributes: {
            ...MOCK_SPAN_TOOL.attributes,
            'genkit:metadata:tool:callId': 'dep-1',
          },
        },
        makeToolSpan({
          spanId: 'pending-tool',
          parentSpanId: 'root-agent',
          name: 'healthCheck',
          callId: 'chk-1',
          endTime: 0,
        }),
        makeModelSpan({
          spanId: 'failed-model',
          parentSpanId: 'root-agent',
          startTime: 330,
          endTime: 350,
          messages: [],
          status: { code: 2, message: 'Quota exceeded' },
        }),
      ],
      { traceId: 'trace-err' }
    );

    const trajErr = extractTrajectoryFromTrace(trace).trajectories['trace-err'];
    expect(trajErr.status).toBe('error');
    expect(trajErr.finishReason).toBe('failed');
    const steps = trajErr.steps;
    expect(steps[0]).toMatchObject({
      source: 'user',
      content: [{ text: 'Deploy build' }],
    });
    expect(steps[1].actions).toMatchObject([
      {
        actionId: 'dep-1',
        status: 'error',
        error: { message: 'Permission denied' },
      },
      { actionId: 'chk-1', status: 'in-progress' },
    ]);
    expect(steps[2]).toMatchObject({
      finishReason: 'failed',
      error: { message: 'Quota exceeded' },
    });
  });

  it('handles nullish, empty, pruned, and model-less fallback traces', () => {
    const nullBundle = extractTrajectoryFromTrace(
      undefined as unknown as TraceData
    );
    expect(TrajectoryBundleSchema.parse(nullBundle).rootTrajectoryId).toBe(
      'root'
    );

    const emptyBundle = extractTrajectoryFromTrace({ traceId: '', spans: {} });
    expect(TrajectoryBundleSchema.parse(emptyBundle).rootTrajectoryId).toBe(
      'root'
    );

    const pruned = extractTrajectoryFromTrace(
      makeTrace(
        [
          makeSpan({
            spanId: 'excluded-root',
            attributes: {
              'genkit:metadata:trajectory:exclude': 'true',
              'session.id': 'sess-pruned',
            },
          }),
        ],
        { traceId: 'trace-pruned' }
      )
    );
    expect(pruned).toMatchObject({
      sessionId: 'sess-pruned',
      trajectories: { 'trace-pruned': { steps: [] } },
    });

    const fallback = extractTrajectoryFromTrace(
      makeTrace(
        [
          makeAgentSpan({
            spanId: 'root-agent',
            input: {
              message: {
                role: 'user',
                content: [{ text: 'Hello before model' }],
              },
            },
          }),
          makeTurnSpan({
            turn: 1,
            input: {
              messages: [
                { role: 'system', content: [{ text: 'System notice' }] },
              ],
            },
          }),
        ],
        { traceId: 'trace-fallback' }
      )
    );
    expect(fallback.trajectories['trace-fallback'].steps).toMatchObject([
      { source: 'user', content: [{ text: 'Hello before model' }] },
      { source: 'system', content: [{ text: 'System notice' }] },
    ]);
  });

  it('projects standalone model traces with per-span stepIds, max_tokens, cancelled, interrupted, and structured errors', () => {
    const trace = makeTrace(
      [
        makeModelSpan({
          spanId: 'm-root',
          parentSpanId: undefined,
          model: 'googleai/gemini-2.5-flash',
          startTime: 100,
          endTime: 200,
          messages: [
            { role: 'system', content: [{ text: 'Be concise.' }] },
            { role: 'user', content: 'Write an essay' },
          ],
          candidates: [
            {
              message: { role: 'model', content: [{ text: 'Truncated...' }] },
              finishReason: 'length',
              usage: {
                inputTokens: 10,
                outputTokens: 50,
                custom: { thoughtsTokens: 7 },
              },
            },
          ],
        }),
        makeModelSpan({
          spanId: 'm-aborted',
          parentSpanId: 'm-root',
          startTime: 210,
          endTime: 250,
          messages: [],
          outputMessage: { role: 'model', content: [{ text: 'Cancelled' }] },
          finishReason: 'aborted',
          usage: { inputTokens: 5, custom: { thoughtsTokens: 3 } },
        }),
        makeModelSpan({
          spanId: 'm-interrupted',
          parentSpanId: 'm-root',
          startTime: 252,
          endTime: 258,
          messages: [],
          outputMessage: { role: 'model', content: [{ text: 'Paused' }] },
          finishReason: 'interrupted',
        }),
        makeModelSpan({
          spanId: 'm-err',
          parentSpanId: 'm-root',
          startTime: 260,
          endTime: 290,
          messages: [],
          outputError: {
            message: 'Quota exceeded',
            details: { stack: 'Error: Quota exceeded\n    at callModel' },
          },
        }),
      ],
      { traceId: 'trace-standalone' }
    );

    const traj =
      extractTrajectoryFromTrace(trace).trajectories['trace-standalone'];
    expect(traj.agentName).toBeUndefined();
    expect(traj.systemInstructions).toBe('Be concise.');
    expect(traj.usage).toEqual({
      inputTokens: 15,
      outputTokens: 50,
      custom: { thoughtsTokens: 10 },
    });
    expect(traj.steps).toMatchObject([
      {
        stepId: 'm-root-1',
        source: 'user',
        content: [{ text: 'Write an essay' }],
      },
      { stepId: 'm-root-2', finishReason: 'length' },
      { stepId: 'm-aborted-1', finishReason: 'aborted' },
      { stepId: 'm-interrupted-1', finishReason: 'interrupted' },
      {
        stepId: 'm-err-1',
        finishReason: 'failed',
        error: {
          message: 'Quota exceeded',
          details: { stack: 'Error: Quota exceeded\n    at callModel' },
        },
      },
    ]);
    expect(MOCK_SPAN_MODEL_TOOL_REQ.spanId).toBe('model-1');
  });

  it('partitions nested subagent spans into linked child trajectories in TrajectoryBundle', () => {
    const bundle = extractTrajectoryFromTrace(MOCK_SUBAGENT_TRACE);

    expect(TrajectoryBundleSchema.parse(bundle)).toEqual(bundle);
    expect(bundle.sessionId).toBe('sess-outer');
    expect(Object.keys(bundle.trajectories)).toEqual([
      'sub-agent',
      'trace-sub',
    ]);
    expect(bundle.trajectories['trace-sub'].steps[1].actions).toEqual([
      expect.objectContaining({
        actionId: 'sub-call-1',
        spanId: 'delegate-tool',
        name: 'delegate_to_researcher',
        type: 'subagent',
        status: 'success',
        subtrajectoryId: 'sub-agent',
      }),
    ]);
    expect(bundle.trajectories['sub-agent']).toMatchObject({
      parentTrajectoryId: 'trace-sub',
      agentName: 'researcher',
      steps: [
        {
          stepId: 'sub-model-1-1',
          content: [{ text: 'Topic: OATR' }],
        },
        {
          stepId: 'sub-model-1-2',
          content: [{ text: 'OATR defines TrajectoryBundle.' }],
        },
      ],
    });
  });

  it('matches plain agent name tool requests to delegate_to_ spans across intermediate wrappers', () => {
    const trace = makeTrace(
      [
        makeAgentSpan({ spanId: 'root-agent', name: 'coordinator' }),
        makeTurnSpan({ spanId: 'turn-1', turn: 1 }),
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: 'turn-1',
          startTime: 120,
          endTime: 160,
          messages: [{ role: 'user', content: [{ text: 'Delegate to mars' }] }],
          outputMessage: {
            role: 'model',
            content: [
              { toolRequest: { name: 'researcher', input: { topic: 'mars' } } },
            ],
          },
        }),
        makeToolSpan({
          spanId: 'tool-delegate',
          parentSpanId: 'turn-1',
          name: 'delegate_to_researcher',
          startTime: 165,
          endTime: 240,
          output: { summary: 'Mars facts' },
        }),
        makeSpan({
          spanId: 'inner-wrapper',
          parentSpanId: 'tool-delegate',
          displayName: 'invokeSubagent',
          startTime: 166,
          endTime: 235,
        }),
        makeAgentSpan({
          spanId: 'sub-researcher',
          parentSpanId: 'inner-wrapper',
          name: 'researcher',
          startTime: 168,
          endTime: 230,
        }),
      ],
      { traceId: 'trace-delegate-wrapper' }
    );

    const bundle = extractTrajectoryFromTrace(trace);
    expect(
      bundle.trajectories['trace-delegate-wrapper'].steps[1].actions[0]
    ).toMatchObject({
      spanId: 'tool-delegate',
      type: 'subagent',
      subtrajectoryId: 'sub-researcher',
      output: { summary: 'Mars facts' },
    });
  });

  describe('Human-in-the-Loop (HITL) interrupts', () => {
    it('projects unresumed ToolInterruptError actions as interrupted without resume', () => {
      const bundle = extractTrajectoryFromTrace(MOCK_HITL_INTERRUPT_TRACE);

      expect(TrajectoryBundleSchema.parse(bundle)).toEqual(bundle);
      const traj = bundle.trajectories['trace-hitl'];
      expect(traj.status).toBe('interrupted');
      expect(traj.finishReason).toBe('interrupted');
      const step = traj.steps[1];
      expect(step.finishReason).toBe('interrupted');
      expect(step.actions[0]).toMatchObject({
        actionId: 'tx-1',
        name: 'transferFunds',
        type: 'tool',
        status: 'interrupted',
        interrupt: { confirmRequired: true },
        input: { to: 'Alice', amount: 500 },
      });
      expect(step.actions[0].resume).toBeUndefined();
    });

    it('pairs in-trace respond and restart HITL resume decisions and restarted tool outputs', () => {
      const trace: TraceData = {
        ...MOCK_HITL_INTERRUPT_TRACE,
        endTime: 650,
        spans: {
          ...MOCK_HITL_INTERRUPT_TRACE.spans,
          'model-1': makeModelSpan({
            spanId: 'model-1',
            parentSpanId: 'turn-1',
            startTime: 110,
            endTime: 200,
            messages: [
              { role: 'user', content: [{ text: 'Pay $500 and book flight' }] },
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
          'flt-1-span': makeToolSpan({
            spanId: 'flt-1-span',
            parentSpanId: 'turn-1',
            name: 'bookFlight',
            callId: 'flt-1',
            startTime: 210,
            endTime: 240,
            interrupt: { seatNeeded: true },
            status: { code: 2, message: 'ToolInterruptError' },
          }),
          'turn-2': makeTurnSpan({
            spanId: 'turn-2',
            turn: 2,
            startTime: 350,
            endTime: 640,
            input: {
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
            },
          }),
          'flt-restarted': makeToolSpan({
            spanId: 'flt-restarted',
            parentSpanId: 'turn-2',
            name: 'bookFlight',
            startTime: 360,
            endTime: 410,
            resumed: { seat: '12A' },
            output: { pnr: 'XYZ987', seat: '12A' },
          }),
          'model-2': makeModelSpan({
            spanId: 'model-2',
            parentSpanId: 'turn-2',
            startTime: 420,
            endTime: 600,
            messages: [
              { role: 'user', content: [{ text: 'Pay $500 and book flight' }] },
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
            outputMessage: {
              role: 'model',
              content: [{ text: 'Paid $500 and booked XYZ987.' }],
            },
          }),
        },
      };

      const steps =
        extractTrajectoryFromTrace(trace).trajectories['trace-hitl'].steps;
      expect(steps[1].finishReason).toBe('interrupted');
      expect(steps[1].actions).toMatchObject([
        {
          actionId: 'tx-1',
          type: 'tool',
          status: 'interrupted',
          interrupt: { confirmRequired: true },
          output: { approved: true, receipt: 'r-1' },
          resume: {
            strategy: 'respond',
            payload: { approved: true, receipt: 'r-1' },
          },
        },
        {
          actionId: 'flt-1',
          type: 'tool',
          status: 'interrupted',
          interrupt: { seatNeeded: true },
          output: { pnr: 'XYZ987', seat: '12A' },
          resume: { strategy: 'restart', payload: { seat: '12A' } },
        },
      ]);
      expect(steps[2]).toMatchObject({
        turnIndex: 1,
        finishReason: 'stop',
        content: [{ text: 'Paid $500 and booked XYZ987.' }],
      });
    });

    it('isolates parallel conflicting refs, detects ancestor turn interrupt metadata, and ignores malformed resume fields', () => {
      const trace = makeTrace(
        [
          makeAgentSpan({ spanId: 'root-agent', name: 'hitlAgent' }),
          makeTurnSpan({
            spanId: 'turn-1',
            turn: 1,
            output: {
              finishReason: 'interrupted',
              message: {
                role: 'model',
                content: [
                  {
                    toolRequest: {
                      ref: 'tx-1',
                      name: 'transfer',
                      input: { amount: 10 },
                    },
                    metadata: { interrupt: { prompt: 'Approve?' } },
                  },
                ],
              },
            },
          }),
          makeModelSpan({
            spanId: 'm1',
            parentSpanId: 'turn-1',
            messages: [{ role: 'user', content: [{ text: 'Transfer twice' }] }],
            outputMessage: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    ref: 'tx-1',
                    name: 'transfer',
                    input: { amount: 10 },
                  },
                },
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
          makeTurnSpan({
            spanId: 'turn-2',
            turn: 2,
            startTime: 300,
            endTime: 490,
            input: {
              resume: {
                respond: 'invalid-non-array',
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
            },
          }),
          makeToolSpan({
            spanId: 'tool-tx-2',
            parentSpanId: 'turn-2',
            name: '/tool.v2/mcp/transfer',
            callId: 'tx-2',
            startTime: 310,
            endTime: 350,
            resumed: true,
            output: { output: { ok: 20 }, content: [] },
          }),
          makeModelSpan({
            spanId: 'm2',
            parentSpanId: 'turn-2',
            startTime: 360,
            endTime: 450,
            messages: [{ role: 'user', content: [{ text: 'Transfer twice' }] }],
            outputMessage: {
              role: 'model',
              content: [{ text: 'Transferred 20.' }],
            },
          }),
        ],
        { traceId: 'trace-ancestor-conflict' }
      );

      const step =
        extractTrajectoryFromTrace(trace).trajectories[
          'trace-ancestor-conflict'
        ].steps[1];
      expect(step.finishReason).toBe('interrupted');
      expect(step.actions[0]).toEqual({
        actionId: 'tx-1',
        name: 'transfer',
        type: 'tool',
        status: 'interrupted',
        interrupt: { prompt: 'Approve?' },
        input: { amount: 10 },
      });
      expect(step.actions[1]).toMatchObject({
        actionId: 'tx-2',
        type: 'tool',
        status: 'interrupted',
        interrupt: true,
        output: { ok: 20 },
        resume: { strategy: 'restart', payload: { amount: 20 } },
      });
    });
  });
});
