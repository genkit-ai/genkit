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
  MOCK_AGENT_TOOL_TRAJECTORY,
  MOCK_SPAN_AGENT,
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

    expect(TrajectoryBundleSchema.parse(bundle)).toEqual(
      MOCK_AGENT_TOOL_TRAJECTORY
    );
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
          finishReason: 'stop',
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

  it('skips _genkit_history messages from prompt step extraction', () => {
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
            { role: 'user', content: [{ text: 'New turn' }] },
          ],
          outputMessage: { role: 'model', content: [{ text: 'Done' }] },
          finishReason: 'stop',
        }),
      ],
      { traceId: 'trace-history' }
    );

    const steps =
      extractTrajectoryFromTrace(trace).trajectories['trace-history'].steps;
    expect(steps.filter((s) => s.source === 'user')).toMatchObject([
      { source: 'user', content: [{ text: 'New turn' }] },
    ]);
  });

  it('surfaces failed tool and model errors on steps and trajectory', () => {
    const trace = makeTrace(
      [
        MOCK_SPAN_AGENT,
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: 'root-agent',
          startTime: 110,
          endTime: 200,
          messages: [{ role: 'user', content: [{ text: 'Deploy build' }] }],
          outputMessage: {
            role: 'model',
            content: [
              { toolRequest: { ref: 'dep-1', name: 'deploy', input: {} } },
            ],
          },
          finishReason: 'stop',
        }),
        makeToolSpan({
          spanId: 'failed-tool',
          parentSpanId: 'root-agent',
          name: 'deploy',
          status: { code: 2, message: 'Permission denied' },
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
    expect(trajErr.steps[1].actions).toMatchObject([
      {
        actionId: 'dep-1',
        status: 'error',
        error: { message: 'Permission denied' },
      },
    ]);
    expect(trajErr.steps[2]).toMatchObject({
      finishReason: 'failed',
      error: { message: 'Quota exceeded' },
    });
  });

  it.each([
    {
      name: 'undefined trace',
      trace: undefined as unknown as TraceData,
      expectedRootId: 'root',
    },
    {
      name: 'empty spans',
      trace: { traceId: '', spans: {} },
      expectedRootId: 'root',
    },
    {
      name: 'all-pruned root span',
      trace: makeTrace(
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
      ),
      expectedRootId: 'trace-pruned',
      expectedSessionId: 'sess-pruned',
    },
  ])(
    'handles degenerate trace: $name',
    ({ trace, expectedRootId, expectedSessionId }) => {
      const bundle = TrajectoryBundleSchema.parse(
        extractTrajectoryFromTrace(trace)
      );
      expect(bundle.rootTrajectoryId).toBe(expectedRootId);
      if (expectedSessionId) expect(bundle.sessionId).toBe(expectedSessionId);
      expect(bundle.trajectories[expectedRootId].steps).toEqual([]);
    }
  );

  it('extracts user and system steps when no model spans exist', () => {
    const trace = makeTrace(
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
    );

    expect(
      extractTrajectoryFromTrace(trace).trajectories['trace-fallback'].steps
    ).toMatchObject([
      { source: 'user', content: [{ text: 'Hello before model' }] },
      { source: 'system', content: [{ text: 'System notice' }] },
    ]);
  });

  it('projects standalone model traces with undefined agentName and per-span stepIds', () => {
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
          outputMessage: { role: 'model', content: [{ text: 'Essay...' }] },
          finishReason: 'stop',
        }),
      ],
      { traceId: 'trace-standalone' }
    );

    const traj =
      extractTrajectoryFromTrace(trace).trajectories['trace-standalone'];
    expect(traj.agentName).toBeUndefined();
    expect(traj.systemInstructions).toBe('Be concise.');
    expect(traj.steps).toMatchObject([
      {
        stepId: 'm-root-1',
        source: 'user',
        content: [{ text: 'Write an essay' }],
      },
      { stepId: 'm-root-2', source: 'agent', finishReason: 'stop' },
    ]);
  });

  it('aggregates standard and custom.thoughtsTokens usage across steps', () => {
    const trace = makeTrace(
      [
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: undefined,
          startTime: 100,
          endTime: 200,
          messages: [{ role: 'user', content: [{ text: 'Step 1' }] }],
          candidates: [
            {
              message: { role: 'model', content: [{ text: 'Part 1' }] },
              finishReason: 'stop',
              usage: {
                inputTokens: 10,
                outputTokens: 50,
                custom: { thoughtsTokens: 7 },
              },
            },
          ],
        }),
        makeModelSpan({
          spanId: 'm2',
          parentSpanId: 'm1',
          startTime: 210,
          endTime: 250,
          messages: [],
          outputMessage: { role: 'model', content: [{ text: 'Part 2' }] },
          finishReason: 'stop',
          usage: { inputTokens: 5, custom: { thoughtsTokens: 3 } },
        }),
      ],
      { traceId: 'trace-usage' }
    );

    expect(
      extractTrajectoryFromTrace(trace).trajectories['trace-usage'].usage
    ).toEqual({
      inputTokens: 15,
      outputTokens: 50,
      custom: { thoughtsTokens: 10 },
    });
  });

  it.each([
    {
      name: 'length',
      spanOverrides: { finishReason: 'length' },
      expectedFinishReason: 'length',
    },
    {
      name: 'aborted',
      spanOverrides: { finishReason: 'aborted' },
      expectedFinishReason: 'aborted',
    },
    {
      name: 'interrupted',
      spanOverrides: { finishReason: 'interrupted' },
      expectedFinishReason: 'interrupted',
    },
    {
      name: 'structured outputError',
      spanOverrides: {
        outputError: {
          message: 'Quota exceeded',
          details: { stack: 'Error: Quota exceeded\n    at callModel' },
        },
      },
      expectedFinishReason: 'failed',
      expectedError: {
        message: 'Quota exceeded',
        details: { stack: 'Error: Quota exceeded\n    at callModel' },
      },
    },
  ])(
    'maps model step outcome for $name',
    ({ spanOverrides, expectedFinishReason, expectedError }) => {
      const trace = makeTrace(
        [
          makeModelSpan({
            spanId: 'm1',
            parentSpanId: undefined,
            messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
            outputMessage: { role: 'model', content: [{ text: 'Reply' }] },
            ...spanOverrides,
          }),
        ],
        { traceId: 'trace-outcome' }
      );

      const agentStep =
        extractTrajectoryFromTrace(trace).trajectories['trace-outcome']
          .steps[1];
      expect(agentStep).toMatchObject({
        finishReason: expectedFinishReason,
        ...(expectedError ? { error: expectedError } : {}),
      });
    }
  );

  it('partitions nested subagent spans into linked child trajectories in TrajectoryBundle', () => {
    const bundle = extractTrajectoryFromTrace(MOCK_SUBAGENT_TRACE);

    expect(TrajectoryBundleSchema.parse(bundle)).toMatchObject({
      sessionId: 'sess-outer',
      trajectories: {
        'trace-sub': {
          steps: [
            {},
            {
              actions: [
                {
                  actionId: 'sub-call-1',
                  spanId: 'delegate-tool',
                  name: 'delegate_to_researcher',
                  type: 'subagent',
                  status: 'success',
                  subtrajectoryId: 'sub-agent',
                },
              ],
            },
            {},
          ],
        },
        'sub-agent': {
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
        },
      },
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

  it('links nested subagent spans invoked directly without an enclosing tool span', () => {
    const trace = makeTrace(
      [
        makeAgentSpan({ spanId: 'root-agent', name: 'coordinator' }),
        makeModelSpan({
          spanId: 'm1',
          parentSpanId: 'root-agent',
          startTime: 120,
          endTime: 160,
          messages: [{ role: 'user', content: [{ text: 'Ask researcher' }] }],
          outputMessage: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  ref: 'direct-sub-1',
                  name: 'researcher',
                  input: { topic: 'direct' },
                },
              },
            ],
          },
        }),
        makeAgentSpan({
          spanId: 'direct-researcher',
          parentSpanId: 'root-agent',
          name: 'researcher',
          startTime: 165,
          endTime: 220,
        }),
      ],
      { traceId: 'trace-direct-subagent' }
    );

    const bundle = extractTrajectoryFromTrace(trace);

    expect(TrajectoryBundleSchema.parse(bundle)).toMatchObject({
      trajectories: {
        'trace-direct-subagent': {
          steps: [
            {},
            {
              actions: [
                {
                  actionId: 'direct-sub-1',
                  spanId: 'direct-researcher',
                  name: 'researcher',
                  type: 'subagent',
                  subtrajectoryId: 'direct-researcher',
                },
              ],
            },
          ],
        },
        'direct-researcher': {
          parentTrajectoryId: 'trace-direct-subagent',
          agentName: 'researcher',
        },
      },
    });
  });
});
