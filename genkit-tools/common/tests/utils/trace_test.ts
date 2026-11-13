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

import { describe, expect, it } from 'vitest';
import {
  TrajectoryBundleSchema,
  type TraceData,
  type TrajectoryBundle,
} from '../../src/types/index.js';
import {
  cleanActionName,
  extractSessionIdFromTrace,
  getSpanName,
  getSpanStatus,
  getSpanType,
  isAgentSpan,
  isModelSpan,
  isToolSpan,
  isTruthyAttr,
  parseSpanJson,
  shortToolName,
  stackTraceSpans,
} from '../../src/utils/index.js';
import {
  BASE_FLOW_SPAN_ID,
  MOCK_AGENT_TOOL_TRACE,
  MOCK_HITL_INTERRUPT_TRACE,
  MOCK_SPAN_AGENT,
  MOCK_SPAN_DEV_WRAPPER,
  MOCK_SPAN_MODEL_REPLY,
  MOCK_SPAN_MODEL_TOOL_REQ,
  MOCK_SPAN_TOOL,
  MOCK_SPAN_TURN_1,
  MOCK_SUBAGENT_TRACE,
  MockTrace,
  makeModelSpan,
  makeSpan,
  makeToolSpan,
  makeTrace,
  makeTurnSpan,
} from './trace.js';

const TEST_TRACE: TraceData = {
  traceId: '7c273c22b219d077c6731a10d46b7d40',
  startTime: 1714059149480,
  endTime: 1714059149485.578,
  displayName: 'multiSteps',
  spans: {},
};

describe('trace utils', () => {
  describe('stackTraceSpans', () => {
    it('returns root span from flat trace spans', () => {
      const trace = new MockTrace('My input', 'My output')
        .addSpan({
          stepName: 'retrieverStep',
          spanType: 'action',
        })
        .addSpan({
          stepName: 'llmStep',
          spanType: 'action',
        })
        .getTrace();

      const span = stackTraceSpans(trace);

      expect(span?.spanId).toEqual(BASE_FLOW_SPAN_ID);
    });

    it('nests child spans under parent spans', () => {
      const root = stackTraceSpans(MOCK_AGENT_TOOL_TRACE);

      expect(root?.spanId).toBe(MOCK_SPAN_AGENT.spanId);
      expect(root?.spans?.[0]?.spanId).toBe(MOCK_SPAN_TURN_1.spanId);
      expect(root?.spans?.[0]?.spans?.map((s) => s.spanId)).toEqual([
        MOCK_SPAN_MODEL_TOOL_REQ.spanId,
        MOCK_SPAN_TOOL.spanId,
        MOCK_SPAN_MODEL_REPLY.spanId,
      ]);
    });

    it('returns undefined if no spans', () => {
      const span = stackTraceSpans(TEST_TRACE);

      expect(span).toBeUndefined();
    });

    it('falls back to orphan span as root when parentSpanId is missing from trace', () => {
      const trace = makeTrace(
        [
          makeSpan({
            spanId: 'orphan',
            parentSpanId: 'missing-parent',
            startTime: 100,
          }),
          makeSpan({
            spanId: 'child',
            parentSpanId: 'orphan',
            startTime: 110,
          }),
        ],
        { traceId: 'trace-orphan' }
      );

      const root = stackTraceSpans(trace);

      expect(root?.spanId).toBe('orphan');
      expect(root?.spans?.map((s) => s.spanId)).toEqual(['child']);
    });

    it('unwraps internal root wrapper spans', () => {
      const wrappedTrace: TraceData = {
        ...MOCK_AGENT_TOOL_TRACE,
        spans: {
          ...MOCK_AGENT_TOOL_TRACE.spans,
          'dev-wrapper': MOCK_SPAN_DEV_WRAPPER,
          'root-agent': {
            ...MOCK_SPAN_AGENT,
            parentSpanId: MOCK_SPAN_DEV_WRAPPER.spanId,
          },
        },
      };

      const root = stackTraceSpans(wrappedTrace);

      expect(root?.spanId).toBe(MOCK_SPAN_AGENT.spanId);
    });

    it('sorts sibling spans chronologically and breaks timestamp ties by spanId', () => {
      const trace = makeTrace([
        MOCK_SPAN_AGENT,
        makeSpan({
          spanId: 'child-b',
          parentSpanId: 'root-agent',
          startTime: 110,
        }),
        makeSpan({
          spanId: 'child-a',
          parentSpanId: 'root-agent',
          startTime: 110,
        }),
      ]);

      const root = stackTraceSpans(trace);

      expect(root?.spans?.map((s) => s.spanId)).toEqual(['child-a', 'child-b']);
    });

    it('prunes excluded spans and their descendant subtrees', () => {
      const traceWithExcludedBranch: TraceData = {
        ...MOCK_SUBAGENT_TRACE,
        spans: {
          ...MOCK_SUBAGENT_TRACE.spans,
          'dev-wrapper': MOCK_SPAN_DEV_WRAPPER,
          'root-agent': {
            ...MOCK_SUBAGENT_TRACE.spans['root-agent'],
            parentSpanId: MOCK_SPAN_DEV_WRAPPER.spanId,
          },
          guardrail: makeTurnSpan({
            spanId: 'guardrail',
            parentSpanId: 'root-agent',
            displayName: 'safetyClassifier',
            startTime: 105,
            endTime: 120,
            attributes: { 'genkit:metadata:trajectory:exclude': 'true' },
          }),
          guardrailModel: makeModelSpan({
            spanId: 'guardrail-model',
            parentSpanId: 'guardrail',
            startTime: 106,
            endTime: 119,
          }),
        },
      };

      const root = stackTraceSpans(traceWithExcludedBranch, {
        exclude: (s) =>
          isTruthyAttr(s.attributes?.['genkit:metadata:trajectory:exclude']),
      });

      expect(root?.spanId).toBe('root-agent');
      expect(root?.spans?.map((s) => s.spanId)).toEqual([
        'model-1',
        'delegate-tool',
        'model-2',
      ]);
    });

    it('breaks parent-child cycles in malformed span trees without looping infinitely', () => {
      const trace = makeTrace(
        [
          makeSpan({ spanId: 'root', startTime: 100 }),
          makeSpan({
            spanId: 'valid-child',
            parentSpanId: 'root',
            startTime: 105,
          }),
          makeSpan({
            spanId: 'self-loop',
            parentSpanId: 'self-loop',
            startTime: 110,
          }),
          makeSpan({
            spanId: 'cycle-a',
            parentSpanId: 'cycle-b',
            startTime: 115,
          }),
          makeSpan({
            spanId: 'cycle-b',
            parentSpanId: 'cycle-a',
            startTime: 120,
          }),
        ],
        { traceId: 'trace-cycle' }
      );

      const root = stackTraceSpans(trace);

      expect(root?.spanId).toBe('root');
      expect(root?.spans?.map((s) => s.spanId)).toEqual(['valid-child']);

      const pureCycleTrace = makeTrace(
        [
          makeSpan({
            spanId: 'cycle-a',
            parentSpanId: 'cycle-b',
            startTime: 115,
          }),
          makeSpan({
            spanId: 'cycle-b',
            parentSpanId: 'cycle-a',
            startTime: 120,
          }),
        ],
        { traceId: 'trace-pure-cycle' }
      );

      const cycleRoot = stackTraceSpans(pureCycleTrace);
      expect(cycleRoot?.spanId).toBe('cycle-a');
      expect(cycleRoot?.spans?.map((s) => s.spanId)).toEqual(['cycle-b']);
      expect(cycleRoot?.spans?.[0].spans).toEqual([]);
    });
  });

  describe('getSpanType', () => {
    it('prefers genkit:metadata:subtype over genkit:type', () => {
      expect(getSpanType(MOCK_SPAN_TOOL)).toBe('tool');
    });

    it('uses genkit:type when subtype is absent', () => {
      expect(getSpanType(MOCK_SPAN_TURN_1)).toBe('flowStep');
    });

    it('falls back to lowercased string spanKind when genkit attributes are absent', () => {
      expect(getSpanType(makeSpan({ spanKind: 'CLIENT' }))).toBe('client');
    });

    it('maps numeric OpenTelemetry spanKind index to kind name', () => {
      expect(getSpanType(makeSpan({ spanKind: 1 as unknown as string }))).toBe(
        'server'
      );
    });

    it('falls back to "span" for unknown numeric spanKind', () => {
      expect(getSpanType(makeSpan({ spanKind: 99 as unknown as string }))).toBe(
        'span'
      );
    });
  });

  describe('getSpanStatus', () => {
    it('returns checkmark for status code 0 (UNSET) or 1 (OK)', () => {
      expect(getSpanStatus(makeSpan({ status: { code: 0 } }))).toBe('✔');
      expect(getSpanStatus(makeSpan({ status: { code: 1 } }))).toBe('✔');
    });

    it('returns cross for status code 2 (ERROR)', () => {
      expect(getSpanStatus(MOCK_HITL_INTERRUPT_TRACE.spans['tool-1'])).toBe(
        '✖'
      );
    });

    it('prefers genkit:state attribute over status code', () => {
      expect(
        getSpanStatus(
          makeSpan({
            attributes: { 'genkit:state': 'success' },
            status: { code: 2 },
          })
        )
      ).toBe('✔');
      expect(
        getSpanStatus(
          makeSpan({
            attributes: { 'genkit:state': 'error' },
            status: { code: 0 },
          })
        )
      ).toBe('✖');
    });

    it('returns empty string when status code is unknown', () => {
      expect(getSpanStatus(makeSpan({ status: { code: 99 } }))).toBe('');
    });
  });

  describe('span classifiers and naming helpers', () => {
    it('classifies agent, model, and tool spans accurately', () => {
      expect(isAgentSpan(MOCK_SPAN_AGENT)).toBe(true);
      expect(isAgentSpan(MOCK_SPAN_MODEL_TOOL_REQ)).toBe(false);

      expect(isModelSpan(MOCK_SPAN_MODEL_TOOL_REQ)).toBe(true);
      expect(isModelSpan(MOCK_SPAN_TOOL)).toBe(false);

      expect(isToolSpan(MOCK_SPAN_TOOL)).toBe(true);
      expect(
        isToolSpan(makeToolSpan({ name: '/tool.v2/mcp-server/read_file' }))
      ).toBe(true);
      expect(isToolSpan(MOCK_SPAN_TURN_1)).toBe(false);
    });

    it('resolves getSpanName from displayName or genkit:name fallback', () => {
      expect(getSpanName(MOCK_SPAN_AGENT)).toBe('/agent/weatherAgent');
      expect(
        getSpanName({
          ...MOCK_SPAN_AGENT,
          displayName: '',
        })
      ).toBe('weatherAgent');
    });

    it('strips leading action prefixes in cleanActionName while preserving provider namespaces', () => {
      expect(cleanActionName('/model/googleai/gemini-2.5-pro')).toBe(
        'googleai/gemini-2.5-pro'
      );
      expect(cleanActionName('/agent/weatherAgent')).toBe('weatherAgent');
      expect(cleanActionName(undefined)).toBe('');
    });

    it('extracts shortToolName across action prefixes and slash-delimited namespaces', () => {
      expect(shortToolName('/tool.v2/mcp-server/read_file')).toBe('read_file');
      expect(shortToolName('getWeather')).toBe('getWeather');
      expect(shortToolName(undefined)).toBe('');
    });

    it('evaluates boolean and string truthy attributes in isTruthyAttr', () => {
      expect(isTruthyAttr(true)).toBe(true);
      expect(isTruthyAttr('TRUE')).toBe(true);
      expect(isTruthyAttr('false')).toBe(false);
      expect(isTruthyAttr(undefined)).toBe(false);
    });
  });

  describe('parseSpanJson', () => {
    it('parses serialized span JSON attributes and handles non-JSON/blank inputs safely', () => {
      expect(parseSpanJson(MOCK_SPAN_TOOL.attributes['genkit:output'])).toEqual(
        {
          tempF: 62,
          condition: 'Rainy',
        }
      );
      expect(parseSpanJson({ already: 'parsed' })).toEqual({
        already: 'parsed',
      });
      expect(parseSpanJson('plain-text')).toBe('plain-text');
      expect(parseSpanJson('   ')).toBeUndefined();
      expect(parseSpanJson(null)).toBeUndefined();
    });
  });

  describe('extractSessionIdFromTrace', () => {
    it('prefers shallower outer agent sessionId over nested subagent sessionId', () => {
      expect(extractSessionIdFromTrace(MOCK_SUBAGENT_TRACE)).toBe('sess-outer');
    });

    it('supports OpenTelemetry session.id attribute fallback', () => {
      const trace = makeTrace([
        makeSpan({
          spanId: 'outer',
          attributes: { 'session.id': 'otel-sess-1' },
        }),
      ]);

      expect(extractSessionIdFromTrace(trace)).toBe('otel-sess-1');
    });

    it('returns undefined when trace has no session attributes', () => {
      expect(extractSessionIdFromTrace(TEST_TRACE)).toBeUndefined();
    });
  });

  describe('TrajectoryBundleSchema', () => {
    it('validates canonical root, subagent, and HITL interrupt bundle structures', () => {
      const bundle: TrajectoryBundle = {
        rootTrajectoryId: 'trace-1',
        sessionId: 'sess-123',
        trajectories: {
          'trace-1': {
            trajectoryId: 'trace-1',
            agentName: 'weatherAgent',
            systemInstructions: 'You are a helpful weather assistant.',
            tools: [
              { name: 'getWeather', description: 'Gets weather for a city' },
            ],
            steps: [
              {
                stepId: 'trace-1-step-1-turn-1',
                stepNumber: 1,
                traceId: 'trace-1',
                spanId: 'turn-1',
                turnIndex: 0,
                source: 'user',
                startTime: 110,
                endTime: 110,
                latencyMs: 0,
                content: [{ text: 'What is the weather in Seattle?' }],
                actions: [],
              },
              {
                stepId: 'trace-1-step-2-model-1',
                stepNumber: 2,
                traceId: 'trace-1',
                spanId: 'model-1',
                turnIndex: 0,
                source: 'agent',
                startTime: 110,
                endTime: 320,
                latencyMs: 210,
                model: 'googleai/gemini-2.5-pro',
                content: [{ reasoning: 'Need to call getWeather.' }],
                actions: [
                  {
                    actionId: 'call-1',
                    spanId: 'tool-1',
                    name: 'getWeather',
                    type: 'interrupt',
                    status: 'interrupted',
                    input: { city: 'Seattle' },
                    output: { tempF: 62 },
                    resume: { strategy: 'respond', payload: { tempF: 62 } },
                  },
                ],
                stopReason: 'interrupted',
                usage: { inputTokens: 20, outputTokens: 15, totalTokens: 35 },
              },
            ],
            usage: { inputTokens: 20, outputTokens: 15, totalTokens: 35 },
            startTime: 100,
            endTime: 500,
          },
          'sub-agent': {
            trajectoryId: 'sub-agent',
            parentTrajectoryId: 'trace-1',
            agentName: 'researcher',
            tools: [],
            steps: [],
            startTime: 215,
            endTime: 590,
          },
        },
      };

      expect(TrajectoryBundleSchema.parse(bundle)).toEqual(bundle);
      expect(
        TrajectoryBundleSchema.safeParse({
          ...bundle,
          trajectories: {
            'trace-1': {
              ...bundle.trajectories['trace-1'],
              steps: [
                {
                  ...bundle.trajectories['trace-1'].steps[0],
                  source: 'invalid-source',
                },
              ],
            },
          },
        }).success
      ).toBe(false);
    });
  });
});
