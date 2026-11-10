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

import { describe, expect, it } from '@jest/globals';
import type { TraceData } from '../../src/types';
import {
  cleanActionName,
  extractSessionIdFromTrace,
  getSpanStatus,
  getSpanType,
  isAgentSpan,
  isModelSpan,
  isToolSpan,
  parseSpanJson,
  shortToolName,
  stackTraceSpans,
} from '../../src/utils';
import { BASE_FLOW_SPAN_ID, MockTrace, makeSpan } from './trace';

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

      expect(span?.spans).toHaveLength(1);
    });

    it('returns undefined if no spans', () => {
      const span = stackTraceSpans(TEST_TRACE);

      expect(span).toBeUndefined();
    });

    it('falls back to orphan span as root when parentSpanId is missing from trace', () => {
      const trace: TraceData = {
        traceId: 'trace-orphan',
        spans: {
          orphan: makeSpan({
            spanId: 'orphan',
            parentSpanId: 'missing-parent',
            startTime: 100,
          }),
          child: makeSpan({
            spanId: 'child',
            parentSpanId: 'orphan',
            startTime: 110,
          }),
        },
      };

      const root = stackTraceSpans(trace);

      expect(root?.spanId).toBe('orphan');
    });

    it('attaches child spans under orphan root span', () => {
      const trace: TraceData = {
        traceId: 'trace-orphan',
        spans: {
          orphan: makeSpan({
            spanId: 'orphan',
            parentSpanId: 'missing-parent',
            startTime: 100,
          }),
          child: makeSpan({
            spanId: 'child',
            parentSpanId: 'orphan',
            startTime: 110,
          }),
        },
      };

      const root = stackTraceSpans(trace);

      expect(root?.spans?.map((s) => s.spanId)).toEqual(['child']);
    });

    it('unwraps internal root wrapper spans', () => {
      const trace: TraceData = {
        traceId: 'trace-unwrap',
        spans: {
          wrapper: makeSpan({
            spanId: 'wrapper',
            startTime: 100,
            attributes: { 'genkit:metadata:genkit-dev-internal': true },
          }),
          agent: makeSpan({
            spanId: 'agent',
            parentSpanId: 'wrapper',
            displayName: '/agent/main',
            startTime: 105,
          }),
        },
      };

      const root = stackTraceSpans(trace);

      expect(root?.spanId).toBe('agent');
    });

    it('sorts sibling spans chronologically and breaks timestamp ties by spanId', () => {
      const trace: TraceData = {
        traceId: 'trace-tiebreak',
        spans: {
          agent: makeSpan({
            spanId: 'agent',
            startTime: 100,
          }),
          childB: makeSpan({
            spanId: 'child-b',
            parentSpanId: 'agent',
            startTime: 110,
          }),
          childA: makeSpan({
            spanId: 'child-a',
            parentSpanId: 'agent',
            startTime: 110,
          }),
        },
      };

      const root = stackTraceSpans(trace);

      expect(root?.spans?.map((s) => s.spanId)).toEqual(['child-a', 'child-b']);
    });

    it('prunes excluded spans and their descendant subtrees', () => {
      const trace: TraceData = {
        traceId: 'trace-prune',
        spans: {
          wrapper: makeSpan({
            spanId: 'wrapper',
            startTime: 100,
            attributes: { 'genkit:metadata:genkit-dev-internal': true },
          }),
          agent: makeSpan({
            spanId: 'agent',
            parentSpanId: 'wrapper',
            displayName: '/agent/main',
            startTime: 105,
          }),
          kept: makeSpan({
            spanId: 'kept',
            parentSpanId: 'agent',
            startTime: 110,
          }),
          excludedBranch: makeSpan({
            spanId: 'excluded-branch',
            parentSpanId: 'agent',
            startTime: 115,
            attributes: { skipMe: true },
          }),
          excludedLeaf: makeSpan({
            spanId: 'excluded-leaf',
            parentSpanId: 'excluded-branch',
            startTime: 116,
          }),
        },
      };

      const root = stackTraceSpans(trace, {
        exclude: (s) => Boolean(s.attributes?.['skipMe']),
      });

      expect(root?.spanId).toBe('agent');
      expect(root?.spans?.map((s) => s.spanId)).toEqual(['kept']);
    });
  });

  describe('getSpanType', () => {
    it('prefers genkit:metadata:subtype over genkit:type', () => {
      expect(
        getSpanType(
          makeSpan({
            attributes: {
              'genkit:type': 'action',
              'genkit:metadata:subtype': 'tool',
            },
          })
        )
      ).toBe('tool');
    });

    it('uses genkit:type when subtype is absent', () => {
      expect(
        getSpanType(
          makeSpan({
            attributes: { 'genkit:type': 'flowStep' },
          })
        )
      ).toBe('flowStep');
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
    it('returns checkmark for status code 0 (UNSET)', () => {
      expect(getSpanStatus(makeSpan({ status: { code: 0 } }))).toBe('✔');
    });

    it('returns checkmark for status code 1 (OK)', () => {
      expect(getSpanStatus(makeSpan({ status: { code: 1 } }))).toBe('✔');
    });

    it('returns cross for status code 2 (ERROR)', () => {
      expect(getSpanStatus(makeSpan({ status: { code: 2 } }))).toBe('✖');
    });

    it('prefers genkit:state success over error status code', () => {
      expect(
        getSpanStatus(
          makeSpan({
            attributes: { 'genkit:state': 'success' },
            status: { code: 2 },
          })
        )
      ).toBe('✔');
    });

    it('prefers genkit:state error over OK status code', () => {
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

  describe('isAgentSpan', () => {
    it('returns true when displayName starts with /agent/', () => {
      expect(isAgentSpan(makeSpan({ displayName: '/agent/planner' }))).toBe(
        true
      );
    });
  });

  describe('isModelSpan', () => {
    it('returns true when genkit:metadata:subtype is model', () => {
      expect(
        isModelSpan(
          makeSpan({
            attributes: { 'genkit:metadata:subtype': 'model' },
          })
        )
      ).toBe(true);
    });
  });

  describe('isToolSpan', () => {
    it('returns true when displayName starts with /tool.v2/', () => {
      expect(isToolSpan(makeSpan({ displayName: '/tool.v2/mcp/search' }))).toBe(
        true
      );
    });
  });

  describe('cleanActionName', () => {
    it('strips leading /model/ prefix while preserving provider namespace', () => {
      expect(cleanActionName('/model/googleai/gemini-2.5-pro')).toBe(
        'googleai/gemini-2.5-pro'
      );
    });

    it('strips leading /agent/ prefix', () => {
      expect(cleanActionName('/agent/weatherAgent')).toBe('weatherAgent');
    });

    it('returns empty string when undefined', () => {
      expect(cleanActionName(undefined)).toBe('');
    });
  });

  describe('shortToolName', () => {
    it('strips leading action prefix and slash-delimited namespace', () => {
      expect(shortToolName('/tool.v2/mcp-server/read_file')).toBe('read_file');
    });

    it('returns plain tool name unchanged', () => {
      expect(shortToolName('getWeather')).toBe('getWeather');
    });

    it('returns empty string when undefined', () => {
      expect(shortToolName(undefined)).toBe('');
    });
  });

  describe('parseSpanJson', () => {
    it('parses valid JSON strings', () => {
      expect(parseSpanJson('{"a":1}')).toEqual({ a: 1 });
    });

    it('returns already-parsed objects as-is', () => {
      expect(parseSpanJson({ already: 'parsed' })).toEqual({
        already: 'parsed',
      });
    });

    it('returns non-JSON strings as-is', () => {
      expect(parseSpanJson('plain-text')).toBe('plain-text');
    });

    it('returns undefined for whitespace-only strings', () => {
      expect(parseSpanJson('   ')).toBeUndefined();
    });

    it('returns undefined for null', () => {
      expect(parseSpanJson(null)).toBeUndefined();
    });
  });

  describe('extractSessionIdFromTrace', () => {
    it('prefers shallower outer session.id over nested subagent sessionId', () => {
      const trace: TraceData = {
        traceId: 't-sid',
        spans: {
          sub: makeSpan({
            spanId: 'sub',
            parentSpanId: 'outer',
            attributes: { 'genkit:metadata:agent:sessionId': 'sub-sess' },
          }),
          outer: makeSpan({
            spanId: 'outer',
            attributes: { 'session.id': 'outer-sess' },
          }),
        },
      };

      expect(extractSessionIdFromTrace(trace)).toBe('outer-sess');
    });

    it('returns undefined when trace has no session attributes', () => {
      expect(extractSessionIdFromTrace({ traceId: 'empty', spans: {} })).toBe(
        undefined
      );
    });
  });
});
