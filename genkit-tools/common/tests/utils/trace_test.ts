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
import type { SpanData, TraceData } from '../../src/types/index.js';
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
import { BASE_FLOW_SPAN_ID, MockTrace, makeSpan, makeTrace } from './trace.js';

const EMPTY_TRACE: TraceData = {
  traceId: '7c273c22b219d077c6731a10d46b7d40',
  startTime: 1714059149480,
  endTime: 1714059149485.578,
  displayName: 'multiSteps',
  spans: {},
};

const PRUNE_BASELINE_TRACE = makeTrace([
  makeSpan({
    spanId: 'agent',
    displayName: '/agent/main',
    startTime: 105,
  }),
  makeSpan({ spanId: 'kept', parentSpanId: 'agent', startTime: 110 }),
  makeSpan({
    spanId: 'excluded-branch',
    parentSpanId: 'agent',
    startTime: 115,
    attributes: { skipMe: true },
  }),
  makeSpan({
    spanId: 'excluded-leaf',
    parentSpanId: 'excluded-branch',
    startTime: 116,
  }),
]);

describe('trace utils', () => {
  describe('stackTraceSpans', () => {
    it('returns root span and nests children from flat trace spans', () => {
      const trace = new MockTrace('My input', 'My output')
        .addSpan({ stepName: 'retrieverStep', spanType: 'action' })
        .addSpan({ stepName: 'llmStep', spanType: 'action' })
        .getTrace();

      const span = stackTraceSpans(trace);

      expect(span?.spanId).toEqual(BASE_FLOW_SPAN_ID);
      expect(span?.spans).toHaveLength(2);
    });

    it('returns undefined if no spans', () => {
      expect(stackTraceSpans(EMPTY_TRACE)).toBeUndefined();
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

    it('sorts sibling spans chronologically and breaks timestamp ties by spanId', () => {
      const trace = makeTrace([
        makeSpan({ spanId: 'agent', startTime: 100 }),
        makeSpan({ spanId: 'child-b', parentSpanId: 'agent', startTime: 110 }),
        makeSpan({ spanId: 'child-a', parentSpanId: 'agent', startTime: 110 }),
      ]);

      expect(stackTraceSpans(trace)?.spans?.map((s) => s.spanId)).toEqual([
        'child-a',
        'child-b',
      ]);
    });

    it('prunes excluded spans and their descendant subtrees', () => {
      const root = stackTraceSpans(PRUNE_BASELINE_TRACE, {
        exclude: (s) => Boolean(s.attributes?.['skipMe']),
      });

      expect(root?.spanId).toBe('agent');
      expect(root?.spans?.map((s) => s.spanId)).toEqual(['kept']);
    });

    it('returns undefined when the root span itself is excluded', () => {
      expect(
        stackTraceSpans(PRUNE_BASELINE_TRACE, {
          exclude: (s) => s.spanId === 'agent',
        })
      ).toBeUndefined();
    });
  });

  describe('getSpanType', () => {
    it.each<[string, Partial<SpanData>, string]>([
      [
        'genkit:metadata:subtype over genkit:type',
        {
          attributes: {
            'genkit:type': 'action',
            'genkit:metadata:subtype': 'tool',
          },
        },
        'tool',
      ],
      [
        'genkit:type',
        { attributes: { 'genkit:type': 'flowStep' } },
        'flowStep',
      ],
      ['string spanKind', { spanKind: 'CLIENT' }, 'client'],
      ['numeric spanKind', { spanKind: 1 as unknown as string }, 'server'],
      [
        'unknown numeric spanKind fallback',
        { spanKind: 99 as unknown as string },
        'span',
      ],
    ])('resolves %s', (_label, overrides, expected) => {
      expect(getSpanType(makeSpan(overrides))).toBe(expected);
    });
  });

  describe('getSpanStatus', () => {
    it.each<[string, Partial<SpanData>, string]>([
      ['status code 0 (UNSET)', { status: { code: 0 } }, '✔'],
      ['status code 1 (OK)', { status: { code: 1 } }, '✔'],
      ['status code 2 (ERROR)', { status: { code: 2 } }, '✖'],
      ['unknown status code', { status: { code: 99 } }, ''],
      [
        'genkit:state success over error status code',
        { attributes: { 'genkit:state': 'success' }, status: { code: 2 } },
        '✔',
      ],
      [
        'genkit:state error over ok status code',
        { attributes: { 'genkit:state': 'error' }, status: { code: 0 } },
        '✖',
      ],
    ])('returns icon for %s', (_label, overrides, expected) => {
      expect(getSpanStatus(makeSpan(overrides))).toBe(expected);
    });
  });

  describe('isAgentSpan', () => {
    it.each<[string, Partial<SpanData>, boolean]>([
      ['/agent/ name prefix', { displayName: '/agent/planner' }, true],
      [
        'agent subtype attribute',
        { attributes: { 'genkit:metadata:subtype': 'agent' } },
        true,
      ],
      ['non-agent span', { displayName: '/tool/search' }, false],
    ])('classifies %s', (_label, overrides, expected) => {
      expect(isAgentSpan(makeSpan(overrides))).toBe(expected);
    });
  });

  describe('isModelSpan', () => {
    it.each<[string, Partial<SpanData>, boolean]>([
      [
        'model subtype attribute',
        { attributes: { 'genkit:metadata:subtype': 'model' } },
        true,
      ],
      [
        '/background-model/ name prefix',
        { displayName: '/background-model/googleai/veo-2.0' },
        true,
      ],
      [
        'background-model subtype attribute',
        { attributes: { 'genkit:metadata:subtype': 'background-model' } },
        true,
      ],
      ['non-model span', { displayName: '/agent/planner' }, false],
    ])('classifies %s', (_label, overrides, expected) => {
      expect(isModelSpan(makeSpan(overrides))).toBe(expected);
    });
  });

  describe('isToolSpan', () => {
    it.each<[string, Partial<SpanData>, boolean]>([
      ['/tool.v2/ name prefix', { displayName: '/tool.v2/mcp/search' }, true],
      [
        'tool subtype attribute',
        { attributes: { 'genkit:metadata:subtype': 'tool' } },
        true,
      ],
      ['non-tool span', { displayName: '/model/gemini' }, false],
    ])('classifies %s', (_label, overrides, expected) => {
      expect(isToolSpan(makeSpan(overrides))).toBe(expected);
    });
  });

  describe('naming and attribute helpers', () => {
    it('getSpanName falls back to genkit:name when displayName is empty', () => {
      expect(
        getSpanName(
          makeSpan({
            displayName: '',
            attributes: { 'genkit:name': 'myAgent' },
          })
        )
      ).toBe('myAgent');
    });

    it.each<[string | undefined, string]>([
      ['/model/googleai/gemini-2.5-pro', 'googleai/gemini-2.5-pro'],
      ['/background-model/googleai/veo-2.0', 'googleai/veo-2.0'],
      ['/agent/weatherAgent', 'weatherAgent'],
      [undefined, ''],
    ])('cleanActionName(%s) -> %s', (input, expected) => {
      expect(cleanActionName(input)).toBe(expected);
    });

    it.each<[string | undefined, string]>([
      ['/tool.v2/mcp-server/read_file', 'read_file'],
      ['getWeather', 'getWeather'],
      [undefined, ''],
    ])('shortToolName(%s) -> %s', (input, expected) => {
      expect(shortToolName(input)).toBe(expected);
    });

    it.each<[unknown, boolean]>([
      [true, true],
      ['TRUE', true],
      ['false', false],
    ])('isTruthyAttr(%s) -> %s', (input, expected) => {
      expect(isTruthyAttr(input)).toBe(expected);
    });
  });

  describe('parseSpanJson', () => {
    it.each<[string, unknown, unknown]>([
      ['valid JSON string', '{"a":1}', { a: 1 }],
      ['already-parsed object', { already: 'parsed' }, { already: 'parsed' }],
      ['non-JSON string', 'plain-text', 'plain-text'],
      ['whitespace-only string', '   ', undefined],
      ['null', null, undefined],
    ])('handles %s', (_label, input, expected) => {
      expect(parseSpanJson(input)).toEqual(expected);
    });
  });

  describe('extractSessionIdFromTrace', () => {
    it('prefers shallower outer session.id over nested subagent sessionId', () => {
      const trace = makeTrace([
        makeSpan({
          spanId: 'sub',
          parentSpanId: 'outer',
          attributes: { 'genkit:metadata:agent:sessionId': 'sub-sess' },
        }),
        makeSpan({
          spanId: 'outer',
          attributes: { 'session.id': 'outer-sess' },
        }),
      ]);
      expect(extractSessionIdFromTrace(trace)).toBe('outer-sess');
    });

    it('terminates safely on cyclic parent links', () => {
      const cyclicTrace = makeTrace([
        makeSpan({
          spanId: 'a',
          parentSpanId: 'b',
          attributes: { 'session.id': 'cyc-sess' },
        }),
        makeSpan({ spanId: 'b', parentSpanId: 'a' }),
      ]);
      expect(extractSessionIdFromTrace(cyclicTrace)).toBe('cyc-sess');
    });

    it('returns undefined when no sessionId attribute is present', () => {
      expect(extractSessionIdFromTrace(EMPTY_TRACE)).toBeUndefined();
    });
  });
});
