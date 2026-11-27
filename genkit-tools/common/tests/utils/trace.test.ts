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
      expect(stackTraceSpans(makeTrace([]))).toBeUndefined();
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

    it('sorts sibling spans chronologically by startTime', () => {
      const trace = makeTrace([
        makeSpan({ spanId: 'agent', startTime: 100 }),
        makeSpan({
          spanId: 'child-late',
          parentSpanId: 'agent',
          startTime: 130,
        }),
        makeSpan({
          spanId: 'child-early',
          parentSpanId: 'agent',
          startTime: 110,
        }),
        makeSpan({
          spanId: 'child-mid',
          parentSpanId: 'agent',
          startTime: 120,
        }),
      ]);

      expect(stackTraceSpans(trace)?.spans?.map((s) => s.spanId)).toEqual([
        'child-early',
        'child-mid',
        'child-late',
      ]);
    });

    it('breaks timestamp ties by spanId when sorting sibling spans', () => {
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
    it.each([
      {
        name: 'genkit:metadata:subtype over genkit:type',
        span: {
          attributes: {
            'genkit:type': 'action',
            'genkit:metadata:subtype': 'tool',
          },
        },
        expected: 'tool',
      },
      {
        name: 'genkit:type',
        span: { attributes: { 'genkit:type': 'flowStep' } },
        expected: 'flowStep',
      },
      {
        name: 'string spanKind',
        span: { spanKind: 'CLIENT' },
        expected: 'client',
      },
      {
        name: 'numeric spanKind',
        span: { spanKind: 1 as unknown as string },
        expected: 'server',
      },
      {
        name: 'unknown numeric spanKind fallback',
        span: { spanKind: 99 as unknown as string },
        expected: 'span',
      },
    ])('resolves $name', ({ span, expected }) => {
      expect(getSpanType(makeSpan(span))).toBe(expected);
    });
  });

  describe('getSpanStatus', () => {
    it.each([
      { code: 0, expected: '✔' },
      { code: 1, expected: '✔' },
      { code: 2, expected: '✖' },
      { code: 99, expected: '' },
    ])('returns "$expected" for status.code $code', ({ code, expected }) => {
      expect(getSpanStatus(makeSpan({ status: { code } }))).toBe(expected);
    });

    it('prefers genkit:state="success" over error status.code', () => {
      const span = makeSpan({
        attributes: { 'genkit:state': 'success' },
        status: { code: 2 },
      });
      expect(getSpanStatus(span)).toBe('✔');
    });

    it('prefers genkit:state="error" over ok status.code', () => {
      const span = makeSpan({
        attributes: { 'genkit:state': 'error' },
        status: { code: 0 },
      });
      expect(getSpanStatus(span)).toBe('✖');
    });
  });

  describe('isAgentSpan', () => {
    it.each([
      {
        name: '/agent/ name prefix',
        span: { displayName: '/agent/planner' },
        expected: true,
      },
      {
        name: 'agent subtype attribute',
        span: { attributes: { 'genkit:metadata:subtype': 'agent' } },
        expected: true,
      },
      {
        name: 'non-agent span',
        span: { displayName: '/tool/search' },
        expected: false,
      },
    ])('classifies $name', ({ span, expected }) => {
      expect(isAgentSpan(makeSpan(span))).toBe(expected);
    });
  });

  describe('isModelSpan', () => {
    it.each([
      {
        name: 'model subtype attribute',
        span: { attributes: { 'genkit:metadata:subtype': 'model' } },
        expected: true,
      },
      {
        name: '/background-model/ name prefix',
        span: { displayName: '/background-model/googleai/veo-2.0' },
        expected: true,
      },
      {
        name: 'background-model subtype attribute',
        span: { attributes: { 'genkit:metadata:subtype': 'background-model' } },
        expected: true,
      },
      {
        name: 'non-model span',
        span: { displayName: '/agent/planner' },
        expected: false,
      },
    ])('classifies $name', ({ span, expected }) => {
      expect(isModelSpan(makeSpan(span))).toBe(expected);
    });
  });

  describe('isToolSpan', () => {
    it.each([
      {
        name: '/tool.v2/ name prefix',
        span: { displayName: '/tool.v2/mcp/search' },
        expected: true,
      },
      {
        name: 'tool subtype attribute',
        span: { attributes: { 'genkit:metadata:subtype': 'tool' } },
        expected: true,
      },
      {
        name: 'non-tool span',
        span: { displayName: '/model/gemini' },
        expected: false,
      },
    ])('classifies $name', ({ span, expected }) => {
      expect(isToolSpan(makeSpan(span))).toBe(expected);
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
    it.each([
      { name: 'valid JSON string', input: '{"a":1}', expected: { a: 1 } },
      {
        name: 'already-parsed object',
        input: { already: 'parsed' },
        expected: { already: 'parsed' },
      },
      {
        name: 'non-JSON string',
        input: 'plain-text',
        expected: 'plain-text',
      },
      {
        name: 'whitespace-only string',
        input: '   ',
        expected: undefined,
      },
      { name: 'null', input: null, expected: undefined },
    ])('handles $name', ({ input, expected }) => {
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
      expect(extractSessionIdFromTrace(makeTrace([]))).toBeUndefined();
    });
  });
});
