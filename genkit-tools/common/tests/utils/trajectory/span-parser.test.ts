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
import type { NestedSpanData } from '../../../src/types/index.js';
import {
  aggregateUsage,
  extractTextFromParts,
  getModelOutputData,
  getTurnContext,
  hasCompatibleRef,
  isExcludedFromTrajectory,
  isExplicitHistoryMessage,
  messageSignature,
  normalizeParts,
  takeMatchingByRefOrName,
} from '../../../src/utils/trajectory/span-parser.js';
import { makeModelSpan, makeSpan, makeTurnSpan } from './mock_data.js';

describe('trajectory span-parser helpers', () => {
  describe('hasCompatibleRef', () => {
    it.each([
      {
        name: 'matching refs',
        reqRef: 'ref-1',
        candidateRef: 'ref-1',
        expected: true,
      },
      {
        name: 'missing candidate ref',
        reqRef: 'ref-1',
        candidateRef: undefined,
        expected: true,
      },
      {
        name: 'conflicting refs',
        reqRef: 'ref-1',
        candidateRef: 'ref-2',
        expected: false,
      },
    ])('returns $expected for $name', ({ reqRef, candidateRef, expected }) => {
      expect(hasCompatibleRef(reqRef, candidateRef)).toBe(expected);
    });
  });

  describe('takeMatchingByRefOrName', () => {
    const makeItems = () => [
      { ref: 'ref-a', name: '/tool/search' },
      { ref: 'ref-b', name: '/tool/search' },
      { ref: undefined, name: '/tool.v2/mcp/lookup' },
    ];

    it('matches by exact ref', () => {
      expect(
        takeMatchingByRefOrName(
          makeItems(),
          'ref-b',
          'search',
          (i) => i.ref,
          (i) => i.name
        )?.ref
      ).toBe('ref-b');
    });

    it('returns undefined on conflicting refs instead of falling back to name', () => {
      expect(
        takeMatchingByRefOrName(
          makeItems(),
          'ref-missing',
          'search',
          (i) => i.ref,
          (i) => i.name
        )
      ).toBeUndefined();
    });

    it('falls back to short tool name when candidate ref is undefined', () => {
      expect(
        takeMatchingByRefOrName(
          makeItems(),
          'ref-c',
          'lookup',
          (i) => i.ref,
          (i) => i.name
        )?.name
      ).toBe('/tool.v2/mcp/lookup');
    });
  });

  describe('getTurnContext', () => {
    it('resolves 0-based turnIndex and turnSpan from enclosing runTurn-<N>', () => {
      const turn2 = makeTurnSpan({ turn: 2 }) as NestedSpanData;
      const model = makeModelSpan({
        parentSpanId: turn2.spanId,
      }) as NestedSpanData;
      const parentMap = new Map<string, NestedSpanData>([
        [model.spanId, turn2],
      ]);

      expect(getTurnContext(model, parentMap)).toEqual({
        turnIndex: 1,
        turnSpan: turn2,
      });
    });

    it('defaults to turnIndex 0 when no runTurn-<N> ancestor exists', () => {
      const model = makeModelSpan() as NestedSpanData;

      expect(getTurnContext(model, new Map())).toEqual({
        turnIndex: 0,
        turnSpan: undefined,
      });
    });

    it('terminates safely on cyclic parentMap links', () => {
      const cycleA = makeSpan({ spanId: 'cyc-a' }) as NestedSpanData;
      const cycleB = makeSpan({ spanId: 'cyc-b' }) as NestedSpanData;
      const cyclicParents = new Map<string, NestedSpanData>([
        ['cyc-a', cycleB],
        ['cyc-b', cycleA],
      ]);

      expect(getTurnContext(cycleA, cyclicParents)).toEqual({
        turnIndex: 0,
        turnSpan: undefined,
      });
    });
  });

  describe('getModelOutputData', () => {
    it('falls back to candidates[0] for message, finishReason, and usage', () => {
      const legacySpan = makeModelSpan({
        candidates: [
          {
            message: { role: 'model', content: [{ text: 'Truncated' }] },
            finishReason: 'length',
            usage: { inputTokens: 10, outputTokens: 5 },
          },
        ],
      });

      expect(getModelOutputData(legacySpan)).toEqual({
        message: { role: 'model', content: [{ text: 'Truncated' }] },
        finishReason: 'length',
        usage: { inputTokens: 10, outputTokens: 5 },
        error: undefined,
      });
    });

    it('extracts structured RuntimeError from genkit:output.error', () => {
      const structuredErrSpan = makeModelSpan({
        outputError: {
          message: 'Quota exceeded',
          details: { stack: 'Error: Quota exceeded' },
        },
      });

      expect(getModelOutputData(structuredErrSpan).error).toEqual({
        message: 'Quota exceeded',
        details: { stack: 'Error: Quota exceeded' },
      });
    });

    it('falls back to span.status.message when status.code === 2', () => {
      const statusErrSpan = makeModelSpan({
        status: { code: 2, message: 'Deadline exceeded' },
      });

      expect(getModelOutputData(statusErrSpan).error).toEqual({
        message: 'Deadline exceeded',
      });
    });
  });

  describe('isExcludedFromTrajectory', () => {
    it.each([
      {
        name: 'trajectory:exclude attribute',
        span: makeSpan({
          attributes: { 'genkit:metadata:trajectory:exclude': 'true' },
        }),
        expected: true,
      },
      {
        name: 'promptTemplate subtype',
        span: makeSpan({
          attributes: { 'genkit:metadata:subtype': 'promptTemplate' },
        }),
        expected: true,
      },
      {
        name: 'model span',
        span: makeModelSpan(),
        expected: false,
      },
    ])('returns $expected for $name', ({ span, expected }) => {
      expect(isExcludedFromTrajectory(span)).toBe(expected);
    });
  });

  describe('aggregateUsage', () => {
    it('sums standard and custom counters across steps', () => {
      expect(
        aggregateUsage([
          { inputTokens: 10, outputTokens: 20, custom: { thoughtsTokens: 4 } },
          undefined,
          { inputTokens: 5, custom: { thoughtsTokens: 6 } },
        ])
      ).toEqual({
        inputTokens: 15,
        outputTokens: 20,
        custom: { thoughtsTokens: 10 },
      });
    });

    it('returns undefined when all inputs are undefined', () => {
      expect(aggregateUsage([undefined])).toBeUndefined();
    });
  });

  describe('normalizeParts', () => {
    it.each([
      {
        name: 'plain string',
        input: 'Hello',
        expected: [{ text: 'Hello' }],
      },
      {
        name: 'mixed array with null and nested arrays filtered out',
        input: ['Hello', { text: 'World' }, null, ['nested']],
        expected: [{ text: 'Hello' }, { text: 'World' }],
      },
    ])('normalizes $name', ({ input, expected }) => {
      expect(normalizeParts(input)).toEqual(expected);
    });
  });

  describe('extractTextFromParts', () => {
    it('joins text parts with newline and trims', () => {
      expect(
        extractTextFromParts([{ text: 'Line 1' }, { text: 'Line 2' }])
      ).toBe('Line 1\nLine 2');
    });
  });

  describe('messageSignature', () => {
    it('ignores part metadata', () => {
      const msgA = {
        role: 'user' as const,
        content: [{ text: 'Hi', metadata: { turn: 1 } }],
      };
      const msgB = {
        role: 'user' as const,
        content: [{ text: 'Hi', metadata: { turn: 2 } }],
      };

      expect(messageSignature(msgA)).toBe(messageSignature(msgB));
    });

    it('returns empty string for undefined', () => {
      expect(messageSignature(undefined)).toBe('');
    });
  });

  describe('isExplicitHistoryMessage', () => {
    it.each([
      {
        name: '_genkit_history: true',
        msg: {
          role: 'user' as const,
          content: [{ text: 'Old' }],
          metadata: { _genkit_history: true },
        },
        expected: true,
      },
      {
        name: 'purpose: history',
        msg: {
          role: 'user' as const,
          content: [{ text: 'Old' }],
          metadata: { purpose: 'history' },
        },
        expected: true,
      },
      {
        name: 'undefined message',
        msg: undefined,
        expected: false,
      },
    ])('returns $expected for $name', ({ msg, expected }) => {
      expect(isExplicitHistoryMessage(msg)).toBe(expected);
    });
  });
});
