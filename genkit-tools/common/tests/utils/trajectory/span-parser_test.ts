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
  getToolCallId,
  getTurnContext,
  hasCompatibleRef,
  isExcludedFromTrajectory,
  isExplicitHistoryMessage,
  messageSignature,
  normalizeParts,
  takeMatchingByRefOrName,
} from '../../../src/utils/trajectory/span-parser.js';
import {
  makeModelSpan,
  makeSpan,
  makeToolSpan,
  makeTurnSpan,
} from './mock_data.js';

describe('trajectory span-parser helpers', () => {
  it('extracts tool call IDs and checks ref compatibility', () => {
    expect(getToolCallId(makeToolSpan({ callId: 'call-1' }))).toBe('call-1');
    expect(getToolCallId(makeToolSpan())).toBeUndefined();

    expect(hasCompatibleRef('ref-1', 'ref-1')).toBe(true);
    expect(hasCompatibleRef('ref-1', undefined)).toBe(true);
    expect(hasCompatibleRef('ref-1', 'ref-2')).toBe(false);
  });

  it('matches items by exact ref before falling back to compatible short tool name', () => {
    const items = [
      { ref: 'ref-a', name: '/tool/search' },
      { ref: 'ref-b', name: '/tool/search' },
      { ref: undefined, name: '/tool.v2/mcp/lookup' },
    ];

    expect(
      takeMatchingByRefOrName(
        items,
        'ref-b',
        'search',
        (i) => i.ref,
        (i) => i.name
      )?.ref
    ).toBe('ref-b');
    expect(
      takeMatchingByRefOrName(
        items,
        'ref-missing',
        'search',
        (i) => i.ref,
        (i) => i.name
      )
    ).toBeUndefined();
    expect(
      takeMatchingByRefOrName(
        items,
        'ref-c',
        'lookup',
        (i) => i.ref,
        (i) => i.name
      )?.name
    ).toBe('/tool.v2/mcp/lookup');
  });

  it('resolves 0-based turnIndex from enclosing runTurn-<N> span and guards against cycles', () => {
    const turn2 = makeTurnSpan({ turn: 2 }) as NestedSpanData;
    const model = makeModelSpan({
      parentSpanId: turn2.spanId,
    }) as NestedSpanData;
    const parentMap = new Map<string, NestedSpanData>([[model.spanId, turn2]]);

    expect(getTurnContext(model, parentMap)).toEqual({
      turnIndex: 1,
      turnSpan: turn2,
    });
    expect(getTurnContext(model, new Map())).toEqual({
      turnIndex: 0,
      turnSpan: undefined,
    });

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

  it('normalizes model output message, candidates fallback, usage, and structured/status errors', () => {
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

    const statusErrSpan = makeModelSpan({
      status: { code: 2, message: 'Deadline exceeded' },
    });
    expect(getModelOutputData(statusErrSpan).error).toEqual({
      message: 'Deadline exceeded',
    });
  });

  it('identifies spans excluded from trajectory projection', () => {
    expect(
      isExcludedFromTrajectory(
        makeSpan({
          attributes: { 'genkit:metadata:trajectory:exclude': 'true' },
        })
      )
    ).toBe(true);
    expect(
      isExcludedFromTrajectory(
        makeSpan({
          attributes: { 'genkit:metadata:subtype': 'promptTemplate' },
        })
      )
    ).toBe(true);
    expect(isExcludedFromTrajectory(makeModelSpan())).toBe(false);
  });

  it('aggregates standard and custom usage counters across steps', () => {
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
    expect(aggregateUsage([undefined])).toBeUndefined();
  });

  it('normalizes parts, computes metadata-invariant signatures, and checks explicit history safely', () => {
    expect(normalizeParts('Hello')).toEqual([{ text: 'Hello' }]);
    expect(normalizeParts(['Hello', { text: 'World' }, null])).toEqual([
      { text: 'Hello' },
      { text: 'World' },
    ]);
    expect(extractTextFromParts([{ text: 'Line 1' }, { text: 'Line 2' }])).toBe(
      'Line 1\nLine 2'
    );

    const msgA = {
      role: 'user' as const,
      content: [{ text: 'Hi', metadata: { turn: 1 } }],
    };
    const msgB = {
      role: 'user' as const,
      content: [{ text: 'Hi', metadata: { turn: 2 } }],
    };
    expect(messageSignature(msgA)).toBe(messageSignature(msgB));
    expect(messageSignature(undefined)).toBe('');

    expect(
      isExplicitHistoryMessage({
        role: 'user',
        content: [{ text: 'Old' }],
        metadata: { _genkit_history: true },
      })
    ).toBe(true);
    expect(isExplicitHistoryMessage(undefined)).toBe(false);
  });
});
