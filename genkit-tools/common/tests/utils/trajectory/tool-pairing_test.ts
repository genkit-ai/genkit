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
import type { TrajectorySpans } from '../../../src/utils/trajectory/span-parser.js';
import {
  collectToolResponses,
  pairStepToolActions,
} from '../../../src/utils/trajectory/tool-pairing.js';
import { makeModelSpan, makeToolSpan, makeTurnSpan } from './mock_data.js';

function makeTrajectorySpans(
  toolSpans: NestedSpanData[],
  parentMap = new Map<string, NestedSpanData>()
): TrajectorySpans {
  return {
    parentMap,
    modelSpans: [],
    toolSpans,
    scopeSpans: toolSpans,
    subagentByToolSpanId: new Map(),
  };
}

describe('trajectory tool-pairing helpers', () => {
  it('deduplicates multi-turn toolResponse parts across input.messages and single input.message', () => {
    const respPart = {
      toolResponse: { ref: 'c1', name: 'getWeather', output: { tempF: 62 } },
    };
    const singleMsgResp = {
      toolResponse: { ref: 'c2', name: 'getTime', output: '10:00' },
    };
    const entries = collectToolResponses([
      { messages: [{ role: 'tool', content: [respPart] }] },
      { messages: [{ role: 'tool', content: [respPart] }] },
      { message: { role: 'tool', content: [singleMsgResp] } },
    ]);

    expect(entries.map((e) => e.part)).toEqual([respPart, singleMsgResp]);
  });

  it('pairs parallel tool spans in chronological order by short tool name', () => {
    const modelSpan = makeModelSpan({
      spanId: 'm1',
      startTime: 110,
      endTime: 200,
    }) as NestedSpanData;
    const toolSpans = [
      makeToolSpan({
        spanId: 'read-a',
        name: 'readFile',
        startTime: 205,
        output: 'A',
      }),
      makeToolSpan({
        spanId: 'read-b',
        name: 'readFile',
        startTime: 210,
        output: 'B',
      }),
      makeToolSpan({
        spanId: 'search-1',
        name: 'search',
        startTime: 215,
        output: ['s1'],
      }),
      makeToolSpan({
        spanId: 'search-2',
        name: 'search',
        startTime: 220,
        output: ['s2'],
      }),
    ] as NestedSpanData[];

    const actions = pairStepToolActions(
      [
        { toolRequest: { ref: 'ref-a', name: 'readFile', input: { p: 'a' } } },
        { toolRequest: { ref: 'ref-b', name: 'readFile', input: { p: 'b' } } },
        { toolRequest: { name: 'search', input: { q: 1 } } },
        { toolRequest: { name: 'search', input: { q: 2 } } },
      ],
      modelSpan,
      undefined,
      0,
      'stop',
      {
        trajectorySpans: makeTrajectorySpans(toolSpans),
        claimedToolSpanIds: new Set(),
        toolResponses: [],
      }
    );

    expect(actions.map((a) => [a.actionId, a.spanId, a.output])).toEqual([
      ['ref-a', 'read-a', 'A'],
      ['ref-b', 'read-b', 'B'],
      ['m1-action-3', 'search-1', ['s1']],
      ['m1-action-4', 'search-2', ['s2']],
    ]);
  });

  it('unwraps MultipartToolResponse .output on tool.v2 spans to match toolResponse.output', () => {
    const modelSpan = makeModelSpan({
      spanId: 'm1',
      startTime: 100,
      endTime: 150,
    }) as NestedSpanData;
    const v2Tool = makeToolSpan({
      spanId: 'mcp-v2',
      name: '/tool.v2/mcp/screenshot',
      startTime: 155,
      endTime: 190,
      output: {
        output: { width: 800, height: 600 },
        content: [{ media: { url: 'data:image/png;base64,AAA' } }],
      },
    }) as NestedSpanData;

    const actions = pairStepToolActions(
      [{ toolRequest: { ref: 'v2-1', name: 'mcp/screenshot', input: {} } }],
      modelSpan,
      undefined,
      0,
      'stop',
      {
        trajectorySpans: makeTrajectorySpans([v2Tool]),
        claimedToolSpanIds: new Set(),
        toolResponses: [],
      }
    );

    expect(actions[0]).toMatchObject({
      actionId: 'v2-1',
      spanId: 'mcp-v2',
      status: 'success',
      output: { width: 800, height: 600 },
      content: [{ media: { url: 'data:image/png;base64,AAA' } }],
    });
  });

  it('falls back to turnIndex matching on skewed clocks and lookahead toolResponse output', () => {
    const turn1 = makeTurnSpan({ spanId: 'turn-1', turn: 1 }) as NestedSpanData;
    const modelSpan = makeModelSpan({
      spanId: 'm1',
      parentSpanId: 'turn-1',
      startTime: 120,
      endTime: 160,
    }) as NestedSpanData;
    const skewedTool = makeToolSpan({
      spanId: 'tool-skewed',
      parentSpanId: 'turn-1',
      name: 'lookupCache',
      startTime: 115,
      endTime: 175,
    }) as NestedSpanData;
    const parentMap = new Map<string, NestedSpanData>([
      ['m1', turn1],
      ['tool-skewed', turn1],
    ]);

    const actions = pairStepToolActions(
      [{ toolRequest: { name: 'lookupCache', input: { key: 'k1' } } }],
      modelSpan,
      undefined,
      0,
      'stop',
      {
        trajectorySpans: makeTrajectorySpans([skewedTool], parentMap),
        claimedToolSpanIds: new Set(),
        toolResponses: [
          {
            part: {
              toolResponse: { name: 'lookupCache', output: { hit: true } },
            },
          },
        ],
      }
    );

    expect(actions[0]).toMatchObject({
      spanId: 'tool-skewed',
      status: 'success',
      output: { hit: true },
      startTime: 115,
      endTime: 175,
    });
  });

  it('classifies failed tool spans as error and unfinished spans as pending', () => {
    const modelSpan = makeModelSpan({
      spanId: 'm1',
      startTime: 100,
    }) as NestedSpanData;
    const failedTool = makeToolSpan({
      spanId: 'failed',
      name: 'deploy',
      status: { code: 2, message: 'Permission denied' },
    }) as NestedSpanData;
    const pendingTool = makeToolSpan({
      spanId: 'pending',
      name: 'healthCheck',
      endTime: 0,
    }) as NestedSpanData;

    const actions = pairStepToolActions(
      [
        { toolRequest: { ref: 'dep-1', name: 'deploy' } },
        { toolRequest: { ref: 'chk-1', name: 'healthCheck' } },
      ],
      modelSpan,
      undefined,
      0,
      'tool_calls',
      {
        trajectorySpans: makeTrajectorySpans([failedTool, pendingTool]),
        claimedToolSpanIds: new Set(),
        toolResponses: [],
      }
    );

    expect(actions[0]).toMatchObject({
      actionId: 'dep-1',
      status: 'error',
      error: { message: 'Permission denied' },
    });
    expect(actions[1]).toMatchObject({
      actionId: 'chk-1',
      status: 'in-progress',
    });
  });
});
