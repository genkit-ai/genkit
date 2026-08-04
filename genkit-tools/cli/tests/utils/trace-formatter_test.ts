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

import { describe, expect, it } from '@jest/globals';
import { formatTraceTree } from '../../src/utils/trace-formatter';

describe('trace-formatter', () => {
  it('should print compact formatted payloads by default in tree format', () => {
    const trace = {
      traceId: 'test-id',
      displayName: 'testSpan',
      startTime: 1000,
      endTime: 2000,
      spans: {
        s1: {
          spanId: 's1',
          displayName: 'testSpan',
          startTime: 1000,
          endTime: 2000,
          attributes: {
            'genkit:input': JSON.stringify({
              query: 'Hello Genkit',
              limit: 10,
            }),
          },
        },
      },
    };
    const output = formatTraceTree(trace as any);
    expect(output).toContain('Hello Genkit');
  });

  it('should format arrays without parts nicely as bullet points', () => {
    const trace = {
      traceId: 'test-id',
      displayName: 'testSpan',
      startTime: 1000,
      endTime: 2000,
      spans: {
        s1: {
          spanId: 's1',
          displayName: 'testSpan',
          startTime: 1000,
          endTime: 2000,
          attributes: {
            'genkit:input': JSON.stringify([{ a: 1 }, { b: 2 }]),
          },
        },
      },
    };
    const output = formatTraceTree(trace as any);
    expect(output).toMatch(/- a: 1\s*- b: 2/s);
  });

  it('should cleanly fallback an empty message content to a generic Role placeholder', () => {
    const trace = {
      traceId: 'test-id',
      displayName: 'testSpan',
      startTime: 1000,
      endTime: 2000,
      spans: {
        s1: {
          spanId: 's1',
          displayName: 'testSpan',
          startTime: 1000,
          endTime: 2000,
          attributes: {
            'genkit:input': JSON.stringify({ role: 'user', content: [] }),
          },
        },
      },
    };
    const output = formatTraceTree(trace as any);
    expect(output).toContain('User: (empty)');
  });
});
