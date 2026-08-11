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
import { getSpanStatus, stackTraceSpans } from '../../src/utils';
import { BASE_FLOW_SPAN_ID, MockTrace } from './trace';

const TEST_TRACE: TraceData = {
  traceId: '7c273c22b219d077c6731a10d46b7d40',
  startTime: 1714059149480,
  endTime: 1714059149485.578,
  displayName: 'multiSteps',
  spans: {},
};

describe('trace utils', () => {
  it('returns root span', async () => {
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

    expect(span).toBeDefined();
    expect(span!.spanId).toEqual(BASE_FLOW_SPAN_ID);
    expect(span?.spans?.length).toBe(1);
  });

  it('returns undefined if no spans', async () => {
    const span = stackTraceSpans(TEST_TRACE);

    expect(span).toBeUndefined();
  });

  describe('getSpanStatus', () => {
    it('returns checkmark for status code 0 (UNSET) or 1 (OK)', () => {
      expect(getSpanStatus({ status: { code: 0 } } as any)).toBe('✔');
      expect(getSpanStatus({ status: { code: 1 } } as any)).toBe('✔');
    });

    it('returns cross for status code 2 (ERROR)', () => {
      expect(getSpanStatus({ status: { code: 2 } } as any)).toBe('✖');
    });

    it('prefers genkit:state attribute if present', () => {
      expect(
        getSpanStatus({
          attributes: { 'genkit:state': 'success' },
          status: { code: 2 },
        } as any)
      ).toBe('✔');
      expect(
        getSpanStatus({
          attributes: { 'genkit:state': 'error' },
          status: { code: 0 },
        } as any)
      ).toBe('✖');
    });
  });
});
