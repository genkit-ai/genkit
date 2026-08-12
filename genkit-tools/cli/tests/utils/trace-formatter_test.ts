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

import type { TraceData } from '@genkit-ai/tools-common';
import { describe, expect, it } from '@jest/globals';
import {
  cleanTraceJson,
  formatTraceTree,
} from '../../src/utils/trace-formatter';

describe('trace-formatter', () => {
  describe('cleanTraceJson', () => {
    it('should deep clone and sanitize base64 media data correctly', () => {
      const rawImage =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
      const originalTrace = {
        traceId: 'test-id',
        displayName: 'testSpan',
        spans: {
          s1: {
            spanId: 's1',
            attributes: {
              'genkit:input': JSON.stringify({ img: rawImage }),
              imgRaw: rawImage,
            },
          },
        },
      } as unknown as TraceData;

      // Without keepBase64 = false (default)
      const sanitized = cleanTraceJson(originalTrace, false);

      expect(sanitized.traceId).toBe('test-id');
      expect(sanitized).not.toBe(originalTrace); // Must be a deep clone

      const attrs = sanitized.spans!.s1.attributes!;

      // Sanitized JSON parsing of base64
      expect((attrs['genkit:input'] as any).img).toContain('base64 data');
      expect((attrs['genkit:input'] as any).img).not.toContain(rawImage);

      // Sanitized raw base64 string
      expect(attrs.imgRaw).toContain('base64 data');
      expect(attrs.imgRaw).not.toContain(rawImage);

      // Verify the type shape remains intact
      expect(typeof sanitized.traceId).toBe('string');
      expect(typeof sanitized.spans).toBe('object');
    });

    it('should deep clone and preserve base64 media data if keepBase64 is true', () => {
      const rawImage =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
      const originalTrace = {
        traceId: 'test-id',
        spans: {
          s1: {
            spanId: 's1',
            attributes: {
              'genkit:input': JSON.stringify({ img: rawImage }),
            },
          },
        },
      } as unknown as TraceData;

      const preserved = cleanTraceJson(originalTrace, true);

      expect(preserved.traceId).toBe('test-id');
      expect((preserved.spans!.s1.attributes!['genkit:input'] as any).img).toBe(
        rawImage
      );
    });
  });

  describe('formatTraceTree', () => {
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
});
