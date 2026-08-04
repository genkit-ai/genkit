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
import {
  formatDuration,
  parseAndSanitizeJson,
  sanitizeBase64DataUrls,
} from '../../src/utils/utils';
describe('utils', () => {
  describe('formatDuration', () => {
    it('formats duration correctly', () => {
      expect(formatDuration(1000, 2500)).toBe('1500ms');
      expect(formatDuration(1000, 1000.5)).toBe('0.50ms');
      expect(formatDuration(1000, 500)).toBe('0.00ms'); // Math.max(0, ...)
    });

    it('returns empty string if start or end time is undefined', () => {
      expect(formatDuration(1000, undefined)).toBe('');
      expect(formatDuration(undefined, 2500)).toBe('');
      expect(formatDuration(undefined, undefined)).toBe('');
    });
  });

  describe('sanitizeBase64DataUrls', () => {
    it('sanitizes base64 data URLs', () => {
      const b64 =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
      const sanitized = sanitizeBase64DataUrls(b64);
      expect(sanitized).toMatch(
        /data:image\/png;base64,<... \d+ B base64 data ...>/
      );
    });

    it('keeps media if keepMedia is true', () => {
      const b64 =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
      expect(sanitizeBase64DataUrls(b64, true)).toBe(b64);
    });

    it('sanitizes deeply nested objects', () => {
      const obj = {
        a: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=',
        b: [
          {
            c: 'data:text/plain;base64,SGVsbG8gV29ybGQ=',
          },
        ],
      };
      const sanitized = sanitizeBase64DataUrls(obj);
      expect(sanitized.a).toMatch(
        /data:image\/png;base64,<... \d+ B base64 data ...>/
      );
      expect(sanitized.b[0].c).toMatch(
        /data:text\/plain;base64,<... \d+ B base64 data ...>/
      );
    });
  });

  describe('parseAndSanitizeJson', () => {
    it('parses and sanitizes JSON strings', () => {
      const json = JSON.stringify({
        a: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=',
      });
      const parsed = parseAndSanitizeJson(json);
      expect(parsed.a).toMatch(
        /data:image\/png;base64,<... \d+ B base64 data ...>/
      );
    });

    it('returns original string if not valid JSON', () => {
      const str = 'not valid json';
      expect(parseAndSanitizeJson(str)).toBe(str);
    });
  });
});
