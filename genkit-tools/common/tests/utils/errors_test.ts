/**
 * Copyright 2025 Google LLC
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
  getErrorCode,
  getErrorDetails,
  isConnectionRefusedError,
} from '../../src/utils/errors';

describe('errors.ts', () => {
  describe('isConnectionRefusedError', () => {
    it('should return false for null/undefined', () => {
      expect(isConnectionRefusedError(null)).toBe(false);
      expect(isConnectionRefusedError(undefined)).toBe(false);
    });

    it('should detect plain objects with connection error codes', () => {
      expect(isConnectionRefusedError({ code: 'ECONNREFUSED' })).toBe(true);
      expect(isConnectionRefusedError({ code: 'ConnectionRefused' })).toBe(
        true
      );
      expect(isConnectionRefusedError({ code: 'ECONNRESET' })).toBe(true);
      expect(isConnectionRefusedError({ code: 'OTHER_ERROR' })).toBe(false);
    });

    it('should detect Error instances with direct code', () => {
      const err = new Error('Connection failed');
      (err as any).code = 'ECONNREFUSED';
      expect(isConnectionRefusedError(err)).toBe(true);

      const err2 = new Error('Connection failed');
      (err2 as any).code = 'ConnectionRefused';
      expect(isConnectionRefusedError(err2)).toBe(true);

      const err3 = new Error('Connection failed');
      (err3 as any).code = 'ECONNRESET';
      expect(isConnectionRefusedError(err3)).toBe(true);
    });

    it('should detect Node.js style errors with cause', () => {
      const err = new Error('Fetch failed');
      (err as any).cause = { code: 'ECONNREFUSED' };
      expect(isConnectionRefusedError(err)).toBe(true);
    });

    it('should fallback to checking error messages', () => {
      expect(
        isConnectionRefusedError(
          new Error('connect ECONNREFUSED 127.0.0.1:3000')
        )
      ).toBe(true);
      expect(
        isConnectionRefusedError(new Error('Connection refused to server'))
      ).toBe(true);
      expect(
        isConnectionRefusedError(
          new Error('ConnectionRefused: Unable to connect')
        )
      ).toBe(true);
      expect(
        isConnectionRefusedError(new Error('Something else went wrong'))
      ).toBe(false);
    });

    it('should handle complex nested structures', () => {
      const err = new Error('Outer error');
      (err as any).cause = new Error('Inner error');
      (err as any).cause.code = 'ECONNREFUSED';
      expect(isConnectionRefusedError(err)).toBe(true);
    });
  });

  describe('getErrorCode', () => {
    it('should return undefined for null/undefined', () => {
      expect(getErrorCode(null)).toBeUndefined();
      expect(getErrorCode(undefined)).toBeUndefined();
    });

    it('should extract code from plain objects', () => {
      expect(getErrorCode({ code: 'ECONNREFUSED' })).toBe('ECONNREFUSED');
      expect(getErrorCode({ code: 'CUSTOM_ERROR' })).toBe('CUSTOM_ERROR');
      expect(getErrorCode({ message: 'No code here' })).toBeUndefined();
    });

    it('should extract code from Error instances', () => {
      const err = new Error('Test error');
      (err as any).code = 'TEST_CODE';
      expect(getErrorCode(err)).toBe('TEST_CODE');
    });

    it('should extract code from cause property', () => {
      const err = new Error('Outer error');
      (err as any).cause = { code: 'INNER_CODE' };
      expect(getErrorCode(err)).toBe('INNER_CODE');
    });

    it('should prioritize direct code over cause code', () => {
      const err = new Error('Test error');
      (err as any).code = 'DIRECT_CODE';
      (err as any).cause = { code: 'CAUSE_CODE' };
      expect(getErrorCode(err)).toBe('DIRECT_CODE');
    });

    it('should handle non-string code values', () => {
      expect(getErrorCode({ code: 123 })).toBeUndefined();
      expect(getErrorCode({ code: null })).toBeUndefined();
      expect(getErrorCode({ code: {} })).toBeUndefined();
    });
  });

  describe('getErrorDetails', () => {
    it('should return "Unknown error" for null/undefined', () => {
      expect(getErrorDetails(null)).toBe('Unknown error');
      expect(getErrorDetails(undefined)).toBe('Unknown error');
    });

    it('should format Error instances with code', () => {
      const err = new Error('Connection failed');
      (err as any).code = 'ECONNREFUSED';
      expect(getErrorDetails(err)).toBe('Connection failed (ECONNREFUSED)');
    });

    it('should format Error instances without code', () => {
      const err = new Error('Simple error');
      expect(getErrorDetails(err)).toBe('Simple error');
    });

    it('should format plain objects with message and code', () => {
      expect(getErrorDetails({ message: 'Failed', code: 'ERR123' })).toBe(
        'Failed (ERR123)'
      );
      expect(getErrorDetails({ message: 'No code here' })).toBe('No code here');
    });

    it('should handle string errors', () => {
      expect(getErrorDetails('String error')).toBe('String error');
    });

    it('should handle number errors', () => {
      expect(getErrorDetails(123)).toBe('123');
    });

    it('should handle boolean errors', () => {
      expect(getErrorDetails(true)).toBe('true');
      expect(getErrorDetails(false)).toBe('false');
    });

    it('should handle objects without message', () => {
      expect(getErrorDetails({ code: 'ERR_NO_MSG' })).toBe('[object Object]');
    });

    it('should extract code from cause for formatting', () => {
      const err = new Error('Outer error');
      (err as any).cause = { code: 'INNER_CODE' };
      expect(getErrorDetails(err)).toBe('Outer error (INNER_CODE)');
    });
  });
});
