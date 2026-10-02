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
import { Command, InvalidArgumentError } from 'commander';
import {
  parseJson,
  parseNonNegativeInt,
  parsePort,
  parsePositiveInt,
  parseTraceStatus,
} from '../../src/utils/option-parsers';

describe('option-parsers', () => {
  describe('parsePositiveInt', () => {
    it('parses positive integers', () => {
      expect(parsePositiveInt('1')).toBe(1);
      expect(parsePositiveInt('100')).toBe(100);
    });

    it.each(['0', '-1', 'abc', '10abc', '1.5', '', '999999999999999999999'])(
      'rejects "%s"',
      (value) => {
        expect(() => parsePositiveInt(value)).toThrow(InvalidArgumentError);
      }
    );
  });

  describe('parseNonNegativeInt', () => {
    it('parses zero and positive integers', () => {
      expect(parseNonNegativeInt('0')).toBe(0);
      expect(parseNonNegativeInt('30000')).toBe(30000);
    });

    it.each(['-1', 'abc', '1.5', '999999999999999999999'])(
      'rejects "%s"',
      (value) => {
        expect(() => parseNonNegativeInt(value)).toThrow(InvalidArgumentError);
      }
    );
  });

  describe('parsePort', () => {
    it('parses ports between 1 and 65535', () => {
      expect(parsePort('1')).toBe(1);
      expect(parsePort('4000')).toBe(4000);
      expect(parsePort('65535')).toBe(65535);
    });

    it.each(['0', '65536', '-1', 'abc', '4000.5'])('rejects "%s"', (value) => {
      expect(() => parsePort(value)).toThrow(InvalidArgumentError);
    });
  });

  describe('parseTraceStatus', () => {
    it('maps "success" and "error" (case-insensitively) and non-negative integer codes', () => {
      expect(parseTraceStatus('success')).toBe(0);
      expect(parseTraceStatus('SUCCESS')).toBe(0);
      expect(parseTraceStatus('error')).toBe(2);
      expect(parseTraceStatus('Error')).toBe(2);
      expect(parseTraceStatus('0')).toBe(0);
      expect(parseTraceStatus('2')).toBe(2);
    });

    it.each(['invalid', '', ' ', '1.5', '-1', 'Infinity'])(
      'rejects "%s"',
      (value) => {
        expect(() => parseTraceStatus(value)).toThrow(InvalidArgumentError);
      }
    );
  });

  describe('parseJson', () => {
    it('parses valid JSON values', () => {
      expect(parseJson('{"auth":{"uid":"u1"}}')).toEqual({
        auth: { uid: 'u1' },
      });
      expect(parseJson('[1, 2]')).toEqual([1, 2]);
    });

    it.each(['', '{invalid}', 'undefined'])('rejects "%s"', (value) => {
      expect(() => parseJson(value)).toThrow(InvalidArgumentError);
    });
  });

  it('ignores the previous value that Commander passes as a second argument', () => {
    const command = new Command()
      .exitOverride()
      .option('--size <size>', 'size', parsePositiveInt);
    command.parse(['node', 'test', '--size', '2', '--size', '8']);
    expect(command.opts().size).toBe(8);
  });

  it('reports invalid values as a Commander option error', () => {
    const command = new Command()
      .exitOverride()
      .configureOutput({ writeErr: () => {} })
      .option('--size <size>', 'size', parsePositiveInt);
    expect(() => command.parse(['node', 'test', '--size', 'abc'])).toThrow(
      "error: option '--size <size>' argument 'abc' is invalid. Must be a positive integer."
    );
  });
});
