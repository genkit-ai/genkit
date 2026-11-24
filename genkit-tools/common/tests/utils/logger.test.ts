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

import { stripVTControlCharacters } from 'util';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { logger } from '../../src/utils/logger.js';

describe('logger', () => {
  const origLevel = logger.level;

  afterEach(() => {
    logger.level = origLevel;
    vi.restoreAllMocks();
  });

  it.each([
    {
      level: 'debug' as const,
      message: 'debug msg',
      expected: 'Debug: debug msg',
    },
    { level: 'info' as const, message: 'info msg', expected: 'info msg' },
    { level: 'warn' as const, message: 'warn msg', expected: 'Warn:' },
    { level: 'error' as const, message: 'error msg', expected: 'Error:' },
  ])(
    'routes $level logs to process.stderr instead of process.stdout',
    ({ level, message, expected }) => {
      logger.level = 'debug';
      const stdoutSpy = vi
        .spyOn(process.stdout, 'write')
        .mockImplementation(() => true);
      const stderrSpy = vi
        .spyOn(process.stderr, 'write')
        .mockImplementation(() => true);

      logger[level](message);

      expect(stdoutSpy).not.toHaveBeenCalled();
      expect(stderrSpy).toHaveBeenCalledOnce();
      const written = String(stderrSpy.mock.calls[0][0]);
      expect(stripVTControlCharacters(written)).toContain(expected);
    }
  );
});
