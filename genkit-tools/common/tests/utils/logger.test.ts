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

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { stripVTControlCharacters } from 'util';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { debugToFile, logger } from '../../src/utils/logger.js';

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
      expected: 'Debug: debug msg\n',
    },
    {
      level: 'info' as const,
      message: 'info msg',
      expected: 'Info: info msg\n',
    },
    {
      level: 'warn' as const,
      message: 'warn msg',
      expected: 'Warn: warn msg\n',
    },
    {
      level: 'error' as const,
      message: 'error msg',
      expected: 'Error: error msg\n',
    },
  ])(
    'routes $level logs with consistent prefix to process.stderr',
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
      expect(stripVTControlCharacters(written)).toBe(expected);
    }
  );

  it('filters out messages below the configured logger.level', () => {
    logger.level = 'warn';
    const stderrSpy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);

    logger.debug('hidden debug');
    logger.info('hidden info');
    logger.warn('visible warn');
    logger.error('visible error');

    expect(stderrSpy).toHaveBeenCalledTimes(2);
    expect(stripVTControlCharacters(String(stderrSpy.mock.calls[0][0]))).toBe(
      'Warn: visible warn\n'
    );
    expect(stripVTControlCharacters(String(stderrSpy.mock.calls[1][0]))).toBe(
      'Error: visible error\n'
    );
  });

  it('formats multiple arguments using util.format', () => {
    logger.level = 'debug';
    const stderrSpy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);

    logger.error('Failed to connect:', new Error('ECONNREFUSED'));

    expect(stderrSpy).toHaveBeenCalledOnce();
    const written = stripVTControlCharacters(
      String(stderrSpy.mock.calls[0][0])
    );
    expect(written).toContain('Error: Failed to connect: Error: ECONNREFUSED');
  });

  it('writes JSONL records and enables debug level when debugToFile is called', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-logger-'));
    const stderrSpy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);

    try {
      debugToFile(tmpDir);
      expect(logger.level).toBe('debug');

      logger.debug('trace details');
      logger.warn('something looks off');

      const logPath = path.join(tmpDir, 'genkit-debug.log');
      const lines = fs
        .readFileSync(logPath, 'utf8')
        .trim()
        .split('\n')
        .map((line) => JSON.parse(line));

      expect(lines).toHaveLength(2);
      expect(lines[0]).toMatchObject({
        level: 'debug',
        message: 'trace details',
      });
      expect(typeof lines[0].timestamp).toBe('string');
      expect(lines[1]).toMatchObject({
        level: 'warn',
        message: 'something looks off',
      });
      expect(stderrSpy).toHaveBeenCalledTimes(2);
    } finally {
      debugToFile(null);
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
