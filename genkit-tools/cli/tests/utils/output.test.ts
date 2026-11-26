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

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  isInteractive,
  isStderrTTY,
  writeStderr,
  writeStdout,
} from '../../src/utils/output.js';

describe('writeStdout', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('writes text with a trailing newline to process.stdout', () => {
    const writeSpy = vi
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);

    writeStdout('hello world');

    expect(writeSpy).toHaveBeenCalledWith('hello world\n');
  });

  it('writes a blank line when called with no arguments', () => {
    const writeSpy = vi
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);

    writeStdout();

    expect(writeSpy).toHaveBeenCalledWith('\n');
  });
});

describe('writeStderr', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('writes text with a trailing newline to process.stderr', () => {
    const writeSpy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);

    writeStderr('status message');

    expect(writeSpy).toHaveBeenCalledWith('status message\n');
  });

  it('writes a blank line when called with no arguments', () => {
    const writeSpy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);

    writeStderr();

    expect(writeSpy).toHaveBeenCalledWith('\n');
  });
});

describe('TTY and interactivity helpers', () => {
  const origStdinTTY = process.stdin.isTTY;
  const origStderrTTY = process.stderr.isTTY;
  const origArgv = process.argv;

  afterEach(() => {
    Object.defineProperty(process.stdin, 'isTTY', {
      value: origStdinTTY,
      configurable: true,
      writable: true,
    });
    Object.defineProperty(process.stderr, 'isTTY', {
      value: origStderrTTY,
      configurable: true,
      writable: true,
    });
    process.argv = origArgv;
  });

  it.each([
    [true, true],
    [false, false],
    [undefined, false],
  ])(
    'isStderrTTY returns %s when process.stderr.isTTY is %s',
    (tty, expected) => {
      Object.defineProperty(process.stderr, 'isTTY', {
        value: tty,
        configurable: true,
        writable: true,
      });

      expect(isStderrTTY()).toBe(expected);
    }
  );

  it.each([
    {
      name: 'both stdin and stderr are TTY without --non-interactive',
      stdinTTY: true,
      stderrTTY: true,
      argv: ['node', 'genkit', 'flow:run'],
      expected: true,
    },
    {
      name: 'stdin is not a TTY',
      stdinTTY: undefined,
      stderrTTY: true,
      argv: ['node', 'genkit', 'flow:run'],
      expected: false,
    },
    {
      name: 'stderr is not a TTY',
      stdinTTY: true,
      stderrTTY: undefined,
      argv: ['node', 'genkit', 'flow:run'],
      expected: false,
    },
    {
      name: '--non-interactive is set',
      stdinTTY: true,
      stderrTTY: true,
      argv: ['node', 'genkit', '--non-interactive', 'flow:run'],
      expected: false,
    },
  ])(
    'isInteractive returns $expected when $name',
    ({ stdinTTY, stderrTTY, argv, expected }) => {
      Object.defineProperty(process.stdin, 'isTTY', {
        value: stdinTTY,
        configurable: true,
        writable: true,
      });
      Object.defineProperty(process.stderr, 'isTTY', {
        value: stderrTTY,
        configurable: true,
        writable: true,
      });
      process.argv = argv;

      expect(isInteractive()).toBe(expected);
    }
  );
});
