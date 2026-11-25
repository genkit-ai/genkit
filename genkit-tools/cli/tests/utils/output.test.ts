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
import { writeStdout } from '../../src/utils/output.js';

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
