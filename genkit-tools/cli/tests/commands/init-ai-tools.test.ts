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

import { logger } from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mocked,
} from 'vitest';
import { initAiTools } from '../../src/commands/init-ai-tools.js';

vi.mock('@genkit-ai/tools-common/utils');

const mockedLogger = logger as Mocked<typeof logger>;

describe('init:ai-tools (deprecated)', () => {
  let tempDir: string;
  let originalCwd: string;

  beforeEach(async () => {
    vi.resetAllMocks();
    originalCwd = process.cwd();
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'genkit-init-ai-tools-'));
    process.chdir(tempDir);
  });

  afterEach(async () => {
    process.chdir(originalCwd);
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  it('logs a deprecation warning and does not write any files', async () => {
    await initAiTools.parseAsync(['node', 'genkit']);

    expect(mockedLogger.warn).toHaveBeenCalledTimes(1);
    expect(mockedLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('`genkit init:ai-tools` is deprecated')
    );
    expect(mockedLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('npx skills add genkit-ai/skills')
    );
    expect(await fs.readdir(tempDir)).toEqual([]);
  });

  it('accepts the legacy -y/--yes flag without error', async () => {
    await initAiTools.parseAsync(['node', 'genkit', '--yes']);

    expect(mockedLogger.warn).toHaveBeenCalledTimes(1);
    expect(await fs.readdir(tempDir)).toEqual([]);
  });

  it('is omitted from top-level help when registered as hidden', () => {
    const program = new Command('genkit');
    program.addCommand(initAiTools, { hidden: true });

    expect(program.helpInformation()).not.toContain('init:ai-tools');
  });
});
