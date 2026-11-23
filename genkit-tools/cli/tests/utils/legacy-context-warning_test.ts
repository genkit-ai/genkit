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
import { warnIfLegacyGenkitContext } from '../../src/utils/legacy-context-warning.js';

vi.mock('@genkit-ai/tools-common/utils');

const mockedLogger = logger as Mocked<typeof logger>;

describe('warnIfLegacyGenkitContext', () => {
  let tempDir: string;

  beforeEach(async () => {
    vi.resetAllMocks();
    tempDir = await fs.mkdtemp(
      path.join(os.tmpdir(), 'genkit-legacy-context-warning-')
    );
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  it('warns without modifying GENKIT.md when legacy bad instructions are present', async () => {
    const genkitMdPath = path.join(tempDir, 'GENKIT.md');
    const content =
      '# Genkit Node.js API Rules\n- Do not run Genkit commands (e.g., `genkit start`) youself.\n';
    await fs.writeFile(genkitMdPath, content, 'utf-8');

    await warnIfLegacyGenkitContext(tempDir);

    expect(mockedLogger.warn).toHaveBeenCalledTimes(1);
    expect(mockedLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('npx skills add genkit-ai/skills')
    );
    expect(mockedLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('delete `GENKIT.md`')
    );
    expect(await fs.readFile(genkitMdPath, 'utf-8')).toBe(content);
  });

  it('does not warn when GENKIT.md does not exist', async () => {
    await warnIfLegacyGenkitContext(tempDir);

    expect(mockedLogger.warn).not.toHaveBeenCalled();
  });

  it('does not warn when GENKIT.md does not contain legacy instructions', async () => {
    await fs.writeFile(
      path.join(tempDir, 'GENKIT.md'),
      '# Custom Project Notes\nUse genkit start during local dev.\n',
      'utf-8'
    );

    await warnIfLegacyGenkitContext(tempDir);

    expect(mockedLogger.warn).not.toHaveBeenCalled();
  });

  it('does not throw if GENKIT.md cannot be read as a file', async () => {
    await fs.mkdir(path.join(tempDir, 'GENKIT.md'));

    await expect(warnIfLegacyGenkitContext(tempDir)).resolves.toBeUndefined();
    expect(mockedLogger.warn).not.toHaveBeenCalled();
  });
});
