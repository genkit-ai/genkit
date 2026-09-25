/**
 * Copyright 2024 Google LLC
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

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

// Define mock function with 'mock' prefix so it can be used in jest.mock
const mockFindProjectRoot = jest.fn();

// Mock both potential resolution paths for utils
jest.mock('../../src/utils', () => {
  const actual = jest.requireActual('../../src/utils') as any;
  return {
    ...actual,
    findProjectRoot: mockFindProjectRoot,
  };
});

jest.mock('../../src/utils/utils', () => {
  const actual = jest.requireActual('../../src/utils/utils') as any;
  return {
    ...actual,
    findProjectRoot: mockFindProjectRoot,
  };
});

import {
  getProjectConfigStore,
  getProjectSettings,
  setProjectSettings,
} from '../../src/utils/configstore';

describe('configstore', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'configstore-test-'));
    (mockFindProjectRoot as any).mockResolvedValue(tempDir);
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
    jest.clearAllMocks();
  });

  describe('getProjectConfigStore', () => {
    it('creates .genkit directory if it does not exist', async () => {
      const dotGenkitDir = path.join(tempDir, '.genkit');
      expect(fs.existsSync(dotGenkitDir)).toBe(false);

      await getProjectConfigStore();

      expect(fs.existsSync(dotGenkitDir)).toBe(true);
    });

    it('returns configstore with correct path', async () => {
      const store = await getProjectConfigStore();
      expect(store.path).toBe(path.join(tempDir, '.genkit', 'genkit.json'));
    });
  });

  describe('getProjectSettings / setProjectSettings', () => {
    it('writes and reads settings', async () => {
      const settings = { runtimeCommand: 'node app.js' };
      await setProjectSettings(settings);

      const readSettings = await getProjectSettings();
      expect(readSettings).toEqual(settings);
    });

    it('returns empty object if no settings found', async () => {
      const readSettings = await getProjectSettings();
      expect(readSettings).toEqual({});
    });

    it('does not create .genkit directory on read if it does not exist', async () => {
      const dotGenkitDir = path.join(tempDir, '.genkit');
      expect(fs.existsSync(dotGenkitDir)).toBe(false);

      const settings = await getProjectSettings();

      expect(settings).toEqual({});
      expect(fs.existsSync(dotGenkitDir)).toBe(false);
    });
  });
});
