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

import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { config } from '../../src/commands/config';

// Mock dependencies
jest.mock('@genkit-ai/tools-common/utils', () => {
  const actual = jest.requireActual('@genkit-ai/tools-common/utils') as any;
  return {
    ...actual,
    getProjectSettings: jest.fn(),
    setProjectSettings: jest.fn(),
    getUserSettings: jest.fn(),
    setUserSettings: jest.fn(),
    logger: {
      info: jest.fn(),
      error: jest.fn(),
      debug: jest.fn(),
      warn: jest.fn(),
    },
    record: jest.fn() as any,
  };
});

import {
  getProjectSettings,
  logger,
  setProjectSettings,
} from '@genkit-ai/tools-common/utils';

describe('config command', () => {
  let command: any;

  beforeEach(() => {
    command = config.exitOverride().configureOutput({
      writeOut: () => {},
      writeErr: () => {},
    });
    jest.clearAllMocks();
  });

  describe('set', () => {
    it('writes runtimeCommand to project config', async () => {
      (getProjectSettings as any).mockResolvedValue({});

      await command.parseAsync([
        'node',
        'config',
        'set',
        'runtimeCommand',
        'node app.js',
      ]);

      expect(setProjectSettings).toHaveBeenCalledWith({
        runtimeCommand: 'node app.js',
      });
      expect(logger.info).toHaveBeenCalled();
      // Use lenient regex to avoid issues with invisible ANSI color codes or escaping in Jest output
      expect((logger.info as any).mock.calls[0][0]).toMatch(
        /Set.*runtimeCommand.*to.*node app.js/
      );
    });

    it('unsets runtimeCommand when passing empty string', async () => {
      (getProjectSettings as any).mockResolvedValue({
        runtimeCommand: 'node app.js',
      });

      await command.parseAsync(['node', 'config', 'set', 'runtimeCommand', '']);

      expect(setProjectSettings).toHaveBeenCalledWith({});
      expect(logger.info).toHaveBeenCalled();
      // Use lenient regex to avoid issues with invisible ANSI color codes or escaping in Jest output
      expect((logger.info as any).mock.calls[0][0]).toMatch(
        /Unset.*runtimeCommand/
      );
    });
  });

  describe('get', () => {
    it('reads runtimeCommand from project config', async () => {
      (getProjectSettings as any).mockResolvedValue({
        runtimeCommand: 'node app.js',
      });

      await command.parseAsync(['node', 'config', 'get', 'runtimeCommand']);

      expect(logger.info).toHaveBeenCalledWith('node app.js');
    });

    it('shows (unset) if not set', async () => {
      (getProjectSettings as any).mockResolvedValue({});

      await command.parseAsync(['node', 'config', 'get', 'runtimeCommand']);

      expect(logger.info).toHaveBeenCalledWith(
        expect.stringContaining('(unset)')
      );
    });
  });
});
