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

import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import * as fs from 'fs/promises';
import { SERVER_HARNESS_COMMAND } from '../../src/commands/server-harness';
import { type CLIRuntimeInfo } from '../../src/utils/runtime-detector';
import {
  buildServerHarnessSpawnConfig,
  validateExecutablePath,
} from '../../src/utils/spawn-config';

jest.mock('fs/promises');
const mockedFs = fs as jest.Mocked<typeof fs>;

describe('spawn-config', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('buildServerHarnessSpawnConfig', () => {
    const mockPort = 4000;
    const mockLogPath = '/path/to/devui.log';

    describe('Node.js CLI runtime', () => {
      it('should build config for Node.js with script path', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'node',
          execPath: '/usr/bin/node',
          scriptPath: '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
          isCompiledBinary: false,
          platform: 'darwin',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('/usr/bin/node');
        expect(config.args).toEqual([
          '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
          SERVER_HARNESS_COMMAND,
          '4000',
          '/path/to/devui.log',
        ]);
        expect(config.options.stdio).toEqual(['ignore', 'ignore', 'ignore']);
        expect(config.options.detached).toBe(false);
        expect(config.options.shell).toBe(false);
      });

      it('should build config for Node.js on Windows', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'node',
          execPath: 'C:\\Program Files\\nodejs\\node.exe',
          scriptPath:
            'C:\\Users\\dev\\AppData\\Roaming\\npm\\node_modules\\genkit-cli\\dist\\bin\\genkit.js',
          isCompiledBinary: false,
          platform: 'win32',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('"C:\\Program Files\\nodejs\\node.exe"');
        expect(config.args).toEqual([
          '"C:\\Users\\dev\\AppData\\Roaming\\npm\\node_modules\\genkit-cli\\dist\\bin\\genkit.js"',
          '"' + SERVER_HARNESS_COMMAND + '"',
          '"4000"',
          '"/path/to/devui.log"',
        ]);
        expect(config.options.shell).toBe(true); // Shell enabled on Windows
      });

      it('should handle Node.js without script path', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'node',
          execPath: '/usr/bin/node',
          scriptPath: undefined,
          isCompiledBinary: false,
          platform: 'linux',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('/usr/bin/node');
        expect(config.args).toEqual([
          SERVER_HARNESS_COMMAND,
          '4000',
          '/path/to/devui.log',
        ]);
        expect(config.options.shell).toBe(false);
      });
    });

    describe('Bun CLI runtime', () => {
      it('should build config for Bun with script path', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'bun',
          execPath: '/usr/local/bin/bun',
          scriptPath: '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
          isCompiledBinary: false,
          platform: 'darwin',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('/usr/local/bin/bun');
        expect(config.args).toEqual([
          '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
          SERVER_HARNESS_COMMAND,
          '4000',
          '/path/to/devui.log',
        ]);
        expect(config.options.stdio).toEqual(['ignore', 'ignore', 'ignore']);
        expect(config.options.detached).toBe(false);
        expect(config.options.shell).toBe(false);
      });

      it('should handle Bun without script path', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'bun',
          execPath: '/opt/homebrew/bin/bun',
          scriptPath: undefined,
          isCompiledBinary: false,
          platform: 'darwin',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('/opt/homebrew/bin/bun');
        expect(config.args).toEqual([
          SERVER_HARNESS_COMMAND,
          '4000',
          '/path/to/devui.log',
        ]);
      });

      it('should handle Bun on Windows', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'bun',
          execPath: 'C:\\Program Files\\Bun\\bun.exe',
          scriptPath: 'C:\\projects\\genkit\\dist\\bin\\genkit.js',
          isCompiledBinary: false,
          platform: 'win32',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('"C:\\Program Files\\Bun\\bun.exe"');
        expect(config.options.shell).toBe(true);
      });
    });

    describe('Compiled binary', () => {
      it('should build config for compiled binary', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'compiled-binary',
          execPath: '/usr/local/bin/genkit',
          scriptPath: undefined,
          isCompiledBinary: true,
          platform: 'linux',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('/usr/local/bin/genkit');
        expect(config.args).toEqual([
          SERVER_HARNESS_COMMAND,
          '4000',
          '/path/to/devui.log',
        ]);
        expect(config.options.stdio).toEqual(['ignore', 'ignore', 'ignore']);
        expect(config.options.detached).toBe(false);
        expect(config.options.shell).toBe(false);
      });

      it('should handle compiled binary on Windows', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'compiled-binary',
          execPath: 'C:\\Tools\\genkit.exe',
          scriptPath: undefined,
          isCompiledBinary: true,
          platform: 'win32',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('"C:\\Tools\\genkit.exe"');
        expect(config.options.shell).toBe(true);
      });

      it('should handle relative path compiled binary', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'compiled-binary',
          execPath: './genkit',
          scriptPath: undefined,
          isCompiledBinary: true,
          platform: 'darwin',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.command).toBe('./genkit');
        expect(config.args).toEqual([
          SERVER_HARNESS_COMMAND,
          '4000',
          '/path/to/devui.log',
        ]);
      });
    });

    describe('Edge cases', () => {
      it('should handle different port numbers', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'node',
          execPath: '/usr/bin/node',
          scriptPath: '/script.js',
          isCompiledBinary: false,
          platform: 'linux',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          8080,
          mockLogPath
        );

        expect(config.args).toContain('8080');
      });

      it('should handle paths with spaces', () => {
        const cliRuntime: CLIRuntimeInfo = {
          type: 'node',
          execPath: '/usr/bin/node',
          scriptPath: '/path with spaces/script.js',
          isCompiledBinary: false,
          platform: 'darwin',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          '/log path/devui.log'
        );

        expect(config.args).toEqual([
          '/path with spaces/script.js',
          SERVER_HARNESS_COMMAND,
          '4000',
          '/log path/devui.log',
        ]);
      });

      it('should handle very long paths', () => {
        const longPath = '/very/long/path/'.repeat(50) + 'script.js';
        const cliRuntime: CLIRuntimeInfo = {
          type: 'node',
          execPath: '/usr/bin/node',
          scriptPath: longPath,
          isCompiledBinary: false,
          platform: 'linux',
        };

        const config = buildServerHarnessSpawnConfig(
          cliRuntime,
          mockPort,
          mockLogPath
        );

        expect(config.args[0]).toBe(longPath);
      });
    });

    describe('Input validation', () => {
      const validRuntime: CLIRuntimeInfo = {
        type: 'node',
        execPath: '/usr/bin/node',
        scriptPath: '/script.js',
        isCompiledBinary: false,
        platform: 'linux',
      };

      it('should throw error for null runtime', () => {
        expect(() =>
          buildServerHarnessSpawnConfig(null as any, mockPort, mockLogPath)
        ).toThrow('CLI runtime info is required');
      });

      it('should throw error for runtime without execPath', () => {
        const invalidRuntime = { ...validRuntime, execPath: '' };
        expect(() =>
          buildServerHarnessSpawnConfig(
            invalidRuntime as any,
            mockPort,
            mockLogPath
          )
        ).toThrow('CLI runtime execPath is required');
      });

      it('should throw error for invalid port numbers', () => {
        expect(() =>
          buildServerHarnessSpawnConfig(validRuntime, -1, mockLogPath)
        ).toThrow('Invalid port number: -1. Must be between 0 and 65535');

        expect(() =>
          buildServerHarnessSpawnConfig(validRuntime, 65536, mockLogPath)
        ).toThrow('Invalid port number: 65536. Must be between 0 and 65535');

        expect(() =>
          buildServerHarnessSpawnConfig(validRuntime, 3.14, mockLogPath)
        ).toThrow('Invalid port number: 3.14. Must be between 0 and 65535');

        expect(() =>
          buildServerHarnessSpawnConfig(validRuntime, NaN, mockLogPath)
        ).toThrow('Invalid port number: NaN. Must be between 0 and 65535');
      });

      it('should throw error for empty log path', () => {
        expect(() =>
          buildServerHarnessSpawnConfig(validRuntime, mockPort, '')
        ).toThrow('Log path is required');

        expect(() =>
          buildServerHarnessSpawnConfig(validRuntime, mockPort, null as any)
        ).toThrow('Log path is required');
      });

      it('should accept valid edge case ports', () => {
        expect(() =>
          buildServerHarnessSpawnConfig(validRuntime, 0, mockLogPath)
        ).not.toThrow();

        expect(() =>
          buildServerHarnessSpawnConfig(validRuntime, 65535, mockLogPath)
        ).not.toThrow();
      });
    });
  });

  describe('validateExecutablePath', () => {
    it('should return true for valid executable path', async () => {
      mockedFs.access.mockResolvedValue(undefined);

      const result = await validateExecutablePath('/usr/bin/node');

      expect(result).toBe(true);
      expect(mockedFs.access).toHaveBeenCalledWith(
        '/usr/bin/node',
        fs.constants.F_OK | fs.constants.X_OK
      );
    });

    it('should return false for non-existent path', async () => {
      mockedFs.access.mockRejectedValue(new Error('ENOENT'));

      const result = await validateExecutablePath('/nonexistent/path');

      expect(result).toBe(false);
    });

    it('should return false for non-executable file', async () => {
      mockedFs.access.mockRejectedValue(new Error('EACCES'));

      const result = await validateExecutablePath('/path/to/non-executable');

      expect(result).toBe(false);
    });

    it('should handle Windows paths', async () => {
      mockedFs.access.mockResolvedValue(undefined);

      const result = await validateExecutablePath(
        'C:\\Windows\\System32\\cmd.exe'
      );

      expect(result).toBe(true);
      expect(mockedFs.access).toHaveBeenCalledWith(
        'C:\\Windows\\System32\\cmd.exe',
        fs.constants.F_OK | fs.constants.X_OK
      );
    });
  });
});
