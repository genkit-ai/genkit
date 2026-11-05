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

import {
  findProjectRoot,
  findServersDir,
  isValidDevToolsInfo,
  logger,
  waitUntilHealthy,
  type DevToolsInfo,
} from '@genkit-ai/tools-common/utils';
import axios from 'axios';
import { spawn, type ChildProcess } from 'child_process';
import * as clc from 'colorette';
import * as fs from 'fs/promises';
import open from 'open';
import path from 'path';
import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mocked,
  type MockedFunction,
} from 'vitest';
import { uiStart } from '../../src/commands/ui-start';
import { detectCLIRuntime } from '../../src/utils/runtime-detector';
import {
  buildServerHarnessSpawnConfig,
  validateExecutablePath,
} from '../../src/utils/spawn-config';

// Mock all external dependencies
vi.mock('@genkit-ai/tools-common/utils');
vi.mock('axios');
vi.mock('child_process');
vi.mock('colorette');
vi.mock('fs/promises');
// Use real getPort - don't mock it
vi.mock('open');
vi.mock('../../src/utils/runtime-detector');
vi.mock('../../src/utils/spawn-config');

const mockedFindProjectRoot = findProjectRoot as MockedFunction<
  typeof findProjectRoot
>;
const mockedFindServersDir = findServersDir as MockedFunction<
  typeof findServersDir
>;
const mockedIsValidDevToolsInfo = isValidDevToolsInfo as MockedFunction<
  typeof isValidDevToolsInfo
>;
const mockedLogger = logger as Mocked<typeof logger>;
const mockedWaitUntilHealthy = waitUntilHealthy as MockedFunction<
  typeof waitUntilHealthy
>;
const mockedAxios = axios as Mocked<typeof axios>;
const mockedSpawn = spawn as MockedFunction<typeof spawn>;
const mockedClc = clc as Mocked<typeof clc>;
const mockedFs = fs as Mocked<typeof fs>;
const mockedOpen = open as MockedFunction<typeof open>;
const mockedDetectCLIRuntime = detectCLIRuntime as MockedFunction<
  typeof detectCLIRuntime
>;
const mockedBuildServerHarnessSpawnConfig =
  buildServerHarnessSpawnConfig as MockedFunction<
    typeof buildServerHarnessSpawnConfig
  >;
const mockedValidateExecutablePath = validateExecutablePath as MockedFunction<
  typeof validateExecutablePath
>;

describe('ui:start', () => {
  const createCommand = () =>
    uiStart.exitOverride().configureOutput({
      writeOut: () => {},
      writeErr: () => {},
    });

  const mockProjectRoot = '/mock/project/root';
  const mockServersDir = '/mock/project/root/.genkit/servers';
  const mockToolsJsonPath = path.join(mockServersDir, 'tools.json');
  const mockLogPath = path.join(mockServersDir, 'devui.log');

  const mockCLIRuntime = {
    type: 'node' as const,
    execPath: '/usr/bin/node',
    scriptPath: '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
    isCompiledBinary: false,
    platform: 'darwin' as const,
  };

  const mockSpawnConfig = {
    command: '/usr/bin/node',
    args: [
      '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
      'server-harness',
      '4000',
      mockLogPath,
    ],
    options: {
      stdio: ['ignore', 'ignore', 'ignore'] as ['ignore', 'ignore', 'ignore'],
      detached: false,
      shell: false,
    },
  };

  const mockChildProcess = {
    on: vi.fn().mockReturnThis(),
    unref: vi.fn(),
    stdin: null,
    stdout: null,
    stderr: null,
    stdio: [null, null, null],
    pid: 12345,
    connected: false,
    exitCode: null,
    signalCode: null,
    spawnargs: [],
    spawnfile: '',
    kill: vi.fn(),
    send: vi.fn(),
    disconnect: vi.fn(),
    ref: vi.fn(),
    addListener: vi.fn(),
    emit: vi.fn(),
    once: vi.fn(),
    prependListener: vi.fn(),
    prependOnceListener: vi.fn(),
    removeListener: vi.fn(),
    removeAllListeners: vi.fn(),
    setMaxListeners: vi.fn(),
    getMaxListeners: vi.fn(),
    listeners: vi.fn(),
    rawListeners: vi.fn(),
    listenerCount: vi.fn(),
    eventNames: vi.fn(),
    off: vi.fn(),
  } as unknown as ChildProcess;

  beforeEach(() => {
    vi.clearAllMocks();
    mockedFindProjectRoot.mockResolvedValue(mockProjectRoot);
    mockedFindServersDir.mockResolvedValue(mockServersDir);
    mockedDetectCLIRuntime.mockReturnValue(mockCLIRuntime);
    mockedBuildServerHarnessSpawnConfig.mockReturnValue(mockSpawnConfig);
    mockedValidateExecutablePath.mockResolvedValue(true);
    mockedSpawn.mockReturnValue(mockChildProcess as any);
    mockedWaitUntilHealthy.mockResolvedValue(true);
    mockedClc.green.mockImplementation((text) => `GREEN:${text}`);
  });

  describe('port validation', () => {
    it('should accept valid port number', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));

      await createCommand().parseAsync(['node', 'ui:start', '--port', '8080']);

      expect(mockedLogger.error).not.toHaveBeenCalled();
    });

    it('should reject invalid port number (NaN)', async () => {
      await createCommand().parseAsync([
        'node',
        'ui:start',
        '--port',
        'invalid',
      ]);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        '"invalid" is not a valid port number'
      );
    });

    it('should reject negative port number', async () => {
      await createCommand().parseAsync(['node', 'ui:start', '--port', '-1']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        '"-1" is not a valid port number'
      );
    });

    it('should accept zero port number', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedFs.mkdir.mockResolvedValue(undefined);
      mockedFs.writeFile.mockResolvedValue(undefined);

      await createCommand().parseAsync(['node', 'ui:start', '--port', '0']);

      expect(mockedLogger.error).not.toHaveBeenCalled();
      expect(mockedBuildServerHarnessSpawnConfig).toHaveBeenCalledWith(
        mockCLIRuntime,
        0,
        mockLogPath
      );
    });

    it('should use default port range when no port specified', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedFs.mkdir.mockResolvedValue(undefined);
      mockedFs.writeFile.mockResolvedValue(undefined);

      await createCommand().parseAsync(['node', 'ui:start']);
    });
  });

  describe('existing server detection', () => {
    it('should detect and report existing healthy server', async () => {
      const mockServerInfo: DevToolsInfo = {
        url: 'http://localhost:4000',
        timestamp: new Date().toISOString(),
      };

      mockedIsValidDevToolsInfo.mockReturnValue(true);
      mockedFs.readFile.mockResolvedValue(JSON.stringify(mockServerInfo));
      mockedAxios.get.mockResolvedValue({ status: 200 });

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.info).toHaveBeenCalledWith(
        expect.stringContaining(
          'Genkit Developer UI is already running at: http://localhost:4000'
        )
      );
      expect(mockedLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('To stop the UI, run `genkit ui:stop`')
      );
    });

    it('should start new server when existing server is unhealthy', async () => {
      const mockServerInfo: DevToolsInfo = {
        url: 'http://localhost:4000',
        timestamp: new Date().toISOString(),
      };

      mockedIsValidDevToolsInfo.mockReturnValue(true);
      mockedFs.readFile.mockResolvedValue(JSON.stringify(mockServerInfo));
      mockedAxios.get.mockRejectedValue(new Error('Connection refused'));

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.debug).toHaveBeenCalledWith(
        'Found UI server metadata but server is not healthy. Starting a new one...'
      );
      expect(mockedLogger.info).toHaveBeenCalledWith('Starting...');
    });

    it('should start new server when tools.json is invalid', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockResolvedValue('invalid json');

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.info).toHaveBeenCalledWith('Starting...');
    });

    it('should start new server when tools.json does not exist', async () => {
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.debug).toHaveBeenCalledWith(
        'No UI running. Starting a new one...'
      );
      expect(mockedLogger.info).toHaveBeenCalledWith('Starting...');
    });
  });

  describe('server startup', () => {
    beforeEach(() => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedFs.mkdir.mockResolvedValue(undefined);
      mockedFs.writeFile.mockResolvedValue(undefined);
    });

    it('should successfully start server and write metadata', async () => {
      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedDetectCLIRuntime).toHaveBeenCalled();

      const spawnConfigCall = mockedBuildServerHarnessSpawnConfig.mock.calls[0];
      const actualPort = spawnConfigCall[1];

      expect(mockedBuildServerHarnessSpawnConfig).toHaveBeenCalledWith(
        mockCLIRuntime,
        actualPort,
        mockLogPath
      );
      expect(mockedValidateExecutablePath).toHaveBeenCalledWith(
        mockSpawnConfig.command
      );
      expect(mockedSpawn).toHaveBeenCalledWith(
        mockSpawnConfig.command,
        mockSpawnConfig.args,
        mockSpawnConfig.options
      );
      expect(mockedWaitUntilHealthy).toHaveBeenCalledWith(
        `http://localhost:${actualPort}`,
        10000
      );
      expect(mockedFs.mkdir).toHaveBeenCalledWith(mockServersDir, {
        recursive: true,
      });
      expect(mockedFs.writeFile).toHaveBeenCalledWith(
        mockToolsJsonPath,
        expect.stringContaining(`"url": "http://localhost:${actualPort}"`)
      );
      expect(mockedLogger.info).toHaveBeenCalledWith(
        expect.stringContaining(
          `Genkit Developer UI started at: http://localhost:${actualPort}`
        )
      );
    });

    it('should open browser when --open flag is provided', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedFs.mkdir.mockResolvedValue(undefined);
      mockedFs.writeFile.mockResolvedValue(undefined);

      await createCommand().parseAsync(['node', 'ui:start', '--open']);

      const spawnConfigCall = mockedBuildServerHarnessSpawnConfig.mock.calls[0];
      const actualPort = spawnConfigCall[1];

      expect(mockedOpen).toHaveBeenCalledWith(`http://localhost:${actualPort}`);
    });

    it('should handle server startup failure', async () => {
      const startupError = new Error('Failed to start server');
      mockedWaitUntilHealthy.mockRejectedValue(startupError);

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Failed to start Genkit Developer UI')
      );
    });

    it('should handle executable validation failure', async () => {
      mockedValidateExecutablePath.mockResolvedValue(false);

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Failed to start Genkit Developer UI')
      );
    });

    it('should handle spawn process error', async () => {
      // Make waitUntilHealthy reject to simulate a failure
      mockedWaitUntilHealthy.mockRejectedValue(
        new Error('Health check failed')
      );

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Failed to start Genkit Developer UI')
      );
    });

    it('should handle spawn process exit', async () => {
      // Make waitUntilHealthy reject to simulate a failure
      mockedWaitUntilHealthy.mockRejectedValue(new Error('Process exited'));

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Failed to start Genkit Developer UI')
      );
    });

    it('should handle health check timeout', async () => {
      mockedWaitUntilHealthy.mockResolvedValue(false);

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Failed to start Genkit Developer UI')
      );
    });
  });

  describe('metadata file operations', () => {
    beforeEach(() => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
    });

    it('should handle metadata write failure gracefully', async () => {
      mockedFs.mkdir.mockRejectedValue(new Error('Permission denied'));

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        'Failed to write UI server metadata. UI server will continue to run.'
      );
      // Should still report success
      expect(mockedLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('Genkit Developer UI started at:')
      );
    });

    it('should write correct metadata format', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedFs.mkdir.mockResolvedValue(undefined);
      mockedFs.writeFile.mockResolvedValue(undefined);

      await createCommand().parseAsync(['node', 'ui:start']);

      const spawnConfigCall = mockedBuildServerHarnessSpawnConfig.mock.calls[0];
      const actualPort = spawnConfigCall[1];

      expect(mockedFs.writeFile).toHaveBeenCalledWith(
        mockToolsJsonPath,
        expect.stringMatching(
          new RegExp(
            `^\\{\\s*"url":\\s*"http://localhost:${actualPort}",\\s*"timestamp":\\s*"[^"]+"\\s*\\}$`
          )
        )
      );
    });
  });

  describe('runtime detection integration', () => {
    beforeEach(() => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
    });

    it('should handle different CLI runtime types', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedFs.mkdir.mockResolvedValue(undefined);
      mockedFs.writeFile.mockResolvedValue(undefined);

      const bunRuntime = {
        type: 'bun' as const,
        execPath: '/usr/local/bin/bun',
        scriptPath: '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
        isCompiledBinary: false,
        platform: 'darwin' as const,
      };

      mockedDetectCLIRuntime.mockReturnValue(bunRuntime);

      await createCommand().parseAsync(['node', 'ui:start']);

      const spawnConfigCall = mockedBuildServerHarnessSpawnConfig.mock.calls[0];
      const actualPort = spawnConfigCall[1];

      expect(mockedBuildServerHarnessSpawnConfig).toHaveBeenCalledWith(
        bunRuntime,
        actualPort,
        mockLogPath
      );
    });

    it('should handle compiled binary runtime', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedFs.mkdir.mockResolvedValue(undefined);
      mockedFs.writeFile.mockResolvedValue(undefined);

      const binaryRuntime = {
        type: 'compiled-binary' as const,
        execPath: '/usr/local/bin/genkit',
        scriptPath: undefined,
        isCompiledBinary: true,
        platform: 'linux' as const,
      };

      mockedDetectCLIRuntime.mockReturnValue(binaryRuntime);

      await createCommand().parseAsync(['node', 'ui:start']);

      const spawnConfigCall = mockedBuildServerHarnessSpawnConfig.mock.calls[0];
      const actualPort = spawnConfigCall[1];

      expect(mockedBuildServerHarnessSpawnConfig).toHaveBeenCalledWith(
        binaryRuntime,
        actualPort,
        mockLogPath
      );
    });
  });

  describe('error handling', () => {
    it('should handle findProjectRoot failure', async () => {
      mockedFindProjectRoot.mockRejectedValue(
        new Error('Project root not found')
      );

      await expect(
        createCommand().parseAsync(['node', 'ui:start'])
      ).rejects.toThrow();
    });

    it('should handle findServersDir failure', async () => {
      mockedFindServersDir.mockRejectedValue(
        new Error('Servers dir not found')
      );

      await expect(
        createCommand().parseAsync(['node', 'ui:start'])
      ).rejects.toThrow();
    });

    it('should handle runtime detection failure', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedDetectCLIRuntime.mockImplementation(() => {
        throw new Error('Runtime detection failed');
      });

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Failed to start Genkit Developer UI')
      );
    });

    it('should handle spawn config build failure', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedBuildServerHarnessSpawnConfig.mockImplementation(() => {
        throw new Error('Invalid spawn config');
      });

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Failed to start Genkit Developer UI')
      );
    });
  });

  describe('logging and debugging', () => {
    beforeEach(() => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
    });

    it('should log debug information for CLI runtime', async () => {
      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.debug).toHaveBeenCalledWith(
        'Detected CLI runtime: node at /usr/bin/node'
      );
      expect(mockedLogger.debug).toHaveBeenCalledWith(
        'Script path: /usr/lib/node_modules/genkit-cli/dist/bin/genkit.js'
      );
    });

    it('should log spawn command for debugging', async () => {
      await createCommand().parseAsync(['node', 'ui:start']);

      // The debug message should contain the spawn command and args
      expect(mockedLogger.debug).toHaveBeenCalledWith(
        expect.stringMatching(
          /^Spawning: \/usr\/bin\/node \/usr\/lib\/node_modules\/genkit-cli\/dist\/bin\/genkit\.js server-harness \d+ \/mock\/project\/root\/\.genkit\/servers\/devui\.log$/
        )
      );
    });

    it('should not log script path when undefined', async () => {
      const runtimeWithoutScript = { ...mockCLIRuntime, scriptPath: undefined };
      mockedDetectCLIRuntime.mockReturnValue(runtimeWithoutScript);

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.debug).not.toHaveBeenCalledWith(
        expect.stringContaining('Script path:')
      );
    });
  });

  describe('health check integration', () => {
    beforeEach(() => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
    });

    it('should check for existing actions after startup', async () => {
      mockedIsValidDevToolsInfo.mockReturnValue(false);
      mockedFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockedFs.mkdir.mockResolvedValue(undefined);
      mockedFs.writeFile.mockResolvedValue(undefined);

      await createCommand().parseAsync(['node', 'ui:start']);

      const spawnConfigCall = mockedBuildServerHarnessSpawnConfig.mock.calls[0];
      const actualPort = spawnConfigCall[1];

      expect(mockedAxios.get).toHaveBeenCalledWith(
        `http://localhost:${actualPort}/api/trpc/listActions`
      );
    });

    it('should show dev environment message when no actions found', async () => {
      mockedAxios.get.mockRejectedValue(new Error('No actions'));

      await createCommand().parseAsync(['node', 'ui:start']);

      expect(mockedLogger.info).toHaveBeenCalledWith(
        'Set env variable `GENKIT_ENV` to `dev` and start your app code to interact with it in the UI.'
      );
    });
  });
});
