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

import { startServer } from '@genkit-ai/tools-common/server';
import { logger } from '@genkit-ai/tools-common/utils';
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import fs from 'fs';
import path from 'path';
import { start } from '../../src/commands/start';
import * as managerUtils from '../../src/utils/manager-utils';

jest.mock('@genkit-ai/tools-common/server');
jest.mock('@genkit-ai/tools-common/utils', () => ({
  findProjectRoot: jest.fn(() => Promise.resolve('/mock/root')),
  logger: {
    warn: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  },
}));
jest.mock('get-port', () => ({
  __esModule: true,
  default: jest.fn(() => Promise.resolve(4000)),
  makeRange: jest.fn(),
}));
jest.mock('open');

describe('start command', () => {
  let startDevProcessManagerSpy: any;
  let startManagerSpy: any;
  let startServerSpy: any;

  beforeEach(() => {
    jest.spyOn(managerUtils, 'getDevEnvVars').mockResolvedValue({
      envVars: {},
      telemetryServerUrl: 'http://localhost:4033',
      reflectionV2Port: 3200,
    });
    startDevProcessManagerSpy = jest
      .spyOn(managerUtils, 'startDevProcessManager')
      .mockResolvedValue({
        manager: {} as any,
        processPromise: Promise.resolve(),
      });
    startManagerSpy = jest
      .spyOn(managerUtils, 'startManager')
      .mockResolvedValue({} as any);
    startServerSpy = startServer as unknown as jest.Mock;

    // Reset args
    start.args = [];
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should start dev process manager when args are provided', async () => {
    await start.parseAsync(['node', 'genkit', 'run', 'app']);

    expect(startDevProcessManagerSpy).toHaveBeenCalledWith(
      '/mock/root',
      'run',
      ['app'],
      expect.objectContaining({ disableRealtimeTelemetry: undefined })
    );
    expect(startServerSpy).toHaveBeenCalled();
  });

  it('should start manager only when no args are provided', async () => {
    let resolveServerStarted: () => void;
    const serverStartedPromise = new Promise<void>((resolve) => {
      resolveServerStarted = resolve;
    });

    startServerSpy.mockImplementation(() => {
      resolveServerStarted();
    });

    start.parseAsync(['node', 'genkit']);

    await serverStartedPromise;

    expect(startManagerSpy).toHaveBeenCalledWith({
      projectRoot: '/mock/root',
      manageHealth: true,
      corsOrigin: undefined,
      experimentalReflectionV2: undefined,
      reflectionV2Port: 3200,
      telemetryServerUrl: 'http://localhost:4033',
    });
    expect(startDevProcessManagerSpy).not.toHaveBeenCalled();
    expect(startServerSpy).toHaveBeenCalled();
  });

  it('should not start server if --noui is provided', async () => {
    let resolveManagerStarted: () => void;
    const managerStartedPromise = new Promise<void>((resolve) => {
      resolveManagerStarted = resolve;
    });

    startManagerSpy.mockImplementation(() => {
      resolveManagerStarted();
      return Promise.resolve({} as any);
    });

    start.parseAsync(['node', 'genkit', '--noui']);

    await managerStartedPromise;
    // Wait for the synchronous continuation after startManager resolves
    await new Promise((resolve) => setImmediate(resolve));

    expect(startServerSpy).not.toHaveBeenCalled();
  });

  it('should pass disableRealtimeTelemetry option', async () => {
    await start.parseAsync([
      'node',
      'genkit',
      'run',
      'app',
      '--disable-realtime-telemetry',
    ]);

    expect(startDevProcessManagerSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.anything(),
      expect.objectContaining({ disableRealtimeTelemetry: true })
    );
  });

  it('should chdir when -C/--directory is provided', async () => {
    const dir = '/some/project';
    const statSpy = jest.spyOn(fs, 'statSync').mockReturnValue({
      isDirectory: () => true,
    } as fs.Stats);
    const chdirSpy = jest
      .spyOn(process, 'chdir')
      .mockImplementation(() => undefined as never);

    try {
      await start.parseAsync(['node', 'genkit', '-C', dir, 'run', 'app']);
      expect(chdirSpy).toHaveBeenCalledWith(path.resolve(dir));
      expect(startDevProcessManagerSpy).toHaveBeenCalledWith(
        '/mock/root',
        'run',
        ['app'],
        expect.anything()
      );
    } finally {
      statSpy.mockRestore();
      chdirSpy.mockRestore();
    }
  });

  it('should error when -C directory does not exist', async () => {
    const statSpy = jest.spyOn(fs, 'statSync').mockImplementation(() => {
      throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
    });
    const chdirSpy = jest.spyOn(process, 'chdir');

    try {
      await start.parseAsync(['node', 'genkit', '-C', '/missing', '--noui']);
      expect(logger.error).toHaveBeenCalledWith('Directory not found: /missing');
      expect(chdirSpy).not.toHaveBeenCalled();
      expect(startManagerSpy).not.toHaveBeenCalled();
      expect(startDevProcessManagerSpy).not.toHaveBeenCalled();
    } finally {
      statSpy.mockRestore();
      chdirSpy.mockRestore();
    }
  });

  it('should error when -C path is not a directory', async () => {
    const statSpy = jest.spyOn(fs, 'statSync').mockReturnValue({
      isDirectory: () => false,
    } as fs.Stats);
    const chdirSpy = jest.spyOn(process, 'chdir');

    try {
      await start.parseAsync([
        'node',
        'genkit',
        '-C',
        '/tmp/file.txt',
        '--noui',
      ]);
      expect(logger.error).toHaveBeenCalledWith(
        '"/tmp/file.txt" is not a directory'
      );
      expect(chdirSpy).not.toHaveBeenCalled();
      expect(startManagerSpy).not.toHaveBeenCalled();
    } finally {
      statSpy.mockRestore();
      chdirSpy.mockRestore();
    }
  });
});
