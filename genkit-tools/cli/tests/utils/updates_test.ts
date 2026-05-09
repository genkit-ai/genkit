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
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';

// Mock dependencies before importing the module
const mockAxiosGet = jest.fn() as jest.MockedFunction<any>;
const mockGetUserSettings = jest.fn();
const mockLoggerDebug = jest.fn();
const mockLoggerInfo = jest.fn();
const mockDetectCLIRuntime = jest.fn();

jest.mock('axios', () => ({
  create: jest.fn(() => ({ get: mockAxiosGet })),
}));

jest.mock('@genkit-ai/tools-common/utils', () => ({
  getUserSettings: mockGetUserSettings,
  logger: {
    debug: mockLoggerDebug,
    info: mockLoggerInfo,
  },
}));

jest.mock('../../src/utils/runtime-detector', () => ({
  detectCLIRuntime: mockDetectCLIRuntime,
}));

jest.mock('../../src/utils/version', () => ({
  version: '1.15.0',
  name: 'genkit-cli',
}));

jest.mock('colorette', () => ({
  yellow: (text: string) => text,
  green: (text: string) => text,
  bold: (text: string) => text,
  dim: (text: string) => text,
  gray: (text: string) => text,
  cyan: (text: string) => text,
}));

jest.mock('os', () => ({
  ...(jest.requireActual('os') as any),
  arch: () => 'x64',
  platform: () => 'linux',
}));

// Now import the functions we want to test
import {
  getCurrentVersion,
  getLatestVersionFromNpm,
  showUpdateNotification,
} from '../../src/utils/updates';

describe('updates', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();

    // Reset process.env
    process.env = { ...originalEnv };
    delete process.env.GENKIT_CLI_DISABLE_UPDATE_NOTIFICATIONS;

    // Set default mock return values
    mockGetUserSettings.mockReturnValue({});
    mockDetectCLIRuntime.mockReturnValue({
      type: 'node',
      execPath: '/usr/bin/node',
      scriptPath: '/usr/bin/genkit',
      isCompiledBinary: false,
      platform: 'linux',
    });
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe('getCurrentVersion', () => {
    it('should return normalized current version', () => {
      const version = getCurrentVersion();
      expect(version).toBe('1.15.0');
    });
  });

  describe('getLatestVersionFromNpm', () => {
    it('should fetch latest version from npm registry', async () => {
      const mockResponse = {
        status: 200,
        data: {
          'dist-tags': {
            latest: '1.16.0',
          },
          versions: {
            '1.16.0': {},
            '1.15.0': {},
          },
        },
      };

      mockAxiosGet.mockResolvedValueOnce(mockResponse);

      const result = await getLatestVersionFromNpm();

      expect(result).toBe('1.16.0');
      expect(mockAxiosGet).toHaveBeenCalledWith(
        'https://registry.npmjs.org/genkit-cli'
      );
    });

    it('should ignore prerelease versions when ignoreRC is true', async () => {
      const mockResponse = {
        status: 200,
        data: {
          'dist-tags': {
            latest: '1.16.0-rc.1',
          },
          versions: {
            '1.16.0-rc.1': {},
            '1.15.0': {},
            '1.14.0': {},
          },
        },
      };

      mockAxiosGet.mockResolvedValueOnce(mockResponse);

      const result = await getLatestVersionFromNpm(true);

      expect(result).toBe('1.15.0');
    });

    it('should include prerelease versions when ignoreRC is false', async () => {
      const mockResponse = {
        status: 200,
        data: {
          'dist-tags': {
            latest: '1.16.0-rc.1',
          },
          versions: {
            '1.16.0-rc.1': {},
            '1.15.0': {},
          },
        },
      };

      mockAxiosGet.mockResolvedValueOnce(mockResponse);

      const result = await getLatestVersionFromNpm(false);

      expect(result).toBe('1.16.0-rc.1');
    });

    it('should return null when no valid versions found', async () => {
      const mockResponse = {
        status: 200,
        data: {
          'dist-tags': {},
          versions: {
            'invalid-version': {},
            'another-invalid': {},
          },
        },
      };

      mockAxiosGet.mockResolvedValueOnce(mockResponse);

      const result = await getLatestVersionFromNpm();

      expect(result).toBeNull();
    });

    it('should handle network errors', async () => {
      const networkError = new Error('Network error');
      mockAxiosGet.mockRejectedValueOnce(networkError);

      await expect(getLatestVersionFromNpm()).rejects.toThrow(
        'Failed to fetch npm versions: Network error'
      );
    });

    it('should handle HTTP error status codes', async () => {
      const mockResponse = {
        status: 404,
        statusText: 'Not Found',
      };

      mockAxiosGet.mockResolvedValueOnce(mockResponse);

      await expect(getLatestVersionFromNpm()).rejects.toThrow(
        'Failed to fetch npm versions: Not Found'
      );
    });
  });

  describe('showUpdateNotification', () => {
    it('should not show notification when disabled via environment variable', async () => {
      process.env.GENKIT_CLI_DISABLE_UPDATE_NOTIFICATIONS = 'true';

      await showUpdateNotification();

      expect(mockLoggerInfo).not.toHaveBeenCalled();
      expect(mockAxiosGet).not.toHaveBeenCalled();
    });

    it('should not show notification when disabled via user config', async () => {
      mockGetUserSettings.mockReturnValueOnce({
        updateNotificationsOptOut: true,
      });

      await showUpdateNotification();

      expect(mockLoggerInfo).not.toHaveBeenCalled();
      expect(mockAxiosGet).not.toHaveBeenCalled();
    });

    it('should show notification for npm installation when update available', async () => {
      const mockNpmResponse = {
        status: 200,
        data: {
          'dist-tags': {
            latest: '1.16.0',
          },
          versions: {
            '1.16.0': {},
            '1.15.0': {},
          },
        },
      };

      mockAxiosGet.mockResolvedValueOnce(mockNpmResponse);

      await showUpdateNotification();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        expect.stringContaining('Update available')
      );
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        expect.stringContaining('v1.15.0 → v1.16.0')
      );
    });

    it('should not show notification when no update available', async () => {
      const mockNpmResponse = {
        status: 200,
        data: {
          'dist-tags': {
            latest: '1.15.0', // Same as current version
          },
          versions: {
            '1.15.0': {},
            '1.14.0': {},
          },
        },
      };

      mockAxiosGet.mockResolvedValueOnce(mockNpmResponse);

      await showUpdateNotification();

      expect(mockLoggerInfo).not.toHaveBeenCalled();
    });

    it('should handle network errors gracefully without throwing', async () => {
      const networkError = new Error('Network error');
      mockAxiosGet.mockRejectedValueOnce(networkError);

      // Should not throw
      await expect(showUpdateNotification()).resolves.toBeUndefined();

      expect(mockLoggerDebug).toHaveBeenCalledWith(
        'Failed to show update notification',
        expect.any(Error)
      );
    });
  });
});
