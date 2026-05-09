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
import * as fs from 'fs';
import { detectCLIRuntime } from '../../src/utils/runtime-detector';

jest.mock('fs');
const mockedFs = fs as jest.Mocked<typeof fs>;

describe('runtime-detector', () => {
  const originalArgv = process.argv;
  const originalExecPath = process.execPath;
  const originalVersions = process.versions;
  const originalPlatform = process.platform;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    process.argv = originalArgv;
    process.execPath = originalExecPath;
    Object.defineProperty(process, 'versions', {
      value: originalVersions,
      writable: true,
      configurable: true,
    });
    Object.defineProperty(process, 'platform', {
      value: originalPlatform,
      writable: true,
      configurable: true,
    });
  });

  describe('Node.js CLI runtime detection', () => {
    it('should detect Node.js CLI runtime with npm global install', () => {
      process.argv = [
        '/usr/bin/node',
        '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
      ];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(true);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBe(
        '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js'
      );
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should detect Node.js CLI runtime with npm link', () => {
      process.argv = [
        '/usr/local/bin/node',
        '/Users/dev/project/node_modules/.bin/genkit',
      ];
      process.execPath = '/usr/local/bin/node';
      mockedFs.existsSync.mockReturnValue(true);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/local/bin/node');
      expect(result.scriptPath).toBe(
        '/Users/dev/project/node_modules/.bin/genkit'
      );
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should detect Node.js CLI runtime with direct execution', () => {
      process.argv = ['node', './dist/bin/genkit.js'];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(true);
      Object.defineProperty(process, 'versions', {
        value: { ...originalVersions, node: '18.0.0' },
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBe('./dist/bin/genkit.js');
      expect(result.isCompiledBinary).toBe(false);
    });
  });

  describe('Bun CLI runtime detection', () => {
    it('should detect Bun CLI runtime via process.versions', () => {
      process.argv = [
        '/usr/local/bin/bun',
        '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js',
      ];
      process.execPath = '/usr/local/bin/bun';
      mockedFs.existsSync.mockReturnValue(true);
      Object.defineProperty(process, 'versions', {
        value: { ...originalVersions, bun: '1.0.0' },
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('bun');
      expect(result.execPath).toBe('/usr/local/bin/bun');
      expect(result.scriptPath).toBe(
        '/usr/lib/node_modules/genkit-cli/dist/bin/genkit.js'
      );
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should detect Bun CLI runtime via execPath', () => {
      process.argv = ['/opt/homebrew/bin/bun', './genkit.js'];
      process.execPath = '/opt/homebrew/bin/bun';
      mockedFs.existsSync.mockReturnValue(true);
      // No bun in process.versions
      Object.defineProperty(process, 'versions', {
        value: { ...originalVersions },
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('bun');
      expect(result.execPath).toBe('/opt/homebrew/bin/bun');
      expect(result.scriptPath).toBe('./genkit.js');
      expect(result.isCompiledBinary).toBe(false);
    });
  });

  describe('Compiled binary detection', () => {
    it('should detect compiled binary when argv[1] is undefined', () => {
      process.argv = ['/usr/local/bin/genkit'];
      process.execPath = '/usr/local/bin/genkit';
      mockedFs.existsSync.mockReturnValue(false);
      Object.defineProperty(process, 'versions', {
        value: {},
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('compiled-binary');
      expect(result.execPath).toBe('/usr/local/bin/genkit');
      expect(result.scriptPath).toBeUndefined();
      expect(result.isCompiledBinary).toBe(true);
    });

    it('should detect compiled binary when argv[1] does not exist', () => {
      process.argv = ['/usr/local/bin/genkit', 'nonexistent.js'];
      process.execPath = '/usr/local/bin/genkit';
      mockedFs.existsSync.mockReturnValue(false);
      Object.defineProperty(process, 'versions', {
        value: {},
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('compiled-binary');
      expect(result.execPath).toBe('/usr/local/bin/genkit');
      expect(result.scriptPath).toBeUndefined();
      expect(result.isCompiledBinary).toBe(true);
    });

    it('should detect Bun-compiled binary', () => {
      process.argv = ['./genkit'];
      process.execPath = './genkit';
      mockedFs.existsSync.mockReturnValue(false);
      Object.defineProperty(process, 'versions', {
        value: {},
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('compiled-binary');
      expect(result.execPath).toBe('./genkit');
      expect(result.scriptPath).toBeUndefined();
      expect(result.isCompiledBinary).toBe(true);
    });
  });

  describe('Edge cases and fallback behavior', () => {
    it('should throw error when execPath is missing', () => {
      const originalExecPath = process.execPath;

      Object.defineProperty(process, 'execPath', {
        value: '',
        writable: true,
        configurable: true,
      });

      expect(() => detectCLIRuntime()).toThrow(
        'Unable to determine CLI runtime executable path'
      );

      Object.defineProperty(process, 'execPath', {
        value: originalExecPath,
        writable: true,
        configurable: true,
      });
    });

    it('should throw error when execPath is whitespace only', () => {
      const originalExecPath = process.execPath;

      Object.defineProperty(process, 'execPath', {
        value: '   ',
        writable: true,
        configurable: true,
      });

      expect(() => detectCLIRuntime()).toThrow(
        'Unable to determine CLI runtime executable path'
      );

      Object.defineProperty(process, 'execPath', {
        value: originalExecPath,
        writable: true,
        configurable: true,
      });
    });

    it('should fall back to node when CLI runtime cannot be determined', () => {
      process.argv = ['/some/unknown/runtime', '/some/script.js'];
      process.execPath = '/some/unknown/runtime';
      mockedFs.existsSync.mockReturnValue(true);
      Object.defineProperty(process, 'versions', {
        value: {},
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/some/unknown/runtime');
      expect(result.scriptPath).toBe('/some/script.js');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle Windows paths correctly', () => {
      process.argv = [
        'C:\\Program Files\\nodejs\\node.exe',
        'C:\\Users\\dev\\genkit\\dist\\cli.js',
      ];
      process.execPath = 'C:\\Program Files\\nodejs\\node.exe';
      mockedFs.existsSync.mockReturnValue(true);
      Object.defineProperty(process, 'platform', {
        value: 'win32',
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('C:\\Program Files\\nodejs\\node.exe');
      expect(result.scriptPath).toBe('C:\\Users\\dev\\genkit\\dist\\cli.js');
      expect(result.isCompiledBinary).toBe(false);
      expect(result.platform).toBe('win32');
    });

    it('should detect platform correctly', () => {
      process.argv = ['/usr/bin/node', '/usr/bin/genkit'];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(true);
      Object.defineProperty(process, 'platform', {
        value: 'darwin',
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.platform).toBe('darwin');
    });
  });

  describe('Additional edge cases', () => {
    it('should handle empty argv array with node executable', () => {
      process.argv = [];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(false);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBeUndefined();
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle empty argv array with compiled binary', () => {
      process.argv = [];
      process.execPath = '/usr/bin/genkit';
      mockedFs.existsSync.mockReturnValue(false);
      Object.defineProperty(process, 'versions', {
        value: {},
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('compiled-binary');
      expect(result.execPath).toBe('/usr/bin/genkit');
      expect(result.scriptPath).toBeUndefined();
      expect(result.isCompiledBinary).toBe(true);
    });

    it('should handle argv with only one element', () => {
      process.argv = ['/usr/bin/node'];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(false);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBeUndefined();
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle fs.existsSync throwing an error', () => {
      process.argv = ['/usr/bin/node', '/path/to/script.js'];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockImplementation(() => {
        throw new Error('Permission denied');
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBe('/path/to/script.js');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle script files with unusual extensions', () => {
      process.argv = ['/usr/bin/node', '/path/to/script.xyz'];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(true);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBe('/path/to/script.xyz');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle executables with version numbers in name', () => {
      process.argv = ['node18', '/script.js'];
      process.execPath = '/usr/bin/node18';
      mockedFs.existsSync.mockReturnValue(true);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node18');
      expect(result.scriptPath).toBe('/script.js');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle bun executables with version numbers', () => {
      process.argv = ['/usr/local/bin/bun1.0', '/script.js'];
      process.execPath = '/usr/local/bin/bun1.0';
      mockedFs.existsSync.mockReturnValue(true);

      const result = detectCLIRuntime();

      expect(result.type).toBe('bun');
      expect(result.execPath).toBe('/usr/local/bin/bun1.0');
      expect(result.scriptPath).toBe('/script.js');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle null or undefined in process.versions', () => {
      process.argv = ['/usr/bin/node', '/script.js'];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(true);
      Object.defineProperty(process, 'versions', {
        value: null,
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBe('/script.js');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle very long file paths', () => {
      const longPath = '/very/long/path/'.repeat(50) + 'script.js';
      process.argv = ['/usr/bin/node', longPath];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(true);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBe(longPath);
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle special characters in paths', () => {
      process.argv = ['/usr/bin/node', '/path with spaces/script (1).js'];
      process.execPath = '/usr/bin/node';
      mockedFs.existsSync.mockReturnValue(true);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/node');
      expect(result.scriptPath).toBe('/path with spaces/script (1).js');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle argv[0] being different from execPath', () => {
      process.argv = ['node', '/script.js'];
      process.execPath = '/usr/local/bin/node';
      mockedFs.existsSync.mockReturnValue(true);

      const result = detectCLIRuntime();

      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/local/bin/node');
      expect(result.scriptPath).toBe('/script.js');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should handle custom Node.js build with unusual script extension', () => {
      // Custom Node.js build named "my-node" with script having .xyz extension
      process.argv = ['/usr/bin/my-node', '/path/to/script.xyz'];
      process.execPath = '/usr/bin/my-node';
      mockedFs.existsSync.mockReturnValue(true);
      Object.defineProperty(process, 'versions', {
        value: { node: '20.0.0' }, // Has node version info
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      // Should correctly detect as Node.js based on version info
      expect(result.type).toBe('node');
      expect(result.execPath).toBe('/usr/bin/my-node');
      expect(result.scriptPath).toBe('/path/to/script.xyz');
      expect(result.isCompiledBinary).toBe(false);
    });

    it('should detect Bun-compiled binary with virtual filesystem path', () => {
      process.argv = ['/usr/local/bin/genkit', '/$bunfs/root/genkit'];
      process.execPath = '/usr/local/bin/genkit';
      mockedFs.existsSync.mockReturnValue(false); // Virtual path doesn't exist on real filesystem
      Object.defineProperty(process, 'versions', {
        value: { ...originalVersions, bun: '1.0.0' },
        writable: true,
        configurable: true,
      });

      const result = detectCLIRuntime();

      expect(result.type).toBe('compiled-binary');
      expect(result.execPath).toBe('/usr/local/bin/genkit');
      expect(result.scriptPath).toBeUndefined();
      expect(result.isCompiledBinary).toBe(true);
    });
  });
});
