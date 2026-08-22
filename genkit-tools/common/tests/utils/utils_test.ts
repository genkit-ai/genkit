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
import {
  findProjectRoot,
  projectNameFromGenkitFilePath,
} from '../../src/utils';
import {
  formatDuration,
  parseAndSanitizeJson,
  sanitizeBase64DataUrls,
} from '../../src/utils/utils';

describe('utils', () => {
  describe('findProjectRoot', () => {
    let tmpDir: string;

    beforeEach(() => {
      // realpathSync resolves symlinks (for example /var -> /private/var on
      // macOS) so the temp paths match what process.cwd() reports.
      tmpDir = fs.realpathSync(
        fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-find-root-'))
      );
    });

    afterEach(() => {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    it('returns the directory containing a pubspec.yaml', async () => {
      const projectDir = path.join(tmpDir, 'dart-app');
      const nestedDir = path.join(projectDir, 'lib', 'src');
      fs.mkdirSync(nestedDir, { recursive: true });
      fs.writeFileSync(path.join(projectDir, 'pubspec.yaml'), 'name: dart_app');

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(nestedDir);

      try {
        expect(await findProjectRoot()).toEqual(projectDir);
      } finally {
        cwdSpy.mockRestore();
      }
    });

    it('returns the nearest project root when a Dart project is nested under a package.json', async () => {
      // Mirrors a Dart project living inside a JS workspace or monorepo. The
      // CLI should stop at the Dart project rather than climbing to the
      // workspace package.json above it.
      const workspaceDir = path.join(tmpDir, 'workspace');
      const dartDir = path.join(workspaceDir, 'dart-app');
      fs.mkdirSync(dartDir, { recursive: true });
      fs.writeFileSync(path.join(workspaceDir, 'package.json'), '{}');
      fs.writeFileSync(path.join(dartDir, 'pubspec.yaml'), 'name: dart_app');

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(dartDir);

      try {
        expect(await findProjectRoot()).toEqual(dartDir);
      } finally {
        cwdSpy.mockRestore();
      }
    });

    it('honors an explicit startDir independent of process.cwd()', async () => {
      const projectDir = path.join(tmpDir, 'js-app');
      const nestedDir = path.join(projectDir, 'src');
      const otherCwd = path.join(tmpDir, 'elsewhere');
      fs.mkdirSync(nestedDir, { recursive: true });
      fs.mkdirSync(otherCwd, { recursive: true });
      fs.writeFileSync(path.join(projectDir, 'package.json'), '{}');

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(otherCwd);

      try {
        expect(await findProjectRoot(nestedDir)).toEqual(projectDir);
      } finally {
        cwdSpy.mockRestore();
      }
    });
  });

  describe('projectNameFromGenkitFilePath', () => {
    it('returns unknown for empty string', () => {
      expect(projectNameFromGenkitFilePath('')).toEqual('unknown');
    });

    it('returns unknown for an invalid path', () => {
      expect(projectNameFromGenkitFilePath('/path/to/nowhere')).toEqual(
        'unknown'
      );
    });

    it('returns project name from a typical runtime file path', () => {
      expect(
        projectNameFromGenkitFilePath(
          '/path/to/test-project/.genkit/runtimes/123.json'
        )
      ).toEqual('test-project');
    });

    it('returns project name from any path that contains a .genkit dir', () => {
      expect(
        projectNameFromGenkitFilePath(
          '/path/to/test-project/.genkit/unexpected/but/valid/location'
        )
      ).toEqual('test-project');
    });
  });

  describe('formatDuration', () => {
    it('formats duration correctly', () => {
      expect(formatDuration(1000, 2500)).toBe('1500ms');
      expect(formatDuration(1000, 1000.5)).toBe('0.50ms');
      expect(formatDuration(1000, 500)).toBe('0.00ms'); // Math.max(0, ...)
    });

    it('returns empty string if start or end time is undefined', () => {
      expect(formatDuration(1000, undefined)).toBe('');
      expect(formatDuration(undefined, 2500)).toBe('');
      expect(formatDuration(undefined, undefined)).toBe('');
    });
  });

  describe('sanitizeBase64DataUrls', () => {
    it('sanitizes base64 data URLs', () => {
      const b64 =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
      const sanitized = sanitizeBase64DataUrls(b64);
      expect(sanitized).toMatch(
        /data:image\/png;base64,<... \d+ B base64 data ...>/
      );
    });

    it('keeps media if keepBase64 is true', () => {
      const b64 =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
      expect(sanitizeBase64DataUrls(b64, true)).toBe(b64);
    });

    it('sanitizes deeply nested objects', () => {
      const obj = {
        a: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=',
        b: [
          {
            c: 'data:text/plain;base64,SGVsbG8gV29ybGQ=',
          },
        ],
      };
      const sanitized = sanitizeBase64DataUrls(obj);
      expect(sanitized.a).toMatch(
        /data:image\/png;base64,<... \d+ B base64 data ...>/
      );
      expect(sanitized.b[0].c).toMatch(
        /data:text\/plain;base64,<... \d+ B base64 data ...>/
      );
    });
  });

  describe('parseAndSanitizeJson', () => {
    it('parses and sanitizes JSON strings', () => {
      const json = JSON.stringify({
        a: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=',
      });
      const parsed = parseAndSanitizeJson(json);
      expect(parsed.a).toMatch(
        /data:image\/png;base64,<... \d+ B base64 data ...>/
      );
    });

    it('returns original string if not valid JSON', () => {
      const str = 'not valid json';
      expect(parseAndSanitizeJson(str)).toBe(str);
    });
  });
});
