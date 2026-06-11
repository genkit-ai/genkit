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

import { afterEach, beforeEach, describe, expect, it } from '@jest/globals';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  findProjectRoot,
  projectNameFromGenkitFilePath,
} from '../../src/utils';

describe('utils', () => {
  describe('findProjectRoot', () => {
    let tmpDir: string;
    let originalCwd: string;

    beforeEach(() => {
      originalCwd = process.cwd();
      // realpathSync resolves symlinks (for example /var -> /private/var on
      // macOS) so the temp paths match what process.cwd() reports.
      tmpDir = fs.realpathSync(
        fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-find-root-'))
      );
    });

    afterEach(() => {
      process.chdir(originalCwd);
      fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    it('returns the directory containing a pubspec.yaml', async () => {
      const projectDir = path.join(tmpDir, 'dart-app');
      const nestedDir = path.join(projectDir, 'lib', 'src');
      fs.mkdirSync(nestedDir, { recursive: true });
      fs.writeFileSync(path.join(projectDir, 'pubspec.yaml'), 'name: dart_app');

      process.chdir(nestedDir);

      expect(await findProjectRoot()).toEqual(projectDir);
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

      process.chdir(dartDir);

      expect(await findProjectRoot()).toEqual(dartDir);
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
});
