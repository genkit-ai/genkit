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

import * as fs from 'fs/promises';
import * as path from 'path';
import type { AgentBackend } from './agent-backend.js';
import type { EditResult, LsResult, ReadResult, WriteResult } from './types.js';

export interface LocalFilesystemBackendOptions {
  cwd: string;
}

export class LocalFilesystemBackend implements AgentBackend {
  readonly id: string;
  readonly cwd: string;
  private readonly securePrefix: string;

  constructor(options: LocalFilesystemBackendOptions) {
    this.cwd = path.resolve(options.cwd);
    this.securePrefix = this.cwd.endsWith(path.sep)
      ? this.cwd
      : this.cwd + path.sep;
    this.id = `local:${this.cwd}`;
  }

  async ls(
    dirPath: string,
    options: {
      recursive?: boolean;
    } = {}
  ): Promise<LsResult> {
    try {
      const targetDir = this.resolvePath(dirPath);
      const files = await this.list(targetDir, !!options.recursive);
      return { files };
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }

  async read(
    filePath: string,
    offset?: number,
    limit?: number
  ): Promise<ReadResult> {
    try {
      const targetFile = this.resolvePath(filePath);
      const contentType = lookupMimeType(targetFile);

      if (contentType.startsWith('image/')) {
        return {
          content: await fs.readFile(targetFile),
          mimeType: contentType,
        };
      }

      let content = await fs.readFile(targetFile, 'utf8');
      if (offset !== undefined || limit !== undefined) {
        const lines = content.split('\n');
        const start = offset ?? 0;
        const end = limit === undefined ? undefined : start + limit;
        content = lines.slice(start, end).join('\n');
      }
      return {
        content,
        mimeType: contentType,
      };
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }

  async write(filePath: string, content: string): Promise<WriteResult> {
    try {
      const targetFile = this.resolvePath(filePath);
      await fs.mkdir(path.dirname(targetFile), { recursive: true });
      await fs.writeFile(targetFile, content, 'utf8');
      return { path: filePath };
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }

  async edit(
    filePath: string,
    oldString: string,
    newString: string,
    replaceAll = false
  ): Promise<EditResult> {
    try {
      const targetFile = this.resolvePath(filePath);
      const content = await fs.readFile(targetFile, 'utf8');
      if (!content.includes(oldString)) {
        return { error: `Search content not found in file ${filePath}.` };
      }

      const occurrences = content.split(oldString).length - 1;
      const updated = replaceAll
        ? content.split(oldString).join(newString)
        : content.replace(oldString, () => newString);
      await fs.writeFile(targetFile, updated, 'utf8');
      return {
        path: filePath,
        occurrences: replaceAll ? occurrences : 1,
      };
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }

  private resolvePath(requestedPath: string) {
    const relativePath = requestedPath.replace(/^[/\\]+/, '');
    const p = path.resolve(this.cwd, relativePath);
    if (!p.startsWith(this.securePrefix) && p !== this.cwd) {
      throw new Error('Access denied: Path is outside of root directory.');
    }
    return p;
  }

  private async list(
    dir: string,
    recursive: boolean,
    base: string = ''
  ): Promise<Array<{ path: string; isDirectory: boolean }>> {
    const results: Array<{ path: string; isDirectory: boolean }> = [];
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const relativePath = path.join(base, entry.name);
      const isDirectory = entry.isDirectory();
      results.push({
        path: relativePath,
        isDirectory,
      });
      if (isDirectory && recursive) {
        results.push(
          ...(await this.list(path.join(dir, entry.name), true, relativePath))
        );
      }
    }
    return results;
  }
}

function lookupMimeType(filePath: string): string {
  switch (path.extname(filePath).toLowerCase()) {
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.png':
      return 'image/png';
    case '.gif':
      return 'image/gif';
    case '.webp':
      return 'image/webp';
    default:
      return 'text/plain';
  }
}
