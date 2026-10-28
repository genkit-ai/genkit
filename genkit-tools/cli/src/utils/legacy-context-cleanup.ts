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

import { logger } from '@genkit-ai/tools-common/utils';
import { createHash } from 'crypto';
import { lstat, readFile, readlink, unlink, writeFile } from 'fs/promises';
import * as path from 'path';

const GENERATED_CONTEXT_HASHES = new Set([
  '71cd594d',
  '313bc8ad',
  '00543dca',
  'c85d0fc9',
  '29edcfe0',
  'fb4c1c37',
  '5fd29dc0',
  'ad0c71b3',
  '78cd09b2',
  '7e95fce8',
  'cd4991bb',
  '48794c66',
]);
const GENERATED_CONTEXT_IMPORT_HASHES = new Set(
  [...GENERATED_CONTEXT_HASHES].map((hash) => calculateHash(hash))
);

const GENERATED_CURSOR_RULE_HASHES = new Set(['c29ffc74', '95dc059b']);
const AUTO_GENERATED_MARKER =
  '<!-- Genkit Context - Auto-generated, do not edit -->';
const LEGACY_IMPORT_CONTENT = 'Genkit Framework Instructions:\n - @./GENKIT.md';
const PROMPT_BLOCK_REGEX =
  /<genkit_prompts(?:\s+hash="([^"]+)")?>([\s\S]*?)<\/genkit_prompts>/g;

/**
 * Removes only files and prompt blocks whose contents match known Genkit
 * generated templates. Cleanup is best-effort and never blocks a CLI command.
 */
export async function cleanupLegacyAiToolContext(
  projectRoot: string
): Promise<void> {
  const rootContextPath = path.join(projectRoot, 'GENKIT.md');
  const rootContextRemoved = await removeKnownFile(
    projectRoot,
    rootContextPath,
    GENERATED_CONTEXT_HASHES
  );

  await cleanupGeminiContextCopy(projectRoot, rootContextPath);
  if (rootContextRemoved) {
    await removeGeminiContextFileName(projectRoot);
  }

  await removeKnownFile(
    projectRoot,
    path.join(projectRoot, '.cursor', 'rules', 'GENKIT.mdc'),
    GENERATED_CURSOR_RULE_HASHES
  );

  for (const relativePath of [
    'CLAUDE.md',
    'GEMINI.md',
    path.join('.idx', 'airules.md'),
  ]) {
    await cleanupPromptBlocks(
      projectRoot,
      path.join(projectRoot, relativePath)
    );
  }
}

async function removeKnownFile(
  projectRoot: string,
  filePath: string,
  hashes: Set<string>
): Promise<boolean> {
  try {
    const stats = await lstat(filePath).catch((error: unknown) => {
      if (getErrorCode(error) === 'ENOENT') {
        return undefined;
      }
      throw error;
    });
    if (!stats || !stats.isFile() || stats.nlink > 1) {
      return false;
    }
    if (await hasSymlinkedParent(projectRoot, filePath)) {
      return false;
    }

    const content = await readFile(filePath, 'utf8');
    if (!hashes.has(calculateHash(content))) {
      return false;
    }

    await unlink(filePath);
    return true;
  } catch (error) {
    logCleanupError(filePath, error);
    return false;
  }
}

async function cleanupGeminiContextCopy(
  projectRoot: string,
  rootContextPath: string
): Promise<void> {
  const copyPath = path.join(projectRoot, '.gemini', 'GENKIT.md');
  try {
    const stats = await lstat(copyPath).catch((error: unknown) => {
      if (getErrorCode(error) === 'ENOENT') {
        return undefined;
      }
      throw error;
    });
    if (!stats) {
      return;
    }
    if (await hasSymlinkedParent(projectRoot, copyPath)) {
      return;
    }

    if (stats.isSymbolicLink()) {
      const target = await readlink(copyPath);
      if (
        path.resolve(path.dirname(copyPath), target) !==
        path.resolve(rootContextPath)
      ) {
        return;
      }

      await unlink(copyPath);
      return;
    }

    if (!stats.isFile() || stats.nlink > 1) {
      return;
    }
    const content = await readFile(copyPath, 'utf8');
    if (!GENERATED_CONTEXT_HASHES.has(calculateHash(content))) {
      return;
    }
    await unlink(copyPath);
  } catch (error) {
    logCleanupError(copyPath, error);
  }
}

async function removeGeminiContextFileName(projectRoot: string): Promise<void> {
  const settingsPath = path.join(projectRoot, '.gemini', 'settings.json');
  try {
    const stats = await lstat(settingsPath).catch((error: unknown) => {
      if (getErrorCode(error) === 'ENOENT') {
        return undefined;
      }
      throw error;
    });
    if (!stats || !stats.isFile() || stats.nlink > 1) {
      return;
    }
    if (await hasSymlinkedParent(projectRoot, settingsPath)) {
      return;
    }

    const content = await readFile(settingsPath, 'utf8');

    const settings: unknown = JSON.parse(content);
    if (!isRecord(settings)) {
      return;
    }

    let changed = false;
    if (settings.contextFileName === 'GENKIT.md') {
      delete settings.contextFileName;
      changed = true;
    } else if (Array.isArray(settings.contextFileName)) {
      const contextFiles = settings.contextFileName.filter(
        (fileName: unknown) => fileName !== 'GENKIT.md'
      );
      if (contextFiles.length !== settings.contextFileName.length) {
        settings.contextFileName = contextFiles;
        changed = true;
      }
    }

    if (!changed) {
      return;
    }

    const trailingNewline = content.endsWith('\n') ? '\n' : '';
    await writeFile(
      settingsPath,
      JSON.stringify(settings, null, 2) + trailingNewline,
      'utf8'
    );
  } catch (error) {
    logCleanupError(settingsPath, error);
  }
}

async function cleanupPromptBlocks(
  projectRoot: string,
  filePath: string
): Promise<void> {
  try {
    const stats = await lstat(filePath).catch((error: unknown) => {
      if (getErrorCode(error) === 'ENOENT') {
        return undefined;
      }
      throw error;
    });
    if (!stats || !stats.isFile() || stats.nlink > 1) {
      return;
    }
    if (await hasSymlinkedParent(projectRoot, filePath)) {
      return;
    }

    const content = await readFile(filePath, 'utf8');

    let removedBlock = false;
    const updated = content.replace(
      PROMPT_BLOCK_REGEX,
      (match, hash: string | undefined, body: string) => {
        if (!isGeneratedPromptBlock(body, hash)) {
          return match;
        }
        removedBlock = true;
        return '';
      }
    );

    if (!removedBlock || updated === content) {
      return;
    }

    if (updated.trim().length === 0) {
      await unlink(filePath);
      return;
    }

    await writeFile(filePath, updated, 'utf8');
  } catch (error) {
    logCleanupError(filePath, error);
  }
}

function isGeneratedPromptBlock(
  body: string,
  hash: string | undefined
): boolean {
  if (!body.includes(AUTO_GENERATED_MARKER)) {
    return false;
  }

  const content = normalizeLineEndings(
    body.replace(AUTO_GENERATED_MARKER, '')
  ).trim();
  return (
    GENERATED_CONTEXT_HASHES.has(calculateHash(content)) ||
    (content === LEGACY_IMPORT_CONTENT &&
      hash !== undefined &&
      GENERATED_CONTEXT_IMPORT_HASHES.has(hash))
  );
}

async function hasSymlinkedParent(
  projectRoot: string,
  filePath: string
): Promise<boolean> {
  const rootPath = path.resolve(projectRoot);
  const relativePath = path.relative(rootPath, path.resolve(filePath));
  if (
    relativePath === '..' ||
    relativePath.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativePath)
  ) {
    return true;
  }

  const parentSegments = path
    .dirname(relativePath)
    .split(path.sep)
    .filter((segment) => segment !== '' && segment !== '.');
  let currentPath = rootPath;
  for (const segment of parentSegments) {
    currentPath = path.join(currentPath, segment);
    const stats = await lstat(currentPath);
    if (stats.isSymbolicLink() || !stats.isDirectory()) {
      return true;
    }
  }

  return false;
}

function calculateHash(content: string): string {
  return createHash('sha256')
    .update(normalizeLineEndings(content).trim())
    .digest('hex')
    .substring(0, 8);
}

function normalizeLineEndings(content: string): string {
  return content.replace(/\r\n/g, '\n');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function logCleanupError(filePath: string, error: unknown): void {
  if (getErrorCode(error) !== 'ENOENT') {
    logger.debug(`Skipping legacy Genkit cleanup for ${filePath}`, error);
  }
}

function getErrorCode(error: unknown): unknown {
  return typeof error === 'object' && error !== null && 'code' in error
    ? error.code
    : undefined;
}
