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

import { existsSync, readFileSync } from 'fs';

import { Runtime } from '@genkit-ai/tools-common/manager';
import * as crypto from 'crypto';
import { writeFile } from 'fs/promises';
import { GENKIT_CONTEXT as GoContext } from './context/go';
import { GENKIT_CONTEXT as NodeContext } from './context/nodejs';

/** Shared location for the GENKIT.md context file */
export const GENKIT_PROMPT_PATH = 'GENKIT.md';

const GENKIT_TAG_REGEX =
  /<genkit_prompts(?:\s+hash="([^"]+)")?>([\s\S]*?)<\/genkit_prompts>/;
/*
 * Deeply compares two JSON-serializable objects. It's a simplified version of a
 * deep equal function, sufficient for comparing the structure of the
 * gemini-extension.json file. It doesn't handle special cases like RegExp,
 * Date, or functions.
 */
export function deepEqual(a: any, b: any): boolean {
  if (a === b) {
    return true;
  }

  if (
    typeof a !== 'object' ||
    a === null ||
    typeof b !== 'object' ||
    b === null
  ) {
    return false;
  }

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) {
    return false;
  }

  for (const key of keysA) {
    if (!keysB.includes(key) || !deepEqual(a[key], b[key])) {
      return false;
    }
  }

  return true;
}

/**
 * Replace an entire prompt file (no user content to preserve). Used for files
 * we fully own like GENKIT.md.
 */
export async function initOrReplaceFile(
  filePath: string,
  content: string
): Promise<{ updated: boolean }> {
  const fileExists = existsSync(filePath);
  if (fileExists) {
    const currentConfig = readFileSync(filePath, 'utf-8');
    if (!deepEqual(currentConfig, content)) {
      await writeFile(filePath, content);
      return { updated: true };
    }
  } else {
    await writeFile(filePath, content);
    return { updated: true };
  }
  return { updated: false };
}

/**
 * Update a file with Genkit prompts section, preserving user content
 * Used for files like CLAUDE.md.
 */
export async function updateContentInPlace(
  filePath: string,
  content: string,
  options?: { hash: string }
): Promise<{ updated: boolean }> {
  const newHash = options?.hash ?? calculateHash(content);
  const newSection = `<genkit_prompts hash="${newHash}">
<!-- Genkit Context - Auto-generated, do not edit -->
${content}
</genkit_prompts>`;

  let currentContent = '';
  const fileExists = existsSync(filePath);
  if (fileExists) {
    currentContent = readFileSync(filePath, 'utf-8');
  }

  // Check if section exists and has same hash
  const match = currentContent.match(GENKIT_TAG_REGEX);
  if (match && match[1] === newHash) {
    return { updated: false };
  }

  // Generate final content
  let finalContent: string;
  if (!currentContent) {
    // New file
    finalContent = newSection;
  } else if (match) {
    // Replace existing section
    finalContent =
      currentContent.substring(0, match.index!) +
      newSection +
      currentContent.substring(match.index! + match[0].length);
  } else {
    // Append to existing file
    const separator = currentContent.endsWith('\n') ? '\n' : '\n\n';
    finalContent = currentContent + separator + newSection;
  }

  await writeFile(filePath, finalContent);
  return { updated: true };
}

/**
 * Generate hash for embedded content.
 */
export function calculateHash(content: string): string {
  return crypto
    .createHash('sha256')
    .update(content.trim())
    .digest('hex')
    .substring(0, 8);
}

/**
 * Get raw prompt content for Genkit
 */
export function getGenkitContext(runtime: Runtime): string {
  switch (runtime) {
    case 'nodejs':
      return NodeContext;
    case 'go':
      return GoContext;
    default:
      throw new Error('Unexpected runtime provided', runtime);
  }
}

/**
 * Initializes the GENKIT.md file
 */
export async function initGenkitFile(runtime: Runtime) {
  const genkitContext = getGenkitContext(runtime);
  const result = await initOrReplaceFile(GENKIT_PROMPT_PATH, genkitContext);
  return { updated: result.updated, hash: calculateHash(genkitContext) };
}
