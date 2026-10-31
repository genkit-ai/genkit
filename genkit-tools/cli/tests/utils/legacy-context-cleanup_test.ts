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

import { afterEach, beforeEach, describe, expect, it } from '@jest/globals';
import { createHash } from 'crypto';
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import { cleanupLegacyAiToolContext } from '../../src/utils/legacy-context-cleanup';

const AUTO_GENERATED_MARKER =
  '<!-- Genkit Context - Auto-generated, do not edit -->';

describe('cleanupLegacyAiToolContext', () => {
  let projectRoot: string;
  let externalRoot: string | undefined;

  beforeEach(async () => {
    projectRoot = await mkdtemp(path.join(os.tmpdir(), 'genkit-cleanup-'));
  });

  afterEach(async () => {
    await rm(projectRoot, { recursive: true, force: true });
    if (externalRoot) {
      await rm(externalRoot, { recursive: true, force: true });
      externalRoot = undefined;
    }
  });

  it('removes known generated files and preserves Gemini MCP settings', async () => {
    const context = await readGeneratedContext();
    expect(calculateHash(context)).toBe('71cd594d');

    const geminiDirectory = path.join(projectRoot, '.gemini');
    const cursorRulesDirectory = path.join(projectRoot, '.cursor', 'rules');
    await mkdir(geminiDirectory, { recursive: true });
    await mkdir(cursorRulesDirectory, { recursive: true });
    await writeFile(path.join(projectRoot, 'GENKIT.md'), context);
    await writeFile(path.join(geminiDirectory, 'GENKIT.md'), context);

    const cursorRules = createCursorRules();
    expect(['c29ffc74', '95dc059b']).toContain(calculateHash(cursorRules));
    await writeFile(path.join(cursorRulesDirectory, 'GENKIT.mdc'), cursorRules);

    const mcpServer = {
      command: 'custom-genkit',
      args: ['mcp', '--custom'],
    };
    const rootMcpConfig = JSON.stringify(
      { mcpServers: { genkit: mcpServer } },
      null,
      2
    );
    const cursorMcpConfig = JSON.stringify(
      { mcpServers: { genkit: { command: 'cursor-genkit' } } },
      null,
      2
    );
    await writeFile(path.join(projectRoot, '.mcp.json'), rootMcpConfig);
    await writeFile(
      path.join(projectRoot, '.cursor', 'mcp.json'),
      cursorMcpConfig
    );
    await writeFile(
      path.join(geminiDirectory, 'settings.json'),
      JSON.stringify(
        {
          contextFileName: ['GENKIT.md', 'PROJECT.md'],
          mcpServers: { genkit: mcpServer, other: { command: 'other' } },
        },
        null,
        2
      )
    );

    await cleanupLegacyAiToolContext(projectRoot);

    await expect(
      lstat(path.join(projectRoot, 'GENKIT.md'))
    ).rejects.toMatchObject({ code: 'ENOENT' });
    await expect(
      lstat(path.join(geminiDirectory, 'GENKIT.md'))
    ).rejects.toMatchObject({ code: 'ENOENT' });
    await expect(
      lstat(path.join(cursorRulesDirectory, 'GENKIT.mdc'))
    ).rejects.toMatchObject({ code: 'ENOENT' });

    const settings = JSON.parse(
      await readFile(path.join(geminiDirectory, 'settings.json'), 'utf8')
    );
    expect(settings.contextFileName).toEqual(['PROJECT.md']);
    expect(settings.mcpServers).toEqual({
      genkit: mcpServer,
      other: { command: 'other' },
    });
    expect(await readFile(path.join(projectRoot, '.mcp.json'), 'utf8')).toBe(
      rootMcpConfig
    );
    expect(
      await readFile(path.join(projectRoot, '.cursor', 'mcp.json'), 'utf8')
    ).toBe(cursorMcpConfig);
  });

  it('removes only known prompt blocks and preserves surrounding instructions', async () => {
    const context = await readGeneratedContext();
    const block = createPromptBlock(context, calculateHash(context));
    const filePaths = [
      path.join(projectRoot, 'CLAUDE.md'),
      path.join(projectRoot, 'GEMINI.md'),
      path.join(projectRoot, '.idx', 'airules.md'),
    ];

    for (const filePath of filePaths) {
      await mkdir(path.dirname(filePath), { recursive: true });
      await writeFile(filePath, `User instructions before.\n${block}\nAfter.`);
    }

    await cleanupLegacyAiToolContext(projectRoot);

    for (const filePath of filePaths) {
      const result = await readFile(filePath, 'utf8');
      expect(result).toContain('User instructions before.');
      expect(result).toContain('After.');
      expect(result).not.toContain('<genkit_prompts');
    }
  });

  it('removes a generated prompt-only file and recognizes the old Claude import', async () => {
    const generatedFilePath = path.join(projectRoot, 'GEMINI.md');
    const context = await readGeneratedContext();
    const block = createPromptBlock(context, calculateHash(context));
    await writeFile(generatedFilePath, block);

    const claudeImport = 'Genkit Framework Instructions:\n - @./GENKIT.md';
    const claudeImportPath = path.join(projectRoot, 'CLAUDE.md');
    await writeFile(
      claudeImportPath,
      createPromptBlock(claudeImport, calculateHash(calculateHash(context)))
    );

    await cleanupLegacyAiToolContext(projectRoot);

    await expect(lstat(generatedFilePath)).rejects.toMatchObject({
      code: 'ENOENT',
    });
    await expect(lstat(claudeImportPath)).rejects.toMatchObject({
      code: 'ENOENT',
    });
  });

  it('preserves edited files, edited blocks, and their settings references', async () => {
    const context = await readGeneratedContext();
    const editedContext = `${context}\nCustom project instructions.`;
    const geminiDirectory = path.join(projectRoot, '.gemini');
    await mkdir(geminiDirectory, { recursive: true });
    await writeFile(path.join(projectRoot, 'GENKIT.md'), editedContext);
    await writeFile(path.join(geminiDirectory, 'GENKIT.md'), editedContext);
    await writeFile(
      path.join(geminiDirectory, 'settings.json'),
      JSON.stringify({
        contextFileName: 'GENKIT.md',
        mcpServers: { genkit: { command: 'genkit' } },
      })
    );

    const editedBlock = createPromptBlock(
      `${context}\nCustom instructions inside the managed section.`,
      calculateHash(context)
    );
    const claudePath = path.join(projectRoot, 'CLAUDE.md');
    await writeFile(claudePath, editedBlock);

    const editedCursorRules = `${createCursorRules()}\nCustom rule.`;
    const cursorPath = path.join(projectRoot, '.cursor', 'rules', 'GENKIT.mdc');
    await mkdir(path.dirname(cursorPath), { recursive: true });
    await writeFile(cursorPath, editedCursorRules);

    await cleanupLegacyAiToolContext(projectRoot);

    expect(await readFile(path.join(projectRoot, 'GENKIT.md'), 'utf8')).toBe(
      editedContext
    );
    expect(
      await readFile(path.join(geminiDirectory, 'GENKIT.md'), 'utf8')
    ).toBe(editedContext);
    expect(await readFile(claudePath, 'utf8')).toBe(editedBlock);
    expect(await readFile(cursorPath, 'utf8')).toBe(editedCursorRules);
    const settings = JSON.parse(
      await readFile(path.join(geminiDirectory, 'settings.json'), 'utf8')
    );
    expect(settings.contextFileName).toBe('GENKIT.md');
    expect(settings.mcpServers.genkit).toEqual({ command: 'genkit' });
  });

  it('removes a Gemini symlink to the project GENKIT.md', async () => {
    const context = `${await readGeneratedContext()}\nCustom project instructions.`;
    const geminiDirectory = path.join(projectRoot, '.gemini');
    const symlinkPath = path.join(geminiDirectory, 'GENKIT.md');
    await mkdir(geminiDirectory, { recursive: true });
    await writeFile(path.join(projectRoot, 'GENKIT.md'), context);
    await symlink('../GENKIT.md', symlinkPath, 'file');

    await cleanupLegacyAiToolContext(projectRoot);

    expect(await readFile(path.join(projectRoot, 'GENKIT.md'), 'utf8')).toBe(
      context
    );
    await expect(lstat(symlinkPath)).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('removes a scalar contextFileName only when the generated root file is removed', async () => {
    const context = await readGeneratedContext();
    const geminiDirectory = path.join(projectRoot, '.gemini');
    await mkdir(geminiDirectory, { recursive: true });
    await writeFile(path.join(projectRoot, 'GENKIT.md'), context);
    await writeFile(
      path.join(geminiDirectory, 'settings.json'),
      JSON.stringify({
        contextFileName: 'GENKIT.md',
        mcpServers: { genkit: { command: 'genkit' } },
      })
    );

    await cleanupLegacyAiToolContext(projectRoot);

    const settings = JSON.parse(
      await readFile(path.join(geminiDirectory, 'settings.json'), 'utf8')
    );
    expect(settings).not.toHaveProperty('contextFileName');
    expect(settings.mcpServers.genkit).toEqual({ command: 'genkit' });
  });

  it('does not clean files through a symlinked configuration directory', async () => {
    const context = await readGeneratedContext();
    externalRoot = await mkdtemp(path.join(os.tmpdir(), 'genkit-external-'));
    const externalContextPath = path.join(externalRoot, 'GENKIT.md');
    await writeFile(externalContextPath, context);
    await symlink(externalRoot, path.join(projectRoot, '.gemini'), 'junction');

    await cleanupLegacyAiToolContext(projectRoot);

    expect(await readFile(externalContextPath, 'utf8')).toBe(context);
  });
});

async function readGeneratedContext(): Promise<string> {
  const contextPath = path.resolve(__dirname, '../../context/GENKIT.js.md');
  return (await readFile(contextPath, 'utf8')).replace(/\r\n/g, '\n').trim();
}

function createPromptBlock(content: string, hash: string): string {
  return `<genkit_prompts hash="${hash}">\n${AUTO_GENERATED_MARKER}\n${content}\n</genkit_prompts>`;
}

function createCursorRules(): string {
  const header =
    '---\ndescription: Genkit project development guidelines\n---\n';
  const importContent = `# Genkit Context\n\n@${path.join('..', '..', 'GENKIT.md')}\n`;
  return `${header}\n${importContent}`;
}

function calculateHash(content: string): string {
  return createHash('sha256')
    .update(content.replace(/\r\n/g, '\n').trim())
    .digest('hex')
    .substring(0, 8);
}
