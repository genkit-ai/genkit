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

import { Runtime } from '@genkit-ai/tools-common/manager';
import { existsSync, readFileSync } from 'fs';
import { mkdir, writeFile } from 'fs/promises';
import * as path from 'path';
import { AIToolConfigResult, AIToolModule, InitConfigOptions } from '../types';
import {
  GENKIT_PROMPT_PATH,
  initGenkitFile,
  initOrReplaceFile,
} from '../utils';

const CURSOR_MCP_PATH = path.join('.cursor', 'mcp.json');
const CURSOR_RULES_DIR = '.cursor/rules';
const GENKIT_MDC_PATH = path.join(CURSOR_RULES_DIR, 'GENKIT.mdc');

const CURSOR_RULES_HEADER = `---
description: Genkit project development guidelines
---
`;

export const cursor: AIToolModule = {
  name: 'cursor',
  displayName: 'Cursor',

  /**
   * Configures Cursor with Genkit context files.
   *
   * This function sets up the necessary files for Cursor to understand the
   * Genkit app and interact with Genkit MCP tools. It creates
   * a `.cursor` directory with the following:
   *
   * - `mcp.json`: Configures the Genkit MCP server for direct Genkit operations from Cursor.
   * - `rules/GENKIT.mdc`: The main entry point for project-specific context, importing the base GENKIT.md file.
   *
   * File ownership:
   * - .cursor/mcp.json: Merges with existing config (preserves user settings)
   * - .cursor/rules/GENKIT.mdc: Fully managed by us (replaced on each update)

   */
  async configure(
    runtime: Runtime,
    options?: InitConfigOptions
  ): Promise<AIToolConfigResult> {
    const files: AIToolConfigResult['files'] = [];

    // Create the base GENKIT context file (GENKIT.md).
    // This file contains fundamental details about the GENKIT project.
    const mdResult = await initGenkitFile(runtime);
    files.push({ path: GENKIT_PROMPT_PATH, updated: mdResult.updated });

    // Handle MCP configuration - merge with existing if present.
    // This allows Cursor to communicate with Genkit tools.
    let mcpUpdated = false;
    let existingConfig: any = {};

    try {
      const fileExists = existsSync(CURSOR_MCP_PATH);
      if (fileExists) {
        existingConfig = JSON.parse(readFileSync(CURSOR_MCP_PATH, 'utf-8'));
      } else {
        await mkdir('.cursor', { recursive: true });
      }
    } catch (e) {
      // File doesn't exist or is invalid JSON, start fresh
    }

    if (!existingConfig.mcpServers?.genkit) {
      if (!existingConfig.mcpServers) {
        existingConfig.mcpServers = {};
      }
      existingConfig.mcpServers.genkit = {
        command: 'genkit',
        args: ['mcp', '--no-update-notification'],
      };
      await writeFile(CURSOR_MCP_PATH, JSON.stringify(existingConfig, null, 2));
      mcpUpdated = true;
    }
    files.push({ path: CURSOR_MCP_PATH, updated: mcpUpdated });

    // Create the main `GENKIT.mdc` file, which acts as an entry point
    // for Cursor's AI and imports the other context files.
    await mkdir(path.join('.cursor', 'rules'), { recursive: true });
    const genkitImport = '@' + path.join('..', '..', GENKIT_PROMPT_PATH);
    const importContent = `# Genkit Context\n\n${genkitImport}\n`;

    const mdcContent = CURSOR_RULES_HEADER + '\n' + importContent;
    const { updated } = await initOrReplaceFile(GENKIT_MDC_PATH, mdcContent);
    files.push({ path: GENKIT_MDC_PATH, updated: updated });

    return { files };
  },
};
