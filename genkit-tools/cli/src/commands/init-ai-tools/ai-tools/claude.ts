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
import { logger } from '@genkit-ai/tools-common/utils';
import { existsSync, readFileSync } from 'fs';
import { writeFile } from 'fs/promises';
import { AIToolConfigResult, AIToolModule, InitConfigOptions } from '../types';
import {
  GENKIT_PROMPT_PATH,
  calculateHash,
  initGenkitFile,
  updateContentInPlace,
} from '../utils';

const CLAUDE_MCP_PATH = '.mcp.json';
const CLAUDE_PROMPT_PATH = 'CLAUDE.md';

/** Configuration module for Claude Code */
export const claude: AIToolModule = {
  name: 'claude',
  displayName: 'Claude Code',

  /**
   * Configures Claude Code with Genkit context.
   *
   * - .mcp.json: Merges with existing MCP config
   * - CLAUDE.local.md: Updates Firebase section only (preserves user content)
   */
  async configure(
    runtime: Runtime,
    options?: InitConfigOptions
  ): Promise<AIToolConfigResult> {
    const files: AIToolConfigResult['files'] = [];

    // Handle MCP configuration - merge with existing if present
    let existingConfig: any = {};
    let settingsUpdated = false;
    try {
      const fileExists = existsSync(CLAUDE_MCP_PATH);
      if (fileExists) {
        existingConfig = JSON.parse(readFileSync(CLAUDE_MCP_PATH, 'utf-8'));
      }
    } catch (e) {
      // File doesn't exist or is invalid JSON, start fresh
    }

    // Check if genkit server already exists
    if (!existingConfig.mcpServers?.genkit) {
      if (!existingConfig.mcpServers) {
        existingConfig.mcpServers = {};
      }
      existingConfig.mcpServers.genkit = {
        command: 'genkit',
        args: ['mcp', '--no-update-notification'],
      };
      await writeFile(CLAUDE_MCP_PATH, JSON.stringify(existingConfig, null, 2));
      settingsUpdated = true;
    }

    files.push({ path: CLAUDE_MCP_PATH, updated: settingsUpdated });

    logger.info('Copying the Genkit context to GENKIT.md...');
    const mdResult = await initGenkitFile(runtime);
    files.push({ path: GENKIT_PROMPT_PATH, updated: mdResult.updated });

    logger.info('Updating CLAUDE.md to include Genkit context...');
    const claudeImportTag = `\nGenkit Framework Instructions:\n - @./GENKIT.md\n`;
    const baseResult = await updateContentInPlace(
      CLAUDE_PROMPT_PATH,
      claudeImportTag,
      { hash: calculateHash(mdResult.hash) }
    );

    files.push({ path: CLAUDE_PROMPT_PATH, updated: baseResult.updated });
    return { files };
  },
};
