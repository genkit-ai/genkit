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

/**
 * Filesystem/import mechanics for agent directories: dynamically loading tool
 * modules and `agent.ts` overrides. A malformed or unloadable file is warned
 * about and skipped - authoring errors in one file never take down plugin
 * initialization.
 *
 * @module @genkit-ai/agent-dirs/loaders
 */

import type { GenkitBeta } from 'genkit/beta';
import { logger } from 'genkit/logging';
import { existsSync, readdirSync } from 'node:fs';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import type { AgentDirOverride, AgentDirTool } from './authoring.js';

/** A tool action registered via `ai.defineTool`. */
export type RegisteredTool = ReturnType<GenkitBeta['defineTool']>;

const TOOL_FILE_EXTENSIONS = ['.ts', '.mts', '.js', '.mjs'];

/**
 * Dynamic import that survives CJS transpilation. esbuild rewrites `import()`
 * to `require()` in the CJS build, which would lose the ability to load ESM
 * tool files; indirecting through `Function` keeps it a native import in both
 * builds.
 */
const dynamicImport = new Function('m', 'return import(m)') as (
  m: string
) => Promise<Record<string, unknown>>;

function isLoadableModule(file: string): boolean {
  return (
    TOOL_FILE_EXTENSIONS.includes(path.extname(file)) &&
    !/\.d\.[cm]?ts$/.test(file)
  );
}

/**
 * Loads every tool module under `toolsDir` and registers it with the registry.
 * Tools default to an agent-prefixed registry name (`<agent>/<file>`) so
 * same-named tool files in different agent directories never collide; the
 * model still sees the short name (the segment after the last '/').
 */
export async function loadTools(
  ai: GenkitBeta,
  toolsDir: string,
  agentName: string
): Promise<RegisteredTool[]> {
  if (!existsSync(toolsDir)) return [];
  const actions: RegisteredTool[] = [];
  for (const file of readdirSync(toolsDir).sort()) {
    if (!isLoadableModule(file)) continue;
    let mod: Record<string, unknown>;
    try {
      mod = await dynamicImport(pathToFileURL(path.join(toolsDir, file)).href);
    } catch (e) {
      logger.warn(
        `[agent-dirs] failed to load ${agentName}/tools/${file}: ${e}; skipping`
      );
      continue;
    }
    const tool = mod.default as AgentDirTool | undefined;
    if (!tool?.config || typeof tool.fn !== 'function') {
      logger.warn(
        `[agent-dirs] ${agentName}/tools/${file} does not default-export ` +
          `{ config, fn } (see defineDirTool); skipping`
      );
      continue;
    }
    const name =
      tool.config.name ??
      `${agentName}/${path.basename(file, path.extname(file))}`;
    actions.push(
      ai.defineTool(
        {
          name,
          description: tool.config.description,
          inputSchema: tool.config.inputSchema,
          outputSchema: tool.config.outputSchema,
        },
        async (input) => tool.fn(input)
      )
    );
  }
  return actions;
}

/**
 * Loads the optional `agent.{ts,mts,js,mjs}` override module from an agent
 * directory, if present.
 */
export async function loadOverride(
  agentPath: string
): Promise<AgentDirOverride | undefined> {
  for (const ext of TOOL_FILE_EXTENSIONS) {
    const overrideFile = path.join(agentPath, `agent${ext}`);
    if (!existsSync(overrideFile)) continue;
    let mod: Record<string, unknown>;
    try {
      mod = await dynamicImport(pathToFileURL(overrideFile).href);
    } catch (e) {
      logger.warn(
        `[agent-dirs] failed to load override ${overrideFile}: ${e}; ignoring`
      );
      return undefined;
    }
    if (typeof mod.default === 'function') {
      return mod.default as AgentDirOverride;
    }
    logger.warn(
      `[agent-dirs] ${overrideFile} exists but does not default-export a ` +
        `function; ignoring`
    );
  }
  return undefined;
}
