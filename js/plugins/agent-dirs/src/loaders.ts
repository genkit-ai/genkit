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
 * modules and `agent.ts` overrides.
 *
 * @module @genkit-ai/agent-dirs/loaders
 */

import type { GenkitBeta } from 'genkit/beta';
import { logger } from 'genkit/logging';
import { existsSync, readdirSync } from 'node:fs';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import type {
  AgentDirOverride,
  AgentDirTool,
  AgentDirToolFactory,
} from './authoring.js';

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
 * Loads every tool module under `toolsDir` and registers it with the
 * registry.
 *
 * Registry names default to `agent-dirs/<agent>/<file>`: prefixing with the
 * plugin name keeps the registry's plugin-segment parsing pointing at a real
 * plugin (a bare `<agent>/<file>` would make the registry treat the agent
 * directory as a phantom plugin), and the per-agent segment keeps same-named
 * tool files in different agent directories from colliding. The model always
 * sees only the short name (the segment after the last '/'). Setting
 * `config.name` opts out of namespacing entirely.
 *
 * Returns `undefined` when a tool failed and `strict` is false (the caller
 * skips the agent); throws when `strict` is true.
 */
export async function loadTools(
  ai: GenkitBeta,
  toolsDir: string,
  agentName: string,
  opts: { strict: boolean }
): Promise<RegisteredTool[] | undefined> {
  if (!existsSync(toolsDir)) return [];
  const fail = (message: string): undefined => {
    const full = `[agent-dirs] agent '${agentName}': ${message}`;
    if (opts.strict) throw new Error(full);
    logger.warn(`${full} - skipping agent`);
    return undefined;
  };

  const actions: RegisteredTool[] = [];
  for (const file of readdirSync(toolsDir).sort()) {
    if (!isLoadableModule(file)) continue;
    let mod: Record<string, unknown>;
    try {
      mod = await dynamicImport(pathToFileURL(path.join(toolsDir, file)).href);
    } catch (e) {
      return fail(`failed to load tools/${file}: ${e}`);
    }
    // Factory form: `export default (ai) => ai.defineTool({...}, fn)` -
    // native API escape hatch; the factory owns naming.
    if (typeof mod.default === 'function') {
      const action = await (mod.default as AgentDirToolFactory)(ai);
      if (!action?.__action?.name) {
        return fail(
          `tools/${file} default-exports a function, but it did not return ` +
            `a tool (expected \`(ai) => ai.defineTool(...)\`)`
        );
      }
      actions.push(action);
      continue;
    }

    const tool = mod.default as AgentDirTool | undefined;
    if (!tool?.config || typeof tool.fn !== 'function') {
      return fail(
        `tools/${file} must \`export default defineDirTool({ ... }, fn)\` ` +
          `(a default export of { config, fn }) or a ` +
          `\`(ai) => ai.defineTool(...)\` factory`
      );
    }
    const name =
      tool.config.name ??
      `agent-dirs/${agentName}/${path.basename(file, path.extname(file))}`;
    actions.push(
      ai.defineTool(
        {
          name,
          description: tool.config.description,
          inputSchema: tool.config.inputSchema,
          outputSchema: tool.config.outputSchema,
        },
        async (input, ctx) => tool.fn(input, ctx)
      )
    );
  }
  return actions;
}

/**
 * Loads the optional `agent.{ts,mts,js,mjs}` override module from an agent
 * directory, if present. When several `agent.*` candidates exist (e.g. a
 * compiled `.js` beside its `.ts` source), the first extension in
 * ts/mts/js/mjs order wins, with a warning.
 */
export async function loadOverride(
  agentPath: string
): Promise<AgentDirOverride | undefined> {
  const candidates = TOOL_FILE_EXTENSIONS.map((ext) =>
    path.join(agentPath, `agent${ext}`)
  ).filter(existsSync);
  if (candidates.length === 0) return undefined;
  if (candidates.length > 1) {
    logger.warn(
      `[agent-dirs] multiple override candidates in ${agentPath} ` +
        `(${candidates.map((c) => path.basename(c)).join(', ')}) - using ` +
        `${path.basename(candidates[0])}`
    );
  }
  const overrideFile = candidates[0];
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
  return undefined;
}
