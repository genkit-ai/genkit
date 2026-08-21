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
 * Dynamic import kept opaque to the bundler. Some transpile configs rewrite
 * `import()` to `require()` in CJS output (our current tsup setup does not,
 * but that is a build detail, not a contract); indirecting through `Function`
 * guarantees a native import in every build, which ESM tool files need.
 */
const dynamicImport = new Function('m', 'return import(m)') as (
  m: string
) => Promise<Record<string, unknown>>;

type TsImport = (
  specifier: string,
  parentURL: string
) => Promise<Record<string, unknown>>;

let tsImportPromise: Promise<TsImport | undefined> | undefined;

/** Resolves tsx's `tsImport` once, or undefined when tsx isn't installed. */
function resolveTsImport(): Promise<TsImport | undefined> {
  return (tsImportPromise ??= dynamicImport('tsx/esm/api').then(
    (mod) => mod.tsImport as TsImport,
    () => undefined
  ));
}

/**
 * Imports a directory module. TypeScript sources need tsx's `tsImport`: a
 * registered loader hook is not enough, because `.ts` files under a CJS (or
 * typeless) package scope are classified as CommonJS and a native `import()`
 * of them dies with a require/import cycle error. `tsImport` side-steps the
 * package scope entirely. Compiled `.js`/`.mjs` files import natively.
 */
async function importModule(file: string): Promise<Record<string, unknown>> {
  const url = pathToFileURL(file).href;
  if (/\.[cm]?ts$/.test(file)) {
    const tsImport = await resolveTsImport();
    if (tsImport) return tsImport(url, url);
    // No tsx (it is an optional peer): a native import still works when the
    // whole process runs under a TS-capable loader (`tsx src/index.ts`).
  }
  return dynamicImport(url);
}

function isLoadableModule(file: string): boolean {
  return (
    TOOL_FILE_EXTENSIONS.includes(path.extname(file)) &&
    !/\.d\.[cm]?ts$/.test(file)
  );
}

/**
 * Collapses `foo.ts` + `foo.mjs` (etc.) to one file per basename, preferring
 * {@link TOOL_FILE_EXTENSIONS} order - the same rule `loadOverride` applies
 * to `agent.*`. Without this, a precompiled tool next to its source would
 * register twice and clobber the registry entry.
 */
function dedupeByBasename(files: string[], agentName: string): string[] {
  const byBase = new Map<string, string[]>();
  for (const file of files) {
    const base = path.basename(file, path.extname(file));
    byBase.set(base, [...(byBase.get(base) ?? []), file]);
  }
  const picked: string[] = [];
  for (const [base, candidates] of byBase) {
    candidates.sort(
      (a, b) =>
        TOOL_FILE_EXTENSIONS.indexOf(path.extname(a)) -
        TOOL_FILE_EXTENSIONS.indexOf(path.extname(b))
    );
    if (candidates.length > 1) {
      logger.warn(
        `[agent-dirs] agent '${agentName}': multiple tool files named ` +
          `'${base}' (${candidates.join(', ')}) - using ${candidates[0]}`
      );
    }
    picked.push(candidates[0]);
  }
  return picked.sort();
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
  const files = dedupeByBasename(
    readdirSync(toolsDir).filter(isLoadableModule),
    agentName
  );
  for (const file of files) {
    let mod: Record<string, unknown>;
    try {
      mod = await importModule(path.join(toolsDir, file));
    } catch (e) {
      const hint = /\.[cm]?ts$/.test(file)
        ? ' (TypeScript tools need tsx installed, or precompile to .mjs)'
        : '';
      return fail(`failed to load tools/${file}: ${e}${hint}`);
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
    mod = await importModule(overrideFile);
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
