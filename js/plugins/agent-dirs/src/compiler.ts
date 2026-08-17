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
 * Compiles agent directories into `ai.defineAgent(...)` registrations.
 *
 * The convention is a table of capabilities: each capability folder or
 * frontmatter key contributes one middleware plus the model-visible tool
 * names it injects. The compiled agent is then just prompt fields + directory
 * tools + the contributed middleware stack. See {@link CAPABILITIES}.
 *
 * Authoring errors fail loudly by default (`strict: true`): a broken file
 * aborts registration with a pointed error instead of silently degrading the
 * agent. Set `strict: false` to warn-and-skip instead.
 *
 * @module @genkit-ai/agent-dirs/compiler
 */

import {
  agents as delegateToAgents,
  skills,
  toolApproval,
} from '@genkit-ai/middleware';
import { FileSessionStore, type GenkitBeta, type SessionStore } from 'genkit/beta';
import { logger } from 'genkit/logging';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import * as path from 'node:path';
import type { CompiledAgentConfig } from './authoring.js';
import { loadOverride, loadTools, type RegisteredTool } from './loaders.js';
import { okfKnowledge } from './okf.js';

/**
 * Last-resort model, used only when nothing else names one: frontmatter
 * `model`, then the plugin's `defaultModel` option, then a `genkit({ model })`
 * instance default all take precedence.
 */
export const DEFAULT_MODEL = 'vertexai/gemini-3.5-flash';

/** Options for the `agentDirs` plugin. */
export interface AgentDirsOptions {
  /** Directory containing one sub-directory per agent. Default `./agents`. */
  dir?: string;
  /**
   * Model for agents whose frontmatter has no `model` key. Setting this
   * pins the model at compile time, overriding any `genkit({ model })`
   * instance default; leaving it unset lets the instance default apply,
   * with {@link DEFAULT_MODEL} as the final fallback.
   */
  defaultModel?: string;
  /**
   * Session store factory, called once per agent. Defaults to a
   * {@link FileSessionStore} under `snapshotDir`.
   */
  store?: (agentName: string) => SessionStore<unknown>;
  /**
   * Root directory for the default file session stores.
   * Default `./.genkit/agent-snapshots`.
   */
  snapshotDir?: string;
  /**
   * When `true` (the default), authoring errors - unparseable frontmatter,
   * broken tool files, unknown `requireApproval`/`delegates` names - throw at
   * plugin initialization. When `false`, they log a warning and the broken
   * piece is skipped.
   */
  strict?: boolean;
}

/** A parsed `instructions.md` file (YAML frontmatter + markdown body). */
type ParsedPrompt = ReturnType<GenkitBeta['registry']['dotprompt']['parse']>;

/** Raw frontmatter of an `instructions.md`, for convention-specific keys. */
type Frontmatter = Record<string, unknown>;

type MiddlewareEntry = NonNullable<CompiledAgentConfig['use']>[number];

/**
 * The convention's frontmatter vocabulary. Keys outside this set are warned
 * about (the parser gives them no diagnostics).
 */
const KNOWN_FRONTMATTER_KEYS = new Set([
  'description',
  'model',
  'tools',
  'config',
  'delegates',
  'requireApproval',
]);

/**
 * What one capability adds to an agent: a middleware for the `use` chain, the
 * model-visible tool names that middleware injects (so tool gating can reason
 * about them), and a label for the registration log line.
 */
interface CapabilityContribution {
  label: string;
  middleware: MiddlewareEntry;
  toolNames: string[];
}

/**
 * A capability inspects the agent directory / frontmatter and either
 * contributes middleware or opts out with `undefined`.
 */
type Capability = (
  agentPath: string,
  frontmatter: Frontmatter
) => CapabilityContribution | undefined;

/** `skills/<name>/SKILL.md` -> first-party skills middleware. */
const skillsCapability: Capability = (agentPath) => {
  const dir = path.join(agentPath, 'skills');
  if (!existsSync(dir)) return undefined;
  return {
    label: 'skills',
    middleware: skills({ skillPaths: [dir] }),
    toolNames: ['use_skill'],
  };
};

/** `knowledge/` (an OKF bundle) -> okfKnowledge middleware. */
const knowledgeCapability: Capability = (agentPath) => {
  const dir = path.join(agentPath, 'knowledge');
  if (!existsSync(dir)) return undefined;
  return {
    label: 'knowledge',
    middleware: okfKnowledge({ knowledgePaths: [dir] }),
    toolNames: ['lookup_knowledge'],
  };
};

/** Frontmatter `delegates: [agent]` -> agents (delegation) middleware. */
const delegatesCapability: Capability = (_agentPath, frontmatter) => {
  const delegates = frontmatter.delegates as string[] | undefined;
  if (!delegates || delegates.length === 0) return undefined;
  return {
    label: `delegates [${delegates.join(', ')}]`,
    middleware: delegateToAgents({ agents: delegates }),
    toolNames: delegates.map((d) => `delegate_to_${d}`),
  };
};

/**
 * Every capability the convention knows. Order matters only for middleware
 * execution order. Tool gating (`requireApproval`) is applied after the
 * `agent.ts` override, because it needs the final tool set.
 */
const CAPABILITIES: Capability[] = [
  skillsCapability,
  knowledgeCapability,
  delegatesCapability,
];

/**
 * Scans `options.dir` and registers one agent per sub-directory. See the
 * package README for the directory layout.
 */
export async function compileAgentDirs(
  ai: GenkitBeta,
  options: AgentDirsOptions
): Promise<void> {
  const rootDir = path.resolve(options.dir ?? './agents');
  if (!existsSync(rootDir)) {
    logger.warn(`[agent-dirs] agents directory not found: ${rootDir}`);
    return;
  }
  const agentNames = readdirSync(rootDir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name);
  for (const name of agentNames) {
    await compileAgentDir(ai, path.join(rootDir, name), name, {
      options,
      siblingAgents: agentNames,
    });
  }
}

interface CompileContext {
  options: AgentDirsOptions;
  siblingAgents: string[];
}

async function compileAgentDir(
  ai: GenkitBeta,
  agentPath: string,
  agentName: string,
  ctx: CompileContext
): Promise<void> {
  const strict = ctx.options.strict ?? true;
  const fail = (message: string): false => {
    const full = `[agent-dirs] agent '${agentName}': ${message}`;
    if (strict) throw new Error(full);
    logger.warn(`${full} - skipping`);
    return false;
  };

  const parsed = parseInstructions(ai, agentPath, agentName, fail);
  if (!parsed) return;

  const frontmatter = validateFrontmatter(
    (parsed.raw ?? {}) as Frontmatter,
    ctx.siblingAgents,
    fail
  );
  if (!frontmatter) return;

  if (/\{\{\s*(role|history)\b/.test(parsed.template)) {
    fail(
      `instructions.md is used as the agent's system prompt; ` +
        `multi-message templates ({{role}}/{{history}}) are not supported here`
    );
    return;
  }

  const tools = await loadTools(ai, path.join(agentPath, 'tools'), agentName, {
    strict,
  });
  if (tools === undefined) return; // strict=false tool failure already logged

  const contributed = CAPABILITIES.map((capability) =>
    capability(agentPath, frontmatter)
  ).filter((c): c is CapabilityContribution => c !== undefined);

  // Model precedence: frontmatter > plugin defaultModel > genkit({ model })
  // instance default (left to apply at generate time by omitting the field)
  // > DEFAULT_MODEL.
  const model =
    parsed.model ??
    ctx.options.defaultModel ??
    (ai.options.model ? undefined : DEFAULT_MODEL);

  let config = assembleConfig(parsed, {
    agentName,
    tools,
    use: contributed.map((c) => c.middleware),
    store: resolveStore(agentName, ctx.options),
    model,
  });

  const override = await loadOverride(agentPath);
  if (override) {
    config = await override(config);
  }

  // Tool gating is applied to the FINAL config (post-override), so tools an
  // override adds or removes are gated correctly.
  const gated = (frontmatter.requireApproval as string[] | undefined) ?? [];
  if (gated.length > 0) {
    const known = finalToolNames(config, contributed);
    const unknown = gated.filter((g) => !known.includes(g));
    if (unknown.length > 0) {
      fail(
        `requireApproval names unknown tools [${unknown.join(', ')}] - ` +
          `known model-visible tool names: [${known.join(', ')}]. ` +
          `Approval gating is fail-closed, so this is an error.`
      );
      return;
    }
    config = {
      ...config,
      use: [
        ...(config.use ?? []),
        toolApproval({ approved: known.filter((n) => !gated.includes(n)) }),
      ],
    };
  }

  // A same-named agent (or prompt) registered by user code would be silently
  // clobbered by the registry - detect and refuse instead.
  if (await ai.registry.lookupAction(`/agent/${config.name}`)) {
    fail(
      `an action '/agent/${config.name}' is already registered - ` +
        `rename the directory or the conflicting defineAgent call`
    );
    return;
  }

  ai.defineAgent(config);
  logSummary(agentPath, config.name, tools, contributed, gated);
}

/**
 * Reads and parses `instructions.md` (YAML frontmatter + markdown body,
 * parsed with dotprompt's splitter). Detects dotprompt's silent YAML-error
 * fallback (it logs to console and returns the whole file - fence included -
 * as the template with empty metadata), which would otherwise become a
 * garbage system prompt.
 */
function parseInstructions(
  ai: GenkitBeta,
  agentPath: string,
  agentName: string,
  fail: (message: string) => false
): ParsedPrompt | undefined {
  const promptFile = path.join(agentPath, 'instructions.md');
  if (!existsSync(promptFile)) {
    fail(`no instructions.md in ${agentPath}`);
    return undefined;
  }
  const source = readFileSync(promptFile, 'utf8');
  let parsed: ParsedPrompt;
  try {
    parsed = ai.registry.dotprompt.parse(source);
  } catch (e) {
    fail(`failed to parse ${promptFile}: ${e}`);
    return undefined;
  }
  if (parsed.template.trimStart().startsWith('---')) {
    fail(
      `frontmatter in ${promptFile} failed to parse (invalid YAML?) - ` +
        `dotprompt fell back to treating the whole file as the template`
    );
    return undefined;
  }
  return parsed;
}

/**
 * Validates the convention's custom frontmatter keys (types and referenced
 * names) and warns on unknown bare keys, which dotprompt never diagnoses.
 * Returns the frontmatter with `delegates`/`requireApproval` normalized to
 * string arrays, or undefined if validation failed.
 */
function validateFrontmatter(
  frontmatter: Frontmatter,
  siblingAgents: string[],
  fail: (message: string) => false
): Frontmatter | undefined {
  for (const key of Object.keys(frontmatter)) {
    if (!KNOWN_FRONTMATTER_KEYS.has(key)) {
      logger.warn(
        `[agent-dirs] unknown frontmatter key '${key}' in instructions.md ` +
          `(known keys: ${[...KNOWN_FRONTMATTER_KEYS].join(', ')}) - ignoring`
      );
    }
  }
  for (const key of ['delegates', 'requireApproval'] as const) {
    const value = frontmatter[key];
    if (value === undefined) continue;
    if (
      !Array.isArray(value) ||
      value.some((v) => typeof v !== 'string')
    ) {
      fail(`frontmatter '${key}' must be a list of strings`);
      return undefined;
    }
  }
  const delegates = (frontmatter.delegates as string[] | undefined) ?? [];
  // A delegate that isn't a sibling directory may still be a code-registered
  // agent (registry contents aren't knowable at compile time), so this warns
  // rather than fails; the delegation middleware resolves names at runtime.
  const unknownDelegates = delegates.filter((d) => !siblingAgents.includes(d));
  if (unknownDelegates.length > 0) {
    logger.warn(
      `[agent-dirs] 'delegates' includes [${unknownDelegates.join(', ')}] ` +
        `which are not agent directories (found: [${siblingAgents.join(', ')}]) - ` +
        `assuming they are registered in code; a typo here will only fail ` +
        `at delegation time`
    );
  }
  return frontmatter;
}

function assembleConfig(
  parsed: ParsedPrompt,
  compiled: {
    agentName: string;
    tools: RegisteredTool[];
    use: MiddlewareEntry[];
    store: SessionStore<unknown>;
    model: string | undefined;
  }
): CompiledAgentConfig {
  const modelConfig =
    parsed.config && typeof parsed.config === 'object'
      ? (parsed.config as Record<string, unknown>)
      : undefined;
  const system = parsed.template.trim();
  return {
    name: compiled.agentName,
    ...(parsed.description && { description: parsed.description }),
    ...(compiled.model && { model: compiled.model }),
    ...(modelConfig && { config: modelConfig }),
    // An empty template (frontmatter-only file) means "no system prompt",
    // typically because an agent.ts override supplies one.
    ...(system && { system }),
    tools: [...compiled.tools, ...(parsed.tools ?? [])],
    ...(compiled.use.length > 0 ? { use: compiled.use } : {}),
    store: compiled.store,
  };
}

/**
 * Model-visible tool names of the final (post-override) config: directory and
 * frontmatter tools by short name, plus names injected by whichever
 * capability middlewares survived the override.
 */
function finalToolNames(
  config: CompiledAgentConfig,
  contributed: CapabilityContribution[]
): string[] {
  const names = new Set<string>();
  for (const tool of config.tools ?? []) {
    if (typeof tool === 'string') {
      names.add(shortName(tool));
    } else if (typeof tool === 'object' && tool !== null && '__action' in tool) {
      names.add(shortName((tool as RegisteredTool).__action.name));
    }
  }
  for (const contribution of contributed) {
    if (config.use?.includes(contribution.middleware)) {
      contribution.toolNames.forEach((n) => names.add(n));
    }
  }
  return [...names];
}

function resolveStore(
  agentName: string,
  options: AgentDirsOptions
): SessionStore<unknown> {
  // The default store is lazy: FileSessionStore mkdirs at construction, and
  // registering an agent must not litter the cwd with snapshot directories
  // before any session exists.
  return (
    options.store?.(agentName) ??
    lazySessionStore(
      () =>
        new FileSessionStore(
          path.join(
            options.snapshotDir ?? './.genkit/agent-snapshots',
            agentName
          )
        )
    )
  );
}

function lazySessionStore(
  make: () => SessionStore<unknown>
): SessionStore<unknown> {
  let store: SessionStore<unknown> | undefined;
  const resolved = () => (store ??= make());
  return {
    getSnapshot: (opts) => resolved().getSnapshot(opts),
    saveSnapshot: (snapshotId, mutator, opts) =>
      resolved().saveSnapshot(snapshotId, mutator, opts),
    onSnapshotStateChange: (snapshotId, callback, opts) =>
      resolved().onSnapshotStateChange?.(snapshotId, callback, opts),
  };
}

/**
 * One registration summary per agent, listing exactly what was compiled -
 * the difference between a working agent and a quietly degraded one should
 * be visible in the log.
 */
function logSummary(
  agentPath: string,
  agentName: string,
  tools: RegisteredTool[],
  contributed: CapabilityContribution[],
  gated: string[]
): void {
  const parts = [
    `tools [${tools.map((t) => shortName(t.__action.name)).join(', ')}]`,
  ];
  const skillNames = scanSkillNames(path.join(agentPath, 'skills'), agentName);
  if (skillNames) parts.push(`skills [${skillNames.join(', ')}]`);
  const knowledgeCount = countKnowledgeDocs(path.join(agentPath, 'knowledge'));
  if (knowledgeCount !== undefined) {
    parts.push(`knowledge ${knowledgeCount} docs`);
  }
  for (const c of contributed) {
    if (c.label.startsWith('delegates')) parts.push(c.label);
  }
  if (gated.length > 0) parts.push(`approval-gated [${gated.join(', ')}]`);
  logger.info(
    `[agent-dirs] registered agent '${agentName}': ${parts.join(', ')}`
  );
}

/** Skill directory names; also warns on layout mistakes the middleware ignores. */
function scanSkillNames(
  skillsDir: string,
  agentName: string
): string[] | undefined {
  if (!existsSync(skillsDir)) return undefined;
  const names: string[] = [];
  for (const entry of readdirSync(skillsDir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      const skillFile = path.join(skillsDir, entry.name, 'SKILL.md');
      if (!existsSync(skillFile)) continue;
      names.push(entry.name);
      const fmName = /^name:\s*(.+)$/m.exec(
        readFileSync(skillFile, 'utf8')
      )?.[1]?.trim();
      if (fmName && fmName !== entry.name) {
        logger.warn(
          `[agent-dirs] agent '${agentName}': skill '${entry.name}' declares ` +
            `frontmatter name '${fmName}', but the model-visible skill name ` +
            `is the directory name ('${entry.name}')`
        );
      }
    } else if (entry.name.endsWith('.md')) {
      logger.warn(
        `[agent-dirs] agent '${agentName}': skills/${entry.name} is ignored - ` +
          `skills live in sub-directories: skills/<name>/SKILL.md`
      );
    }
  }
  return names;
}

function countKnowledgeDocs(knowledgeDir: string): number | undefined {
  if (!existsSync(knowledgeDir)) return undefined;
  let count = 0;
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory() && !entry.name.startsWith('.')) {
        walk(path.join(dir, entry.name));
      } else if (
        entry.isFile() &&
        entry.name.endsWith('.md') &&
        entry.name !== 'log.md' &&
        entry.name !== 'index.md'
      ) {
        count++;
      }
    }
  };
  walk(knowledgeDir);
  return count;
}

/** The model-visible tool name: the segment after the last '/'. */
function shortName(name: string): string {
  return name.substring(name.lastIndexOf('/') + 1);
}
