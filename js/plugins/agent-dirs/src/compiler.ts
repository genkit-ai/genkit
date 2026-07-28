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

/** Options for the `agentDirs` plugin. */
export interface AgentDirsOptions {
  /** Directory containing one sub-directory per agent. Default `./agents`. */
  dir?: string;
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
}

/** A parsed `agent.prompt` file. */
type ParsedPrompt = ReturnType<GenkitBeta['registry']['dotprompt']['parse']>;

/** Raw frontmatter of an `agent.prompt`, for convention-specific keys. */
type Frontmatter = Record<string, unknown>;

type MiddlewareEntry = NonNullable<CompiledAgentConfig['use']>[number];

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
  const delegates = stringList(frontmatter.delegates);
  if (delegates.length === 0) return undefined;
  return {
    label: `${delegates.length} delegates`,
    middleware: delegateToAgents({ agents: delegates }),
    toolNames: delegates.map((d) => `delegate_to_${d}`),
  };
};

/**
 * Every capability the convention knows. Order matters only for middleware
 * execution order. Tool gating (`requireApproval`) is applied after these, in
 * {@link approvalContribution}, because it needs the union of all tool names.
 */
const CAPABILITIES: Capability[] = [
  skillsCapability,
  knowledgeCapability,
  delegatesCapability,
];

/**
 * Frontmatter `requireApproval: [toolName]` -> toolApproval middleware. The
 * middleware takes an allow-list, so the approved set is the complement:
 * every model-visible tool name the agent has that is not listed.
 */
function approvalContribution(
  frontmatter: Frontmatter,
  allToolNames: string[]
): Pick<CapabilityContribution, 'label' | 'middleware'> | undefined {
  const gated = stringList(frontmatter.requireApproval);
  if (gated.length === 0) return undefined;
  return {
    label: 'tool approval',
    middleware: toolApproval({
      approved: allToolNames.filter((n) => !gated.includes(n)),
    }),
  };
}

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
  for (const entry of readdirSync(rootDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    await compileAgentDir(
      ai,
      path.join(rootDir, entry.name),
      entry.name,
      options
    );
  }
}

async function compileAgentDir(
  ai: GenkitBeta,
  agentPath: string,
  agentName: string,
  options: AgentDirsOptions
): Promise<void> {
  const parsed = parseAgentPrompt(ai, agentPath, agentName);
  if (!parsed) return;
  const frontmatter: Frontmatter = (parsed.raw ?? {}) as Frontmatter;

  const tools = await loadTools(ai, path.join(agentPath, 'tools'), agentName);

  const contributed = CAPABILITIES.map((capability) =>
    capability(agentPath, frontmatter)
  ).filter((c): c is CapabilityContribution => c !== undefined);

  const use = contributed.map((c) => c.middleware);
  const labels = contributed.map((c) => c.label);

  const modelVisibleTools = [
    ...tools.map((t) => shortName(t.__action.name)),
    ...(parsed.tools ?? []).map(shortName),
    ...contributed.flatMap((c) => c.toolNames),
  ];
  const approval = approvalContribution(frontmatter, modelVisibleTools);
  if (approval) {
    use.push(approval.middleware);
    labels.push(approval.label);
  }

  let config = assembleConfig(parsed, {
    agentName,
    tools,
    use,
    store: resolveStore(agentName, options),
  });

  const override = await loadOverride(agentPath);
  if (override) {
    config = await override(config);
  }

  ai.defineAgent(config);
  logger.info(
    `[agent-dirs] registered agent '${config.name}' ` +
      `(${[`${tools.length} tools`, ...labels].join(', ')})`
  );
}

/** Reads and parses `agent.prompt`; warns and skips the agent on failure. */
function parseAgentPrompt(
  ai: GenkitBeta,
  agentPath: string,
  agentName: string
): ParsedPrompt | undefined {
  const promptFile = path.join(agentPath, 'agent.prompt');
  if (!existsSync(promptFile)) {
    logger.warn(
      `[agent-dirs] skipping '${agentName}': no agent.prompt in ${agentPath}`
    );
    return undefined;
  }
  try {
    return ai.registry.dotprompt.parse(readFileSync(promptFile, 'utf8'));
  } catch (e) {
    logger.warn(
      `[agent-dirs] skipping '${agentName}': failed to parse ${promptFile}: ${e}`
    );
    return undefined;
  }
}

function assembleConfig(
  parsed: ParsedPrompt,
  compiled: {
    agentName: string;
    tools: RegisteredTool[];
    use: MiddlewareEntry[];
    store: SessionStore<unknown>;
  }
): CompiledAgentConfig {
  const modelConfig =
    parsed.config && typeof parsed.config === 'object'
      ? (parsed.config as Record<string, unknown>)
      : undefined;
  return {
    name: compiled.agentName,
    ...(parsed.description && { description: parsed.description }),
    ...(parsed.model && { model: parsed.model }),
    ...(modelConfig && { config: modelConfig }),
    system: parsed.template.trim(),
    tools: [...compiled.tools, ...(parsed.tools ?? [])],
    ...(compiled.use.length > 0 ? { use: compiled.use } : {}),
    store: compiled.store,
  };
}

function resolveStore(
  agentName: string,
  options: AgentDirsOptions
): SessionStore<unknown> {
  return (
    options.store?.(agentName) ??
    new FileSessionStore(
      path.join(options.snapshotDir ?? './.genkit/agent-snapshots', agentName)
    )
  );
}

/** The model-visible tool name: the segment after the last '/'. */
function shortName(name: string): string {
  return name.substring(name.lastIndexOf('/') + 1);
}

function stringList(v: unknown): string[] {
  return Array.isArray(v)
    ? v.filter((x): x is string => typeof x === 'string')
    : [];
}
