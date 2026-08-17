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
 * Experimental "agent = directory" convention layer for Genkit.
 *
 * ```ts
 * import { genkit } from 'genkit/beta';
 * import { agentDirs, directoryAgent } from '@genkit-ai/agent-dirs';
 *
 * const ai = genkit({ plugins: [agentDirs({ dir: './agents' })] });
 * const support = await directoryAgent(ai, 'support');
 * const chat = support.chat();
 * ```
 *
 * Every sub-directory of `dir` compiles to one `ai.defineAgent(...)` call:
 * `instructions.md` (YAML frontmatter + markdown system prompt) supplies
 * model/config/system prompt, `tools/*.ts`
 * supply tools, `skills/*.md` supply progressively-disclosed instructions, and
 * an optional `agent.ts` can amend the compiled config in code.
 *
 * @module @genkit-ai/agent-dirs
 */

import type { Genkit } from 'genkit';
import type { GenkitBeta } from 'genkit/beta';
import { genkitPlugin } from 'genkit/plugin';
import { compileAgentDirs, type AgentDirsOptions } from './compiler.js';

export {
  defineDirTool,
  type AgentDirOverride,
  type AgentDirTool,
  type AgentDirToolConfig,
  type AgentDirToolContext,
  type AgentDirToolFactory,
  type CompiledAgentConfig,
} from './authoring.js';
export { type AgentDirsOptions } from './compiler.js';
export { directoryAgent, listAgents } from './lookup.js';
export {
  serveAgents,
  type AgentServer,
  type ServeAgentsOptions,
} from './server.js';
export {
  okfKnowledge,
  OkfKnowledgeOptionsSchema,
  type OkfKnowledgeOptions,
} from './okf.js';

/**
 * Genkit plugin that registers every agent directory under `options.dir`
 * (default `./agents`) as an agent.
 */
export function agentDirs(options: AgentDirsOptions = {}) {
  return genkitPlugin('agent-dirs', async (ai: Genkit) => {
    await compileAgentDirs(ai as GenkitBeta, options);
  });
}
