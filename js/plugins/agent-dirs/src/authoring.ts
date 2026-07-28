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
 * The authoring contract for files inside an agent directory. This module is
 * everything a directory author needs to know; the compiler machinery lives in
 * `compiler.ts` / `loaders.ts`.
 *
 * @module @genkit-ai/agent-dirs/authoring
 */

import { z } from 'genkit';
import type { GenkitBeta } from 'genkit/beta';

/**
 * The config half of a directory tool module. Mirrors the first argument of
 * `ai.defineTool` except that `name` is optional - it defaults to the tool's
 * filename (registered as `<agent>/<file>`; the model sees the short name).
 */
export interface AgentDirToolConfig<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
> {
  name?: string;
  description: string;
  inputSchema?: I;
  outputSchema?: O;
}

/**
 * The shape a `tools/*.ts` file must default-export: the same `(config, fn)`
 * pair `ai.defineTool` takes, packaged as a plain object so the file stays a
 * pure module (no registry access, unit-testable in isolation).
 */
export interface AgentDirTool<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
> {
  config: AgentDirToolConfig<I, O>;
  fn: (input: z.infer<I>) => Promise<z.infer<O>> | z.infer<O>;
}

/**
 * Typed authoring helper for directory tool files:
 *
 * ```ts
 * export default defineDirTool(
 *   { description: '...', inputSchema: z.object({...}) },
 *   async (input) => ...
 * );
 * ```
 */
export function defineDirTool<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
>(
  config: AgentDirToolConfig<I, O>,
  fn: (input: z.infer<I>) => Promise<z.infer<O>> | z.infer<O>
): AgentDirTool<I, O> {
  return { config, fn };
}

/** The `ai.defineAgent` config type, without repeating its generics. */
export type CompiledAgentConfig = Parameters<GenkitBeta['defineAgent']>[0];

/**
 * The shape an optional `agent.ts` override file must default-export: a
 * function that receives the convention-compiled config and returns the config
 * to register. Code wins, files fill in.
 */
export type AgentDirOverride = (
  config: CompiledAgentConfig
) => CompiledAgentConfig | Promise<CompiledAgentConfig>;
