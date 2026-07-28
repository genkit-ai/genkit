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
 * Registry lookup helpers for agents registered by the plugin.
 *
 * @module @genkit-ai/agent-dirs/lookup
 */

import type { Genkit } from 'genkit';
import type { Agent } from 'genkit/beta';

/**
 * Looks up a directory-defined agent by name.
 *
 * Directory agents are registered by the plugin rather than user code, so
 * there is no `defineAgent` return value to hold on to; this resolves the
 * registered agent action (which implements the full `AgentAPI`: `chat`,
 * `loadChat`, `getSnapshotData`, `abort`).
 *
 * Named after the `remoteAgent` idiom - a noun-phrase factory for an Agent
 * handle you didn't `defineAgent` yourself, described by where it lives
 * (`remoteAgent` = behind a URL, `directoryAgent` = defined by a directory).
 * A future first-party equivalent would more naturally be an
 * `ai.agent(name)` instance method, mirroring `ai.prompt(name)`.
 */
export async function directoryAgent<State = unknown>(
  ai: Genkit,
  name: string
): Promise<Agent<State>> {
  await ai.registry.initializeAllPlugins();
  const action = await ai.registry.lookupAction(`/agent/${name}`);
  if (!action) {
    throw new Error(
      `[agent-dirs] no agent named '${name}' is registered. Is its ` +
        `directory under the plugin's 'dir' and does it contain agent.prompt?`
    );
  }
  if (typeof (action as unknown as Partial<Agent<State>>).chat !== 'function') {
    throw new Error(
      `[agent-dirs] action '/agent/${name}' is not an agent (no chat method).`
    );
  }
  return action as unknown as Agent<State>;
}

/**
 * Lists every registered agent, keyed by name. Useful for exposing all
 * directory agents over HTTP without naming them one by one.
 */
export async function listAgents(ai: Genkit): Promise<Record<string, Agent>> {
  const actions = await ai.registry.listActions();
  const agents: Record<string, Agent> = {};
  for (const [key, action] of Object.entries(actions)) {
    if (!key.startsWith('/agent/')) continue;
    if (typeof (action as unknown as Partial<Agent>).chat !== 'function')
      continue;
    agents[key.slice('/agent/'.length)] = action as unknown as Agent;
  }
  return agents;
}
