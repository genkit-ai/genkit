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

import type { AgentCard } from '@a2a-js/sdk';

/**
 * The A2A protocol version this plugin targets.
 */
export const A2A_PROTOCOL_VERSION = '0.3.0';

/**
 * The minimal shape this plugin reads off a Genkit agent to derive its card.
 * A Genkit `Agent` is a callable action carrying its name/description under
 * `__action`.
 */
export interface AgentLike {
  __action?: { name?: string; description?: string };
}

/**
 * Reads the agent's registered name (from its `__action` metadata).
 */
export function getAgentName(agent: AgentLike): string {
  const name = agent.__action?.name;
  if (!name) {
    throw new Error(
      'Unable to derive an A2A agent card: the provided agent has no name. ' +
        'Pass an explicit `card` to GenkitA2ARequestHandler.'
    );
  }
  return name;
}

/**
 * Reads the agent's registered description (from its `__action` metadata).
 */
export function getAgentDescription(agent: AgentLike): string | undefined {
  return agent.__action?.description;
}

/**
 * Options for deriving (and optionally overriding) the A2A {@link AgentCard}.
 */
export interface DeriveAgentCardOptions {
  /**
   * The base URL where the agent's A2A JSON-RPC endpoint is hosted, e.g.
   * `http://localhost:3000`. Required unless a fully-formed `card` (with a
   * `url`) is provided.
   */
  url?: string;
  /**
   * An explicit card (full or partial) to use/merge. Any provided fields
   * override the values derived from the agent; omitted fields fall back to
   * the derived defaults.
   */
  card?: Partial<AgentCard>;
  /** The agent's own version string. Defaults to `'0.0.0'`. */
  version?: string;
}

/**
 * Derives a complete A2A {@link AgentCard} from a Genkit agent.
 *
 * The card's `name`/`description` come from the agent (the description falls
 * back to a generic string), `url` from the options, and a single skill is
 * synthesized from the agent's identity. Any fields supplied via
 * `options.card` take precedence, so callers can override or extend the
 * derived card (e.g. add skills, security schemes, or a provider).
 */
export function deriveAgentCard(
  agent: AgentLike,
  options: DeriveAgentCardOptions = {}
): AgentCard {
  const override = options.card ?? {};

  const name = override.name ?? getAgentName(agent);
  const description =
    override.description ?? getAgentDescription(agent) ?? `The ${name} agent.`;

  const url = override.url ?? options.url;
  if (!url) {
    throw new Error(
      'Unable to derive an A2A agent card: no `url` provided. Pass `url` ' +
        '(the agent endpoint) to GenkitA2ARequestHandler or include it in `card`.'
    );
  }

  const defaultSkill = {
    id: name,
    name,
    description,
    tags: ['genkit'],
  };

  return {
    protocolVersion: override.protocolVersion ?? A2A_PROTOCOL_VERSION,
    name,
    description,
    url,
    version: override.version ?? options.version ?? '0.0.0',
    preferredTransport: override.preferredTransport ?? 'JSONRPC',
    capabilities: {
      streaming: true,
      pushNotifications: false,
      stateTransitionHistory: false,
      ...override.capabilities,
    },
    defaultInputModes: override.defaultInputModes ?? ['text/plain'],
    defaultOutputModes: override.defaultOutputModes ?? ['text/plain'],
    skills: override.skills ?? [defaultSkill],
    // Pass through any remaining optional fields the caller supplied.
    ...(override.provider && { provider: override.provider }),
    ...(override.documentationUrl && {
      documentationUrl: override.documentationUrl,
    }),
    ...(override.iconUrl && { iconUrl: override.iconUrl }),
    ...(override.additionalInterfaces && {
      additionalInterfaces: override.additionalInterfaces,
    }),
    ...(override.security && { security: override.security }),
    ...(override.securitySchemes && {
      securitySchemes: override.securitySchemes,
    }),
    ...(override.supportsAuthenticatedExtendedCard !== undefined && {
      supportsAuthenticatedExtendedCard:
        override.supportsAuthenticatedExtendedCard,
    }),
  };
}
