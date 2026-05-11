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

import { z, type Action } from 'genkit';
import type { AgentInit, AgentInput, AgentOutput } from 'genkit/beta';

// Re-define the agent schemas locally so we don't need a direct dependency on
// `@genkit-ai/ai` or `@genkit-ai/core`.  These mirror the canonical schemas in
// `@genkit-ai/ai/agent` and must be kept in sync.

const AgentInitSchema = z.object({
  snapshotId: z.string().optional(),
  newSnapshotId: z.string().optional(),
  state: z.any().optional(),
});

const AgentInputSchema = z.object({
  messages: z.array(z.any()).optional(),
  resume: z
    .object({
      respond: z.array(z.any()).optional(),
      restart: z.array(z.any()).optional(),
    })
    .optional(),
  detach: z.boolean().optional(),
});

/**
 * Envelope schema that combines agent `init` and `input` into a single input
 * payload.  This is used by the {@link agentAdapter} to bridge transports that
 * do not support a separate `init` channel (e.g. Firebase callable functions
 * via `onCallGenkit`).
 */
export const AgentInputEnvelopeSchema = z.object({
  /** Initialization data (snapshotId, state, etc.). */
  init: AgentInitSchema.optional(),
  /** Agent input (messages, resume, detach). */
  input: AgentInputSchema,
});

/**
 * Envelope type that combines agent `init` and `input` into a single payload.
 *
 * Use this type on the client side when calling an agent deployed via
 * `onCallGenkit(agentAdapter(myAgent))`.
 */
export interface AgentInputEnvelope {
  /** Initialization data (snapshotId, state, etc.). */
  init?: AgentInit;
  /** Agent input (messages, resume, detach). */
  input: AgentInput;
}

/**
 * Wraps a Genkit Agent (a bidi action) into a plain {@link Action}
 * whose input is an {@link AgentInputEnvelope}.  The adapter unpacks the
 * envelope and forwards `init` and `input` to the underlying agent.
 *
 * This is designed for transports that only support a single `data` payload
 * and have no concept of a separate `init` field — most notably Firebase
 * callable functions (`onCallGenkit`).
 *
 * ### Server usage
 *
 * ```ts
 * import { agentAdapter } from '@genkit-ai/firebase/beta';
 *
 * const myAgent = ai.defineAgent({ name: 'myAgent', ... });
 *
 * export const myAgentFn = onCallGenkit(agentAdapter(myAgent));
 * ```
 *
 * ### Client usage
 *
 * ```ts
 * import type { AgentInputEnvelope } from '@genkit-ai/firebase/beta';
 *
 * const result = await myAgentFn.stream({
 *   init: { snapshotId: '...' },
 *   input: { messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
 * });
 * ```
 */
export function agentAdapter(agent: Action): Action {
  // We build a wrapper action by cloning the agent's callable interface
  // and re-mapping input/output schemas.  The wrapper intercepts every
  // call, unpacks the envelope, and delegates to the real agent.

  const wrapperRun = async (
    envelope: AgentInputEnvelope,
    options?: any
  ): Promise<{ result: AgentOutput; telemetry: any }> => {
    return agent.run(envelope.input as any, {
      ...options,
      init: envelope.init,
    });
  };

  const wrapperFn = async (
    envelope?: AgentInputEnvelope,
    options?: any
  ): Promise<AgentOutput> => {
    const result = await wrapperRun(envelope!, options);
    return result.result;
  };

  const wrapperStream = (envelope?: AgentInputEnvelope, opts?: any) => {
    return agent.stream(envelope?.input as any, {
      ...opts,
      init: envelope?.init,
    });
  };

  // Attach Action metadata so transports (e.g. onCallGenkit) can inspect
  // the schemas and treat this as a regular flow/action.
  (wrapperFn as any).__action = {
    ...agent.__action,
    actionType: 'flow',
    inputSchema: AgentInputEnvelopeSchema,
    // Keep output and stream schemas from the agent
  };
  (wrapperFn as any).run = wrapperRun;
  (wrapperFn as any).stream = wrapperStream;

  return wrapperFn as unknown as Action;
}
