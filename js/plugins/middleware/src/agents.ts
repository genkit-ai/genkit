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

import {
  generateMiddleware,
  ToolInterruptError,
  z,
  type Action,
  type GenerateMiddleware,
} from 'genkit';
import { tool, type AgentOutput } from 'genkit/beta';

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/**
 * An agent reference: either a plain name string or an object with
 * `name` and an optional `description` override.
 */
const AgentRefSchema = z.union([
  z.string(),
  z.object({
    name: z.string().describe('Name of the registered agent.'),
    description: z
      .string()
      .optional()
      .describe(
        'Custom description for this agent. Overrides the auto-discovered description from the registry.'
      ),
  }),
]);

export const AgentsOptionsSchema = z.object({
  agents: z
    .array(AgentRefSchema)
    .describe(
      'Agents available for delegation. Each entry can be a name string ' +
        'or an object with a name and optional description override.'
    ),
  toolPrefix: z
    .string()
    .optional()
    .describe(
      'Prefix for generated delegation tool names. Defaults to "delegate_to" ' +
        '(tools become delegate_to_<agent>). Set to empty string to use bare agent names.'
    ),
  maxDelegations: z
    .number()
    .optional()
    .describe(
      'Maximum sub-agent delegations allowed per generate call. ' +
        'Prevents runaway delegation loops.'
    ),
  historyLength: z
    .number()
    .optional()
    .describe(
      'Number of recent conversation messages (user/model only) to forward ' +
        'to sub-agents as additional context. 0 or omitted means only the ' +
        'task description is sent.'
    ),
});

export type AgentsOptions = z.infer<typeof AgentsOptionsSchema>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface NormalizedAgentRef {
  name: string;
  description?: string;
}

function normalizeRef(
  ref: string | { name: string; description?: string }
): NormalizedAgentRef {
  return typeof ref === 'string' ? { name: ref } : ref;
}

function makeToolName(prefix: string, agentName: string): string {
  return prefix ? `${prefix}_${agentName}` : agentName;
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/**
 * Creates a middleware that enables sub-agent delegation.
 *
 * For every agent listed in the configuration the middleware injects a
 * dedicated delegation tool (e.g. `delegate_to_researcher`) whose description
 * is automatically populated from the agent's registry metadata — or can be
 * overridden in configuration. A `<sub-agents>` block is appended to the
 * system prompt listing the available agents and their descriptions.
 *
 * When the model calls a delegation tool the middleware:
 *
 * 1. Resolves the target agent from the registry.
 * 2. Optionally forwards recent conversation history as context.
 * 3. Runs the sub-agent with the task.
 * 4. Returns the sub-agent's response (including artifact metadata) as the
 *    tool result.
 *
 * If a sub-agent triggers an interrupt, the interrupt is propagated to the
 * caller as a `ToolInterruptError`.
 *
 * @example
 * ```typescript
 * const researcher = ai.defineAgent({
 *   name: 'researcher',
 *   description: 'Searches the web and summarizes findings.',
 *   ...
 * });
 * const coder = ai.defineAgent({ name: 'coder', ... });
 *
 * const orchestrator = ai.defineAgent({
 *   name: 'orchestrator',
 *   system: 'You are a helpful project assistant.',
 *   use: [
 *     agents({
 *       agents: [
 *         'researcher',                                           // auto-discovered description
 *         { name: 'coder', description: 'Writes TypeScript code' }, // explicit override
 *       ],
 *       maxDelegations: 5,
 *       historyLength: 4,
 *     }),
 *   ],
 * });
 * ```
 */
export const agents: GenerateMiddleware<typeof AgentsOptionsSchema> =
  generateMiddleware(
    {
      name: 'agents',
      description:
        'Injects per-agent delegation tools for calling registered sub-agents.',
      configSchema: AgentsOptionsSchema,
    },
    ({ config, ai }) => {
      if (!config?.agents || config.agents.length === 0) {
        throw new Error(
          'agents middleware requires at least one agent in the "agents" option.'
        );
      }

      const agentRefs = config.agents.map(normalizeRef);
      const prefix = config.toolPrefix ?? 'delegate_to';
      const maxDelegations = config.maxDelegations;
      const historyLength = config.historyLength ?? 0;

      // Shared mutable state — safe because `instantiate()` is called per
      // `generate()` invocation, giving each call its own closure.
      const shared = {
        delegationCount: 0,
        conversationMessages: [] as any[],
      };

      // Caches (persist across turns within the same generate cycle).
      const agentCache = new Map<string, Action>();
      const descriptionCache = new Map<string, string>();

      async function resolveAgent(name: string): Promise<Action | undefined> {
        const cached = agentCache.get(name);
        if (cached) return cached;

        const action = await ai.registry.lookupAction(`/agent/${name}`);
        if (action) {
          agentCache.set(name, action);
        }
        return action;
      }

      async function discoverDescription(
        name: string
      ): Promise<string | undefined> {
        const cached = descriptionCache.get(name);
        if (cached !== undefined) return cached;

        // Try the agent action first.
        const agentAction = await ai.registry.lookupAction(`/agent/${name}`);
        let desc = agentAction?.__action?.description;

        // Fallback: `defineAgent` stores the description on the prompt action.
        if (!desc) {
          const promptAction = await ai.registry.lookupAction(
            `/prompt/${name}`
          );
          desc = promptAction?.__action?.description;
        }

        if (desc) {
          descriptionCache.set(name, desc);
        }
        return desc;
      }

      // ── Per-agent delegation tools ────────────────────────────────────

      const delegationTools = agentRefs.map((ref) => {
        const toolName = makeToolName(prefix, ref.name);
        const staticDescription =
          ref.description ?? `Delegates a task to the "${ref.name}" sub-agent.`;

        return tool(
          {
            name: toolName,
            description: staticDescription,
            inputSchema: z.object({
              task: z
                .string()
                .describe(
                  'A clear, self-contained description of the task to delegate.'
                ),
            }),
            outputSchema: z.object({
              response: z.string().describe("The sub-agent's text response."),
              artifacts: z
                .array(
                  z.object({
                    name: z
                      .string()
                      .optional()
                      .describe('Name of the artifact.'),
                  })
                )
                .optional()
                .describe('Artifacts produced by the sub-agent, if any.'),
            }),
          },
          async (input) => {
            // ── Guard rail ──────────────────────────────────────────
            if (
              maxDelegations !== undefined &&
              shared.delegationCount >= maxDelegations
            ) {
              return {
                response:
                  `Delegation limit reached (${maxDelegations}). ` +
                  `Complete the task using information already gathered.`,
              };
            }
            shared.delegationCount++;

            const agentAction = await resolveAgent(ref.name);
            if (!agentAction) {
              return {
                response: `Error: Agent '${ref.name}' not found in registry.`,
              };
            }

            try {
              // Build messages for the sub-agent.
              const messages: any[] = [];

              // Optionally include recent conversation history as context.
              if (historyLength > 0 && shared.conversationMessages.length > 0) {
                const contextMsgs = shared.conversationMessages
                  .filter((m: any) => m.role === 'user' || m.role === 'model')
                  .slice(-historyLength);
                messages.push(...contextMsgs);
              }

              // The task itself as a user message.
              messages.push({
                role: 'user' as const,
                content: [{ text: input.task }],
              });

              const actionResult = await agentAction.run(
                { messages },
                { init: {} }
              );
              const agentOutput: AgentOutput = actionResult.result;

              // Extract text content from the agent's response.
              const textContent = (agentOutput.message?.content ?? [])
                .map((p: any) => p.text)
                .filter(
                  (t: unknown): t is string =>
                    typeof t === 'string' && (t as string).length > 0
                )
                .join('\n');

              // Extract artifact metadata for the model.
              const artifacts = agentOutput.artifacts
                ?.filter((a: any) => a.name)
                .map((a: any) => ({ name: a.name }));

              return {
                response: textContent || '(no response)',
                ...(artifacts?.length ? { artifacts } : {}),
              };
            } catch (e: unknown) {
              // If the sub-agent triggered an interrupt, propagate it up.
              if (e instanceof ToolInterruptError) {
                throw new ToolInterruptError({
                  source: 'subagent',
                  agentName: ref.name,
                  task: input.task,
                  originalInterrupt: e.metadata,
                });
              }

              // Other errors: return as tool output so the model can recover.
              const message = e instanceof Error ? e.message : String(e);
              return {
                response: `Error calling agent '${ref.name}': ${message}`,
              };
            }
          }
        );
      });

      return {
        tools: delegationTools,

        generate: async (envelope, ctx, next) => {
          const { request } = envelope;

          // Capture the latest messages for optional history forwarding.
          // Note: delegationCount is NOT reset here — the generate hook runs
          // on every turn of the tool loop, but the count must accumulate
          // across the entire generate() call.  The initial value of 0 is
          // set when instantiate() creates the closure.
          shared.conversationMessages = request.messages ?? [];

          // ── Auto-discover descriptions for the system prompt ──────
          const agentDescriptions = await Promise.all(
            agentRefs.map(async (ref) => {
              const description =
                ref.description ??
                (await discoverDescription(ref.name)) ??
                'No description available.';
              return {
                name: ref.name,
                toolName: makeToolName(prefix, ref.name),
                description,
              };
            })
          );

          const agentList = agentDescriptions
            .map((a) => `  - ${a.toolName}: ${a.description}`)
            .join('\n');

          const agentsInstructions =
            `<sub-agents>\n` +
            `You can delegate tasks to specialized sub-agents using their ` +
            `delegation tools:\n` +
            `${agentList}\n` +
            `\n` +
            `When a task is better handled by a specialized agent, delegate ` +
            `it using the appropriate tool. Provide a clear, self-contained ` +
            `task description.\n` +
            `</sub-agents>`;

          // ── Inject into system message ────────────────────────────
          const messages = [...request.messages];
          const MARKER_KEY = 'agents-middleware-instructions';

          // Check if we've already injected (multi-turn).
          const alreadyInjected = messages.some((msg) =>
            msg.content.some(
              (part) => part.text && part.metadata?.[MARKER_KEY] === true
            )
          );

          if (!alreadyInjected) {
            const systemIdx = messages.findIndex((m) => m.role === 'system');
            if (systemIdx !== -1) {
              messages[systemIdx] = {
                ...messages[systemIdx],
                content: [
                  ...messages[systemIdx].content,
                  {
                    text: agentsInstructions,
                    metadata: { [MARKER_KEY]: true },
                  },
                ],
              };
            } else {
              messages.unshift({
                role: 'system',
                content: [
                  {
                    text: agentsInstructions,
                    metadata: { [MARKER_KEY]: true },
                  },
                ],
              });
            }
          }

          return next({ ...envelope, request: { ...request, messages } }, ctx);
        },
      };
    }
  );
