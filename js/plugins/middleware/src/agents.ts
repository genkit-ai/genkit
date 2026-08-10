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
  z,
  type Action,
  type GenerateMiddleware,
  type MessageData,
  type Part,
} from 'genkit';
import { tool, type AgentOutput, type Artifact } from 'genkit/beta';

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
  artifactStrategy: z
    .enum(['inline', 'session'])
    .optional()
    .describe(
      'How sub-agent artifacts are handled:\n' +
        '  - "inline" (default): artifact content is included in the delegation ' +
        'tool result so the orchestrator model can see it, AND artifacts are ' +
        'merged into the parent session.\n' +
        '  - "session": artifacts are merged into the parent session only. ' +
        'The tool result mentions artifact names but not content. Use the ' +
        '"artifacts" middleware to give the model read/write access to session artifacts.'
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

/**
 * Generates a short, unique invocation ID for a sub-agent call.
 * Format: `{agentName}_{random4}` — e.g. `researcher_k9m2`
 */
function makeInvocationId(agentName: string): string {
  const random = Math.random().toString(36).slice(2, 6);
  return `${agentName}_${random}`;
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
 * 4. Returns the sub-agent's response as the tool result.
 *
 * Artifact handling is controlled by the `artifactStrategy` option:
 *
 * - `"inline"` (default): Artifact content is included in the tool result
 *   so the orchestrator model can reason about it, AND artifacts are merged
 *   into the parent session (prefixed with an invocation ID for namespacing).
 * - `"session"`: Artifacts are merged into the parent session only. The tool
 *   result mentions artifact names but not content. Pair with the `artifacts`
 *   middleware to give the model `read_artifact` / `write_artifact` tools.
 *
 * If a sub-agent triggers an interrupt, it is reported back to the orchestrator
 * as a normal tool response (not propagated as a `ToolInterruptError`). There is
 * no stateful sub-agent runtime to resume into, so interactive, back-and-forth
 * interaction with an interrupted sub-agent is a future feature.
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
 *       artifactStrategy: 'session', // pair with artifacts() middleware
 *     }),
 *     artifacts(),
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
      const artifactStrategy = config.artifactStrategy ?? 'inline';

      // Shared mutable state — safe because `instantiate()` is called per
      // `generate()` invocation, giving each call its own closure.
      const shared = {
        delegationCount: 0,
        conversationMessages: [] as MessageData[],
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

      // -- Build the output schema based on artifactStrategy ----------------

      const inlineArtifactSchema = z.object({
        name: z.string().optional().describe('Name of the artifact.'),
        content: z
          .string()
          .optional()
          .describe('Text content of the artifact.'),
      });

      const sessionArtifactSchema = z.object({
        name: z.string().optional().describe('Name of the artifact.'),
      });

      const toolOutputSchema = z.object({
        response: z.string().describe("The sub-agent's text response."),
        artifacts: z
          .array(
            artifactStrategy === 'inline'
              ? inlineArtifactSchema
              : sessionArtifactSchema
          )
          .optional()
          .describe(
            artifactStrategy === 'inline'
              ? 'Artifacts produced by the sub-agent, including their content.'
              : 'Names of artifacts produced by the sub-agent. Use read_artifact to access content.'
          ),
      });

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
            outputSchema: toolOutputSchema,
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
              // Optionally include recent conversation history as context.
              // Only user/model messages are forwarded, and each is reduced to
              // its text parts. This avoids leaking tool/tool-request parts —
              // a model message mid-tool-loop can carry dangling `toolRequest`
              // parts with no matching tool response, which would confuse the
              // sub-agent model.
              const historyMsgs: MessageData[] = [];
              if (historyLength > 0 && shared.conversationMessages.length > 0) {
                const contextMsgs = shared.conversationMessages
                  .filter((m) => m.role === 'user' || m.role === 'model')
                  .slice(-historyLength)
                  .map((m) => ({
                    role: m.role,
                    content: m.content.filter(
                      (p): p is Part & { text: string } =>
                        typeof p.text === 'string' && p.text.length > 0
                    ),
                  }))
                  .filter((m) => m.content.length > 0);
                historyMsgs.push(...contextMsgs);
              }

              // The agent input accepts a single `message` (the task). Prior
              // conversation is seeded via the session state (`init.state`),
              // which only client-managed agents (no persistent store) accept —
              // sending `state` to a server-managed agent throws a precondition
              // error. Server-managed sub-agents can't be seeded with ad-hoc
              // per-delegation history, so history forwarding is skipped for
              // them (the task is still delivered).
              const stateManagement =
                agentAction.__action?.metadata?.agent?.stateManagement;
              const init =
                historyMsgs.length > 0 && stateManagement !== 'server'
                  ? { state: { messages: historyMsgs } }
                  : {};

              const actionResult = await agentAction.run(
                {
                  message: {
                    role: 'user' as const,
                    content: [{ text: input.task }],
                  },
                },
                { init }
              );
              const agentOutput: AgentOutput = actionResult.result;

              // The agent runtime resolves gracefully rather than throwing:
              // a failed turn returns `finishReason: 'failed'` with structured
              // error details, and an interrupted turn returns
              // `finishReason: 'interrupted'`. Handle both explicitly here (the
              // `catch` below only fires for exceptions thrown outside the
              // agent's graceful handling).

              // ── Interrupt: surface as a normal tool response ──────
              // We deliberately do NOT propagate the interrupt to the parent.
              // Throwing would interrupt the parent's generate, but there is no
              // stateful sub-agent runtime to resume back into — the sub-agent's
              // turn state is already gone — so the parent could never satisfy
              // the interrupt. Interactive, stateful sub-agent interaction is a
              // future feature. For now we report it as text the orchestrator
              // can reason about.
              if (agentOutput.finishReason === 'interrupted') {
                return {
                  response:
                    `Sub-agent '${ref.name}' interrupted for additional input ` +
                    `and could not complete the task. Interactive sub-agent ` +
                    `interrupts are not currently supported; try delegating a ` +
                    `more self-contained task.`,
                };
              }

              // ── Failure: surface the error to the orchestrator ────
              if (agentOutput.finishReason === 'failed') {
                const message =
                  agentOutput.error?.message ?? 'Unknown sub-agent failure.';
                return {
                  response: `Error calling agent '${ref.name}': ${message}`,
                };
              }

              // Extract text content from the agent's response.
              const textContent = (agentOutput.message?.content ?? [])
                .map((p) => p.text)
                .filter(
                  (t): t is string => typeof t === 'string' && t.length > 0
                )
                .join('\n');

              // ── Artifact handling ─────────────────────────────────
              const subArtifacts: Artifact[] = (
                agentOutput.artifacts ?? []
              ).filter((a) => a.name);

              // Generate a unique invocation ID to namespace artifacts
              const invocationId = makeInvocationId(ref.name);

              // Merge artifacts into the parent session (both strategies).
              // `ai.currentSession()` throws when there is no active session,
              // so guard with try/catch rather than a falsy check.
              if (subArtifacts.length > 0) {
                try {
                  const session = ai.currentSession();
                  const namespacedArtifacts: Artifact[] = subArtifacts.map(
                    (a) => ({
                      ...a,
                      name: `${invocationId}/${a.name}`,
                      metadata: {
                        ...a.metadata,
                        source: ref.name,
                        invocationId,
                      },
                    })
                  );
                  session.addArtifacts(namespacedArtifacts);
                } catch {
                  // No active session — artifacts can't be merged into a
                  // parent session. With the "inline" strategy the content is
                  // still returned in the tool result below.
                }
              }

              // Build tool result based on strategy
              let artifacts: { name: string; content?: string }[] | undefined;
              if (subArtifacts.length > 0) {
                if (artifactStrategy === 'inline') {
                  // Include full content in tool result
                  artifacts = subArtifacts.map((a) => ({
                    name: `${invocationId}/${a.name}`,
                    content: (a.parts ?? [])
                      .map((p) => p.text ?? '')
                      .filter((t) => t.length > 0)
                      .join('\n'),
                  }));
                } else {
                  // Session strategy: names only
                  artifacts = subArtifacts.map((a) => ({
                    name: `${invocationId}/${a.name}`,
                  }));
                }
              }

              return {
                response: textContent || '(no response)',
                ...(artifacts?.length ? { artifacts } : {}),
              };
            } catch (e: unknown) {
              // The agent runtime resolves failures and interrupts gracefully
              // (see above), so this only fires for exceptions thrown outside
              // that handling (e.g. schema parse errors on `run`). Return them
              // as tool output so the model can recover.
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
