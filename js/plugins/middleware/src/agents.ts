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

export const AgentsOptionsSchema = z.object({
  agents: z
    .array(z.string())
    .describe('Names of registered agents that can be called as sub-agents.'),
  toolName: z
    .string()
    .optional()
    .describe(
      'Custom name for the injected delegation tool. Defaults to "call_agent".'
    ),
});

export type AgentsOptions = z.infer<typeof AgentsOptionsSchema>;

/**
 * Creates a middleware that enables sub-agent delegation.
 *
 * Injects a `call_agent` tool that allows the model to delegate tasks to
 * registered sub-agents. The middleware intercepts the tool call, runs the
 * sub-agent via its `.run()` method, and returns the sub-agent's response
 * (including any artifacts) back to the calling model as the tool response.
 *
 * If a sub-agent triggers an interrupt, the interrupt is propagated up to the
 * main agent's caller as a `ToolInterruptError`.
 *
 * @example
 * ```typescript
 * const researcher = ai.defineAgent({ name: 'researcher', ... });
 * const coder = ai.defineAgent({ name: 'coder', ... });
 *
 * const orchestrator = ai.defineAgent({
 *   name: 'orchestrator',
 *   system: 'Delegate research to researcher, coding to coder.',
 *   use: [agents({ agents: ['researcher', 'coder'] })],
 * });
 * ```
 */
export const agents: GenerateMiddleware<typeof AgentsOptionsSchema> =
  generateMiddleware(
    {
      name: 'agents',
      description:
        'Injects a tool for delegating tasks to registered sub-agents.',
      configSchema: AgentsOptionsSchema,
    },
    ({ config, ai }) => {
      if (!config?.agents || config.agents.length === 0) {
        throw new Error(
          'agents middleware requires at least one agent name in the "agents" option.'
        );
      }

      const agentNames = config.agents;
      const delegationToolName = config.toolName ?? 'call_agent';

      // Cache resolved agent actions for performance across turns.
      const agentCache = new Map<string, Action>();

      async function resolveAgent(name: string): Promise<Action | undefined> {
        const cached = agentCache.get(name);
        if (cached) return cached;

        // Agents are registered as session-flows with actionType 'agent'.
        const action = await ai.registry.lookupAction(`/agent/${name}`);
        if (!action) {
          return undefined;
        }
        agentCache.set(name, action);
        return action;
      }

      const callAgentTool = tool(
        {
          name: delegationToolName,
          description:
            `Delegates a task to a sub-agent. Available agents: ${agentNames.join(', ')}. ` +
            `Provide the agent name and a clear task description. The agent will execute ` +
            `the task and return its response.`,
          inputSchema: z.object({
            agent: z
              .enum(agentNames as [string, ...string[]])
              .describe('The name of the sub-agent to call.'),
            task: z
              .string()
              .describe(
                'A clear description of the task to delegate to the sub-agent.'
              ),
          }),
          outputSchema: z.object({
            response: z.string().describe("The sub-agent's text response."),
            agentName: z
              .string()
              .describe('The name of the agent that was called.'),
            hasArtifacts: z
              .boolean()
              .optional()
              .describe('Whether the sub-agent produced artifacts.'),
          }),
        },
        async (input) => {
          const agentAction = await resolveAgent(input.agent);
          if (!agentAction) {
            return {
              response: `Error: Agent '${input.agent}' not found. Available agents: ${agentNames.join(', ')}`,
              agentName: input.agent,
            };
          }

          try {
            // Call the agent's run method with the task as a user message.
            const actionResult = await agentAction.run(
              {
                messages: [{ role: 'user', content: [{ text: input.task }] }],
              },
              { init: {} }
            );
            const agentOutput: AgentOutput = actionResult.result;

            // Extract text content from the agent's response message.
            const textContent = (agentOutput.message?.content ?? [])
              .map((p) => p.text)
              .filter((t): t is string => typeof t === 'string' && t.length > 0)
              .join('\n');

            return {
              response: textContent || '(no response)',
              agentName: input.agent,
              hasArtifacts: (agentOutput.artifacts?.length ?? 0) > 0,
            };
          } catch (e: unknown) {
            // If the sub-agent triggered an interrupt, propagate it up.
            if (e instanceof ToolInterruptError) {
              throw new ToolInterruptError({
                source: 'subagent',
                agentName: input.agent,
                task: input.task,
                originalInterrupt: e.metadata,
              });
            }

            // Other errors: return as tool error so the model can self-correct.
            const message = e instanceof Error ? e.message : String(e);
            return {
              response: `Error calling agent '${input.agent}': ${message}`,
              agentName: input.agent,
            };
          }
        }
      );

      return {
        tools: [callAgentTool],
        generate: async (envelope, ctx, next) => {
          const { request } = envelope;

          // Build agent descriptions for the system prompt.
          const agentList = agentNames.map((name) => `  - ${name}`).join('\n');

          const agentsInstructions =
            `<sub-agents>\n` +
            `You have access to the following sub-agents that you can delegate tasks to ` +
            `using the "${delegationToolName}" tool:\n` +
            `${agentList}\n` +
            `\n` +
            `When a task is better handled by a specialized agent, delegate it rather than ` +
            `attempting it yourself. Provide a clear, self-contained task description.\n` +
            `</sub-agents>`;

          // Inject into system message.
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
