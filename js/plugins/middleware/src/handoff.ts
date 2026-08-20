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
  type GenerateMiddleware,
  type MessageData,
} from 'genkit';
import { tool } from 'genkit/beta';

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/**
 * A single specialized agent ("persona") the conversation can be handed off to.
 *
 * Unlike the `agents` middleware (which *delegates* a one-shot subtask and gets
 * a result back), a handoff *swaps the active persona*: its system instructions
 * and tools replace the active set for all subsequent turns, so the user keeps
 * talking directly to the specialist until it transfers again.
 */
const PersonaSchema = z.object({
  name: z
    .string()
    .describe('Unique persona name (e.g. "triage", "refund", "billing").'),
  description: z
    .string()
    .optional()
    .describe(
      'Short description of what this persona handles. Shown to other ' +
        'personas as the description of the transfer tool that targets it, ' +
        'so they know when to hand off to it.'
    ),
  system: z
    .string()
    .describe(
      'System instructions that drive this persona while it is active.'
    ),
  tools: z
    .array(z.string())
    .optional()
    .describe(
      'Names of registered tools (from ai.defineTool) this persona can use. ' +
        'Only the active persona\u2019s tools are exposed to the model; other ' +
        "personas' tools are hidden until the conversation is handed to them."
    ),
});

export const HandoffOptionsSchema = z.object({
  personas: z
    .array(PersonaSchema)
    .describe(
      'The specialized agents the conversation can be handed off between.'
    ),
  defaultPersona: z
    .string()
    .optional()
    .describe(
      'Name of the persona that drives the conversation before any transfer ' +
        'happens. Defaults to the first persona in the list.'
    ),
  toolPrefix: z
    .string()
    .optional()
    .describe(
      'Prefix for generated transfer tool names. Defaults to "transfer_to" ' +
        '(tools become transfer_to_<persona>). Set to empty string to use ' +
        'bare persona names.'
    ),
  maxTransfers: z
    .number()
    .optional()
    .describe(
      'Maximum number of transfers allowed per generate call. Prevents ' +
        'runaway transfer loops (e.g. two personas bouncing back and forth).'
    ),
});

export type HandoffOptions = z.infer<typeof HandoffOptionsSchema>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface Persona {
  name: string;
  description?: string;
  system: string;
  tools: string[];
}

function makeToolName(prefix: string, personaName: string): string {
  return prefix ? `${prefix}_${personaName}` : personaName;
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/**
 * Creates a middleware that enables the "agent transfer" / handoff pattern.
 *
 * A single host agent hosts several **personas** (specialized sub-agents). At
 * any moment exactly one persona is *active*: its system instructions and tools
 * drive the model. The middleware injects a `transfer_to_<persona>` tool for
 * every persona; when the model calls one, the active persona swaps for all
 * subsequent turns, so the user keeps talking directly to the new specialist.
 * Personas can transfer back to the default (e.g. a triage agent) or to any
 * other persona, enabling multi-turn conversations that flow between
 * specialists.
 *
 * The active persona is tracked statelessly from the conversation history (the
 * most recent successful transfer tool response wins), so it survives multi-turn
 * conversations and works with both client- and server-managed agents, with or
 * without a session.
 *
 * How it differs from the `agents` middleware: `agents` *delegates* a
 * self-contained subtask to a sub-agent and returns its result to the
 * orchestrator (the orchestrator stays in control). `handoff` *transfers
 * control*: the persona itself changes and stays changed across turns.
 *
 * Notes / limitations:
 *
 * - Persona `tools` must be names of tools registered via `ai.defineTool`. Only
 *   the active persona's tools are exposed to the model; the transfer tools are
 *   always available so any persona can hand off.
 * - The middleware chain is fixed for the duration of a `generate()` call, so
 *   per-persona *middleware* (e.g. a different `artifacts()` per persona) is not
 *   supported. Put shared middleware on the host agent. The system prompt and
 *   tools do swap per persona.
 *
 * @example
 * ```typescript
 * ai.defineTool({ name: 'lookupOrder', ... }, async (i) => { ... });
 * ai.defineTool({ name: 'issueRefund', ... }, async (i) => { ... });
 * ai.defineTool({ name: 'getInvoice',  ... }, async (i) => { ... });
 *
 * const customerService = ai.defineAgent({
 *   name: 'customerService',
 *   system: 'You are Acme support. Always be friendly and concise.',
 *   use: [
 *     handoff({
 *       personas: [
 *         {
 *           name: 'triage',
 *           description: 'Figures out the user\u2019s issue and routes them.',
 *           system: 'You triage support requests and transfer to a specialist.',
 *         },
 *         {
 *           name: 'refund',
 *           description: 'Handles refund requests.',
 *           system: 'You handle refunds. Look up the order before refunding.',
 *           tools: ['lookupOrder', 'issueRefund'],
 *         },
 *         {
 *           name: 'billing',
 *           description: 'Handles billing and invoice questions.',
 *           system: 'You answer billing questions.',
 *           tools: ['getInvoice'],
 *         },
 *       ],
 *       defaultPersona: 'triage',
 *       maxTransfers: 5,
 *     }),
 *   ],
 * });
 * ```
 */
export const handoff: GenerateMiddleware<typeof HandoffOptionsSchema> =
  generateMiddleware(
    {
      name: 'handoff',
      description:
        'Enables the agent transfer pattern: swaps the active persona ' +
        '(system prompt + tools) when the model calls a transfer tool.',
      configSchema: HandoffOptionsSchema,
    },
    ({ config }) => {
      if (!config?.personas || config.personas.length === 0) {
        throw new Error(
          'handoff middleware requires at least one persona in the "personas" option.'
        );
      }

      const prefix = config.toolPrefix ?? 'transfer_to';
      const maxTransfers = config.maxTransfers;

      const personas: Persona[] = config.personas.map((p) => ({
        name: p.name,
        description: p.description,
        system: p.system,
        tools: p.tools ?? [],
      }));

      const personasByName = new Map<string, Persona>(
        personas.map((p) => [p.name, p])
      );

      const defaultPersonaName = config.defaultPersona ?? personas[0].name;
      if (!personasByName.has(defaultPersonaName)) {
        throw new Error(
          `handoff middleware: defaultPersona "${defaultPersonaName}" is not ` +
            `one of the configured personas.`
        );
      }

      // Map transfer tool name -> target persona name (for active-persona
      // detection from history and the system-prompt listing).
      const toolNameToPersona = new Map<string, string>();
      for (const p of personas) {
        toolNameToPersona.set(makeToolName(prefix, p.name), p.name);
      }

      // Union of every persona's tools. These are "managed": only the active
      // persona's tools are exposed each turn, the rest are stripped from
      // `request.tools`. Tools NOT in this set (e.g. host-agent shared tools)
      // are left untouched.
      const managedToolNames = new Set<string>(
        personas.flatMap((p) => p.tools)
      );

      // Shared per-generate state (instantiate runs once per generate() call).
      const shared = { transferCount: 0 };

      const MARKER_KEY = 'handoff-instructions';

      /**
       * Determines the active persona by scanning history backwards for the
       * most recent *successful* transfer tool response. Falls back to the
       * default persona when no transfer has happened.
       */
      function getActivePersona(messages: MessageData[]): Persona {
        for (let i = messages.length - 1; i >= 0; i--) {
          const m = messages[i];
          if (m.role !== 'tool') continue;
          for (let j = m.content.length - 1; j >= 0; j--) {
            const tr = m.content[j].toolResponse;
            if (!tr || !toolNameToPersona.has(tr.name)) continue;
            const output = tr.output as { transferred?: boolean } | undefined;
            if (output?.transferred === true) {
              const target = toolNameToPersona.get(tr.name)!;
              return (
                personasByName.get(target) ??
                personasByName.get(defaultPersonaName)!
              );
            }
            // A refused/failed transfer doesn't change the persona; keep
            // looking for an earlier successful one.
          }
        }
        return personasByName.get(defaultPersonaName)!;
      }

      function buildInstructions(active: Persona): string {
        const transferTargets = personas
          .filter((p) => p.name !== active.name)
          .map((p) => {
            const toolName = makeToolName(prefix, p.name);
            const desc = p.description ?? `The "${p.name}" agent.`;
            return `  - ${toolName}: ${desc}`;
          })
          .join('\n');

        const transferSection = transferTargets
          ? `\nIf the user's needs fall outside your specialty, transfer the ` +
            `conversation to another agent using one of these tools:\n` +
            `${transferTargets}\n` +
            `When you transfer, the other agent takes over the conversation ` +
            `directly. Provide a brief reason so they have context.\n`
          : '';

        return (
          `<active-agent name="${active.name}">\n` +
          `You are now acting as the "${active.name}" agent.\n` +
          `${active.system}\n` +
          transferSection +
          `</active-agent>`
        );
      }

      // ── Transfer tools (always available so any persona can hand off) ──
      const transferTools = personas.map((persona) => {
        const toolName = makeToolName(prefix, persona.name);
        const description =
          `Transfer the conversation to the "${persona.name}" agent. ` +
          (persona.description ?? '') +
          ` After transferring, the "${persona.name}" agent takes over and ` +
          `responds to the user directly.`;

        return tool(
          {
            name: toolName,
            description,
            inputSchema: z.object({
              reason: z
                .string()
                .optional()
                .describe(
                  'Brief context for the receiving agent: why you are ' +
                    'transferring and what the user needs.'
                ),
            }),
            outputSchema: z.object({
              transferred: z.boolean(),
              to: z.string(),
              message: z.string(),
            }),
          },
          async (input) => {
            if (
              maxTransfers !== undefined &&
              shared.transferCount >= maxTransfers
            ) {
              return {
                transferred: false,
                to: persona.name,
                message:
                  `Transfer limit reached (${maxTransfers}). Continue ` +
                  `handling the request as the current agent.`,
              };
            }
            shared.transferCount++;
            return {
              transferred: true,
              to: persona.name,
              message:
                `Transferred to the "${persona.name}" agent.` +
                (input.reason ? ` Context: ${input.reason}` : ''),
            };
          }
        );
      });

      return {
        tools: transferTools,

        generate: async (envelope, ctx, next) => {
          const { request } = envelope;
          const messages = [...request.messages];

          const active = getActivePersona(messages);
          const instructions = buildInstructions(active);

          // ── Expose only the active persona's tools ──────────────────
          // Strip every managed (persona-owned) tool, then add back the
          // active persona's tools. Idempotent across turns and leaves
          // non-persona tools (host shared tools) untouched. Transfer tools
          // are injected separately via `tools` above and stay available.
          const baseTools = (request.tools ?? []).filter(
            (t) => !managedToolNames.has(t)
          );
          const tools = [...baseTools, ...active.tools];

          // ── Inject / refresh the active persona's instructions ──────
          let markerMsgIndex = -1;
          let markerPartIndex = -1;
          for (let i = 0; i < messages.length && markerMsgIndex === -1; i++) {
            const content = messages[i].content;
            for (let j = 0; j < content.length; j++) {
              const p = content[j];
              if (p.text && p.metadata?.[MARKER_KEY] === true) {
                markerMsgIndex = i;
                markerPartIndex = j;
                break;
              }
            }
          }

          if (markerMsgIndex !== -1) {
            // Refresh in place if the active persona changed.
            const existing = messages[markerMsgIndex].content[markerPartIndex];
            if (existing.text !== instructions) {
              const newContent = [...messages[markerMsgIndex].content];
              newContent[markerPartIndex] = {
                text: instructions,
                metadata: { [MARKER_KEY]: true },
              };
              messages[markerMsgIndex] = {
                ...messages[markerMsgIndex],
                content: newContent,
              };
            }
          } else {
            const systemIdx = messages.findIndex((m) => m.role === 'system');
            if (systemIdx !== -1) {
              messages[systemIdx] = {
                ...messages[systemIdx],
                content: [
                  ...messages[systemIdx].content,
                  { text: instructions, metadata: { [MARKER_KEY]: true } },
                ],
              };
            } else {
              messages.unshift({
                role: 'system',
                content: [
                  { text: instructions, metadata: { [MARKER_KEY]: true } },
                ],
              });
            }
          }

          return next(
            { ...envelope, request: { ...request, messages, tools } },
            ctx
          );
        },
      };
    }
  );
