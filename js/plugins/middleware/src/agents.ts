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
  GenkitError,
  generateMiddleware,
  z,
  type GenerateMiddleware,
  type MessageData,
  type Part,
} from 'genkit';
import {
  tool,
  type Agent,
  type AgentFinishReason,
  type AgentOutput,
  type Artifact,
  type SessionSnapshot,
} from 'genkit/beta';

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
        '(tools become delegate_to_<agent>). Set to empty string to use bare agent names. ' +
        'A non-empty prefix also namespaces the background-task tools added by "async".'
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
  async: z
    .boolean()
    .optional()
    .describe(
      'Enables background delegation: delegation tools accept a "background" ' +
        'flag that starts the sub-agent in the background and returns a taskId ' +
        'immediately, and the check_background_tasks / wait_for_background_tasks / ' +
        'abort_background_tasks tools are added. Background delegation requires ' +
        'server-managed sub-agents (agents defined with a session store).'
    ),
});

export type AgentsOptions = z.infer<typeof AgentsOptionsSchema>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Names of the shared background-task tools added when `async` is set. */
const CHECK_BACKGROUND_TASKS_TOOL = 'check_background_tasks';
const WAIT_FOR_BACKGROUND_TASKS_TOOL = 'wait_for_background_tasks';
const ABORT_BACKGROUND_TASKS_TOOL = 'abort_background_tasks';

/**
 * The report status for a task that could not be resolved (malformed ID,
 * unconfigured agent, missing snapshot, or read error). Settled for waiting
 * purposes: it only arrives once a read failure was classified unhelpable.
 */
const TASK_STATUS_UNKNOWN = 'unknown';

/** Guidance returned when a background-task tool is called without task IDs. */
const NO_TASK_IDS_NOTE =
  'No task IDs given. Pass the taskId values returned by background delegations.';

/**
 * The longest delay a timer can represent. A wait timeout beyond it is
 * treated as unbounded, the same as 0: `setTimeout` would otherwise fire it
 * at once, and the wait would return instantly with nothing settled.
 */
const MAX_TIMEOUT_MS = 2 ** 31 - 1;

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
 * Generates a short, unique invocation ID for a synchronous sub-agent call.
 * Format: `{agentName}_{random4}` — e.g. `researcher_k9m2`
 */
function makeInvocationId(agentName: string): string {
  const random = Math.random().toString(36).slice(2, 6);
  return `${agentName}_${random}`;
}

/**
 * The artifact namespace of a background task's run: the agent name and a
 * prefix of the run's snapshot ID. Deterministic, unlike the synchronous
 * path's random ID: `addArtifacts` replaces by name, so a re-check of the same
 * task, in this call or after the orchestrator restarts, overwrites the same
 * artifact names instead of duplicating them.
 */
function snapshotNamespace(agentName: string, snapshotId: string): string {
  return `${agentName}_${snapshotId.slice(0, 8)}`;
}

/**
 * The model-facing handle of a background task (`<agent>:<snapshotId>`).
 * Self-contained, so it can be parsed back after the orchestrator is
 * re-instantiated with nothing but its conversation history.
 */
function formatTaskId(agentName: string, snapshotId: string): string {
  return `${agentName}:${snapshotId}`;
}

/**
 * Whether a turn that ended for `reason` produced an answer its caller can
 * use: the agent spoke and stopped. Named for the reasons that do carry a
 * result, so a reason added later defaults to "no answer"; mistaking an
 * explanation for an answer hands the orchestrator partial work as though it
 * were final, while the reverse only asks it to look at the text.
 */
function carriesResult(reason?: AgentFinishReason): boolean {
  return (
    reason === undefined ||
    reason === 'stop' ||
    reason === 'other' ||
    reason === 'unknown'
  );
}

/**
 * Whether a snapshot or task-report status can no longer change on its own,
 * which is the rule the wait tool counts by. `pending` is the only status
 * that can; an absent status is the `completed` default, and the report-only
 * `unknown` is settled too (see {@link TASK_STATUS_UNKNOWN}).
 */
function isSettled(status: string | undefined): boolean {
  return status !== 'pending';
}

/** Joins a message's non-empty text parts with newlines. */
function messageText(message?: MessageData): string {
  return (message?.content ?? [])
    .map((p) => p.text)
    .filter((t): t is string => typeof t === 'string' && t.length > 0)
    .join('\n');
}

/**
 * The persisted conversation's final model message: what the sub-agent last
 * said. The transcript's tip is not that for every agent (a custom agent can
 * end its turn on a tool response, or on input it appended itself), and
 * reporting either as the answer would put someone else's words in the
 * sub-agent's mouth.
 */
function lastModelMessage(snapshot: SessionSnapshot): MessageData | undefined {
  const messages = snapshot.state?.messages ?? [];
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'model') return messages[i];
  }
  return undefined;
}

/**
 * The tool text reported when a sub-agent interrupted for input the
 * orchestrator can never provide.
 */
function interruptedResponse(agentName: string): string {
  return (
    `Sub-agent '${agentName}' interrupted for additional input ` +
    `and could not complete the task. Interactive sub-agent ` +
    `interrupts are not currently supported; try delegating a ` +
    `more self-contained task.`
  );
}

/**
 * Explains to the orchestrator why a sub-agent turn produced no answer. It
 * prefers the structured failure, falls back to whatever the agent managed to
 * say before it stopped, and names the finish reason when it has neither. A
 * snapshot the runtime finalizes as completed carries no error even when its
 * finish reason says the turn failed, so for a background task the agent's
 * last message is often the only account of what happened.
 */
function subAgentFailureMessage(
  reason: AgentFinishReason | undefined,
  error: { message?: string } | undefined,
  last: MessageData | undefined
): string {
  if (error?.message) return error.message;
  if (!reason) return 'Unknown sub-agent failure.';
  let msg = `the turn ended as '${reason}' without completing the task`;
  const text = messageText(last);
  if (text) msg += `; the agent's last message was: ${text}`;
  return msg + '.';
}

/**
 * The tool text reported for a run that settled on a result-carrying reason
 * without a final model text: a custom agent that returned no message, or a
 * model whose last message holds only tool requests. It says outright that
 * the run succeeded and where its result is, so the orchestrator neither
 * mistakes the silence for a failure nor repeats finished work.
 */
function noFinalMessageResponse(artifacts: number): string {
  switch (artifacts) {
    case 0:
      return 'The task completed, but the agent gave no final message and produced no artifacts.';
    case 1:
      return 'The task completed, but the agent gave no final message; its result is in the one artifact it produced.';
    default:
      return `The task completed, but the agent gave no final message; its result is in the ${artifacts} artifacts it produced.`;
  }
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** The canonical status name an error carries, if any (`GenkitError.status`). */
function errorStatus(e: unknown): string | undefined {
  const status = (e as { status?: unknown } | undefined)?.status;
  return typeof status === 'string' ? status : undefined;
}

/**
 * Whether an error is how an aborted `AbortSignal` surfaces: the signal's own
 * reason (an `AbortError`, or a `TimeoutError` from `AbortSignal.timeout`).
 */
function isAbortError(e: unknown): boolean {
  const name = (e as { name?: unknown } | undefined)?.name;
  return name === 'AbortError' || name === 'TimeoutError';
}

/**
 * Returns up to `n` of the most recent user/model messages, each reduced to
 * its non-empty text parts. Tool and tool-request parts are dropped: a model
 * message mid-tool-loop can carry a `toolRequest` part with no matching
 * response, which would confuse the sub-agent model.
 */
function recentTextHistory(messages: MessageData[], n: number): MessageData[] {
  if (n <= 0) return [];
  return messages
    .filter((m) => m.role === 'user' || m.role === 'model')
    .slice(-n)
    .map((m) => ({
      role: m.role,
      content: m.content.filter(
        (p): p is Part & { text: string } =>
          typeof p.text === 'string' && p.text.length > 0
      ),
    }))
    .filter((m) => m.content.length > 0);
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
 * With `async: true`, delegation tools additionally accept a `background`
 * flag. A background delegation starts the sub-agent through its detach
 * support (`detach: true`): the sub-agent's runtime persists a pending
 * snapshot, hands back a task ID immediately, and keeps working in the
 * background, so the orchestrator can continue calling tools and collect the
 * result later through the added `check_background_tasks` and
 * `wait_for_background_tasks` tools, or drop it with `abort_background_tasks`.
 * The pending snapshot (heartbeated while the worker lives, finalized in place
 * with the cumulative state when the work settles) is the durable record, and
 * task IDs are self-contained (`<agent>:<snapshotId>`), so a re-instantiated
 * orchestrator can pick results up using nothing but the IDs recorded in its
 * conversation history. Background delegation requires server-managed
 * sub-agents (defined with a `store`); a launch on any other agent is refused
 * as tool text that points the model at a synchronous delegation.
 *
 * Task handles are not access-scoped: the background-task tools read any
 * snapshot ID belonging to a configured sub-agent, whether or not this
 * conversation launched it (mirroring the sub-agent's `getSnapshot` companion
 * action, which is itself unscoped). In multi-tenant deployments treat
 * snapshot IDs as capability-like secrets: text that reaches the orchestrator
 * model can steer these tools at any ID it names.
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
      const async = config.async ?? false;

      // The shared background-task tools take an explicitly set prefix and
      // none by default: the default delegate_to prefix is a delegation verb,
      // not an instance namespace. Two async instances in one generate call
      // therefore need distinct, explicit prefixes; left at the default they
      // both emit the bare names and the request is rejected for duplicate
      // tools.
      const sharedPrefix = config.toolPrefix ?? '';
      const taskTools = {
        check: makeToolName(sharedPrefix, CHECK_BACKGROUND_TASKS_TOOL),
        wait: makeToolName(sharedPrefix, WAIT_FOR_BACKGROUND_TASKS_TOOL),
        abort: makeToolName(sharedPrefix, ABORT_BACKGROUND_TASKS_TOOL),
      };

      // Every generated tool name is validated as it is claimed: a collision
      // (two agents mapping to one delegation tool name, or a delegation tool
      // landing on a background-task tool's name) would otherwise surface only
      // at generate time as a duplicate-tool rejection of the whole request.
      const claimedNames = new Map<string, string>();
      function claimName(name: string, owner: string): void {
        const previous = claimedNames.get(name);
        if (previous) {
          throw new GenkitError({
            status: 'INVALID_ARGUMENT',
            message:
              `agents middleware: tool name '${name}' for ${owner} collides ` +
              `with ${previous}; use a different toolPrefix or agent name.`,
          });
        }
        claimedNames.set(name, owner);
      }

      // Shared mutable state — safe because `instantiate()` is called per
      // `generate()` invocation, giving each call its own closure.
      const shared = {
        delegationCount: 0,
        conversationMessages: [] as MessageData[],
        // Terminal background-task reports by task ID for the rest of the
        // generate call: completed, failed, and aborted rows never change, so
        // a re-check skips the snapshot fetch and artifact re-merge (and cannot
        // clobber a merged artifact the orchestrator has since edited).
        // Pending, expired, and unresolvable reports can still change and are
        // never cached.
        settledReports: new Map<string, BackgroundTaskReport>(),
      };

      // Caches (persist across turns within the same generate cycle).
      const agentCache = new Map<string, Agent>();
      const descriptionCache = new Map<string, string>();

      async function resolveAgent(name: string): Promise<Agent | undefined> {
        const cached = agentCache.get(name);
        if (cached) return cached;

        const action = (await ai.registry.lookupAction(`/agent/${name}`)) as
          | Agent
          | undefined;
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

      /** Who owns the agent's session state, from its action metadata. */
      function stateManagementOf(agent: Agent): string | undefined {
        return agent.__action?.metadata?.agent?.stateManagement;
      }

      // -- Schemas ------------------------------------------------------------

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

      const artifactsField = z
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
        );

      // The result schema is what the model reads, so a synchronous-only
      // instance must not advertise task handles or background-task tools it
      // does not have.
      const syncDelegationResultSchema = z.object({
        response: z.string().describe("The sub-agent's text response."),
        artifacts: artifactsField,
      });
      const asyncDelegationResultSchema = syncDelegationResultSchema.extend({
        response: z
          .string()
          .describe(
            "The sub-agent's text response. For a background delegation it describes the launch instead; the sub-agent's response arrives later via the background-task tools."
          ),
        taskId: z
          .string()
          .optional()
          .describe(
            'Handle of the background task ("<agent>:<snapshotId>") when the delegation was started with background: true. Pass it to the background-task tools.'
          ),
        status: z
          .string()
          .optional()
          .describe('"pending" when a background delegation was started.'),
      });
      const delegationResultSchema = async
        ? asyncDelegationResultSchema
        : syncDelegationResultSchema;
      type DelegationResult = z.infer<typeof asyncDelegationResultSchema>;

      const delegateInputSchema = z.object({
        task: z
          .string()
          .describe(
            'A clear, self-contained description of the task to delegate.'
          ),
      });

      // The async variant carries the extra "background" flag, so the two
      // modes need distinct input schemas (tool schemas are static).
      const asyncDelegateInputSchema = delegateInputSchema.extend({
        background: z
          .boolean()
          .optional()
          .describe(
            `Run the delegation in the background. The tool returns immediately with a taskId; collect the result later with ${taskTools.check} or ${taskTools.wait}.`
          ),
      });

      // taskIds is optional so the schema does not mark it required. A model
      // that calls one of these tools with no arguments is making a
      // recoverable mistake, answered with guidance; a required field would
      // instead fail validation, which surfaces as a tool error that fails the
      // whole generate call rather than a turn the model can correct.
      const backgroundTasksInputSchema = z.object({
        taskIds: z
          .array(z.string())
          .optional()
          .describe(
            'Task IDs returned by background delegations (form "<agent>:<snapshotId>").'
          ),
      });

      const waitBackgroundTasksInputSchema = backgroundTasksInputSchema.extend({
        timeoutSeconds: z
          .number()
          .optional()
          .describe(
            'Maximum seconds to wait before returning the current statuses. 0 or omitted waits until every task settles; a negative value returns the current statuses immediately. Values too large to represent are treated as unbounded.'
          ),
        // A free string rather than an enum: an unknown value is answered
        // with guidance the model can correct, not a validation failure that
        // kills the turn.
        waitFor: z
          .string()
          .optional()
          .describe(
            '"all" (default) waits until every listed task settles. "first" returns as soon as any one settles; the remaining tasks report their current status and keep running.'
          ),
      });

      const backgroundTaskReportSchema = z.object({
        taskId: z.string().describe('The handle the report describes.'),
        agent: z
          .string()
          .optional()
          .describe('The sub-agent running the task.'),
        status: z
          .string()
          .describe(
            'The task\'s lifecycle state: "pending", "completed", "failed", "aborted", "expired" (worker presumed dead), or "unknown" (the ID could not be resolved; see error). "completed" always carries a response.'
          ),
        response: z
          .string()
          .optional()
          .describe(
            "The sub-agent's final text response, for completed tasks."
          ),
        artifacts: artifactsField,
        error: z
          .string()
          .optional()
          .describe(
            'Why no response is available (failure, abort, expiry, or an unresolvable task ID).'
          ),
      });
      type BackgroundTaskReport = z.infer<typeof backgroundTaskReportSchema>;

      const backgroundTasksResultSchema = z.object({
        tasks: z.array(backgroundTaskReportSchema).optional(),
        timedOut: z
          .boolean()
          .optional()
          .describe(
            'Set when the wait returned because timeoutSeconds elapsed while some tasks were still pending.'
          ),
        note: z
          .string()
          .optional()
          .describe(
            'Usage guidance when the call itself was unusable (e.g. no task IDs given).'
          ),
      });
      type BackgroundTasksResult = z.infer<typeof backgroundTasksResultSchema>;

      // -- Delegation ---------------------------------------------------------

      /**
       * Namespaces the sub-agent's artifacts by invocation ID, tags them with
       * their source, and merges them into the active session. No-op when
       * there is no active session (`ai.currentSession()` throws then).
       */
      function mergeArtifacts(
        source: string,
        invocationId: string,
        artifacts: Artifact[]
      ): void {
        try {
          const session = ai.currentSession();
          session.addArtifacts(
            artifacts.map((a) => ({
              ...a,
              name: `${invocationId}/${a.name}`,
              metadata: { ...a.metadata, source, invocationId },
            }))
          );
        } catch {
          // No active session — artifacts can't be merged into a parent
          // session. With the "inline" strategy the content is still returned
          // in the tool result.
        }
      }

      /**
       * Builds the tool-result artifact list, including content only under
       * the inline strategy.
       */
      function delegatedArtifacts(
        invocationId: string,
        artifacts: Artifact[]
      ): { name: string; content?: string }[] {
        return artifacts.map((a) => ({
          name: `${invocationId}/${a.name}`,
          ...(artifactStrategy === 'inline' && {
            content: (a.parts ?? [])
              .map((p) => p.text ?? '')
              .filter((t) => t.length > 0)
              .join('\n'),
          }),
        }));
      }

      /**
       * Turns a settled sub-agent output into a delegation tool result:
       * interrupts and failures become explanatory text, and artifacts are
       * merged into the parent session under `invocationId` and surfaced per
       * the configured strategy. Shared by the synchronous path and the
       * background-task reports, so a delegation reports the same answer and
       * the same artifacts whether it ran in the background or not.
       */
      function foldDelegationOutput(
        ref: NormalizedAgentRef,
        out: AgentOutput,
        invocationId: string
      ): DelegationResult {
        // The agent runtime resolves gracefully rather than throwing: a failed
        // turn returns `finishReason: 'failed'` with structured error details,
        // and an interrupted turn returns `finishReason: 'interrupted'`.

        // Interrupted first: it is one of the reasons that carry no result,
        // and the one with an explanation of its own worth giving. It is
        // deliberately NOT propagated to the parent: there is no stateful
        // sub-agent runtime to resume back into, so the parent could never
        // satisfy it.
        if (out.finishReason === 'interrupted') {
          return { response: interruptedResponse(ref.name) };
        }
        if (!carriesResult(out.finishReason)) {
          // Blocked, truncated, aborted, or failed. The turn's last message is
          // whatever the agent got out before it stopped, so it explains the
          // outcome rather than answering the task, and reporting it as the
          // answer would hand the orchestrator partial work as if it were
          // final.
          return {
            response: `Error calling agent '${ref.name}': ${subAgentFailureMessage(
              out.finishReason,
              out.error,
              out.message
            )}`,
          };
        }

        const subArtifacts = (out.artifacts ?? []).filter((a) => a.name);
        const response =
          messageText(out.message) ||
          noFinalMessageResponse(subArtifacts.length);
        if (subArtifacts.length === 0) {
          return { response };
        }
        // Merge into the parent session under both strategies.
        mergeArtifacts(ref.name, invocationId, subArtifacts);
        return {
          response,
          artifacts: delegatedArtifacts(invocationId, subArtifacts),
        };
      }

      /**
       * The prologue every delegation shares, synchronous or background: it
       * enforces `maxDelegations` and resolves the sub-agent. A refusal is the
       * tool result to return as is. A resolution failure keeps its slot: the
       * agent is misconfigured or missing, so every retry fails the same way,
       * and refunding would mean the cap never bites on exactly the runaway
       * loop it exists to stop.
       */
      async function beginDelegation(
        ref: NormalizedAgentRef
      ): Promise<{ agent: Agent } | { refusal: DelegationResult }> {
        if (
          maxDelegations !== undefined &&
          shared.delegationCount >= maxDelegations
        ) {
          return {
            refusal: {
              response:
                `Delegation limit reached (${maxDelegations}). ` +
                `Complete the task using information already gathered.`,
            },
          };
        }
        shared.delegationCount++;

        const agent = await resolveAgent(ref.name);
        if (!agent) {
          return {
            refusal: {
              response: `Error: Agent '${ref.name}' not found in registry.`,
            },
          };
        }
        return { agent };
      }

      /**
       * Returns a reserved cap slot to a delegation whose refusal names a retry
       * that can succeed: a background launch refused because the sub-agent
       * cannot run in the background, whose refusal tells the model to
       * delegate synchronously instead. Every other refusal keeps its slot.
       */
      function releaseDelegation(): void {
        shared.delegationCount--;
      }

      /**
       * Runs one turn of the sub-agent with the task. History rides as
       * client-managed init state, which only client-managed agents accept;
       * `detach` asks the sub-agent runtime to move the work to the background
       * at once, so the output carries the pending snapshot's ID and
       * `finishReason: 'detached'` while the sub-agent keeps working.
       */
      async function runSubAgent(
        agent: Agent,
        task: string,
        opts: { history?: MessageData[]; detach?: boolean } = {}
      ): Promise<AgentOutput> {
        const init = opts.history?.length
          ? { state: { messages: opts.history } }
          : {};
        const { result } = await agent.run(
          {
            message: { role: 'user' as const, content: [{ text: task }] },
            ...(opts.detach && { detach: true }),
          },
          { init }
        );
        return result;
      }

      /** The synchronous delegation body. */
      async function runDelegation(
        ref: NormalizedAgentRef,
        task: string
      ): Promise<DelegationResult> {
        const begun = await beginDelegation(ref);
        if ('refusal' in begun) return begun.refusal;
        const { agent } = begun;

        try {
          // Prior conversation is seeded via the session state (`init.state`),
          // which only client-managed agents (no persistent store) accept —
          // sending `state` to a server-managed agent throws a precondition
          // error. Server-managed sub-agents can't be seeded with ad-hoc
          // per-delegation history, so history forwarding is skipped for them
          // (the task is still delivered).
          const history =
            stateManagementOf(agent) !== 'server'
              ? recentTextHistory(shared.conversationMessages, historyLength)
              : [];
          const out = await runSubAgent(agent, task, { history });
          return foldDelegationOutput(ref, out, makeInvocationId(ref.name));
        } catch (e: unknown) {
          // The agent runtime resolves failures and interrupts gracefully (see
          // foldDelegationOutput), so this only fires for exceptions thrown
          // outside that handling (e.g. schema parse errors on `run`). Return
          // them as tool output so the model can recover.
          return {
            response: `Error calling agent '${ref.name}': ${errorMessage(e)}`,
          };
        }
      }

      /**
       * Starts a background delegation through the sub-agent's detach support
       * and returns the task handle without waiting for the work. Launches
       * count against `maxDelegations` like synchronous delegations, except
       * for a launch the sub-agent cannot support at all: that refusal returns
       * its slot, so the synchronous fallback it hints at is not refused by a
       * cap the refusal consumed. History is never forwarded: detach requires
       * a server-managed sub-agent, and server-managed init rejects seeded
       * state.
       */
      async function launchDelegation(
        ref: NormalizedAgentRef,
        task: string
      ): Promise<DelegationResult> {
        const begun = await beginDelegation(ref);
        if ('refusal' in begun) return begun.refusal;
        const { agent } = begun;

        const withoutBackground = `Delegate to it without "background" instead.`;
        // Pre-flight from the agent's own metadata: the runtime accepts a
        // detach only on a store-backed agent, so a genkit-defined agent
        // without one is refused here deterministically, without a wasted
        // invocation and without the hedged wording below (which remains only
        // for agents that publish no metadata).
        const stateManagement = stateManagementOf(agent);
        if (stateManagement !== undefined && stateManagement !== 'server') {
          releaseDelegation();
          return {
            response:
              `Error calling agent '${ref.name}': this agent has no session ` +
              `store, so it cannot run tasks in the background. ${withoutBackground}`,
          };
        }

        let out: AgentOutput;
        try {
          out = await runSubAgent(agent, task, { detach: true });
        } catch (e: unknown) {
          // A thrown rejection (e.g. a schema parse error on `run`) carries
          // the same status a graceful one does, so it takes the failed shape
          // and is judged once below.
          out = {
            finishReason: 'failed',
            error: { status: errorStatus(e), message: errorMessage(e) },
          };
        }

        switch (out.finishReason) {
          case 'detached': {
            if (!out.snapshotId) {
              return {
                response: `Error calling agent '${ref.name}': the background launch returned no task handle.`,
              };
            }
            const taskId = formatTaskId(ref.name, out.snapshotId);
            return {
              taskId,
              status: 'pending',
              response:
                `Background task ${taskId} started for agent '${ref.name}'. ` +
                `Collect the result with ${taskTools.check} or ${taskTools.wait}, ` +
                `or stop it with ${taskTools.abort}.`,
            };
          }
          case 'failed': {
            // FAILED_PRECONDITION is how the runtime rejects a detach on an
            // agent that cannot support it. Only a metadata-less agent reaches
            // this (one that publishes metadata and cannot detach was refused
            // above), and only this failure earns its slot back: the retry it
            // points at is the synchronous launch. Every other failure keeps
            // the slot.
            const msg = subAgentFailureMessage(
              out.finishReason,
              out.error,
              out.message
            );
            if (
              stateManagement === undefined &&
              out.error?.status === 'FAILED_PRECONDITION'
            ) {
              releaseDelegation();
              return {
                response:
                  `Error calling agent '${ref.name}': ${msg} If this agent has ` +
                  `no session store, it cannot run in the background. ${withoutBackground}`,
              };
            }
            return { response: `Error calling agent '${ref.name}': ${msg}` };
          }
          default:
            // The invocation settled before the detach landed; fold it like a
            // synchronous delegation.
            return foldDelegationOutput(ref, out, makeInvocationId(ref.name));
        }
      }

      // -- Background tasks ---------------------------------------------------

      /**
       * How a task's snapshot is obtained: read once for the check tool,
       * waited for by the wait tool, aborted first by the abort tool. All three
       * dispatch companion actions of the sub-agent, so all three apply the
       * runtime's read shaping (a pending row whose heartbeat went stale reads
       * as expired) and throw NOT_FOUND for a missing row. Ending on the row,
       * rather than on each tool's own idea of an outcome, is what lets one
       * report path serve every tool.
       */
      type SnapshotFetch = (
        agent: Agent,
        snapshotId: string,
        signal?: AbortSignal
      ) => Promise<SessionSnapshot>;

      const readSnapshotOnce: SnapshotFetch = async (agent, snapshotId) =>
        (await agent.getSnapshotDataAction.run({ snapshotId })).result;

      const awaitSnapshot: SnapshotFetch = async (agent, snapshotId, signal) =>
        (
          await agent.waitForSnapshotAction.run(
            { snapshotId },
            { abortSignal: signal }
          )
        ).result;

      // The abort reads before it stops anything, because there are rows an
      // abort must not touch. Expiry is decided on read, not stored: a worker
      // that stopped heartbeating leaves a row that is still pending in the
      // store and reads as expired, and aborting it would overwrite the one
      // signal telling the model the work is gone. A task that already
      // settled needs no abort at all and is answered from the row alone.
      const abortSnapshot: SnapshotFetch = async (agent, snapshotId) => {
        const current = await readSnapshotOnce(agent, snapshotId);
        if (isSettled(current.status)) {
          return current;
        }
        const { result } = await agent.abortAgentAction.run({ snapshotId });
        // The abort action answers with the status the row had before the
        // attempt: `pending` means the flip landed, so the row just read is
        // handed back restamped rather than re-read (a re-read could fail on
        // its own and turn a delivered stop into "unknown"). Anything else
        // means the task settled between the read and the abort, and a re-read
        // fetches the answer it now carries.
        if (result.status === 'pending') {
          return { ...current, status: 'aborted' };
        }
        return readSnapshotOnce(agent, snapshotId);
      };

      /**
       * Parses a task handle by matching it against the configured agents,
       * taking the longest matching name so a configured name containing ':'
       * cannot have its tasks claimed by a shorter configured prefix of it.
       * The runtime mints snapshot IDs as UUIDs (never containing ':'), so the
       * longest configured prefix is always the launching agent; anchoring the
       * parse on the finite set of configured names also confines the
       * background-task tools to the agents this middleware was configured
       * with.
       */
      function resolveTaskId(
        taskId: string
      ): { ref: NormalizedAgentRef; snapshotId: string } | undefined {
        let best: NormalizedAgentRef | undefined;
        let bestLength = 0;
        for (const ref of agentRefs) {
          const candidate = `${ref.name}:`;
          if (
            taskId.length > candidate.length &&
            taskId.startsWith(candidate) &&
            candidate.length > bestLength
          ) {
            best = ref;
            bestLength = candidate.length;
          }
        }
        return best
          ? { ref: best, snapshotId: taskId.slice(bestLength) }
          : undefined;
      }

      /**
       * Resolves one task handle, obtains its snapshot through `fetch`, and
       * shapes the result into a report. Completed tasks surface the
       * sub-agent's final response and artifacts; terminal non-success
       * statuses surface an explanatory error instead. The raw failure rides
       * alongside for the wait tool, which classifies it.
       */
      async function reportTask(
        taskId: string,
        fetch: SnapshotFetch,
        signal?: AbortSignal
      ): Promise<{ report: BackgroundTaskReport; error?: unknown }> {
        const cached = shared.settledReports.get(taskId);
        if (cached) return { report: cached };

        const resolved = resolveTaskId(taskId);
        if (!resolved) {
          const error = `Task ID '${taskId}' does not match any configured agent (expected "<agent>:<snapshotId>").`;
          return {
            report: { taskId, status: TASK_STATUS_UNKNOWN, error },
            error: new Error(error),
          };
        }
        const { ref, snapshotId } = resolved;
        const report: BackgroundTaskReport = {
          taskId,
          agent: ref.name,
          status: TASK_STATUS_UNKNOWN,
        };

        // Resolving the agent and reading its snapshot fail for unrelated
        // reasons, so they are reported separately: an unregistered agent
        // must not get the missing-snapshot advice, which would tell the model
        // to delegate again into a delegation tool that fails identically.
        const agent = await resolveAgent(ref.name);
        if (!agent) {
          report.error =
            `Agent '${ref.name}' is configured on the agents middleware but is ` +
            `not registered. This task cannot be collected here; report it as ` +
            `unavailable rather than delegating it again.`;
          return { report, error: new Error(report.error) };
        }
        if (
          !agent.getSnapshotDataAction ||
          !agent.waitForSnapshotAction ||
          !agent.abortAgentAction
        ) {
          report.error =
            `Agent '${ref.name}' does not expose the snapshot companion ` +
            `actions, so its tasks cannot be collected here; report the task ` +
            `as unavailable.`;
          return { report, error: new Error(report.error) };
        }

        let snapshot: SessionSnapshot;
        try {
          snapshot = await fetch(agent, snapshotId, signal);
        } catch (e: unknown) {
          // The agent resolved above, so NOT_FOUND here is the snapshot and
          // nothing else, and re-delegating is genuinely the way to get the
          // work done. A rejected request is reported as is; anything else is
          // presumed transient.
          const status = errorStatus(e);
          if (status === 'NOT_FOUND') {
            report.error = `No record of this task exists (${errorMessage(e)}). Delegate the task again if the result is still needed.`;
          } else if (
            status === 'FAILED_PRECONDITION' ||
            status === 'INVALID_ARGUMENT'
          ) {
            report.error = errorMessage(e);
          } else {
            report.error = `Could not read the task's status: ${errorMessage(e)}. Check again later.`;
          }
          return { report, error: e };
        }

        // An absent status is the runtime's `completed` default.
        const snapshotStatus = snapshot.status ?? 'completed';
        report.status = snapshotStatus;
        switch (snapshotStatus) {
          case 'pending':
            // Still running; nothing to report yet.
            break;
          case 'completed': {
            // Fold the settled snapshot exactly as a synchronous delegation
            // folds its output, under the deterministic namespace of the run.
            // The response is what the sub-agent last said, not whatever the
            // transcript happens to end on. One caveat: this reads through
            // the sub-agent's companion action, so a sub-agent with a
            // `clientTransform.state` has already shaped what is read here,
            // while the synchronous path sees the output unshaped.
            const folded = foldDelegationOutput(
              ref,
              {
                finishReason: snapshot.finishReason,
                message: lastModelMessage(snapshot),
                artifacts: snapshot.state?.artifacts,
              },
              snapshotNamespace(ref.name, snapshotId)
            );
            if (carriesResult(snapshot.finishReason)) {
              report.response = folded.response;
              if (folded.artifacts) report.artifacts = folded.artifacts;
            } else {
              // The row committed, so the stored status is completed, but the
              // agent declared a reason that carries no answer. Report the
              // outcome the reader has to act on, not the row's bookkeeping:
              // a model that sees "completed" moves on and never reads the
              // error. Which reason it was, and what the agent last said, is
              // in the folded text.
              report.status = 'failed';
              report.error = folded.response;
            }
            break;
          }
          case 'failed':
            report.error = subAgentFailureMessage(
              snapshot.finishReason,
              snapshot.error,
              lastModelMessage(snapshot)
            );
            break;
          case 'aborted':
            report.error = 'The task was aborted before it finished.';
            break;
          case 'expired':
            report.error =
              'The background worker stopped reporting progress and is presumed dead. Delegate the task again if the result is still needed.';
            break;
        }

        // Expired is the one terminal read that can still change its mind:
        // the worker may be alive and merely slow to beat, so a later read can
        // find it settled properly. Everything else terminal is final.
        if (isSettled(snapshotStatus) && snapshotStatus !== 'expired') {
          shared.settledReports.set(taskId, report);
        }
        return { report };
      }

      /**
       * Builds one report per entry of `taskIds`, fetching each distinct ID
       * once and copying its report to every duplicate (the IDs are
       * model-authored, so repeats happen). The fetches run concurrently, so
       * the slowest distinct task sets the wall clock rather than the sum, and
       * failures stay isolated per task: one bad handle cannot hide the status
       * of the others.
       */
      async function collectReports(
        taskIds: string[],
        report: (taskId: string) => Promise<BackgroundTaskReport>
      ): Promise<BackgroundTaskReport[]> {
        const distinct = [...new Set(taskIds)];
        const fetched = new Map(
          await Promise.all(
            distinct.map(async (id) => [id, await report(id)] as const)
          )
        );
        return taskIds.map((id) => fetched.get(id)!);
      }

      /**
       * One report per task with a single fetch each; no waiting. The body of
       * the check and abort tools, and the wait tool's don't-wait path.
       */
      async function reportTasks(
        taskIds: string[],
        fetch: SnapshotFetch
      ): Promise<BackgroundTasksResult> {
        if (taskIds.length === 0) {
          return { note: NO_TASK_IDS_NOTE };
        }
        return {
          tasks: await collectReports(
            taskIds,
            async (taskId) => (await reportTask(taskId, fetch)).report
          ),
        };
      }

      /**
       * Follows one task to its end and returns its report. A fetch error
       * that reaches this level is a dead end worth reporting, with one
       * exception: `signal` ending the wait (its timeout, or a won race). The
       * follow was cut short rather than finished then, so the task is
       * reported as it stands from one more plain read: the runtime's wait
       * checks the signal before its first read, and a deadline that beats
       * the dispatch would otherwise report an already-settled task as
       * pending. A read that fails there leaves the task pending, since it
       * was still running the last time anyone saw it. That is decided from
       * the failure itself, not from the signal alone: a handle that never
       * resolved is not the signal's doing and keeps its error however the
       * wait ended, or the model would be told to keep re-checking an ID that
       * can never settle.
       */
      async function awaitTask(
        taskId: string,
        signal: AbortSignal
      ): Promise<BackgroundTaskReport> {
        const { report, error } = await reportTask(
          taskId,
          awaitSnapshot,
          signal
        );
        if (error === undefined || !signal.aborted || !isAbortError(error)) {
          return report;
        }
        const current = await reportTask(taskId, readSnapshotOnce);
        if (current.error === undefined) return current.report;
        return { ...report, status: 'pending', error: undefined };
      }

      /**
       * The blocking status tool: follows every task to its end, or returns
       * the current statuses when the optional timeout elapses. Each task is
       * followed by the sub-agent's `waitForSnapshot` companion action, so the
       * waiting happens next to the store that knows when the work finished:
       * one action dispatch per task for the whole wait, and a settlement is
       * observed as it happens. The waits run concurrently, so the slowest
       * task sets the wall clock. A timeout returns the current statuses
       * rather than an error so the orchestrator can do other work and come
       * back; the calling tool's own abort signal ending propagates as an
       * error.
       */
      async function waitForBackgroundTasks(
        input: z.infer<typeof waitBackgroundTasksInputSchema>,
        toolSignal?: AbortSignal
      ): Promise<BackgroundTasksResult> {
        const taskIds = input.taskIds ?? [];
        if (taskIds.length === 0) {
          return { note: NO_TASK_IDS_NOTE };
        }
        const waitFor = input.waitFor ?? 'all';
        if (waitFor !== 'all' && waitFor !== 'first') {
          // A recoverable mistake, answered like an empty ID list.
          return {
            note: `Unknown waitFor value '${waitFor}'. Use 'all' (the default) to wait for every task, or 'first' to return when any one settles.`,
          };
        }
        const first = waitFor === 'first';

        // A negative timeout means "don't wait": report the current statuses.
        const timeoutSeconds = input.timeoutSeconds ?? 0;
        if (timeoutSeconds < 0) {
          return reportTasks(taskIds, readSnapshotOnce);
        }
        const timeoutMs = timeoutSeconds * 1000;

        // One signal ends every follow: the deadline, the caller hanging up,
        // or (with waitFor "first") the first settlement, after which the
        // remaining follows report their tasks as they stand. The controller
        // is aborted once the collection is over however it ended, so no
        // follow outlives the call that started it, and the deadline timer is
        // cleared rather than left to fire into a finished wait.
        const controller = new AbortController();
        const signal = toolSignal
          ? AbortSignal.any([controller.signal, toolSignal])
          : controller.signal;
        const deadline =
          timeoutMs > 0 && timeoutMs <= MAX_TIMEOUT_MS
            ? setTimeout(
                () =>
                  controller.abort(
                    new DOMException('wait timed out', 'TimeoutError')
                  ),
                timeoutMs
              )
            : undefined;

        let reports: BackgroundTaskReport[];
        try {
          reports = await collectReports(taskIds, async (taskId) => {
            const report = await awaitTask(taskId, signal);
            if (first && isSettled(report.status)) {
              controller.abort(
                new DOMException('first task settled', 'AbortError')
              );
            }
            return report;
          });
        } finally {
          clearTimeout(deadline);
          controller.abort(new DOMException('wait ended', 'AbortError'));
        }

        // The calling tool ending is cancellation, not a timeout; let it fail
        // the tool call rather than dressing it up as a settled result.
        toolSignal?.throwIfAborted();

        const pending = reports.filter((r) => !isSettled(r.status)).length;
        if (pending === 0) {
          return { tasks: reports };
        }
        if (first && pending < reports.length) {
          // The race was won, not timed out.
          return {
            tasks: reports,
            note: 'Returned on the first settled task; the remaining tasks report their current status and keep running.',
          };
        }
        return {
          tasks: reports,
          timedOut: true,
          note: 'Stopped waiting; the pending tasks are still running. Check them again later.',
        };
      }

      // ── Per-agent delegation tools ────────────────────────────────────

      const delegationTools = agentRefs.map((ref) => {
        const toolName = makeToolName(prefix, ref.name);
        claimName(toolName, `agent '${ref.name}'`);
        return tool(
          {
            name: toolName,
            description:
              ref.description ??
              `Delegates a task to the "${ref.name}" sub-agent.`,
            inputSchema: async ? asyncDelegateInputSchema : delegateInputSchema,
            outputSchema: delegationResultSchema,
          },
          (input: z.infer<typeof asyncDelegateInputSchema>) =>
            input.background
              ? launchDelegation(ref, input.task)
              : runDelegation(ref, input.task)
        );
      });

      // ── Shared background-task tools ──────────────────────────────────

      function defineBackgroundTools() {
        for (const name of Object.values(taskTools)) {
          claimName(name, 'the background-task tools');
        }
        return [
          tool(
            {
              name: taskTools.check,
              description:
                'Returns the current status of background sub-agent tasks without waiting, including results for tasks that finished.',
              inputSchema: backgroundTasksInputSchema,
              outputSchema: backgroundTasksResultSchema,
            },
            (input) => reportTasks(input.taskIds ?? [], readSnapshotOnce)
          ),
          tool(
            {
              name: taskTools.wait,
              description:
                'Waits until the given background sub-agent tasks finish and returns their results. Set timeoutSeconds to bound the wait; on timeout the current statuses are returned. Set waitFor to "first" to return as soon as any one task settles.',
              inputSchema: waitBackgroundTasksInputSchema,
              outputSchema: backgroundTasksResultSchema,
            },
            (input, ctx) => waitForBackgroundTasks(input, ctx.abortSignal)
          ),
          tool(
            {
              name: taskTools.abort,
              description:
                'Stops background sub-agent tasks whose results are no longer needed, and returns where that left each one. A task that had already finished is unaffected and reports its result.',
              inputSchema: backgroundTasksInputSchema,
              outputSchema: backgroundTasksResultSchema,
            },
            (input) => reportTasks(input.taskIds ?? [], abortSnapshot)
          ),
        ];
      }
      const backgroundTools = async ? defineBackgroundTools() : [];

      return {
        tools: [...delegationTools, ...backgroundTools],

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

          const asyncInstructions = async
            ? `\n` +
              `Delegations can run in the background: set "background": true ` +
              `on a delegation tool call to get a taskId back immediately ` +
              `while the sub-agent keeps working. Continue with other work, ` +
              `then collect results with ${taskTools.check} (returns current ` +
              `status without waiting) or ${taskTools.wait} (blocks until the ` +
              `tasks settle). Use ${taskTools.abort} to stop tasks whose ` +
              `results are no longer needed. Background tasks keep running ` +
              `across turns, and task IDs from earlier tool results stay ` +
              `valid: check them before delegating the same work again.\n`
            : '';

          const agentsInstructions =
            `<sub-agents>\n` +
            `You can delegate tasks to specialized sub-agents using their ` +
            `delegation tools:\n` +
            `${agentList}\n` +
            `\n` +
            `When a task is better handled by a specialized agent, delegate ` +
            `it using the appropriate tool. Provide a clear, self-contained ` +
            `task description.\n` +
            asyncInstructions +
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
