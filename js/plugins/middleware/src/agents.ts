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
import { logger } from 'genkit/logging';

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
        'A non-empty prefix also namespaces the shared tools: the background-task ' +
        'tools added by "async" and the continue_task tool.'
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
  maxWaitSeconds: z
    .number()
    .positive()
    .optional()
    .describe(
      'Upper bound on how long one wait_for_background_tasks call blocks, ' +
        'whatever timeoutSeconds the model asks for (including 0, "until ' +
        'every task settles"). At the bound the wait returns the current ' +
        'statuses with timedOut set. Omitted means the model decides.'
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

/** Name of the shared continue tool, namespaced like the background-task tools. */
const CONTINUE_TASK_TOOL = 'continue_task';

/** The continue tool's model-facing description. */
const CONTINUE_TASK_TOOL_DESCRIPTION =
  'Continues a sub-agent task by its taskId: a failed or aborted task picks up from its last saved progress (omit instructions to retry it as it stood, or pass instructions to steer it), and a completed task accepts follow-up instructions inside its own session. A task that stopped on an interrupt cannot be continued.';

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
 * Maps a settled finish reason onto the snapshot-status vocabulary that
 * delegation results and background-task reports share: `completed` for every
 * reason that carries a result, `aborted` for an aborted run, and `failed` for
 * the rest. A blocked or truncated turn commits a completed row, but it
 * carries no answer, and a model told "completed" moves on without reading
 * the explanation; the explanation names the finish reason itself.
 */
function settledStatus(reason?: AgentFinishReason): string {
  if (carriesResult(reason)) return 'completed';
  return reason === 'aborted' ? 'aborted' : 'failed';
}

/**
 * Whether a snapshot or task-report status can no longer change on its own,
 * which is the rule the wait tool counts by. `pending` and `aborting` (a
 * stopped task winding down toward its finalize) are the two still in flight;
 * an absent status is the `completed` default, and the report-only `unknown`
 * is settled too (see {@link TASK_STATUS_UNKNOWN}).
 */
function isSettled(status: string | undefined): boolean {
  return status !== 'pending' && status !== 'aborting';
}

/**
 * How a task settles once an abort reaches it, in the words the abort tool's
 * description and the aborting report share.
 */
const STOPPED_TASK_SETTLES = 'settles as "aborted" with the progress it saved';

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
 * Whether a snapshot read failure cannot be helped by retrying: the row is
 * gone or the request itself is rejected. Anything else (a store blip, a
 * timed-out read) is presumed transient. It is the policy the runtime's own
 * wait applies to its re-reads.
 */
function isDeadEndRead(e: unknown): boolean {
  const status = errorStatus(e);
  return (
    status === 'NOT_FOUND' ||
    status === 'FAILED_PRECONDITION' ||
    status === 'INVALID_ARGUMENT'
  );
}

/**
 * The user message a continuation delivers: the instructions when given,
 * otherwise none, so the runtime re-attempts the conversation as committed,
 * which is the retry for a run that stopped short.
 *
 * The exception is a background retry. A detached input with no payload of
 * its own is a pure detach signal to the runtime and runs no turn, so an
 * empty background continuation would finalize the loaded state untouched
 * instead of retrying it. It gets the smallest honest payload instead, which
 * also records in the sub-agent's transcript why the run picked back up.
 */
function continueMessage(
  instructions: string | undefined,
  detach: boolean
): string | undefined {
  if (instructions) return instructions;
  return detach ? 'Continue the task from where it stopped.' : undefined;
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
 * A delegation to a server-managed sub-agent leaves the same kind of handle
 * behind when it settles, synchronous or not, and the shared `continue_task`
 * tool spends it: a failed or aborted task continues from its last saved
 * progress (retried as it stood, or steered with instructions), a completed
 * one takes follow-up instructions inside its own session, and an expired one
 * is fenced with an abort and continued from the last snapshot it committed.
 * A delegation may carry a `name` label, echoed on its result, its reports,
 * and its continuations.
 *
 * The background-task tools and the continue tool accept only the task IDs
 * this conversation minted: ones this generate call's delegations and
 * continuations returned, or ones a delegation or continue tool's result in
 * the conversation history carries. Text that reaches the orchestrator model
 * therefore cannot steer them at another conversation's task. A history
 * compacted past those results loses those tasks' handles. The sub-agent's
 * own companion actions (`getSnapshot`, `waitForSnapshot`, `abort`) stay
 * unscoped, so in multi-tenant deployments treat snapshot IDs as
 * capability-like secrets where those are exposed.
 *
 * `maxWaitSeconds` bounds one `wait_for_background_tasks` call whatever
 * timeout the model asks for, so a sub-agent that keeps running cannot hold
 * the orchestrator's turn open past it.
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
      if (agentRefs.some((ref) => !ref.name)) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message: 'agents middleware: every agent reference must have a name.',
        });
      }
      const prefix = config.toolPrefix ?? 'delegate_to';
      const maxDelegations = config.maxDelegations;
      const historyLength = config.historyLength ?? 0;
      const artifactStrategy = config.artifactStrategy ?? 'inline';
      const async = config.async ?? false;
      const maxWaitSeconds = config.maxWaitSeconds;

      // The shared tools (the background-task tools and the continue tool)
      // take an explicitly set prefix and none by default: the default
      // delegate_to prefix is a delegation verb, not an instance namespace.
      // Two instances in one generate call therefore need distinct, explicit
      // prefixes; left at the default they both emit the bare names and the
      // request is rejected for duplicate tools.
      const sharedPrefix = config.toolPrefix ?? '';
      const taskTools = {
        check: makeToolName(sharedPrefix, CHECK_BACKGROUND_TASKS_TOOL),
        wait: makeToolName(sharedPrefix, WAIT_FOR_BACKGROUND_TASKS_TOOL),
        abort: makeToolName(sharedPrefix, ABORT_BACKGROUND_TASKS_TOOL),
      };
      const continueTool = makeToolName(sharedPrefix, CONTINUE_TASK_TOOL);

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
        // Pending, aborting, expired, and unresolvable reports can still change
        // and are never cached.
        settledReports: new Map<string, BackgroundTaskReport>(),
        // Task IDs this generate call launched; see launchedHere.
        launchedTaskIds: new Set<string>(),
        // Caller-chosen delegation labels by task handle, echoed on
        // background-task reports and carried onto continuations. A per-call
        // reading aid: after a restart the transcript still pairs each label
        // with its taskId at the delegation that minted it.
        labels: new Map<string, string>(),
      };
      const delegationToolNames = new Set(
        agentRefs.map((ref) => makeToolName(prefix, ref.name))
      );

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

      /**
       * Whether the agent's store can signal a running worker to stop, from
       * its action metadata; undefined when the agent publishes none.
       */
      function abortableOf(agent: Agent): boolean | undefined {
        const abortable = agent.__action?.metadata?.agent?.abortable;
        return typeof abortable === 'boolean' ? abortable : undefined;
      }

      /**
       * Whether the agent exposes the companion actions the background-task
       * tools and the continue tool dispatch. Every genkit-defined agent does.
       */
      function hasCompanionActions(agent: Agent): boolean {
        return [
          agent.getSnapshotDataAction,
          agent.waitForSnapshotAction,
          agent.abortAgentAction,
        ].every((action) => action !== undefined);
      }

      /**
       * Whether any configured sub-agent can leave a continuable task handle
       * behind: only server-managed sub-agents (those with a session store)
       * commit durable snapshots. An agent that does not resolve, or that
       * publishes no metadata, counts as continuable, so a wrong guess costs
       * a refusal at call time rather than a silently missing tool.
       * Resolved once per generate call.
       */
      let continuable: Promise<boolean> | undefined;
      function anyContinuableAgent(): Promise<boolean> {
        continuable ??= (async () => {
          for (const ref of agentRefs) {
            const agent = await resolveAgent(ref.name);
            if (!agent || stateManagementOf(agent) !== 'client') return true;
          }
          return false;
        })();
        return continuable;
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
      // instance must not advertise background launches or background-task
      // tools it does not have.
      const delegationResultSchema = z.object({
        response: z
          .string()
          .describe(
            async
              ? "The sub-agent's text response. For a background delegation it describes the launch instead; the sub-agent's response arrives later via the background-task tools."
              : "The sub-agent's text response."
          ),
        artifacts: artifactsField,
        taskId: z
          .string()
          .optional()
          .describe(
            'The delegation\'s handle ("<agent>:<snapshotId>"). ' +
              (async
                ? "For a background delegation it names the pending task; for a synchronous delegation to a sub-agent with a session store it names the run's last saved progress, whatever the outcome. "
                : "It names the run's last saved progress, whatever the outcome. ") +
              `Pass it to ${continueTool}` +
              (async ? ' or to the background-task tools. ' : '. ') +
              'Absent when nothing addressable stands behind the result: a sub-agent without a session store, a run that saved no progress, or an interrupt.'
          ),
        status: z
          .string()
          .optional()
          .describe(
            (async
              ? '"pending" when a background delegation was started; otherwise the '
              : 'The ') +
              'settled outcome behind taskId ("completed", "failed", or "aborted").'
          ),
        name: z
          .string()
          .optional()
          .describe('The label given to this delegation, echoed back.'),
      });
      type DelegationResult = z.infer<typeof delegationResultSchema>;

      const delegateInputSchema = z.object({
        task: z
          .string()
          .describe(
            'A clear, self-contained description of the task to delegate.'
          ),
        // A reading aid, never identity: the taskId stays the handle; the
        // label just keeps several concurrent tasks readable.
        name: z
          .string()
          .optional()
          .describe(
            `Optional short label for this delegation (e.g. "sources-sweep"). Echoed on the result${async ? ' and on background-task reports next to the taskId' : ''}, to keep several tasks readable. Not an identifier.`
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
            'Maximum seconds to wait before returning the current statuses. 0 or omitted waits until every task settles' +
              (maxWaitSeconds === undefined
                ? ''
                : `, for at most ${maxWaitSeconds} seconds`) +
              '; a negative value returns the current statuses immediately. Values too large to represent are treated as unbounded.'
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

      // taskId is optional for the reason taskIds is above.
      const continueInputSchema = z.object({
        taskId: z
          .string()
          .optional()
          .describe(
            'The task handle to continue ("<agent>:<snapshotId>"), from a delegation result or a background-task report.'
          ),
        instructions: z
          .string()
          .optional()
          .describe(
            'Optional guidance delivered to the sub-agent as it continues. Omit it to retry a failed or aborted task exactly as it stood; required when following up on a completed task.'
          ),
      });

      const asyncContinueInputSchema = continueInputSchema.extend({
        background: z
          .boolean()
          .optional()
          .describe(
            'Continue the task in the background. The tool returns immediately with a new taskId; collect the result later with the background-task tools.'
          ),
      });

      const backgroundTaskReportSchema = z.object({
        taskId: z.string().describe('The handle the report describes.'),
        agent: z
          .string()
          .optional()
          .describe('The sub-agent running the task.'),
        name: z
          .string()
          .optional()
          .describe(
            'The label given to the delegation behind taskId, if one was given.'
          ),
        status: z
          .string()
          .describe(
            `The task's lifecycle state: "pending", "completed", "failed", "aborted", "expired" (worker presumed dead), "aborting" (the stop was delivered and the task is winding down; it ${STOPPED_TASK_SETTLES}), or "unknown" (the ID could not be resolved; see error). "completed" always carries a response.`
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
       * merged into the parent session and surfaced per the configured
       * strategy. Shared by the synchronous path, the continue tool, and the
       * background-task reports, so a delegation reports the same answer and
       * the same artifacts whether it ran in the background or not.
       *
       * A server-managed output names the run's last committed snapshot, so
       * every settled result but an interrupt is stamped with the same
       * `<agent>:<snapshotId>` handle background delegations mint, plus the
       * outcome it settled in. The handle makes a delegation addressable after
       * the fact: the background-task tools accept it, and it is what the
       * continue tool spends. The same snapshot namespaces the run's
       * artifacts, so one run merges identical artifact names whichever path
       * folds it, and `addArtifacts`' replace-by-name makes a later re-check of
       * the handle idempotent instead of duplicative. A run with no snapshot
       * behind it (a client-managed sub-agent) gets a random invocation ID.
       */
      function foldDelegationOutput(
        ref: NormalizedAgentRef,
        out: AgentOutput
      ): DelegationResult {
        // The agent runtime resolves gracefully rather than throwing: a failed
        // turn returns `finishReason: 'failed'` with structured error details,
        // and an interrupted turn returns `finishReason: 'interrupted'`.

        // Interrupted first: it is one of the reasons that carry no result,
        // and the one with an explanation of its own worth giving. It is
        // deliberately NOT propagated to the parent: there is no stateful
        // sub-agent runtime to resume back into, so the parent could never
        // satisfy it. For the same reason it is the one settled outcome that
        // carries no handle: continuing past it means answering the interrupt.
        if (out.finishReason === 'interrupted') {
          return { response: interruptedResponse(ref.name) };
        }
        const result: DelegationResult = { response: '' };
        let namespace = makeInvocationId(ref.name);
        // A detached output names a pending row, not a settled turn.
        const settledId =
          out.finishReason === 'detached' ? undefined : out.snapshotId;
        if (settledId) {
          result.taskId = formatTaskId(ref.name, settledId);
          result.status = settledStatus(out.finishReason);
          namespace = snapshotNamespace(ref.name, settledId);
        }
        if (!carriesResult(out.finishReason)) {
          // Blocked, truncated, aborted, or failed. The turn's last message is
          // whatever the agent got out before it stopped, so it explains the
          // outcome rather than answering the task, and reporting it as the
          // answer would hand the orchestrator partial work as if it were
          // final.
          result.response = `Error calling agent '${ref.name}': ${subAgentFailureMessage(
            out.finishReason,
            out.error,
            out.message
          )}`;
          if (result.taskId) {
            result.response += ` The run's progress up to that point is saved; call ${continueTool} with this taskId to continue it, optionally with instructions.`;
          }
          return result;
        }

        const subArtifacts = (out.artifacts ?? []).filter((a) => a.name);
        result.response =
          messageText(out.message) ||
          noFinalMessageResponse(subArtifacts.length);
        if (subArtifacts.length > 0) {
          // Merge into the parent session under both strategies.
          mergeArtifacts(ref.name, namespace, subArtifacts);
          result.artifacts = delegatedArtifacts(namespace, subArtifacts);
        }
        return result;
      }

      /**
       * Stamps the caller-chosen label on a result and records it against the
       * result's handle, so background-task reports and continuations can
       * echo it for the rest of the call. A label with no handle still rides
       * the result (the transcript keeps the pairing).
       */
      function labelTask(result: DelegationResult, name?: string): void {
        if (!name) return;
        result.name = name;
        if (result.taskId) shared.labels.set(result.taskId, name);
      }

      /**
       * The prologue every delegation shares, synchronous, background, or a
       * continuation: it enforces `maxDelegations` and resolves the sub-agent.
       * A refusal is the tool result to return as is. A resolution failure
       * keeps its slot: the agent is misconfigured or missing, so every retry
       * fails the same way, and refunding would mean the cap never bites on
       * exactly the runaway loop it exists to stop.
       */
      async function beginDelegation(
        ref: NormalizedAgentRef
      ): Promise<{ agent: Agent } | { refusal: DelegationResult }> {
        if (
          maxDelegations !== undefined &&
          shared.delegationCount >= maxDelegations
        ) {
          logger.warn(
            `agents middleware: delegation to '${ref.name}' refused, limit ${maxDelegations} reached.`
          );
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
          logger.warn(
            `agents middleware: sub-agent '${ref.name}' is not registered.`
          );
          return {
            refusal: {
              response: `Error: Agent '${ref.name}' not found in registry.`,
            },
          };
        }
        return { agent };
      }

      /**
       * Returns a reserved cap slot to a delegation whose refusal ran no
       * sub-agent work and names a corrected retry that can succeed: a
       * background launch the sub-agent cannot support (the retry is the
       * synchronous call), or a continuation refused for a wrong flag, missing
       * instructions, a task still running, or a transient read failure.
       * Every other refusal keeps its slot, because its retry fails
       * identically and refunding it would leave the cap unable to bite.
       */
      function releaseDelegation(): void {
        shared.delegationCount--;
      }

      /**
       * Runs one turn of the sub-agent. `text` is the turn's user message;
       * none runs the turn on the session as it stands, which is how a failed
       * turn is re-attempted. `history` rides as client-managed init state,
       * which only client-managed agents accept, and `snapshotId` names the
       * snapshot a continuation resumes. `detach` asks the sub-agent runtime
       * to move the work to the background at once, so the output carries the
       * pending snapshot's ID and `finishReason: 'detached'` while the
       * sub-agent keeps working. `abortSignal` is the tool call's: a stopped
       * orchestrator stops a synchronous sub-agent with it, and the runtime
       * ignores it once a detach is requested, so a background task outlives
       * the call.
       */
      async function runSubAgent(
        agent: Agent,
        text: string | undefined,
        opts: {
          history?: MessageData[];
          snapshotId?: string;
          detach?: boolean;
          abortSignal?: AbortSignal;
        } = {}
      ): Promise<AgentOutput> {
        const init = opts.snapshotId
          ? { snapshotId: opts.snapshotId }
          : opts.history?.length
            ? { state: { messages: opts.history } }
            : {};
        const { result } = await agent.run(
          {
            ...(text !== undefined && {
              message: { role: 'user' as const, content: [{ text }] },
            }),
            ...(opts.detach && { detach: true }),
          },
          { init, abortSignal: opts.abortSignal }
        );
        return result;
      }

      /** The synchronous delegation body. */
      async function runDelegation(
        ref: NormalizedAgentRef,
        task: string,
        name: string | undefined,
        abortSignal?: AbortSignal
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
          logger.debug(
            `agents middleware: delegating to '${ref.name}' with ${history.length} history messages.`
          );
          const start = Date.now();
          const out = await runSubAgent(agent, task, { history, abortSignal });
          const result = foldDelegationOutput(ref, out);
          labelTask(result, name);
          logger.debug(
            `agents middleware: delegation to '${ref.name}' finished as '${out.finishReason}' in ${Date.now() - start}ms.`
          );
          return result;
        } catch (e: unknown) {
          // The agent runtime resolves failures and interrupts gracefully (see
          // foldDelegationOutput), so this only fires for exceptions thrown
          // outside that handling (e.g. schema parse errors on `run`). Return
          // them as tool output so the model can recover.
          logger.warn(
            `agents middleware: delegation to '${ref.name}' failed: ${errorMessage(e)}`
          );
          return {
            response: `Error calling agent '${ref.name}': ${errorMessage(e)}`,
          };
        }
      }

      /**
       * The model-facing phrasings that differ between the two background
       * launches, a delegation and a continuation, of the one launch protocol:
       * {@link refuseUndetachable} before the run, {@link foldDetachOutcome}
       * after it.
       */
      interface LaunchWords {
        /** Opens every refusal: "Error calling agent ..." or "Error continuing task ...". */
        errPrefix: string;
        /** The sentence naming the synchronous retry the refusals point at. */
        withoutBackground: string;
        /** Renders the sentence announcing the pending handle. */
        started: (taskId: string) => string;
        /** The caller-chosen label to stamp on the result. */
        label?: string;
      }

      /**
       * The pre-flight both background launches run, from the agent's own
       * metadata: the runtime accepts a detach only on a store-backed agent,
       * so a genkit-defined agent without one is refused here
       * deterministically, without a wasted invocation and without the hedged
       * wording of {@link foldDetachOutcome} (which remains only for agents
       * that publish no metadata). The refusal names the synchronous retry, so
       * it returns its slot.
       */
      function refuseUndetachable(
        ref: NormalizedAgentRef,
        agent: Agent,
        words: LaunchWords
      ): DelegationResult | undefined {
        const stateManagement = stateManagementOf(agent);
        if (stateManagement === undefined || stateManagement === 'server') {
          return undefined;
        }
        releaseDelegation();
        logger.warn(
          `agents middleware: background launch on '${ref.name}' refused; it has no session store.`
        );
        return {
          response:
            `${words.errPrefix}: this agent has no session store, so it ` +
            `cannot run tasks in the background. ${words.withoutBackground}`,
        };
      }

      /**
       * Turns the output of a detached run into the launch's tool result: the
       * pending handle when the detach landed, a hedged refusal when a
       * metadata-less agent may have rejected the detach, and the ordinary
       * fold when the run settled before detaching or failed on its own.
       */
      function foldDetachOutcome(
        ref: NormalizedAgentRef,
        agent: Agent,
        out: AgentOutput,
        words: LaunchWords
      ): DelegationResult {
        if (out.finishReason === 'detached') {
          if (!out.snapshotId) {
            return {
              response: `${words.errPrefix}: the background launch returned no task handle.`,
            };
          }
          const taskId = formatTaskId(ref.name, out.snapshotId);
          shared.launchedTaskIds.add(taskId);
          logger.debug(`agents middleware: background task ${taskId} started.`);
          const result: DelegationResult = {
            taskId,
            status: 'pending',
            response:
              `${words.started(taskId)} ` +
              `Collect the result with ${taskTools.check} or ${taskTools.wait}, ` +
              `or stop it with ${taskTools.abort}.`,
          };
          labelTask(result, words.label);
          return result;
        }
        // FAILED_PRECONDITION is how the runtime rejects a detach on an agent
        // that cannot support it. Only a metadata-less agent reaches this (one
        // that publishes metadata and cannot detach was refused by the
        // pre-flight), and only this failure earns its slot back: the retry it
        // points at is the synchronous call. Every other failure is the
        // sub-agent's own and keeps the slot, or an agent that always fails
        // could be launched forever.
        if (
          out.finishReason === 'failed' &&
          stateManagementOf(agent) === undefined &&
          out.error?.status === 'FAILED_PRECONDITION'
        ) {
          const msg = subAgentFailureMessage(
            out.finishReason,
            out.error,
            out.message
          );
          logger.warn(
            `agents middleware: background launch on '${ref.name}' rejected: ${msg}`
          );
          releaseDelegation();
          return {
            response:
              `${words.errPrefix}: ${msg} If this agent has no session ` +
              `store, it cannot run in the background. ${words.withoutBackground}`,
          };
        }
        // The run settled before the detach landed, or failed on its own;
        // fold it like a synchronous delegation, handle and all.
        logger.debug(
          `agents middleware: background launch on '${ref.name}' settled synchronously as '${out.finishReason}'.`
        );
        const result = foldDelegationOutput(ref, out);
        labelTask(result, words.label);
        return result;
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
        task: string,
        name: string | undefined,
        abortSignal?: AbortSignal
      ): Promise<DelegationResult> {
        const begun = await beginDelegation(ref);
        if ('refusal' in begun) return begun.refusal;
        const { agent } = begun;

        const words: LaunchWords = {
          errPrefix: `Error calling agent '${ref.name}'`,
          withoutBackground: `Delegate to it without "background" instead.`,
          started: (taskId) =>
            `Background task ${taskId} started for agent '${ref.name}'.`,
          label: name,
        };
        const refusal = refuseUndetachable(ref, agent, words);
        if (refusal) return refusal;

        let out: AgentOutput;
        try {
          out = await runSubAgent(agent, task, { detach: true, abortSignal });
        } catch (e: unknown) {
          // A thrown rejection (e.g. a schema parse error on `run`) carries
          // the same status a graceful one does, so it takes the failed shape
          // and is judged once in foldDetachOutcome.
          out = {
            finishReason: 'failed',
            error: { status: errorStatus(e), message: errorMessage(e) },
          };
        }
        return foldDetachOutcome(ref, agent, out, words);
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

      const readSnapshotOnce: SnapshotFetch = async (
        agent,
        snapshotId,
        signal
      ) =>
        (
          await agent.getSnapshotDataAction.run(
            { snapshotId },
            { abortSignal: signal }
          )
        ).result;

      // The companion action holds one request for at most the sub-agent's
      // maxSnapshotWaitMs and then answers with the row as it stands, so the
      // follow asks again until the row settles or the signal ends it.
      const awaitSnapshot: SnapshotFetch = async (
        agent,
        snapshotId,
        signal
      ) => {
        while (true) {
          const { result } = await agent.waitForSnapshotAction.run(
            { snapshotId },
            { abortSignal: signal }
          );
          if (isSettled(result.status) || signal?.aborted) return result;
        }
      };

      // The abort reads before it stops anything, because there are rows an
      // abort must not touch. Expiry is decided on read, not stored: a worker
      // that stopped heartbeating leaves a row that is still pending or
      // aborting in the store and reads as expired, and aborting it would overwrite the one
      // signal telling the model the work is gone. A task that already
      // settled needs no abort at all and is answered from the row alone.
      const abortSnapshot: SnapshotFetch = async (
        agent,
        snapshotId,
        signal
      ) => {
        const current = await readSnapshotOnce(agent, snapshotId, signal);
        if (isSettled(current.status)) {
          return current;
        }
        // A cancelled call must not stop a task on its way out.
        signal?.throwIfAborted();
        const { result } = await agent.abortAgentAction.run(
          { snapshotId },
          { abortSignal: signal }
        );
        // The abort action answers with the status the row had before the
        // attempt: `pending` means the flip to `aborting` landed, and
        // `aborting` means an earlier one had. Either way the stop is durable,
        // so the row just read is handed back restamped rather than re-read (a
        // re-read could fail on its own and turn a delivered stop into
        // "unknown"), and the report says the task is winding down. The abort
        // never waits for the finalize; the wait tool is the one that waits.
        // Anything else means the task settled between the read and the
        // abort, and a re-read fetches the answer it now carries.
        if (result.status === 'pending' || result.status === 'aborting') {
          return { ...current, status: 'aborting' };
        }
        return readSnapshotOnce(agent, snapshotId, signal);
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
       * Whether this conversation launched the task: this generate call did,
       * or a delegation tool's result in the conversation names it. The
       * background-task tools accept only those handles, so text that reaches
       * the model (a sub-agent's result, a retrieved document) cannot steer
       * them at another conversation's task. A re-instantiated orchestrator
       * still collects its tasks, since its history carries the launch
       * results.
       */
      function launchedHere(taskId: string): boolean {
        if (shared.launchedTaskIds.has(taskId)) return true;
        return shared.conversationMessages.some((message) =>
          message.content?.some((part) => {
            const response = part.toolResponse;
            return (
              !!response &&
              delegationToolNames.has(response.name) &&
              (response.output as { taskId?: unknown } | undefined)?.taskId ===
                taskId
            );
          })
        );
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
        if (!launchedHere(taskId)) {
          const error = `Task ID '${taskId}' was not started in this conversation; only tasks a delegation here launched can be checked, awaited, or stopped.`;
          return {
            report: { taskId, status: TASK_STATUS_UNKNOWN, error },
            error: new Error(error),
          };
        }
        const { ref, snapshotId } = resolved;
        const report: BackgroundTaskReport = {
          taskId,
          agent: ref.name,
          ...(shared.labels.has(taskId) && { name: shared.labels.get(taskId) }),
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
        if (!hasCompanionActions(agent)) {
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
          logger.debug(
            `agents middleware: reading background task ${taskId} failed: ${errorMessage(e)}`
          );
          if (errorStatus(e) === 'NOT_FOUND') {
            report.error = `No record of this task exists (${errorMessage(e)}). Delegate the task again if the result is still needed.`;
          } else if (isDeadEndRead(e)) {
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
            // The output names the row's own snapshot, so the fold namespaces
            // the artifacts exactly as the synchronous fold of the same run
            // does. The response is what the sub-agent last said, not
            // whatever the transcript happens to end on. One caveat: this
            // reads through the sub-agent's companion action, so a sub-agent
            // with a `clientTransform.state` has already shaped what is read
            // here, while the synchronous path sees the output unshaped.
            const folded = foldDelegationOutput(ref, {
              snapshotId,
              finishReason: snapshot.finishReason,
              message: lastModelMessage(snapshot),
              artifacts: snapshot.state?.artifacts,
            });
            if (carriesResult(snapshot.finishReason)) {
              report.response = folded.response;
              if (folded.artifacts) report.artifacts = folded.artifacts;
            } else {
              // The row committed, so the stored status is completed, but the
              // agent declared a reason that carries no answer. Report the
              // outcome the reader has to act on, not the row's bookkeeping:
              // a model that sees "completed" moves on and never reads the
              // error. Which reason it was, what the agent last said, and how
              // to continue (an interrupt cannot be) is the folded text.
              report.status = settledStatus(snapshot.finishReason);
              report.error = folded.response;
            }
            break;
          }
          case 'failed':
            report.error =
              subAgentFailureMessage(
                snapshot.finishReason,
                snapshot.error,
                lastModelMessage(snapshot)
              ) +
              ` The task's progress up to the failure is saved; continue it with ${continueTool} using this taskId.`;
            break;
          case 'aborting':
            // The runtime stops a worker through its store's change feed.
            // Where the store has none, the flip still lands but the worker
            // is not reached, so the row settles only once the work runs to
            // its end; say so, or a model that aborted to save cost would
            // assume it had.
            report.error =
              abortableOf(agent) === false
                ? "The task is marked for abort, but this agent's store " +
                  'cannot signal its worker, so the work runs to completion ' +
                  `and then ${STOPPED_TASK_SETTLES}. `
                : `The stop signal reached the task and it is winding down; it ${STOPPED_TASK_SETTLES}. `;
            report.error +=
              `No further action is needed to stop it; collect the settled ` +
              `state with ${taskTools.wait} if you need it.`;
            break;
          case 'aborted':
            report.error =
              abortableOf(agent) === false
                ? "The task was aborted, but this agent's store cannot " +
                  'signal its worker, so the work may have run to completion ' +
                  'first.'
                : 'The task was aborted before it finished.';
            report.error += ` Continue it with ${continueTool} using this taskId to pick up from its last saved progress.`;
            break;
          case 'expired':
            report.error = `The background worker stopped reporting progress and is presumed dead. Attempt ${continueTool} with this taskId to recover saved progress, or delegate the task again.`;
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
        fetch: SnapshotFetch,
        toolSignal?: AbortSignal
      ): Promise<BackgroundTasksResult> {
        if (taskIds.length === 0) {
          return { note: NO_TASK_IDS_NOTE };
        }
        const tasks = await collectReports(
          taskIds,
          async (taskId) => (await reportTask(taskId, fetch, toolSignal)).report
        );
        // A cancelled call fails its dispatches, and each failure reads back
        // as "could not read this task, check again later". Reported together
        // that is a settled-looking answer claiming live tasks are unreadable,
        // so the cancellation is the result instead. The rule lives here,
        // with the fan-out, so a tool added later cannot forget it.
        toolSignal?.throwIfAborted();
        return { tasks };
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
       * can never settle. When `toolSignal` ended the wait, the tool call
       * fails as a whole, so there is no report to refresh and no re-read.
       */
      async function awaitTask(
        taskId: string,
        signal: AbortSignal,
        toolSignal?: AbortSignal
      ): Promise<BackgroundTaskReport> {
        const { report, error } = await reportTask(
          taskId,
          awaitSnapshot,
          signal
        );
        if (error === undefined || !signal.aborted || !isAbortError(error)) {
          return report;
        }
        if (toolSignal?.aborted) {
          return { ...report, status: 'pending', error: undefined };
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
        // The operator's bound clamps any other value, "until every task
        // settles" included, so a prompt cannot decide how long a request
        // hangs.
        let timeoutSeconds = input.timeoutSeconds ?? 0;
        if (timeoutSeconds < 0) {
          return reportTasks(taskIds, readSnapshotOnce, toolSignal);
        }
        if (
          maxWaitSeconds !== undefined &&
          (timeoutSeconds === 0 || timeoutSeconds > maxWaitSeconds)
        ) {
          timeoutSeconds = maxWaitSeconds;
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

        const start = Date.now();
        logger.debug(
          `agents middleware: waiting for ${taskIds.length} background tasks (timeoutSeconds: ${timeoutSeconds}).`
        );
        let reports: BackgroundTaskReport[];
        try {
          reports = await collectReports(taskIds, async (taskId) => {
            const report = await awaitTask(taskId, signal, toolSignal);
            logger.debug(
              `agents middleware: background task ${taskId} reported '${report.status}' after ${Date.now() - start}ms.`
            );
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
        const elapsed = `${Date.now() - start}ms`;
        if (pending === 0) {
          logger.debug(
            `agents middleware: wait for ${taskIds.length} background tasks finished after ${elapsed}.`
          );
          return { tasks: reports };
        }
        if (first && pending < reports.length) {
          // The race was won, not timed out.
          logger.debug(
            `agents middleware: wait for background tasks returned on the first settle after ${elapsed}; ${pending} still pending.`
          );
          return {
            tasks: reports,
            note: 'Returned on the first settled task; the remaining tasks report their current status and keep running.',
          };
        }
        logger.debug(
          `agents middleware: wait for background tasks timed out after ${elapsed}; ${pending} still pending.`
        );
        return {
          tasks: reports,
          timedOut: true,
          note: 'Stopped waiting; the pending tasks are still running. Check them again later.',
        };
      }

      // -- Continuation -------------------------------------------------------
      //
      // A delegation that settled leaves a handle behind (`<agent>:<snapshotId>`),
      // and the sub-agent runtime makes the snapshot behind it a resume point:
      // a failed or aborted run holds the state through its last committed
      // turn, and a completed run holds the whole conversation. The continue
      // tool spends that handle: it retries a failed or aborted task from its
      // saved progress (no instructions re-attempts the turn as committed;
      // instructions steer the retry), and follows up on a completed task
      // inside the sub-agent's own session, so the orchestrator presses on
      // without re-buying work that already happened, and without the
      // sub-agent's conversation ever entering its own context window.
      //
      // Only server-managed sub-agents are continuable. A client-managed
      // delegation settles inline and leaves nothing durable a handle could
      // name, so its result carries no taskId.

      /** One continuation in flight: the tool input, resolved. */
      interface Continuation {
        ref: NormalizedAgentRef;
        agent: Agent;
        taskId: string;
        instructions?: string;
        background: boolean;
        abortSignal?: AbortSignal;
      }

      /**
       * The continue tool body. It spends a delegation slot like any
       * delegation (a continuation is a real sub-agent run, and an
       * always-failing task continued forever is exactly the runaway
       * `maxDelegations` bounds), resolves the handle, and continues the
       * snapshot behind it.
       *
       * Refusal slot policy follows the background launch's: a refusal that
       * ran no sub-agent work and names a corrected retry that can succeed
       * (wrong background flag, missing instructions, task still running, a
       * transient read failure) returns its slot; a dead end (unknown handle,
       * unresolvable or client-managed agent, no saved progress) keeps it.
       */
      async function runContinue(
        input: z.infer<typeof asyncContinueInputSchema>,
        abortSignal?: AbortSignal
      ): Promise<DelegationResult> {
        const taskId = input.taskId ?? '';
        if (!taskId) {
          return {
            response:
              'No taskId given. Pass the taskId from a delegation result or a background-task report.',
          };
        }
        const resolved = resolveTaskId(taskId);
        if (!resolved) {
          return {
            response: `Error: task ID '${taskId}' does not match any configured agent (expected "<agent>:<snapshotId>").`,
          };
        }
        const { ref, snapshotId } = resolved;
        const begun = await beginDelegation(ref);
        if ('refusal' in begun) return begun.refusal;
        const { agent } = begun;
        if (stateManagementOf(agent) === 'client') {
          // Nothing durable exists behind a client-managed delegation, so no
          // handle can name a resume point; a dead end keeps its slot.
          return {
            response: `Error: agent '${ref.name}' manages its state on the client and its delegations cannot be continued; delegate the task again. Only sub-agents with a session store leave continuable task handles.`,
          };
        }
        if (!hasCompanionActions(agent)) {
          return {
            response: `Error: agent '${ref.name}' does not expose the snapshot companion actions, so its tasks cannot be continued here; delegate the task again.`,
          };
        }
        return continueFromStore(
          {
            ref,
            agent,
            taskId,
            instructions: input.instructions || undefined,
            background: !!input.background,
            abortSignal,
          },
          snapshotId
        );
      }

      /**
       * Continues a server-managed task from the snapshot behind its handle.
       * The read goes through the sub-agent's companion action, so the
       * runtime's shaping applies: a pending row whose heartbeat went stale
       * reads as expired here rather than as forever-running.
       */
      async function continueFromStore(
        c: Continuation,
        snapshotId: string
      ): Promise<DelegationResult> {
        let snapshot: SessionSnapshot;
        try {
          snapshot = await readSnapshotOnce(c.agent, snapshotId, c.abortSignal);
        } catch (e: unknown) {
          c.abortSignal?.throwIfAborted();
          logger.debug(
            `agents middleware: reading task ${c.taskId} to continue it failed: ${errorMessage(e)}`
          );
          if (errorStatus(e) === 'NOT_FOUND') {
            return {
              response: `Error: no record of task '${c.taskId}' exists (${errorMessage(e)}). Delegate the task again if the work is still needed.`,
            };
          }
          if (isDeadEndRead(e)) {
            return {
              response: `Error continuing task '${c.taskId}': ${errorMessage(e)}`,
            };
          }
          releaseDelegation();
          return {
            response: `Error: could not read task '${c.taskId}' (${errorMessage(e)}). Try again later.`,
          };
        }

        // Expired before the other settled statuses: expiry is a settled
        // verdict on the row, but its recovery has its own path.
        switch (snapshot.status ?? 'completed') {
          case 'pending': {
            releaseDelegation();
            const hint = async
              ? ` Collect it with ${taskTools.check} or ${taskTools.wait}, or stop it with ${taskTools.abort} first.`
              : '';
            return {
              response: `Task '${c.taskId}' is still running; only a settled task can be continued.${hint}`,
            };
          }
          case 'expired':
            return continueExpired(c, snapshotId);
          case 'aborting':
            return windingDownRefusal(c.taskId);
          default:
            return continueSettled(c, snapshotId, snapshot);
        }
      }

      /**
       * Continues a handle whose row is settled (completed, failed, or
       * aborted), applying the interrupt refusal and the completed-task
       * instructions gate before the run.
       */
      async function continueSettled(
        c: Continuation,
        snapshotId: string,
        snapshot: SessionSnapshot
      ): Promise<DelegationResult> {
        if (snapshot.finishReason === 'interrupted') {
          // Continuing past an interrupt means answering it, which the
          // orchestrator cannot do; a retry fails identically, so this dead
          // end keeps its slot.
          return {
            response: `Error: task '${c.taskId}' stopped on an interrupt (a tool request that needs an answer from outside the sub-agent), and continuing interrupted tasks is not supported. Delegate a more self-contained task instead.`,
          };
        }
        const refusal = refuseEmptyFollowUp(
          c,
          snapshot,
          `Task '${c.taskId}' already completed. To follow up in the sub-agent's session, call this tool again with instructions; re-running it without instructions would only repeat the finished work.`
        );
        if (refusal) return refusal;
        return runContinueFrom(c, snapshotId);
      }

      /**
       * Recovers a task whose worker is presumed dead. "Presumed" is the
       * word: a slow worker may still be alive and beating late, and
       * continuing past it would fork the run against it. The row is aborted
       * first as a fence (a live worker observes the flip and stops; a dead
       * one is unaffected), and one re-read then decides the recovery point:
       *
       * - Still expired: the worker really is gone, and the recovery falls
       *   back to the dead row's parent, the last snapshot committed before
       *   the detach.
       * - Settled (completed, failed, or aborted): a finalize landed in the
       *   window, so the worker was alive after all and the row itself holds
       *   the run's full state. It continues like any settled handle.
       * - Aborting: the fence reached a live worker, whose beats keep the row
       *   reading as aborting because its finalize is coming. That settled
       *   row will be the continuation point, so the refusal names the retry
       *   and returns the slot.
       *
       * The fence is the one write standing between the recovery and a live
       * worker, so a fence that fails is a refusal: proceeding without it
       * risks two live branches of one session.
       */
      async function continueExpired(
        c: Continuation,
        snapshotId: string
      ): Promise<DelegationResult> {
        if (abortableOf(c.agent) === false) {
          // The flip would land, but this store cannot signal a worker, so a
          // worker that is only late would never see it and the recovery
          // would race it. Retrying cannot change that, so the slot is kept.
          return {
            response: `Error: task '${c.taskId}' reads as expired, but this agent's store cannot signal its worker, so the recovery cannot be fenced against a worker that is only late. Delegate the task again if the work is still needed.`,
          };
        }
        // A cancelled call must not fence a task on its way out.
        c.abortSignal?.throwIfAborted();
        try {
          await c.agent.abortAgentAction.run(
            { snapshotId },
            { abortSignal: c.abortSignal }
          );
        } catch (e: unknown) {
          c.abortSignal?.throwIfAborted();
          logger.debug(
            `agents middleware: fencing task ${c.taskId} failed: ${errorMessage(e)}`
          );
          return refuseRead(
            e,
            `Error: could not fence task '${c.taskId}' before recovering it (${errorMessage(e)}).`
          );
        }
        let current: SessionSnapshot;
        try {
          current = await readSnapshotOnce(c.agent, snapshotId, c.abortSignal);
        } catch (e: unknown) {
          c.abortSignal?.throwIfAborted();
          logger.debug(
            `agents middleware: re-reading task ${c.taskId} after its fence failed: ${errorMessage(e)}`
          );
          return refuseRead(
            e,
            `Error: could not read task '${c.taskId}' after fencing it (${errorMessage(e)}).`
          );
        }
        if (current.status === 'expired') {
          return continueFromParent(c, current.parentId);
        }
        if (isSettled(current.status)) {
          return continueSettled(c, snapshotId, current);
        }
        // The fence reached a live worker: its finalize is coming and the
        // parent must not be raced.
        return windingDownRefusal(c.taskId);
      }

      /**
       * Turns a failed pre-read, fence, or parent read into the refusal
       * `msg`, with the slot following the failure's kind: a transient
       * failure names a retry that can succeed, so the refusal says so and
       * returns the slot; a classified dead end fails identically on retry
       * and keeps it, so the cap can still bite.
       */
      function refuseRead(e: unknown, msg: string): DelegationResult {
        if (isDeadEndRead(e)) return { response: msg };
        releaseDelegation();
        return { response: `${msg} Try again later.` };
      }

      /**
       * Refuses to continue a task whose row is aborting: the stop landed and
       * the worker is draining toward the finalize that makes the row a
       * continuation point, so the same handle is the thing to come back to,
       * and the refund reflects that.
       */
      function windingDownRefusal(taskId: string): DelegationResult {
        releaseDelegation();
        const hint = async
          ? ` Collect its settled state with ${taskTools.wait}, then continue that.`
          : '';
        return {
          response: `Task '${taskId}' is winding down after a stop signal; its progress is being saved. Once it settles, continue this taskId: as is if it stopped or failed, with instructions if it completed.${hint}`,
        };
      }

      /**
       * Refuses an instructions-less continuation of a snapshot whose last
       * committed turn finished (completed, with a result-carrying reason).
       * An empty input re-attempts the last committed turn, which is the
       * right retry for a run that stopped short and pure duplicate work for
       * one that finished; the refusal delivers `msg` (which names the fix)
       * and returns the slot, since the corrected call can succeed.
       */
      function refuseEmptyFollowUp(
        c: Continuation,
        snapshot: SessionSnapshot,
        msg: string
      ): DelegationResult | undefined {
        if (
          (snapshot.status ?? 'completed') !== 'completed' ||
          !carriesResult(snapshot.finishReason) ||
          c.instructions
        ) {
          return undefined;
        }
        releaseDelegation();
        return { response: msg };
      }

      /**
       * Recovers a dead task from its pending row's parent: the last snapshot
       * committed before the detach. The session's latest row cannot serve
       * here, because it is the dead pending row itself, so the parent
       * pointer is the one durable path back to the committed work. A
       * background delegation detaches at turn zero and has no parent, which
       * is the honest nothing-was-saved case. The parent is read first so a
       * finished parent turn gets the same instructions gate as a completed
       * task.
       */
      async function continueFromParent(
        c: Continuation,
        parentId: string | undefined
      ): Promise<DelegationResult> {
        if (!parentId) {
          return {
            response: `Error: task '${c.taskId}' saved no progress to continue from (it detached at the start of the run, and its worker died before finalizing). Delegate the task again if the work is still needed.`,
          };
        }
        let parent: SessionSnapshot;
        try {
          parent = await readSnapshotOnce(c.agent, parentId, c.abortSignal);
        } catch (e: unknown) {
          c.abortSignal?.throwIfAborted();
          return refuseRead(
            e,
            `Error: task '${c.taskId}' kept its progress in snapshot '${parentId}', which could not be read (${errorMessage(e)}).`
          );
        }
        const refusal = refuseEmptyFollowUp(
          c,
          parent,
          `Task '${c.taskId}' kept progress only up to its last finished turn (from before the background work started). Call this tool again with instructions to continue from there; an empty retry would only re-run that finished turn.`
        );
        if (refusal) return refusal;
        return runContinueFrom(c, parentId);
      }

      /**
       * The shared tail of every continuation: runs the sub-agent from the
       * named snapshot, with the instructions as the turn's user message, and
       * folds the outcome like a synchronous delegation, or launches it in the
       * background through the same launch protocol as a background
       * delegation.
       */
      async function runContinueFrom(
        c: Continuation,
        snapshotId: string
      ): Promise<DelegationResult> {
        const words: LaunchWords = {
          errPrefix: `Error continuing task '${c.taskId}'`,
          withoutBackground: `Continue it without "background" instead.`,
          started: (taskId) =>
            `Task ${c.taskId} continued in the background as ${taskId}.`,
          // The continuation is the same undertaking; its label follows the
          // handle.
          label: shared.labels.get(c.taskId),
        };
        if (c.background) {
          const refusal = refuseUndetachable(c.ref, c.agent, words);
          if (refusal) return refusal;
        }

        c.abortSignal?.throwIfAborted();
        logger.debug(
          `agents middleware: continuing task ${c.taskId} from snapshot ${snapshotId} (background: ${c.background}).`
        );
        let out: AgentOutput;
        try {
          out = await runSubAgent(
            c.agent,
            continueMessage(c.instructions, c.background),
            {
              snapshotId,
              detach: c.background,
              abortSignal: c.abortSignal,
            }
          );
        } catch (e: unknown) {
          logger.warn(
            `agents middleware: continuing task ${c.taskId} failed: ${errorMessage(e)}`
          );
          if (errorStatus(e) === 'FAILED_PRECONDITION') {
            // The runtime rejected the resume point itself (nothing behind
            // it, or a still-live worker); its message says which.
            return {
              response: `${words.errPrefix}: ${errorMessage(e)} If no progress was saved, delegate the task again.`,
            };
          }
          return { response: `${words.errPrefix}: ${errorMessage(e)}` };
        }
        if (c.background) {
          return foldDetachOutcome(c.ref, c.agent, out, words);
        }
        const result = foldDelegationOutput(c.ref, out);
        labelTask(result, words.label);
        return result;
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
          (input: z.infer<typeof asyncDelegateInputSchema>, { abortSignal }) =>
            input.background
              ? launchDelegation(ref, input.task, input.name, abortSignal)
              : runDelegation(ref, input.task, input.name, abortSignal)
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
            (input, { abortSignal }) =>
              reportTasks(input.taskIds ?? [], readSnapshotOnce, abortSignal)
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
              description: `Stops background sub-agent tasks whose results are no longer needed, and returns where that left each one. A live task reports "aborting" while it winds down and ${STOPPED_TASK_SETTLES}; a task that had already finished is unaffected and reports its result.`,
              inputSchema: backgroundTasksInputSchema,
              outputSchema: backgroundTasksResultSchema,
            },
            (input, { abortSignal }) =>
              reportTasks(input.taskIds ?? [], abortSnapshot, abortSignal)
          ),
        ];
      }
      const backgroundTools = async ? defineBackgroundTools() : [];

      // ── Shared continue tool ──────────────────────────────────────────

      // Instantiation is synchronous and the registry is read asynchronously,
      // so the tool is always registered: whether any sub-agent can leave a
      // continuable handle is decided per request in the generate hook, which
      // withholds the tool from the model when none can. Like the delegation
      // tools, its input schema depends on whether background execution
      // exists.
      claimName(continueTool, 'the continue tool');
      const continueTaskTool = tool(
        {
          name: continueTool,
          description: CONTINUE_TASK_TOOL_DESCRIPTION,
          inputSchema: async ? asyncContinueInputSchema : continueInputSchema,
          outputSchema: delegationResultSchema,
        },
        (input: z.infer<typeof asyncContinueInputSchema>, { abortSignal }) =>
          runContinue(input, abortSignal)
      );

      return {
        tools: [...delegationTools, ...backgroundTools, continueTaskTool],

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

          const continueInstructions = (await anyContinuableAgent())
            ? `\n` +
              `Results of delegations to sub-agents that keep sessions carry ` +
              `a taskId where the sub-agent's progress is addressable. If ` +
              `such a delegation fails or is aborted, its saved progress is ` +
              `not lost: call ${continueTool} with the taskId to continue it ` +
              `from where it stopped, either as-is or steered with ` +
              `instructions. A completed task accepts follow-up instructions ` +
              `in its own session the same way, without repeating the ` +
              `finished work. A task that stopped on an interrupt cannot be ` +
              `continued; delegate a more self-contained task instead. A ` +
              `result without a taskId is not continuable; delegate again to ` +
              `redo that work.\n`
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
            continueInstructions +
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

        // The continue tool can never succeed when every configured sub-agent
        // is client-managed, so the model is not shown it then.
        model: async (req, ctx, next) => {
          if (
            req.tools?.some((t) => t.name === continueTool) &&
            !(await anyContinuableAgent())
          ) {
            req = {
              ...req,
              tools: req.tools.filter((t) => t.name !== continueTool),
            };
          }
          return next(req, ctx);
        },
      };
    }
  );
