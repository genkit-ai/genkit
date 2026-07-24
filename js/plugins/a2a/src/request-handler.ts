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

import type {
  Artifact as A2AArtifact,
  Message as A2AMessage,
  Part as A2APart,
  AgentCard,
  DeleteTaskPushNotificationConfigParams,
  GetTaskPushNotificationConfigParams,
  ListTaskPushNotificationConfigParams,
  MessageSendParams,
  Task,
  TaskArtifactUpdateEvent,
  TaskIdParams,
  TaskPushNotificationConfig,
  TaskQueryParams,
  TaskStatusUpdateEvent,
} from '@a2a-js/sdk';
import type { A2ARequestHandler, ServerCallContext } from '@a2a-js/sdk/server';
import type {
  AgentChat,
  AgentInput,
  AgentResponse,
  Artifact,
  MessageData,
  SessionSnapshot,
} from 'genkit/beta';

import {
  deriveAgentCard,
  type AgentLike,
  type DeriveAgentCardOptions,
} from './agent-card.js';
import {
  a2aMessageToGenkit,
  a2aMessageToResumeInput,
  finishReasonToA2AState,
  genkitMessageToA2AParts,
  genkitPartToA2A,
  genkitPartsToA2A,
  genkitRoleToA2A,
} from './mapping.js';
import { InMemoryA2ATaskStore, type A2ATaskStore } from './task-store.js';

/**
 * How a snapshot is looked up on the agent.
 */
type SnapshotLookup = string | { snapshotId: string } | { sessionId: string };

/**
 * A Genkit agent capable of streaming turns over the transport-agnostic
 * {@link AgentChat} surface (the value returned by `ai.defineAgent`).
 */
export interface GenkitAgent extends AgentLike {
  chat(init?: { sessionId?: string; snapshotId?: string }): AgentChat;
  /**
   * Reads a persisted snapshot. Present on every agent, but only functional for
   * a server-managed agent (one configured with a `SessionStore`).
   */
  getSnapshot?(lookup: SnapshotLookup): Promise<SessionSnapshot | undefined>;
}

/**
 * Configuration for {@link GenkitA2ARequestHandler}.
 */
export interface GenkitA2ARequestHandlerOptions extends DeriveAgentCardOptions {
  /** The Genkit agent to expose over A2A. */
  agent: GenkitAgent;
  /**
   * The store mapping an A2A `taskId` to the Genkit snapshot that currently
   * backs it. Only consulted for server-managed agents, and only written when
   * a task advances past its originating snapshot (i.e. an interrupted task is
   * resumed). Defaults to an {@link InMemoryA2ATaskStore}; supply a durable
   * implementation to survive restarts / span multiple processes.
   */
  taskStore?: A2ATaskStore;
}

/**
 * Generates an A2A id (a UUID), used for message ids and artifact ids (and, for
 * client-managed agents, task ids) when the caller does not supply one.
 */
function newId(): string {
  return crypto.randomUUID();
}

/**
 * Builds an A2A agent message carrying the given parts.
 */
function agentMessage(
  parts: A2APart[],
  taskId: string,
  contextId: string
): A2AMessage {
  return {
    kind: 'message',
    messageId: newId(),
    role: 'agent',
    taskId,
    contextId,
    parts,
  };
}

/**
 * An A2A {@link A2ARequestHandler} that runs a Genkit agent.
 *
 * ## Task identity
 *
 * The A2A `contextId` is used as the Genkit `sessionId`, so a server-managed
 * agent (one with a `store`) resumes its session across tasks that share a
 * context. For a server-managed agent an A2A `taskId` **is** the Genkit
 * snapshot id of the turn that originated the task (reserved up front and
 * surfaced on the turn's `turnStart` stream chunk). `getTask` and interrupt
 * resume therefore read straight from the agent's own snapshot store via
 * `agent.getSnapshot` - no task bookkeeping is needed for the common case.
 *
 * A {@link A2ATaskStore} pointer is written only when a task *advances* past
 * its originating snapshot - i.e. an interrupted task is resumed and the
 * resumed turn persists a new snapshot. Resolving a task is then
 * `taskStore.get(taskId)?.snapshotId ?? taskId`.
 *
 * For a **client-managed** agent (no store) there are no snapshots, so the
 * handler keeps a best-effort in-memory record of tasks (lost on restart) so
 * `getTask` still works within a single process.
 *
 * @example
 * ```ts
 * import { jsonRpcHandler, agentCardHandler, UserBuilder } from '@a2a-js/sdk/server/express';
 * import { GenkitA2ARequestHandler } from '@genkit-ai/a2a';
 *
 * const handler = new GenkitA2ARequestHandler({ agent: weatherAgent, url });
 * app.use('/', jsonRpcHandler({ requestHandler: handler, userBuilder: UserBuilder.noAuthentication }));
 * app.use('/.well-known/agent-card.json', agentCardHandler({ agentCardProvider: handler }));
 * ```
 */
export class GenkitA2ARequestHandler implements A2ARequestHandler {
  private readonly agent: GenkitAgent;
  private readonly card: AgentCard;
  private readonly taskStore: A2ATaskStore;
  private readonly serverManaged: boolean;
  /**
   * Best-effort in-memory task cache for client-managed (no-store) agents,
   * which have no snapshots to read tasks back from. Not durable.
   */
  private readonly noStoreTasks = new Map<string, Task>();

  constructor(options: GenkitA2ARequestHandlerOptions) {
    this.agent = options.agent;
    this.card = deriveAgentCard(options.agent, options);
    this.taskStore = options.taskStore ?? new InMemoryA2ATaskStore();
    const stateManagement = (options.agent as AgentLike).__action?.metadata
      ?.agent?.stateManagement;
    this.serverManaged =
      stateManagement === 'server' &&
      typeof options.agent.getSnapshot === 'function';
  }

  // ── Agent card ─────────────────────────────────────────────────────────

  async getAgentCard(): Promise<AgentCard> {
    return this.card;
  }

  async getAuthenticatedExtendedAgentCard(): Promise<AgentCard> {
    // No separate authenticated card; return the public one.
    return this.card;
  }

  // ── Messaging ──────────────────────────────────────────────────────────

  /**
   * Blocking send: runs a turn to completion and returns the resulting
   * {@link Task}. Internally drains {@link sendMessageStream}, accumulating the
   * streamed events into the final task.
   */
  async sendMessage(
    params: MessageSendParams,
    _context?: ServerCallContext
  ): Promise<A2AMessage | Task> {
    let task: Task | undefined;
    for await (const event of this.sendMessageStream(params)) {
      if (event.kind === 'task') {
        task = event;
      } else if (event.kind === 'status-update' && task) {
        task = { ...task, status: event.status };
      } else if (event.kind === 'artifact-update' && task) {
        task = { ...task, artifacts: mergeArtifact(task.artifacts, event) };
      }
    }
    if (!task) {
      throw new Error('Agent produced no task.');
    }
    return task;
  }

  /**
   * Streaming send: runs a Genkit agent turn and yields A2A events
   * (a `Task`, `working` status, streamed artifact updates, and a terminal
   * status update derived from the turn's finish reason).
   */
  async *sendMessageStream(
    params: MessageSendParams,
    _context?: ServerCallContext
  ): AsyncGenerator<
    A2AMessage | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent,
    void,
    undefined
  > {
    const userMessage = params.message;
    const contextId = userMessage.contextId ?? newId();
    const incomingTaskId = userMessage.taskId;

    // Resolve the agent input and the chat init for this turn. A task id
    // arriving on the message is one the server previously assigned (A2A
    // clients omit it on the first message of a task), so it identifies an
    // existing task to continue or resume.
    let agentInput: AgentInput;
    let chatInit: { sessionId?: string; snapshotId?: string };
    let taskId: string | undefined;

    if (incomingTaskId) {
      const { input, init } = await this.resolveContinuation(
        incomingTaskId,
        contextId,
        userMessage
      );
      agentInput = input;
      chatInit = init;
      taskId = incomingTaskId;
    } else {
      agentInput = { message: a2aMessageToGenkit(userMessage) };
      // Server-managed: contextId is the sessionId, resuming the session's
      // latest snapshot (or seeding a fresh session under this id).
      // Client-managed: no durable session; a new chat starts fresh.
      chatInit = this.serverManaged ? { sessionId: contextId } : {};
      // Server-managed task ids are the reserved snapshot id, learned from the
      // turn's first `turnStart` chunk. Client-managed has no snapshot, so mint
      // an opaque id up front.
      taskId = this.serverManaged ? undefined : newId();
    }

    const chat = this.agent.chat(chatInit);
    const artifactId = newId();
    let firstArtifactChunk = true;
    let task: Task | undefined;

    // Emits the initial Task + `working` status once the task id is known,
    // recording the task for the client-managed (no-store) read path.
    const emitStart = function* (
      this: GenkitA2ARequestHandler,
      id: string
    ): Generator<Task | TaskStatusUpdateEvent> {
      task = {
        kind: 'task',
        id,
        contextId,
        status: { state: 'submitted', timestamp: new Date().toISOString() },
        history: [userMessage],
      };
      this.rememberTask(task);
      yield task;
      const working = statusUpdate(id, contextId, 'working', undefined, false);
      task = applyStatus(task, working);
      this.rememberTask(task);
      yield working;
    }.bind(this);

    if (taskId) {
      yield* emitStart(taskId);
    }

    let response: AgentResponse;
    try {
      const turn = chat.sendStream(agentInput);
      for await (const chunk of turn.stream) {
        // Learn the server-managed task id from the first `turnStart` chunk
        // (carries the reserved snapshot id), emitting the Task before content.
        if (!taskId && chunk.snapshotId) {
          taskId = chunk.snapshotId;
          yield* emitStart(taskId);
        }

        // Only stream user-facing content (text / reasoning / media) as
        // artifact updates. Internal tool-call mechanics (toolRequest /
        // toolResponse / data) are dropped here: they are noise for a generic
        // A2A consumer and would otherwise accumulate into the task's result
        // artifact. Interrupts are still surfaced separately on the terminal
        // `input-required` status (see finalStatus).
        if (!taskId) continue;
        const parts = genkitPartsToA2A(
          userFacingParts(chunk.raw.modelChunk?.content)
        );
        if (parts.length > 0) {
          const evt = artifactUpdate(
            taskId,
            contextId,
            artifactId,
            parts,
            !firstArtifactChunk
          );
          task = applyArtifact(task, evt);
          this.rememberTask(task);
          yield evt;
          firstArtifactChunk = false;
        }
      }
      response = await turn.response;
    } catch (err) {
      // Ensure a task exists so the failure is reportable.
      if (!taskId) {
        taskId = newId();
        yield* emitStart(taskId);
      }
      const failed = statusUpdate(
        taskId,
        contextId,
        'failed',
        agentMessage(
          [{ kind: 'text', text: errorText(err) }],
          taskId,
          contextId
        ),
        true
      );
      task = applyStatus(task, failed);
      this.rememberTask(task);
      yield failed;
      return;
    }

    // A turn that produced no chunks at all (e.g. an immediate interrupt) still
    // needs a task id; fall back to the response's snapshot id.
    if (!taskId) {
      taskId = response.snapshotId ?? newId();
      yield* emitStart(taskId);
    }

    // Advance the task pointer if the turn persisted a new snapshot past the
    // task's originating one (server-managed interrupt resume).
    if (
      this.serverManaged &&
      response.snapshotId &&
      response.snapshotId !== taskId
    ) {
      await this.taskStore.set(taskId, {
        contextId,
        snapshotId: response.snapshotId,
      });
    }

    const final = this.finalStatus(taskId, contextId, response);
    task = applyStatus(task, final);
    this.rememberTask(task);
    yield final;
  }

  // ── Task queries ───────────────────────────────────────────────────────

  async getTask(
    params: TaskQueryParams,
    _context?: ServerCallContext
  ): Promise<Task> {
    const taskId = params.id;
    if (this.serverManaged) {
      const snapshotId = await this.resolve(taskId);
      const snapshot = await this.agent.getSnapshot!(snapshotId);
      if (!snapshot) {
        throw new Error(
          `Task ${taskId} not found (snapshot ${snapshotId} does not exist).`
        );
      }
      return this.snapshotToTask(taskId, snapshot);
    }
    const task = this.noStoreTasks.get(taskId);
    if (!task) {
      throw new Error(`Task ${taskId} not found.`);
    }
    return task;
  }

  async cancelTask(
    _params: TaskIdParams,
    _context?: ServerCallContext
  ): Promise<Task> {
    // Cancellation of an in-flight Genkit turn over A2A is not yet supported.
    throw new Error('Task cancellation is not supported.');
  }

  // ── Push notifications (unsupported) ─────────────────────────────────────

  async setTaskPushNotificationConfig(): Promise<TaskPushNotificationConfig> {
    throw new Error('Push notifications are not supported.');
  }

  async getTaskPushNotificationConfig(
    _params: TaskIdParams | GetTaskPushNotificationConfigParams
  ): Promise<TaskPushNotificationConfig> {
    throw new Error('Push notifications are not supported.');
  }

  async listTaskPushNotificationConfigs(
    _params: ListTaskPushNotificationConfigParams
  ): Promise<TaskPushNotificationConfig[]> {
    throw new Error('Push notifications are not supported.');
  }

  async deleteTaskPushNotificationConfig(
    _params: DeleteTaskPushNotificationConfigParams
  ): Promise<void> {
    throw new Error('Push notifications are not supported.');
  }

  async *resubscribe(): AsyncGenerator<
    Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent,
    void,
    undefined
  > {
    throw new Error('Stream resubscription is not supported.');
  }

  // ── Internals ────────────────────────────────────────────────────────────

  /**
   * Resolves an A2A task id to the snapshot id currently backing it. The
   * identity for the common case (`taskId` is the originating snapshot);
   * advanced only for resumed (interrupt) tasks recorded in the task store.
   */
  private async resolve(taskId: string): Promise<string> {
    const record = await this.taskStore.get(taskId);
    return record?.snapshotId ?? taskId;
  }

  /**
   * Builds the agent input and chat init for a message that continues or
   * resumes an existing task. Throws if the referenced task cannot be found.
   */
  private async resolveContinuation(
    taskId: string,
    contextId: string,
    userMessage: A2AMessage
  ): Promise<{
    input: AgentInput;
    init: { sessionId?: string; snapshotId?: string };
  }> {
    if (this.serverManaged) {
      const snapshotId = await this.resolve(taskId);
      const snapshot = await this.agent.getSnapshot!(snapshotId);
      if (!snapshot) {
        throw new Error(
          `Cannot continue task ${taskId}: snapshot ${snapshotId} does not exist.`
        );
      }
      const input =
        snapshot.finishReason === 'interrupted'
          ? a2aMessageToResumeInput(userMessage)
          : { message: a2aMessageToGenkit(userMessage) };
      return { input, init: { snapshotId } };
    }

    const task = this.noStoreTasks.get(taskId);
    if (!task) {
      throw new Error(`Cannot continue task ${taskId}: task not found.`);
    }
    const input =
      task.status.state === 'input-required'
        ? a2aMessageToResumeInput(userMessage)
        : { message: a2aMessageToGenkit(userMessage) };
    // Client-managed agents have no durable session to resume; a fresh chat is
    // the best effort within a single process.
    return { input, init: {} };
  }

  /**
   * Records a task in the in-memory cache used by client-managed (no-store)
   * agents. A no-op for server-managed agents, which read tasks from snapshots.
   */
  private rememberTask(task: Task): void {
    if (this.serverManaged) return;
    this.noStoreTasks.set(task.id, task);
    if (this.noStoreTasks.size > 10000) {
      const oldest = this.noStoreTasks.keys().next().value;
      if (oldest !== undefined) {
        this.noStoreTasks.delete(oldest);
      }
    }
  }

  /**
   * Derives the terminal A2A status from a completed Genkit turn's finish
   * reason, attaching interrupt tool requests on the `input-required` path and
   * the error text on the `failed` path.
   */
  private finalStatus(
    taskId: string,
    contextId: string,
    response: AgentResponse
  ): TaskStatusUpdateEvent {
    const state = finishReasonToA2AState(response.finishReason);

    if (state === 'input-required') {
      // Surface the interrupt tool requests so the client can resolve them and
      // resume the task with a follow-up message.
      const interruptParts = interruptPartsOf(response.message);
      return statusUpdate(
        taskId,
        contextId,
        'input-required',
        agentMessage(interruptParts, taskId, contextId),
        true
      );
    }

    if (state === 'failed') {
      return statusUpdate(
        taskId,
        contextId,
        'failed',
        agentMessage(
          [
            {
              kind: 'text',
              text: response.finishMessage ?? 'Agent turn failed.',
            },
          ],
          taskId,
          contextId
        ),
        true
      );
    }

    if (state === 'canceled') {
      return statusUpdate(taskId, contextId, 'canceled', undefined, true);
    }

    // completed (and rejected, mapped to its state). Echo the final assistant
    // message so non-streaming clients see the result.
    const finalParts = genkitPartsToA2A(response.message?.content);
    return statusUpdate(
      taskId,
      contextId,
      state,
      finalParts.length > 0
        ? agentMessage(finalParts, taskId, contextId)
        : undefined,
      true
    );
  }

  /**
   * Reconstructs an A2A {@link Task} from a persisted Genkit snapshot (the
   * server-managed `getTask` read path).
   */
  private snapshotToTask(taskId: string, snapshot: SessionSnapshot): Task {
    const contextId = snapshot.sessionId ?? snapshot.state?.sessionId ?? '';
    const state = a2aStateFromSnapshot(snapshot);
    const messages = snapshot.state?.messages ?? [];
    const lastModel = lastModelMessage(messages);

    let statusMessage: A2AMessage | undefined;
    if (state === 'input-required') {
      statusMessage = agentMessage(
        interruptPartsOf(lastModel),
        taskId,
        contextId
      );
    } else if (state === 'failed') {
      statusMessage = agentMessage(
        [
          {
            kind: 'text',
            text: snapshot.error?.message ?? 'Agent turn failed.',
          },
        ],
        taskId,
        contextId
      );
    } else if (state === 'completed' && lastModel) {
      const parts = genkitPartsToA2A(lastModel.content);
      if (parts.length > 0) {
        statusMessage = agentMessage(parts, taskId, contextId);
      }
    }

    const artifacts = (snapshot.state?.artifacts ?? []).map((a) =>
      genkitArtifactToA2A(a)
    );

    return {
      kind: 'task',
      id: taskId,
      contextId,
      status: {
        state,
        ...(statusMessage && { message: statusMessage }),
        timestamp:
          snapshot.updatedAt ?? snapshot.createdAt ?? new Date().toISOString(),
      },
      history: messages.map((m) => genkitMessageToA2A(m, taskId, contextId)),
      ...(artifacts.length > 0 && { artifacts }),
    };
  }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/**
 * Builds a status-update event.
 */
function statusUpdate(
  taskId: string,
  contextId: string,
  state: Task['status']['state'],
  message: A2AMessage | undefined,
  final: boolean
): TaskStatusUpdateEvent {
  return {
    kind: 'status-update',
    taskId,
    contextId,
    final,
    status: {
      state,
      ...(message && { message }),
      timestamp: new Date().toISOString(),
    },
  };
}

/**
 * Builds an artifact-update event.
 */
function artifactUpdate(
  taskId: string,
  contextId: string,
  artifactId: string,
  parts: A2APart[],
  append: boolean
): TaskArtifactUpdateEvent {
  return {
    kind: 'artifact-update',
    taskId,
    contextId,
    append,
    lastChunk: false,
    artifact: { artifactId, parts },
  };
}

/**
 * Returns a copy of `task` with the given status applied.
 */
function applyStatus(task: Task | undefined, evt: TaskStatusUpdateEvent): Task {
  return { ...(task as Task), status: evt.status };
}

/**
 * Returns a copy of `task` with the artifact-update event's parts accumulated
 * onto the matching artifact (by id).
 */
function applyArtifact(
  task: Task | undefined,
  evt: TaskArtifactUpdateEvent
): Task {
  const base = task as Task;
  return { ...base, artifacts: mergeArtifact(base.artifacts, evt) };
}

/**
 * Merges an artifact-update event's parts into an artifact list, appending to
 * the matching artifact (by id) or adding a new one.
 */
function mergeArtifact(
  artifacts: A2AArtifact[] | undefined,
  evt: TaskArtifactUpdateEvent
): A2AArtifact[] {
  const out = (artifacts ?? []).map((a) => ({ ...a, parts: [...a.parts] }));
  const existing = out.find((a) => a.artifactId === evt.artifact.artifactId);
  if (existing) {
    existing.parts.push(...evt.artifact.parts);
  } else {
    out.push({ ...evt.artifact, parts: [...evt.artifact.parts] });
  }
  return out;
}

/**
 * Maps a Genkit snapshot's lifecycle to an A2A task state. A terminal
 * persistence status (`failed` / `aborted`) wins; a `pending` snapshot is
 * still `working`; otherwise the semantic finish reason drives the state.
 */
function a2aStateFromSnapshot(
  snapshot: SessionSnapshot
): Task['status']['state'] {
  if (snapshot.status === 'failed') return 'failed';
  if (snapshot.status === 'aborted') return 'canceled';
  if (snapshot.status === 'pending') return 'working';
  return finishReasonToA2AState(snapshot.finishReason);
}

/**
 * Extracts the interrupt tool-request parts from a model message (the parts a
 * client must resolve to resume an `input-required` task).
 */
function interruptPartsOf(message: MessageData | undefined): A2APart[] {
  return (message?.content ?? [])
    .filter((p) => !!p.toolRequest && !!(p.metadata as any)?.interrupt)
    .map((p) => genkitPartToA2A(p))
    .filter((p): p is A2APart => p !== undefined);
}

/**
 * Returns the last `model`-role message in a history, or `undefined`.
 */
function lastModelMessage(messages: MessageData[]): MessageData | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'model') return messages[i];
  }
  return undefined;
}

/**
 * Maps a Genkit message to an A2A message bound to the given task/context.
 */
function genkitMessageToA2A(
  message: MessageData,
  taskId: string,
  contextId: string
): A2AMessage {
  return {
    kind: 'message',
    messageId: newId(),
    role: genkitRoleToA2A(message.role),
    taskId,
    contextId,
    parts: genkitMessageToA2AParts(message),
  };
}

/**
 * Maps a Genkit artifact to an A2A artifact.
 */
function genkitArtifactToA2A(artifact: Artifact): A2AArtifact {
  return {
    artifactId: artifact.name ?? newId(),
    ...(artifact.name && { name: artifact.name }),
    parts: genkitPartsToA2A(artifact.parts),
  };
}

/**
 * Filters a model chunk's Genkit parts down to user-facing content
 * (`text` / `reasoning` / `media`), dropping the internal tool-call mechanics
 * (`toolRequest` / `toolResponse` / `data` / `custom`). Used when streaming
 * artifact updates so a generic A2A consumer sees only the assistant's
 * response, not the tool plumbing.
 */
function userFacingParts(
  content: NonNullable<AgentResponse['message']>['content'] | undefined
): NonNullable<AgentResponse['message']>['content'] {
  return (content ?? []).filter(
    (p) =>
      typeof p.text === 'string' || typeof p.reasoning === 'string' || !!p.media
  );
}

/**
 * Extracts a human-readable error message from an unknown thrown value.
 */
function errorText(err: unknown): string {
  if (err instanceof Error) return err.message || 'Agent execution failed';
  if (typeof err === 'string') return err || 'Agent execution failed';
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}
