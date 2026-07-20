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
import type { AgentChat, AgentInput, AgentResponse } from 'genkit/beta';

import {
  deriveAgentCard,
  type AgentLike,
  type DeriveAgentCardOptions,
} from './agent-card.js';
import {
  a2aMessageToGenkit,
  a2aMessageToResumeInput,
  genkitPartToA2A,
  genkitPartsToA2A,
} from './mapping.js';

/**
 * A Genkit agent capable of streaming turns over the transport-agnostic
 * {@link AgentChat} surface (the value returned by `ai.defineAgent`).
 */
export interface GenkitAgent extends AgentLike {
  chat(init?: { sessionId?: string }): AgentChat;
}

/**
 * Configuration for {@link GenkitA2ARequestHandler}.
 */
export interface GenkitA2ARequestHandlerOptions extends DeriveAgentCardOptions {
  /** The Genkit agent to expose over A2A. */
  agent: GenkitAgent;
}

/**
 * Generates an A2A id (a UUID), used for task ids, message ids, and artifact
 * ids when the caller does not supply one.
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
 * Each A2A task corresponds to a single Genkit agent turn; the A2A `contextId`
 * is used as the Genkit `sessionId`, so a server-managed agent (one with a
 * `store`) resumes its session across tasks that share a context. Conversation
 * state lives in the agent's own `SessionStore`; this handler keeps a small
 * in-memory record of tasks so `getTask` and interrupt-resume detection work.
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
  /** In-memory task store: taskId -> latest Task snapshot. */
  private readonly tasks = new Map<string, Task>();

  constructor(options: GenkitA2ARequestHandlerOptions) {
    this.agent = options.agent;
    this.card = deriveAgentCard(options.agent, options);
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
   * {@link Task}. Internally drains {@link sendMessageStream}.
   */
  async sendMessage(
    params: MessageSendParams,
    _context?: ServerCallContext
  ): Promise<A2AMessage | Task> {
    let lastTask: Task | undefined;
    for await (const event of this.sendMessageStream(params)) {
      if (event.kind === 'task') {
        lastTask = event;
      } else if (
        event.kind === 'status-update' ||
        event.kind === 'artifact-update'
      ) {
        const existing = lastTask;
        if (existing) {
          lastTask = this.tasks.get(existing.id) ?? existing;
        }
      }
    }
    if (!lastTask) {
      throw new Error('Agent produced no task.');
    }
    return this.tasks.get(lastTask.id) ?? lastTask;
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
    const taskId = userMessage.taskId ?? newId();

    // Determine whether this message resumes a task we paused for input.
    const referencedTask = userMessage.taskId
      ? this.tasks.get(userMessage.taskId)
      : undefined;
    const isResume = referencedTask?.status.state === 'input-required';

    const agentInput: AgentInput = isResume
      ? a2aMessageToResumeInput(userMessage)
      : { message: a2aMessageToResumeInputFresh(userMessage) };

    // Seed/refresh the task record and emit the initial Task event.
    const task: Task = {
      kind: 'task',
      id: taskId,
      contextId,
      status: { state: 'submitted', timestamp: new Date().toISOString() },
      history: [userMessage],
    };
    this.tasks.set(taskId, task);
    if (this.tasks.size > 10000) {
      const oldestKey = this.tasks.keys().next().value;
      if (oldestKey !== undefined) {
        this.tasks.delete(oldestKey);
      }
    }
    yield task;

    // Transition to `working`.
    yield this.setStatus(taskId, contextId, 'working', undefined, false);

    // The A2A contextId is the Genkit sessionId, so a server-managed agent
    // resumes its session across tasks sharing this context.
    const chat = this.agent.chat({ sessionId: contextId });

    const artifactId = newId();
    let firstChunk = true;

    let response: AgentResponse;
    try {
      const turn = chat.sendStream(agentInput);
      for await (const chunk of turn.stream) {
        // Only stream user-facing content (text / reasoning / media) as
        // artifact updates. Internal tool-call mechanics (toolRequest /
        // toolResponse / data) are dropped here: they are noise for a generic
        // A2A consumer and would otherwise accumulate into the task's result
        // artifact. Interrupts are still surfaced separately on the terminal
        // `input-required` status (see finalStatus).
        const parts = genkitPartsToA2A(
          userFacingParts(chunk.raw.modelChunk?.content)
        );
        if (parts.length > 0) {
          yield this.artifactUpdate(
            taskId,
            contextId,
            artifactId,
            parts,
            !firstChunk
          );
          firstChunk = false;
        }
      }
      response = await turn.response;
    } catch (err) {
      yield this.setStatus(
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
      return;
    }

    yield this.finalStatus(taskId, contextId, response);
  }

  // ── Task queries ───────────────────────────────────────────────────────

  async getTask(
    params: TaskQueryParams,
    _context?: ServerCallContext
  ): Promise<Task> {
    const task = this.tasks.get(params.id);
    if (!task) {
      throw new Error(`Task ${params.id} not found.`);
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
   * Builds (and records) a status-update event, updating the stored task's
   * status so a later `getTask` reflects it.
   */
  private setStatus(
    taskId: string,
    contextId: string,
    state: Task['status']['state'],
    message: A2AMessage | undefined,
    final: boolean
  ): TaskStatusUpdateEvent {
    const status: Task['status'] = {
      state,
      ...(message && { message }),
      timestamp: new Date().toISOString(),
    };
    const task = this.tasks.get(taskId);
    if (task) {
      task.status = status;
    }
    return {
      kind: 'status-update',
      taskId,
      contextId,
      final,
      status,
    };
  }

  /**
   * Builds an artifact-update event and accumulates its parts onto the stored
   * task so a later `getTask` includes the produced artifact.
   */
  private artifactUpdate(
    taskId: string,
    contextId: string,
    artifactId: string,
    parts: A2APart[],
    append: boolean
  ): TaskArtifactUpdateEvent {
    const task = this.tasks.get(taskId);
    if (task) {
      const artifacts = task.artifacts ?? (task.artifacts = []);
      const existing = artifacts.find((a) => a.artifactId === artifactId);
      if (existing) {
        existing.parts.push(...parts);
      } else {
        artifacts.push({ artifactId, parts: [...parts] });
      }
    }
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
   * Derives the terminal A2A status from a completed Genkit turn's finish
   * reason, attaching interrupt tool requests on the `input-required` path and
   * the error text on the `failed` path.
   */
  private finalStatus(
    taskId: string,
    contextId: string,
    response: AgentResponse
  ): TaskStatusUpdateEvent {
    const finishReason = response.finishReason;

    if (finishReason === 'interrupted') {
      // Surface the interrupt tool requests so the client can resolve them and
      // resume the task with a follow-up message.
      const interruptParts = (response.message?.content ?? [])
        .filter((p) => !!p.toolRequest && !!(p.metadata as any)?.interrupt)
        .map((p) => genkitPartToA2A(p))
        .filter((p): p is A2APart => p !== undefined);
      return this.setStatus(
        taskId,
        contextId,
        'input-required',
        agentMessage(interruptParts, taskId, contextId),
        true
      );
    }

    if (finishReason === 'failed') {
      return this.setStatus(
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

    if (finishReason === 'aborted') {
      return this.setStatus(taskId, contextId, 'canceled', undefined, true);
    }

    // stop / length / other / unknown -> completed. Echo the final assistant
    // message so non-streaming clients see the result.
    const finalParts = genkitPartsToA2A(response.message?.content);
    return this.setStatus(
      taskId,
      contextId,
      'completed',
      finalParts.length > 0
        ? agentMessage(finalParts, taskId, contextId)
        : undefined,
      true
    );
  }
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
 * Maps a fresh (non-resume) A2A user message to a Genkit message. Split out so
 * {@link a2aMessageToResumeInput} can stay focused on the resume path.
 */
function a2aMessageToResumeInputFresh(message: A2AMessage) {
  return a2aMessageToGenkit(message);
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
