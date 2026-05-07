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
  AgentCard,
  Artifact as A2AArtifact,
  DeleteTaskPushNotificationConfigParams,
  GetTaskPushNotificationConfigParams,
  ListTaskPushNotificationConfigParams,
  Message as A2AMessage,
  MessageSendParams,
  Task as A2ATask,
  TaskArtifactUpdateEvent,
  TaskIdParams,
  TaskPushNotificationConfig,
  TaskQueryParams,
  TaskStatusUpdateEvent,
} from '@a2a-js/sdk';
import {
  A2AError,
  type A2ARequestHandler,
  type ServerCallContext,
} from '@a2a-js/sdk/server';
import type { AgentOutput, SessionSnapshot } from 'genkit/beta';

import {
  mapA2APartToGenkit,
  mapGenkitArtifactToA2A,
  mapGenkitPartToA2A,
  type GenkitArtifact,
} from './mapping.js';

// ---------------------------------------------------------------------------
// Minimal Agent interface — the subset of the Genkit Agent we need.
// We define this locally rather than importing `Agent` from internal packages
// because external plugins must only import from `genkit` / `genkit/beta`.
// ---------------------------------------------------------------------------

/**
 * Minimal interface for a Genkit Agent — the subset used by the request handler.
 *
 * We use `any` for the `run()` parameters to avoid TypeScript variance issues
 * when assigning a real `Agent<State>` (whose `run()` uses narrow literal
 * types for roles and typed Part arrays) to this interface.
 */
export interface GenkitAgentLike {
  /** Run a single turn to completion and return the result. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  run(input?: any, options?: any): Promise<{ result: AgentOutput }>;

  /** Retrieve snapshot data by snapshotId. */
  getSnapshotData(snapshotId: string): Promise<SessionSnapshot | undefined>;

  /** Abort a running agent by snapshotId. Returns the previous status. */
  abort(snapshotId: string): Promise<string | undefined>;

  /** Action metadata — used to derive a default AgentCard. */
  __action: { name: string; description?: string };
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * Configuration for `GenkitA2ARequestHandler`.
 *
 * Either supply a full `card`, or let the handler derive one from the agent's
 * metadata (`__action.name` / `__action.description`).  When auto-deriving,
 * you **must** provide `url` so the card knows where the agent is hosted.
 */
export interface GenkitA2AHandlerConfig {
  /** The Genkit Agent to expose. */
  agent: GenkitAgentLike;

  /**
   * Full A2A Agent Card.  When provided, it is used as-is.
   * When omitted, a minimal card is derived from the agent's metadata
   * (requires `url` to be set).
   */
  card?: AgentCard;

  /**
   * Base URL where this A2A endpoint is served (e.g.
   * `http://localhost:41245`).  Required when `card` is not provided so the
   * auto-generated card has a valid `url` field.
   */
  url?: string;
}

// ---------------------------------------------------------------------------
// Genkit → A2A state mapping helpers
// ---------------------------------------------------------------------------

/**
 * Maps a Genkit snapshot status to an A2A TaskState.
 */
function snapshotStatusToTaskState(
  status?: string
): A2ATask['status']['state'] {
  switch (status) {
    case 'done':
      return 'completed';
    case 'pending':
      return 'working';
    case 'failed':
      return 'failed';
    case 'aborted':
      return 'canceled';
    default:
      return 'completed';
  }
}

/**
 * Builds an A2A Task from a Genkit AgentOutput.
 */
function buildTaskFromOutput(
  output: AgentOutput,
  taskId: string,
  contextId: string
): A2ATask {
  // Build history from state messages
  const stateMessages = (output.state as any)?.messages || [];
  const history: A2AMessage[] = stateMessages.map(
    (m: { role: string; content: any[] }, i: number) => ({
      kind: 'message' as const,
      messageId: `msg-${taskId}-${i}`,
      role: m.role === 'user' || m.role === 'system' ? 'user' : 'agent',
      parts: (m.content || []).map(mapGenkitPartToA2A),
      contextId,
      taskId,
    })
  );

  // Build artifacts
  const stateArtifacts: GenkitArtifact[] =
    (output.state as any)?.artifacts ||
    (output.artifacts as GenkitArtifact[]) ||
    [];
  const artifacts: A2AArtifact[] = stateArtifacts
    .filter((a: GenkitArtifact) => a.name) // A2A requires artifactId
    .map(mapGenkitArtifactToA2A);

  return {
    kind: 'task',
    id: taskId,
    contextId,
    status: {
      state: 'completed',
      timestamp: new Date().toISOString(),
    },
    ...(history.length > 0 && { history }),
    ...(artifacts.length > 0 && { artifacts }),
  };
}

/**
 * Builds an A2A Task from a Genkit SessionSnapshot.
 */
function buildTaskFromSnapshot(
  snapshot: SessionSnapshot,
  taskId: string,
  contextId?: string
): A2ATask {
  const effectiveContextId =
    contextId || (snapshot.state as any)?.custom?.a2aContextId || taskId;

  const stateMessages = (snapshot.state as any)?.messages || [];
  const history: A2AMessage[] = stateMessages.map(
    (m: { role: string; content: any[] }, i: number) => ({
      kind: 'message' as const,
      messageId: `msg-${taskId}-${i}`,
      role: m.role === 'user' || m.role === 'system' ? 'user' : 'agent',
      parts: (m.content || []).map(mapGenkitPartToA2A),
      contextId: effectiveContextId,
      taskId,
    })
  );

  const stateArtifacts: GenkitArtifact[] =
    (snapshot.state as any)?.artifacts || [];
  const artifacts: A2AArtifact[] = stateArtifacts
    .filter((a: GenkitArtifact) => a.name)
    .map(mapGenkitArtifactToA2A);

  return {
    kind: 'task',
    id: taskId,
    contextId: effectiveContextId,
    status: {
      state: snapshotStatusToTaskState(snapshot.status),
      timestamp: snapshot.createdAt || new Date().toISOString(),
    },
    ...(history.length > 0 && { history }),
    ...(artifacts.length > 0 && { artifacts }),
  };
}

// ---------------------------------------------------------------------------
// GenkitA2ARequestHandler
// ---------------------------------------------------------------------------

/**
 * Exposes a Genkit Agent as an A2A endpoint by implementing the
 * `A2ARequestHandler` interface from `@a2a-js/sdk`.
 *
 * This handler translates A2A protocol calls directly to the Genkit agent's
 * `run()` / `getSnapshotData()` / `abort()` methods, keeping Genkit's
 * `SessionStore` as the single source of truth (no dual-write with a
 * separate A2A `TaskStore`).
 *
 * The initial implementation supports blocking `sendMessage` only.
 * Streaming (`sendMessageStream`) falls back to blocking behavior.
 *
 * @example
 * ```ts
 * import { GenkitA2ARequestHandler } from '@genkit-ai/a2a';
 * import { jsonRpcHandler } from '@a2a-js/sdk/server/express';
 *
 * const handler = new GenkitA2ARequestHandler({
 *   agent: myAgent,
 *   card: {
 *     name: 'My Agent',
 *     description: 'A helpful assistant',
 *     url: 'https://my-agent.example.com/a2a',
 *     protocolVersion: '0.2.2',
 *     capabilities: {},
 *     defaultInputModes: ['text/plain'],
 *     defaultOutputModes: ['text/plain'],
 *   },
 * });
 *
 * app.use('/a2a', jsonRpcHandler({ requestHandler: handler }));
 * ```
 */
export class GenkitA2ARequestHandler implements A2ARequestHandler {
  private readonly agent: GenkitAgentLike;
  private readonly card: AgentCard;

  constructor(config: GenkitA2AHandlerConfig) {
    this.agent = config.agent;

    if (config.card) {
      this.card = config.card;
    } else {
      const url = config.url;
      if (!url) {
        throw new Error(
          'GenkitA2ARequestHandler: either `card` or `url` must be provided.'
        );
      }
      const name = config.agent.__action.name;
      const description =
        config.agent.__action.description || `${name} A2A agent`;
      this.card = {
        name,
        description,
        url,
        version: '1.0.0',
        protocolVersion: '0.2.2',
        capabilities: { streaming: false },
        defaultInputModes: ['text/plain'],
        defaultOutputModes: ['text/plain'],
        skills: [],
      };
    }
  }

  // -------------------------------------------------------------------------
  // Agent card
  // -------------------------------------------------------------------------

  async getAgentCard(): Promise<AgentCard> {
    return this.card;
  }

  async getAuthenticatedExtendedAgentCard(
    _context?: ServerCallContext
  ): Promise<AgentCard> {
    return this.card;
  }

  // -------------------------------------------------------------------------
  // sendMessage (blocking)
  // -------------------------------------------------------------------------

  async sendMessage(
    params: MessageSendParams,
    _context?: ServerCallContext
  ): Promise<A2AMessage | A2ATask> {
    const incoming = params.message;
    const contextId =
      (incoming as any).contextId || (incoming as any).taskId || crypto.randomUUID();
    const taskId = (incoming as any).taskId || crypto.randomUUID();

    // Map A2A parts → Genkit parts
    const genkitParts = incoming.parts.map(mapA2APartToGenkit);
    const isToolResponse = genkitParts.some(
      (p) => 'toolResponse' in p && (p as any).toolResponse !== undefined
    );

    // Build init — resume from existing snapshot if taskId references one
    const init: { snapshotId?: string; newSnapshotId?: string } = {};
    if ((incoming as any).taskId) {
      init.snapshotId = (incoming as any).taskId;
    }
    init.newSnapshotId = taskId; // Use taskId as snapshotId for 1:1 mapping

    // Run agent to completion
    const result = await this.agent.run(
      {
        messages: [
          {
            role: isToolResponse ? 'tool' : 'user',
            content: genkitParts,
          },
        ],
      },
      { init }
    );

    const output = result.result;
    return buildTaskFromOutput(output, taskId, contextId);
  }

  // -------------------------------------------------------------------------
  // sendMessageStream — falls back to blocking for now
  // -------------------------------------------------------------------------

  async *sendMessageStream(
    params: MessageSendParams,
    context?: ServerCallContext
  ): AsyncGenerator<
    A2AMessage | A2ATask | TaskStatusUpdateEvent | TaskArtifactUpdateEvent,
    void,
    undefined
  > {
    const result = await this.sendMessage(params, context);
    yield result;
  }

  // -------------------------------------------------------------------------
  // getTask — maps taskId → snapshotId
  // -------------------------------------------------------------------------

  async getTask(
    params: TaskQueryParams,
    _context?: ServerCallContext
  ): Promise<A2ATask> {
    const snapshot = await this.agent.getSnapshotData(params.id);
    if (!snapshot) {
      throw A2AError.taskNotFound(params.id);
    }
    const task = buildTaskFromSnapshot(snapshot, params.id);

    // Respect historyLength if specified
    if (params.historyLength !== undefined && task.history) {
      task.history = task.history.slice(-params.historyLength);
    }

    return task;
  }

  // -------------------------------------------------------------------------
  // cancelTask — maps to agent.abort()
  // -------------------------------------------------------------------------

  async cancelTask(
    params: TaskIdParams,
    _context?: ServerCallContext
  ): Promise<A2ATask> {
    const previousStatus = await this.agent.abort(params.id);
    if (previousStatus === undefined) {
      throw A2AError.taskNotFound(params.id);
    }
    const snapshot = await this.agent.getSnapshotData(params.id);
    if (!snapshot) {
      throw A2AError.taskNotFound(params.id);
    }
    return buildTaskFromSnapshot(snapshot, params.id);
  }

  // -------------------------------------------------------------------------
  // Push notifications — not supported
  // -------------------------------------------------------------------------

  async setTaskPushNotificationConfig(
    _params: TaskPushNotificationConfig,
    _context?: ServerCallContext
  ): Promise<TaskPushNotificationConfig> {
    throw A2AError.pushNotificationNotSupported();
  }

  async getTaskPushNotificationConfig(
    _params: TaskIdParams | GetTaskPushNotificationConfigParams,
    _context?: ServerCallContext
  ): Promise<TaskPushNotificationConfig> {
    throw A2AError.pushNotificationNotSupported();
  }

  async listTaskPushNotificationConfigs(
    _params: ListTaskPushNotificationConfigParams,
    _context?: ServerCallContext
  ): Promise<TaskPushNotificationConfig[]> {
    throw A2AError.pushNotificationNotSupported();
  }

  async deleteTaskPushNotificationConfig(
    _params: DeleteTaskPushNotificationConfigParams,
    _context?: ServerCallContext
  ): Promise<void> {
    throw A2AError.pushNotificationNotSupported();
  }

  // -------------------------------------------------------------------------
  // resubscribe — not supported
  // -------------------------------------------------------------------------

  async *resubscribe(
    _params: TaskIdParams,
    _context?: ServerCallContext
  ): AsyncGenerator<
    A2ATask | TaskStatusUpdateEvent | TaskArtifactUpdateEvent,
    void,
    undefined
  > {
    throw A2AError.unsupportedOperation('resubscribe');
  }
}
