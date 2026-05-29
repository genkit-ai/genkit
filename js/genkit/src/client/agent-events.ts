/**
 * @license
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

/**
 * Client-side helpers for consuming the agent event stream.
 *
 * Each chunk emitted by `streamFlow` against an agent endpoint is a tagged
 * discriminated union — `walkAgentEvent` dispatches it to typed handlers
 * so consumers don't open-code `switch (chunk.type) { ... }`.
 */

/**
 * The union of events an agent may emit during a turn. Mirrors the
 * `AgentStreamChunkSchema` discriminated union defined server-side in
 * `@genkit-ai/ai`. Kept inlined here so client-only callers don't pull in
 * the AI package.
 */
/**
 * Structured continuation handed back to the next turn.
 *
 *   `{ kind: 'snapshot', snapshotId }` — server-stored agents (small handle)
 *   `{ kind: 'state',    state }`      — client-managed agents (inline state)
 *
 * Clients round-trip whichever the server returns. The `snapshot` variant
 * is URL-fit; the `state` variant is not.
 */
export type AgentContinuation =
  | { kind: 'snapshot'; snapshotId: string }
  | { kind: 'state'; state: unknown };

export type AgentEvent =
  | { type: 'model-chunk'; chunk: ModelChunkData }
  | { type: 'status'; status: unknown }
  | { type: 'artifact-emitted'; artifact: any }
  | {
      type: 'snapshot';
      snapshotId: string;
      continuation: AgentContinuation;
    }
  | {
      type: 'interrupt';
      toolCallId: string;
      toolName: string;
      input: unknown;
      kind: 'respond' | 'restart';
      metadata?: unknown;
    }
  | {
      type: 'tool-error';
      toolCallId: string;
      toolName: string;
      errorText: string;
      errorCode?: string;
      details?: unknown;
    }
  | {
      type: 'detached';
      snapshotId: string;
      continuation: AgentContinuation;
    }
  | {
      type: 'turn-end';
      snapshotId?: string;
      continuation?: AgentContinuation;
    }
  | { type: 'error'; errorText: string };

/**
 * Minimal shape of a model chunk's content array. Mirrors Genkit's
 * `GenerateResponseChunkData` for the parts consumers actually walk.
 */
export interface ModelChunkData {
  role?: string;
  index?: number;
  content?: Array<{
    text?: string;
    reasoning?: string;
    toolRequest?: {
      name: string;
      input?: unknown;
      ref?: string;
      partial?: boolean;
    };
    toolResponse?: { name: string; output?: unknown; ref?: string };
    [k: string]: unknown;
  }>;
  [k: string]: unknown;
}

/**
 * Handlers for the agent event union. All optional — unhandled events are
 * silently dropped.
 */
export interface AgentEventHandlers {
  onModelChunk?: (chunk: ModelChunkData) => void;
  onText?: (delta: string) => void;
  onReasoning?: (delta: string) => void;
  onToolRequest?: (req: {
    toolCallId: string;
    toolName: string;
    input: unknown;
    partial?: boolean;
  }) => void;
  onToolResponse?: (res: {
    toolCallId: string;
    toolName: string;
    output: unknown;
  }) => void;
  /**
   * Application-defined status payload. Shape is per-agent; consumers
   * narrow it through their own type parameter (e.g. the React hook's
   * `<TStatus>` generic).
   */
  onStatus?: (status: unknown) => void;
  onArtifact?: (artifact: unknown) => void;
  onSnapshot?: (snapshot: {
    snapshotId: string;
    continuation: AgentContinuation;
  }) => void;
  onInterrupt?: (interrupt: {
    toolCallId: string;
    toolName: string;
    input: unknown;
    kind: 'respond' | 'restart';
    metadata?: unknown;
  }) => void;
  onToolError?: (err: {
    toolCallId: string;
    toolName: string;
    errorText: string;
    errorCode?: string;
    details?: unknown;
  }) => void;
  onDetached?: (detached: {
    snapshotId: string;
    continuation: AgentContinuation;
  }) => void;
  onTurnEnd?: (turn: {
    snapshotId?: string;
    continuation?: AgentContinuation;
  }) => void;
  onError?: (error: { errorText: string }) => void;
}

/**
 * Dispatch a single agent event to the provided handlers.
 *
 * For `model-chunk` events, `onModelChunk` fires (raw), AND the content
 * parts are further dispatched to `onText` / `onReasoning` /
 * `onToolRequest` / `onToolResponse`. Use whichever is more convenient.
 */
export function walkAgentEvent(
  event: AgentEvent,
  handlers: AgentEventHandlers
): void {
  switch (event.type) {
    case 'model-chunk': {
      handlers.onModelChunk?.(event.chunk);
      for (const part of event.chunk.content ?? []) {
        // Defensive: model adapters have occasionally emitted holes in
        // content arrays. Skip rather than crashing the whole stream.
        if (!part) continue;
        if (typeof part.text === 'string' && part.text.length > 0) {
          handlers.onText?.(part.text);
        }
        if (typeof part.reasoning === 'string' && part.reasoning.length > 0) {
          handlers.onReasoning?.(part.reasoning);
        }
        if (part.toolRequest) {
          handlers.onToolRequest?.({
            toolCallId: part.toolRequest.ref ?? part.toolRequest.name,
            toolName: part.toolRequest.name,
            input: part.toolRequest.input,
            partial: part.toolRequest.partial,
          });
        }
        if (part.toolResponse) {
          handlers.onToolResponse?.({
            toolCallId: part.toolResponse.ref ?? part.toolResponse.name,
            toolName: part.toolResponse.name,
            output: part.toolResponse.output,
          });
        }
      }
      return;
    }
    case 'status':
      handlers.onStatus?.(event.status);
      return;
    case 'artifact-emitted':
      handlers.onArtifact?.(event.artifact);
      return;
    case 'snapshot':
      handlers.onSnapshot?.({
        snapshotId: event.snapshotId,
        continuation: event.continuation,
      });
      return;
    case 'interrupt':
      handlers.onInterrupt?.({
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        input: event.input,
        kind: event.kind,
        metadata: event.metadata,
      });
      return;
    case 'tool-error':
      handlers.onToolError?.({
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        errorText: event.errorText,
        errorCode: event.errorCode,
        details: event.details,
      });
      return;
    case 'detached':
      handlers.onDetached?.({
        snapshotId: event.snapshotId,
        continuation: event.continuation,
      });
      return;
    case 'turn-end':
      handlers.onTurnEnd?.({
        snapshotId: event.snapshotId,
        continuation: event.continuation,
      });
      return;
    case 'error':
      handlers.onError?.({ errorText: event.errorText });
      return;
  }
}
