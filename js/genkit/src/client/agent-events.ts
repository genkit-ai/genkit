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
 * Client-side helpers for consuming the v2 Agent event stream.
 *
 * The server emits a tagged-union `event` field on each `AgentStreamChunk`
 * (in addition to the legacy fields). This module provides a typed visitor
 * so consumers can `switch (event.type)` without duplicating the dispatch
 * across every UI.
 */

/**
 * The union of events an agent may emit during a turn. Mirrors the
 * `AgentEventSchema` defined server-side in `@genkit-ai/ai`. Kept inlined
 * here so client-only callers don't pull in the AI package.
 */
export type AgentEvent =
  | { type: 'model-chunk'; chunk: ModelChunkData }
  | { type: 'status'; label: string; key?: string }
  | { type: 'progress'; label?: string; current: number; total: number }
  | { type: 'phase'; phase: string }
  | { type: 'artifact-emitted'; artifact: any }
  | { type: 'artifact-start'; id: string; name: string; mediaType?: string }
  | { type: 'artifact-delta'; id: string; delta: any }
  | { type: 'artifact-complete'; id: string }
  | { type: 'snapshot'; snapshotId: string; continuationId: string }
  | {
      type: 'interrupt';
      toolCallId: string;
      toolName: string;
      input: unknown;
      kind: 'respond' | 'restart';
      metadata?: unknown;
    }
  | { type: 'detached'; snapshotId: string; continuationId: string }
  | { type: 'turn-end'; snapshotId?: string; continuationId?: string }
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
    toolRequest?: { name: string; input?: unknown; ref?: string; partial?: boolean };
    toolResponse?: { name: string; output?: unknown; ref?: string };
    [k: string]: unknown;
  }>;
  [k: string]: unknown;
}

/**
 * Shape of a v2 chunk from `streamFlow`. Both legacy fields and `event` are
 * populated by the server during the migration window.
 */
export interface AgentStreamChunkV2 {
  modelChunk?: ModelChunkData;
  status?: unknown;
  artifact?: unknown;
  turnEnd?: { snapshotId?: string };
  event?: AgentEvent;
}

/**
 * Handlers for the v2 event union. All optional — unhandled events are
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
  onStatus?: (status: { label: string; key?: string }) => void;
  onProgress?: (progress: {
    label?: string;
    current: number;
    total: number;
  }) => void;
  onPhase?: (phase: string) => void;
  onArtifact?: (artifact: unknown) => void;
  onSnapshot?: (snapshot: {
    snapshotId: string;
    continuationId: string;
  }) => void;
  onInterrupt?: (interrupt: {
    toolCallId: string;
    toolName: string;
    input: unknown;
    kind: 'respond' | 'restart';
    metadata?: unknown;
  }) => void;
  onDetached?: (detached: {
    snapshotId: string;
    continuationId: string;
  }) => void;
  onTurnEnd?: (turn: {
    snapshotId?: string;
    continuationId?: string;
  }) => void;
  onError?: (error: { errorText: string }) => void;
}

/**
 * Walk a single chunk and dispatch its event to the provided handlers.
 *
 * For chunks that contain a `model-chunk` event, the `onModelChunk` handler
 * fires (raw), AND the content parts are further dispatched to `onText`,
 * `onReasoning`, `onToolRequest`, `onToolResponse`. Use whichever is more
 * convenient for your UI.
 *
 * For backwards compatibility: if a chunk has the legacy fields but no v2
 * `event` field, this synthesizes the equivalent event on the fly.
 */
export function walkAgentEvent(
  chunk: AgentStreamChunkV2,
  handlers: AgentEventHandlers
): void {
  // Prefer the explicit v2 event; fall back to synthesizing from legacy fields
  // for older servers.
  const event = chunk.event ?? synthesizeEvent(chunk);
  if (!event) return;

  switch (event.type) {
    case 'model-chunk': {
      handlers.onModelChunk?.(event.chunk);
      for (const part of event.chunk.content ?? []) {
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
      handlers.onStatus?.({ label: event.label, key: event.key });
      return;
    case 'progress':
      handlers.onProgress?.({
        label: event.label,
        current: event.current,
        total: event.total,
      });
      return;
    case 'phase':
      handlers.onPhase?.(event.phase);
      return;
    case 'artifact-emitted':
    case 'artifact-start':
    case 'artifact-delta':
    case 'artifact-complete':
      handlers.onArtifact?.(
        'artifact' in event ? event.artifact : event
      );
      return;
    case 'snapshot':
      handlers.onSnapshot?.({
        snapshotId: event.snapshotId,
        continuationId: event.continuationId,
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
    case 'detached':
      handlers.onDetached?.({
        snapshotId: event.snapshotId,
        continuationId: event.continuationId,
      });
      return;
    case 'turn-end':
      handlers.onTurnEnd?.({
        snapshotId: event.snapshotId,
        continuationId: event.continuationId,
      });
      return;
    case 'error':
      handlers.onError?.({ errorText: event.errorText });
      return;
  }
}

function synthesizeEvent(chunk: AgentStreamChunkV2): AgentEvent | undefined {
  if (chunk.modelChunk) {
    return { type: 'model-chunk', chunk: chunk.modelChunk };
  }
  if (chunk.artifact) {
    return { type: 'artifact-emitted', artifact: chunk.artifact };
  }
  if (chunk.turnEnd) {
    return {
      type: 'turn-end',
      snapshotId: chunk.turnEnd.snapshotId,
    };
  }
  if (chunk.status !== undefined && chunk.status !== null) {
    const s = chunk.status as { label?: string; key?: string };
    return {
      type: 'status',
      label: typeof s === 'string' ? s : (s?.label ?? JSON.stringify(s)),
      key: s?.key,
    };
  }
  return undefined;
}
