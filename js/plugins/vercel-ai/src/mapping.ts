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

import type { UIMessage } from 'ai';
import type { MessageData, Part } from 'genkit';

// ---------------------------------------------------------------------------
// UIMessage → Genkit
// ---------------------------------------------------------------------------

/**
 * Maps a Vercel AI SDK role to a Genkit role.
 */
function mapRole(role: UIMessage['role']): MessageData['role'] {
  switch (role) {
    case 'assistant':
      return 'model';
    case 'user':
      return 'user';
    case 'system':
      return 'system';
    default:
      // Unknown roles fall back to 'user' so content is not silently dropped.
      return 'user';
  }
}

/**
 * Maps a Genkit role to a Vercel AI SDK role.
 */
function mapGenkitRole(role: MessageData['role']): UIMessage['role'] {
  switch (role) {
    case 'model':
      return 'assistant';
    case 'system':
      return 'system';
    case 'user':
    default:
      return 'user';
  }
}

/**
 * Extracts a plain-object `metadata` value from an AI SDK UI part, if present.
 *
 * AI SDK part types don't all formally declare a `metadata` field, but the
 * transport (and middleware-aware UIs) may attach one. We read it defensively
 * so Genkit part metadata survives a UI → Genkit round-trip.
 */
function extractPartMetadata(
  part: unknown
): Record<string, unknown> | undefined {
  const md = (part as { metadata?: unknown } | null)?.metadata;
  if (md && typeof md === 'object' && !Array.isArray(md)) {
    return md as Record<string, unknown>;
  }
  return undefined;
}

/**
 * Maps a single Vercel AI SDK v6 UIMessagePart to zero or more Genkit Parts.
 *
 * Handles `text`, `file`, and tool parts (both `ToolUIPart` with
 * `type: 'tool-<name>'` and `DynamicToolUIPart` with `type: 'dynamic-tool'`).
 *
 * A tool part with state `output-available` produces both a `toolRequest`
 * and a `toolResponse` part (matching Genkit's conversation model).
 */
export function mapUIPartToGenkit(part: UIMessage['parts'][number]): Part[] {
  // Genkit part metadata that may have been attached to the UI part (e.g. by
  // a middleware-aware UI or a previous Genkit → UI mapping). Preserved so it
  // survives a UI → Genkit round-trip.
  const metadata = extractPartMetadata(part);
  const withMetadata = <T extends Part>(p: T): T =>
    metadata ? ({ ...p, metadata } as T) : p;

  if (part.type === 'text') {
    return [withMetadata({ text: part.text })];
  }

  if (part.type === 'reasoning') {
    // Map AI SDK reasoning parts to Genkit reasoning parts so the model's
    // thinking is preserved across turns (e.g. when replaying history).
    // Empty reasoning is dropped to avoid emitting stray, contentless
    // reasoning blocks (mirrors the streaming side in client.ts, which also
    // skips `reasoning === ''`).
    const reasoning = (part as { text?: unknown }).text;
    return typeof reasoning === 'string' && reasoning !== ''
      ? [withMetadata({ reasoning })]
      : [];
  }

  if (part.type === 'file') {
    return [
      withMetadata({
        media: {
          url: part.url,
          contentType: part.mediaType,
        },
      }),
    ];
  }

  // Handle v6 tool parts:
  //   ToolUIPart:    type: 'tool-<name>', toolCallId, state, input?, output?
  //   DynamicToolUIPart: type: 'dynamic-tool', toolName, toolCallId, state, input?, output?
  //
  // Use runtime property checks via Record cast since TypeScript can't
  // narrow on prefix match for template literal types.
  if (part.type === 'dynamic-tool' || part.type.startsWith('tool-')) {
    const p = part as unknown as Record<string, unknown>;
    const toolCallId =
      typeof p.toolCallId === 'string' ? p.toolCallId : crypto.randomUUID();
    const toolName =
      typeof p.toolName === 'string' && p.toolName
        ? p.toolName
        : part.type.startsWith('tool-')
          ? part.type.slice('tool-'.length)
          : 'unknown';

    const parts: Part[] = [
      withMetadata({
        toolRequest: {
          ref: toolCallId,
          name: toolName,
          input: p.input,
        },
      }),
    ];

    if (p.state === 'output-available' && p.output !== undefined) {
      parts.push(
        withMetadata({
          toolResponse: {
            ref: toolCallId,
            name: toolName,
            output: p.output,
          },
        })
      );
    }

    return parts;
  }

  // Unsupported part types (source-*, step-start, data-*) are silently
  // skipped — they don't have a direct Genkit equivalent.
  return [];
}

/**
 * Maps a Vercel AI SDK UIMessage to a Genkit MessageData.
 */
export function mapUIMessageToGenkit(msg: UIMessage): MessageData {
  const role = mapRole(msg.role);
  const genkitParts: Part[] = msg.parts.flatMap(mapUIPartToGenkit);
  return { role, content: genkitParts };
}

// ---------------------------------------------------------------------------
// Genkit → UIMessage (session restore)
// ---------------------------------------------------------------------------

/**
 * Maps a single Genkit {@link Part} to zero or more AI SDK UI parts.
 *
 * `toolResponse` outputs live in separate (`tool`-role) Genkit messages, so a
 * `toolRequest` is paired with its response via `responsesByRef` to produce a
 * single AI SDK tool part: `output-available` when the response is known, or
 * `input-available` when it is still pending (e.g. an unresolved interrupt).
 *
 * Standalone `toolResponse` parts are skipped here — they're merged into the
 * matching request's tool part instead of producing a separate one.
 */
function mapGenkitPartToUI(
  part: Part,
  responsesByRef?: Map<string, unknown>
): UIMessage['parts'] {
  // Preserve Genkit part metadata on the UI part so it survives a
  // Genkit → UI → Genkit round-trip (read back by extractPartMetadata).
  const metadata =
    part.metadata && typeof part.metadata === 'object'
      ? (part.metadata as Record<string, unknown>)
      : undefined;
  const withMetadata = (
    p: Record<string, unknown>
  ): UIMessage['parts'][number] =>
    (metadata ? { ...p, metadata } : p) as UIMessage['parts'][number];

  if (typeof part.text === 'string') {
    return [withMetadata({ type: 'text', text: part.text })];
  }

  if (typeof part.reasoning === 'string') {
    return part.reasoning !== ''
      ? [withMetadata({ type: 'reasoning', text: part.reasoning })]
      : [];
  }

  if (part.media) {
    return [
      withMetadata({
        type: 'file',
        url: part.media.url,
        mediaType: part.media.contentType,
      }),
    ];
  }

  if (part.toolRequest) {
    const tr = part.toolRequest;
    const toolCallId = tr.ref ?? crypto.randomUUID();
    const base: Record<string, unknown> = {
      type: `tool-${tr.name}`,
      toolCallId,
      input: tr.input,
    };
    if (responsesByRef?.has(toolCallId)) {
      return [
        withMetadata({
          ...base,
          state: 'output-available',
          output: responsesByRef.get(toolCallId),
        }),
      ];
    }
    return [withMetadata({ ...base, state: 'input-available' })];
  }

  // Standalone toolResponse (paired with a request above) or any other part
  // type without a direct UI equivalent is skipped.
  return [];
}

/**
 * Maps a single Genkit {@link MessageData} to an AI SDK {@link UIMessage}.
 *
 * Pass `responsesByRef` (built across the whole conversation) so each
 * `toolRequest` can be paired with the `toolResponse` that resolved it, even
 * though they live in different messages. When omitted, all tool parts are
 * emitted in the `input-available` state.
 */
export function mapGenkitMessageToUI(
  msg: MessageData,
  responsesByRef?: Map<string, unknown>,
  id?: string
): UIMessage {
  const parts = (msg.content ?? []).flatMap((part) =>
    mapGenkitPartToUI(part, responsesByRef)
  );
  return {
    id: id ?? `restored-${crypto.randomUUID()}`,
    role: mapGenkitRole(msg.role),
    parts,
  };
}

/**
 * Converts the `messages` of a Genkit `SessionSnapshot` (an array of
 * {@link MessageData}) into AI SDK {@link UIMessage}s suitable for seeding
 * `useChat` (via its `messages` option or `setMessages`) to rehydrate a
 * previous conversation after a reload.
 *
 * Tool requests and their responses (which Genkit stores in separate
 * `model`/`tool` messages) are merged into single AI SDK tool parts, so the
 * UI renders each tool call as one element with both its input and output.
 * `tool`-role messages are therefore omitted from the result. Unresolved tool
 * requests (e.g. a pending interrupt) are emitted in the `input-available`
 * state.
 *
 * Conversation state is server-managed by `sessionId` (the `useChat` `id`), so
 * to resume the *next* turn you only need to reuse the same chat `id` — there
 * is no snapshot to restore on the client.
 *
 * @example
 * ```ts
 * const snapshot = await runFlow({ url: '/api/chat/weather/state', input: sessionId });
 * const messages = messagesFromSnapshot(snapshot.state.messages);
 * // pass `messages` to useChat({ id: sessionId, messages })
 * ```
 */
export function messagesFromSnapshot(messages: MessageData[]): UIMessage[] {
  // Collect every tool response by ref so a request in one message can be
  // paired with its response in another (Genkit stores them separately).
  const responsesByRef = new Map<string, unknown>();
  for (const msg of messages) {
    for (const part of msg.content ?? []) {
      if (part.toolResponse?.ref) {
        responsesByRef.set(part.toolResponse.ref, part.toolResponse.output);
      }
    }
  }

  const result: UIMessage[] = [];
  let index = 0;
  for (const msg of messages) {
    // tool-role messages only carry responses, which are merged into the
    // matching request's tool part — skip them.
    if (msg.role === 'tool') continue;
    const ui = mapGenkitMessageToUI(msg, responsesByRef, `restored-${index}`);
    index++;
    // Drop messages that produced no renderable parts.
    if (ui.parts.length > 0) result.push(ui);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Interrupt restart support
// ---------------------------------------------------------------------------

/**
 * Marker placed on a tool output to signal that the user chose to *restart*
 * (re-run) an interrupted tool rather than supply a final response.
 *
 * `Symbol.for` is intentionally avoided because tool outputs are serialized to
 * JSON before reaching the transport; a string marker survives that round-trip.
 */
const RESTART_MARKER = '__genkitVercelAiRestart__';

/**
 * The serialized shape produced by {@link restartInterrupt}.
 */
export interface RestartInterruptOutput {
  [RESTART_MARKER]: true;
  /**
   * Optional metadata surfaced to the tool as its `resumed` argument when it
   * re-runs server-side. Defaults to `true` when omitted.
   */
  metadata?: unknown;
}

/**
 * Builds a tool output that instructs the Genkit agent to **restart** (re-run)
 * an interrupted tool instead of accepting a user-supplied response.
 *
 * Pass the result to the AI SDK's `addToolResult` as the tool `output`:
 *
 * ```tsx
 * addToolResult({
 *   tool: 'userApproval',
 *   toolCallId,
 *   output: restartInterrupt({ note: 'user revised the request' }),
 * });
 * ```
 *
 * The transport detects this marker and emits a `resume.restart` entry (with
 * the original tool input, which the server requires to match exactly) instead
 * of a `resume.respond` entry. Any `metadata` is passed through to the tool as
 * its `resumed` value.
 */
export function restartInterrupt(metadata?: unknown): RestartInterruptOutput {
  return metadata === undefined
    ? { [RESTART_MARKER]: true }
    : { [RESTART_MARKER]: true, metadata };
}

/**
 * Returns the restart payload if `output` was produced by
 * {@link restartInterrupt}, otherwise `undefined`.
 */
export function asRestartInterrupt(
  output: unknown
): RestartInterruptOutput | undefined {
  if (
    output &&
    typeof output === 'object' &&
    (output as Record<string, unknown>)[RESTART_MARKER] === true
  ) {
    return output as RestartInterruptOutput;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * A resolved tool invocation extracted from the frontend message history.
 */
export interface ResolvedToolResult {
  toolCallId: string;
  toolName: string;
  /** The original tool input (needed to build a `resume.restart`). */
  input: unknown;
  /** The user-supplied output (or a {@link restartInterrupt} marker). */
  result: unknown;
}

/**
 * Returns true if a user message carries no meaningful content — i.e. it has
 * no non-empty text and no file parts. The AI SDK's `sendMessage({ text: '' })`
 * (used to nudge a resume) produces such a "phantom" user message, which must
 * not be treated as a real user turn.
 */
function isEmptyUserMessage(msg: UIMessage): boolean {
  return !msg.parts.some((part) => {
    if (part.type === 'text') {
      return typeof part.text === 'string' && part.text.trim() !== '';
    }
    if (part.type === 'file') {
      return true;
    }
    return false;
  });
}

/**
 * Extracts the resolved tool invocations carried by a single UIMessage.
 *
 * Supports the v6 per-tool format (`type === 'tool-<toolName>'`) as well as
 * dynamic tools (`type === 'dynamic-tool'`, with an explicit `toolName`),
 * matching `state === 'output-available'`. De-duplicates by `toolCallId`.
 */
function resolvedToolsFromMessage(msg: UIMessage): ResolvedToolResult[] {
  const seen = new Set<string>();
  const results: ResolvedToolResult[] = [];
  for (const part of msg.parts) {
    const p = part as unknown as Record<string, unknown>;
    const isTool =
      part.type === 'dynamic-tool' || part.type.startsWith('tool-');
    if (
      isTool &&
      typeof p.toolCallId === 'string' &&
      p.toolCallId &&
      !seen.has(p.toolCallId) &&
      p.state === 'output-available' &&
      p.output !== undefined
    ) {
      const toolName =
        typeof p.toolName === 'string' && p.toolName
          ? p.toolName
          : part.type.startsWith('tool-')
            ? part.type.slice('tool-'.length)
            : 'unknown';
      seen.add(p.toolCallId);
      results.push({
        toolCallId: p.toolCallId,
        toolName,
        input: p.input,
        result: p.output,
      });
    }
  }
  return results;
}

/**
 * Extracts resolved tool invocations from a UIMessage array.
 *
 * Supports the v6 per-tool format (`type === 'tool-<toolName>'`) as well as
 * dynamic tools (`type === 'dynamic-tool'`, with an explicit `toolName`),
 * matching `state === 'output-available'`.
 *
 * Scans **all** messages (most-recent first) and returns every resolved tool
 * result, de-duplicated by `toolCallId` (keeping the most recent).
 *
 * Prefer {@link currentTurnResolvedTools} for interrupt-resume detection — it
 * scopes the scan to the most recent turn so prior turns' (already-completed)
 * tool calls are not re-sent as a resume payload.
 */
export function extractResolvedToolResults(
  messages: UIMessage[]
): ResolvedToolResult[] {
  const seen = new Set<string>();
  const results: ResolvedToolResult[] = [];

  // Walk backwards so the most recent resolution for a given toolCallId wins.
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== 'assistant') continue;
    for (const tr of resolvedToolsFromMessage(msg)) {
      if (seen.has(tr.toolCallId)) continue;
      seen.add(tr.toolCallId);
      results.push(tr);
    }
  }

  return results;
}

/**
 * Returns the resolved tool invocations that constitute an interrupt resume
 * for the *current* turn, or an empty array if this is a fresh user turn.
 *
 * An interrupt resume is identified by the last assistant message carrying
 * resolved tool outputs (e.g. supplied via `addToolResult`) with no *non-empty*
 * user message after it. A real user message following the assistant turn means
 * the user started a new turn, so no resume is performed.
 *
 * This deliberately scopes detection to the most recent turn: tool calls from
 * earlier turns are already part of the server-side session state and must not
 * be replayed as a resume payload.
 */
export function currentTurnResolvedTools(
  messages: UIMessage[]
): ResolvedToolResult[] {
  // Locate the most recent assistant message.
  let lastAssistantIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'assistant') {
      lastAssistantIdx = i;
      break;
    }
  }
  if (lastAssistantIdx === -1) return [];

  // A non-empty user message after the assistant turn means a fresh user turn.
  for (let i = lastAssistantIdx + 1; i < messages.length; i++) {
    if (messages[i].role === 'user' && !isEmptyUserMessage(messages[i])) {
      return [];
    }
  }

  return resolvedToolsFromMessage(messages[lastAssistantIdx]);
}

/**
 * Finds the last user message in a UIMessage array.
 */
export function findLastUserMessage(
  messages: UIMessage[]
): UIMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return messages[i];
  }
  return undefined;
}

/**
 * Finds the last *non-empty* user message in a UIMessage array, skipping
 * phantom empty messages produced by `sendMessage({ text: '' })`.
 */
export function findLastNonEmptyUserMessage(
  messages: UIMessage[]
): UIMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'user' && !isEmptyUserMessage(msg)) return msg;
  }
  return undefined;
}
