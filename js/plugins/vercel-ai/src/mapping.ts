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
 * Maps a single Vercel AI SDK v6 UIMessagePart to zero or more Genkit Parts.
 *
 * Handles `text`, `file`, and tool parts (both `ToolUIPart` with
 * `type: 'tool-<name>'` and `DynamicToolUIPart` with `type: 'dynamic-tool'`).
 *
 * A tool part with state `output-available` produces both a `toolRequest`
 * and a `toolResponse` part (matching Genkit's conversation model).
 */
export function mapUIPartToGenkit(part: UIMessage['parts'][number]): Part[] {
  if (part.type === 'text') {
    return [{ text: part.text }];
  }

  if (part.type === 'reasoning') {
    // Map AI SDK reasoning parts to Genkit reasoning parts so the model's
    // thinking is preserved across turns (e.g. when replaying history).
    const reasoning = (part as { text?: unknown }).text;
    return typeof reasoning === 'string' ? [{ reasoning }] : [];
  }

  if (part.type === 'file') {
    return [
      {
        media: {
          url: part.url,
          contentType: part.mediaType,
        },
      },
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
      {
        toolRequest: {
          ref: toolCallId,
          name: toolName,
          input: p.input,
        },
      },
    ];

    if (p.state === 'output-available' && p.output !== undefined) {
      parts.push({
        toolResponse: {
          ref: toolCallId,
          name: toolName,
          output: p.output,
        },
      });
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
 * Extracts resolved tool invocations from a UIMessage array.
 *
 * Supports the v6 per-tool format (`type === 'tool-<toolName>'`,
 * `state === 'output-available'`).
 *
 * Scans **all** messages (most-recent first) and returns every resolved tool
 * result, de-duplicated by `toolCallId` (keeping the most recent). The
 * transport then filters these against the set of pending interrupts so only
 * the relevant resolutions are sent back as a resume payload — this avoids
 * misclassifying an auto-executed tool's output as an interrupt response.
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

    for (const part of msg.parts) {
      // v6 per-tool format: { type: 'tool-<name>', toolCallId,
      //   state: 'output-available', output: ... }
      // In SDK v6, ToolUIPart encodes the tool name in the `type` field
      // (e.g. `type: 'tool-userApproval'`) and does NOT have a separate
      // `toolName` property.  We derive the name from `type` when
      // `toolName` is not explicitly present.
      const p = part as unknown as Record<string, unknown>;
      if (
        part.type.startsWith('tool-') &&
        typeof p.toolCallId === 'string' &&
        p.toolCallId &&
        !seen.has(p.toolCallId) &&
        p.state === 'output-available' &&
        p.output !== undefined
      ) {
        const toolName =
          typeof p.toolName === 'string' && p.toolName
            ? p.toolName
            : part.type.slice('tool-'.length);
        seen.add(p.toolCallId);
        results.push({
          toolCallId: p.toolCallId,
          toolName,
          input: p.input,
          result: p.output,
        });
      }
    }
  }

  return results;
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
