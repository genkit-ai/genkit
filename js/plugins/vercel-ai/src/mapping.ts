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

import type { MessageData, Part } from 'genkit';

// ---------------------------------------------------------------------------
// Vercel AI SDK types — defined locally to avoid a hard dependency on the
// `ai` package.  These mirror the wire format used by `useChat`.
// ---------------------------------------------------------------------------

/**
 * A single part within a Vercel AI SDK UIMessage.
 */
export interface UIMessagePart {
  type: string;
  /** Present when type === 'text'. */
  text?: string;
  /** Present when type === 'tool-invocation'. */
  toolInvocation?: {
    toolCallId: string;
    toolName: string;
    args: unknown;
    state: 'call' | 'partial-call' | 'result';
    result?: unknown;
  };
  /** Present when type === 'file'. */
  mediaType?: string;
  /** Present when type === 'file'. */
  url?: string;
}

/**
 * Vercel AI SDK UIMessage — the wire format used by `useChat`.
 */
export interface UIMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'data';
  /** @deprecated Removed in AI SDK v6; kept optional for backward compat. */
  content?: string;
  parts: UIMessagePart[];
  createdAt?: string | Date;
}

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
      return 'user';
  }
}

/**
 * Maps a single UIMessagePart to zero or more Genkit Parts.
 *
 * A `tool-invocation` with state `result` produces both a `toolRequest`
 * and a `toolResponse` part (matching Genkit's conversation model).
 */
export function mapUIPartToGenkit(part: UIMessagePart): Part[] {
  if (part.type === 'text' && part.text !== undefined) {
    return [{ text: part.text }];
  }

  if (part.type === 'tool-invocation' && part.toolInvocation) {
    const ti = part.toolInvocation;
    const parts: Part[] = [
      {
        toolRequest: {
          ref: ti.toolCallId,
          name: ti.toolName,
          input: ti.args,
        },
      },
    ];
    if (ti.state === 'result' && ti.result !== undefined) {
      parts.push({
        toolResponse: {
          ref: ti.toolCallId,
          name: ti.toolName,
          output: ti.result,
        },
      });
    }
    return parts;
  }

  if (part.type === 'file' && part.url) {
    return [
      {
        media: {
          url: part.url,
          contentType: part.mediaType,
        },
      },
    ];
  }

  // Unsupported part types (reasoning, source-*, step-start, data-*) are
  // silently skipped — they don't have a direct Genkit equivalent.
  return [];
}

/**
 * Maps a Vercel AI SDK UIMessage to a Genkit MessageData.
 */
export function mapUIMessageToGenkit(msg: UIMessage): MessageData {
  const role = mapRole(msg.role);

  // Prefer structured `parts` when available; fall back to `content` string.
  let genkitParts: Part[];
  if (msg.parts && msg.parts.length > 0) {
    genkitParts = msg.parts.flatMap(mapUIPartToGenkit);
  } else if (msg.content) {
    genkitParts = [{ text: msg.content }];
  } else {
    genkitParts = [];
  }

  return { role, content: genkitParts };
}

// ---------------------------------------------------------------------------
// Genkit → UIMessage (for state reconstruction)
// ---------------------------------------------------------------------------

/**
 * Maps a Genkit Part to a UIMessagePart.
 */
export function mapGenkitPartToUI(part: Part): UIMessagePart | null {
  if ('text' in part && part.text !== undefined) {
    return { type: 'text', text: part.text };
  }

  if ('toolRequest' in part && part.toolRequest) {
    return {
      type: 'tool-invocation',
      toolInvocation: {
        toolCallId: part.toolRequest.ref || crypto.randomUUID(),
        toolName: part.toolRequest.name,
        args: part.toolRequest.input,
        state: 'call',
      },
    };
  }

  if ('toolResponse' in part && part.toolResponse) {
    // Tool responses in Genkit are separate parts; map them as resolved
    // tool invocations.
    return {
      type: 'tool-invocation',
      toolInvocation: {
        toolCallId: part.toolResponse.ref || crypto.randomUUID(),
        toolName: part.toolResponse.name,
        args: {},
        state: 'result',
        result: part.toolResponse.output,
      },
    };
  }

  if ('media' in part && part.media) {
    return {
      type: 'file',
      url: part.media.url,
      mediaType: part.media.contentType,
    };
  }

  return null;
}

/**
 * Maps a Genkit MessageData to a Vercel AI SDK UIMessage.
 */
export function mapGenkitMessageToUI(msg: MessageData, id?: string): UIMessage {
  const role =
    msg.role === 'model'
      ? 'assistant'
      : msg.role === 'system'
        ? 'system'
        : 'user';

  const parts: UIMessagePart[] = (msg.content || [])
    .map(mapGenkitPartToUI)
    .filter((p): p is UIMessagePart => p !== null);

  const textContent = parts
    .filter((p) => p.type === 'text')
    .map((p) => p.text || '')
    .join('');

  return {
    id: id || crypto.randomUUID(),
    role,
    content: textContent,
    parts,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Extracts resolved tool invocations from a UIMessage array.
 *
 * Supports both the classic format (`type === 'tool-invocation'`,
 * `state === 'result'`) and the v6 per-tool format
 * (`type === 'tool-<toolName>'`, `state === 'output-available'`).
 *
 * Used to detect interrupt resume submissions from the frontend.
 */
export function extractResolvedToolResults(
  messages: UIMessage[]
): Array<{ toolCallId: string; toolName: string; result: unknown }> {
  // Walk backwards to find the last assistant message.
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== 'assistant') continue;

    const results: Array<{
      toolCallId: string;
      toolName: string;
      result: unknown;
    }> = [];

    for (const part of msg.parts || []) {
      // Classic format: { type: 'tool-invocation', toolInvocation: { state: 'result', ... } }
      if (
        part.type === 'tool-invocation' &&
        part.toolInvocation?.state === 'result' &&
        part.toolInvocation.result !== undefined
      ) {
        results.push({
          toolCallId: part.toolInvocation.toolCallId,
          toolName: part.toolInvocation.toolName,
          result: part.toolInvocation.result,
        });
        continue;
      }

      // v6 per-tool format: { type: 'tool-<name>', toolCallId,
      //   state: 'output-available', output: ... }
      // In SDK v6, ToolUIPart encodes the tool name in the `type` field
      // (e.g. `type: 'tool-userApproval'`) and does NOT have a separate
      // `toolName` property.  We derive the name from `type` when
      // `toolName` is not explicitly present.
      const p = part as unknown as Record<string, unknown>;
      if (
        part.type.startsWith('tool-') &&
        part.type !== 'tool-invocation' &&
        typeof p.toolCallId === 'string' &&
        p.toolCallId &&
        p.state === 'output-available' &&
        p.output !== undefined
      ) {
        const toolName =
          typeof p.toolName === 'string' && p.toolName
            ? p.toolName
            : part.type.slice('tool-'.length);
        results.push({
          toolCallId: p.toolCallId as string,
          toolName,
          result: p.output,
        });
      }
    }

    if (results.length > 0) return results;
  }

  return [];
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
