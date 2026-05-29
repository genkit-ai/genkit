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
      typeof p.toolCallId === 'string'
        ? p.toolCallId
        : crypto.randomUUID();
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

  // Unsupported part types (reasoning, source-*, step-start, data-*) are
  // silently skipped — they don't have a direct Genkit equivalent.
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
// Helpers
// ---------------------------------------------------------------------------

/**
 * Extracts resolved tool invocations from a UIMessage array.
 *
 * Supports the v6 per-tool format (`type === 'tool-<toolName>'`,
 * `state === 'output-available'`).
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
