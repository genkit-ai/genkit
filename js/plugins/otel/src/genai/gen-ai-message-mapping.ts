/**
 * Copyright 2025 Google LLC
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

/**
 * Normalizes Genkit messages to the OpenTelemetry GenAI content schema.
 *
 * Pure mapping logic with no OpenTelemetry dependency, so it can be unit
 * tested directly. Only used when content capture is enabled.
 */

import type { MessageData, Part, Role } from 'genkit/model';

/**
 * The result of splitting a message list into system instructions and the
 * remaining conversation messages, both in the GenAI content schema.
 */
export interface NormalizedMessages {
  messages: Record<string, unknown>[];
  systemInstructions: Record<string, unknown>[];
}

/**
 * Maps a Genkit role to the GenAI role name.
 *
 * Genkit `model` becomes `assistant`; other roles pass through.
 */
export function mapRole(role: Role): string {
  return role === 'model' ? 'assistant' : role;
}

/**
 * Converts a single Genkit part to a GenAI content part.
 *
 * Discriminates on which field is present rather than on a nominal type, since
 * a `Part` is a union of shapes. Unknown part kinds fall back to a generic
 * `text` part so nothing is silently dropped from captured content.
 */
export function mapPart(part: Part): Record<string, unknown> {
  if (part.text != null) {
    return { type: 'text', content: part.text };
  }
  if (part.reasoning != null) {
    return { type: 'reasoning', content: part.reasoning };
  }
  if (part.toolRequest) {
    const { ref, name, input } = part.toolRequest;
    return {
      type: 'tool_call',
      ...(ref != null ? { id: ref } : {}),
      name,
      arguments: input,
    };
  }
  if (part.toolResponse) {
    const { ref, output } = part.toolResponse;
    return {
      type: 'tool_call_response',
      ...(ref != null ? { id: ref } : {}),
      response: output,
    };
  }
  if (part.media) {
    const { url, contentType } = part.media;
    return {
      type: 'media',
      content: url,
      ...(contentType != null ? { content_type: contentType } : {}),
    };
  }
  // Unknown/opaque part: represent it structurally without losing the fact
  // that it existed.
  return { type: 'text', content: JSON.stringify(part) };
}

/** Whether `part` is a tool-request part. */
export function isToolRequestPart(part: Part): boolean {
  return !!part.toolRequest;
}

/** Converts a single Genkit message to a GenAI message. */
export function mapMessage(message: MessageData): Record<string, unknown> {
  return {
    role: mapRole(message.role),
    parts: message.content.map(mapPart),
  };
}

/**
 * Splits `messages` into `system_instructions` (messages with role `system`)
 * and the remaining conversation `messages`, each normalized to the GenAI
 * content schema.
 */
export function normalizeMessages(messages: MessageData[]): NormalizedMessages {
  const system: Record<string, unknown>[] = [];
  const rest: Record<string, unknown>[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      // System instructions are represented by their parts directly.
      system.push(...message.content.map(mapPart));
    } else {
      rest.push(mapMessage(message));
    }
  }
  return { messages: rest, systemInstructions: system };
}

/**
 * Maps a response message to a GenAI output message, attaching the mapped
 * `finishReason`.
 */
export function mapOutputMessage(
  message: MessageData,
  finishReason: string
): Record<string, unknown> {
  return { ...mapMessage(message), finish_reason: finishReason };
}
