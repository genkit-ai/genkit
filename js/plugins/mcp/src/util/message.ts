/**
 * Copyright 2024 Google LLC
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

import type { PromptMessage } from '@modelcontextprotocol/sdk/types.js' with { 'resolution-mode': 'import' };
import { type MessageData, type Part } from 'genkit';

const ROLE_MAP = {
  user: 'user',
  assistant: 'model',
} as const;

/**
 * Converts an MCP (Model Context Protocol) PromptMessage into Genkit's
 * MessageData format. This involves mapping MCP roles (user, assistant) to
 * Genkit roles (user, model) and transforming the MCP content part into a
 * Genkit Part.
 *
 * @param message The MCP PromptMessage to convert.
 * @returns The corresponding Genkit MessageData object.
 */
export function fromMcpPromptMessage(message: PromptMessage): MessageData {
  return {
    role: ROLE_MAP[message.role],
    content: [fromMcpPart(message.content)],
  };
}

/**
 * Converts an MCP message content part (text, image, or resource) into a Genkit
 * Part.
 * - Text parts are directly mapped.
 * - Image parts are converted to Genkit media parts with a data URL.
 * - Resource parts currently result in an empty Genkit Part (further
 *   implementation may be needed).
 *
 * @param part The MCP PromptMessage content part to convert.
 * @returns The corresponding Genkit Part.
 */
export function fromMcpPart(part: PromptMessage['content']): Part {
  switch (part.type) {
    case 'text':
      return { text: part.text };
    case 'image':
      return {
        media: {
          contentType: part.mimeType,
          url: `data:${part.mimeType};base64,${part.data}`,
        },
      };
    case 'resource':
      return {
        resource: {
          uri: part.resource.uri,
        },
      };
    default:
      return {};
  }
}
