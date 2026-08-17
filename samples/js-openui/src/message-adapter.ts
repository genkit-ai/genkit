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

export interface WireMessage {
  role: 'user' | 'assistant';
  content: string;
}

const contentMarker = ']]>openui:content';
const contextMarker = ']]>openui:context';

interface InlineMessage {
  content: string;
  context: unknown | null;
}

function markerPayloadStart(
  raw: string,
  marker: string,
  markerIndex: number
): number {
  const markerEnd = markerIndex + marker.length;
  if (raw.startsWith('\r\n', markerEnd)) {
    return markerEnd + 2;
  }
  return raw[markerEnd] === '\n' ? markerEnd + 1 : markerEnd;
}

/** Separates AgentInterface's display content from its action/form context. */
function parseInlineMessage(raw: string): InlineMessage {
  const contentIndex = raw.lastIndexOf(contentMarker);
  const contextIndex = raw.lastIndexOf(contextMarker);

  if (contentIndex === -1 && contextIndex === -1) {
    return { content: raw, context: null };
  }

  const contentStart =
    contentIndex === -1
      ? 0
      : markerPayloadStart(raw, contentMarker, contentIndex);
  const contentEnd = contextIndex === -1 ? raw.length : contextIndex;
  const content = raw.slice(contentStart, contentEnd).trimEnd();

  if (contextIndex === -1) {
    return { content, context: null };
  }

  const contextStart = markerPayloadStart(raw, contextMarker, contextIndex);
  try {
    return { content, context: JSON.parse(raw.slice(contextStart)) };
  } catch {
    return { content, context: raw.slice(contextStart) };
  }
}

/** Converts one AgentInterface wire message into model-friendly Genkit text. */
export function toGenkitContent(message: WireMessage): string {
  const parsed = parseInlineMessage(message.content);
  if (message.role === 'assistant' || parsed.context === null) {
    return parsed.content;
  }

  return `${parsed.content}\n\nOpenUI action context:\n${JSON.stringify(parsed.context)}`;
}
