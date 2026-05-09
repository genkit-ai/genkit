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
 * Shared utilities for converting Anthropic content blocks to Genkit Parts.
 * Uses structural typing so both stable and beta API types work with these functions.
 */

import type { Part } from 'genkit';
import type {
  AnthropicCitation,
  AnthropicDocumentOptions,
} from '../../types.js';
import { MEDIA_TYPES, MediaTypeSchema } from '../../types.js';
import {
  fromAnthropicCitation,
  type AnthropicCitationInput,
} from './citations.js';

// Re-export citation utilities for backward compatibility
export {
  fromAnthropicCitation,
  type AnthropicCitationInput,
} from './citations.js';

/**
 * Converts a text block to a Genkit Part, including citations if present.
 * Uses structural typing for compatibility with both stable and beta APIs.
 */
export function textBlockToPart(block: {
  text: string;
  citations?: AnthropicCitationInput[] | null;
}): Part {
  if (block.citations && block.citations.length > 0) {
    const citations = block.citations
      .map((c) => fromAnthropicCitation(c))
      .filter((c): c is AnthropicCitation => c !== undefined);
    if (citations.length > 0) {
      return {
        text: block.text,
        metadata: { citations },
      };
    }
  }
  return { text: block.text };
}

/**
 * Converts a tool_use block to a Genkit Part.
 */
export function toolUseBlockToPart(block: {
  id: string;
  name: string;
  input: unknown;
}): Part {
  return {
    toolRequest: {
      ref: block.id,
      name: block.name,
      input: block.input,
    },
  };
}

/**
 * Converts a thinking block to a Genkit Part, including signature metadata if present.
 */
export function thinkingBlockToPart(block: {
  thinking: string;
  signature?: string;
}): Part {
  if (block.signature !== undefined) {
    return {
      reasoning: block.thinking,
      metadata: { thoughtSignature: block.signature },
    };
  }
  return { reasoning: block.thinking };
}

/**
 * Converts a redacted thinking block to a Genkit Part.
 */
export function redactedThinkingBlockToPart(block: { data: string }): Part {
  return { custom: { redactedThinking: block.data } };
}

/**
 * Converts a web_search_tool_result block to a Genkit Part.
 */
export function webSearchToolResultBlockToPart(block: {
  tool_use_id: string;
  content: unknown;
}): Part {
  return {
    text: `[Anthropic server tool result ${block.tool_use_id}] ${JSON.stringify(block.content)}`,
    metadata: {
      anthropicServerToolResult: {
        type: 'web_search_tool_result',
        toolUseId: block.tool_use_id,
        content: block.content,
      },
    },
  };
}

// --- Delta converters for streaming ---

/**
 * Converts a text_delta to a Genkit Part.
 */
export function textDeltaToPart(delta: { text: string }): Part {
  return { text: delta.text };
}

/**
 * Converts a thinking_delta to a Genkit Part.
 */
export function thinkingDeltaToPart(delta: { thinking: string }): Part {
  return { reasoning: delta.thinking };
}

/**
 * Converts a citations_delta to a Genkit Part for streaming.
 * Returns a text part with empty text and citation data in metadata.
 * Empty text is intentional: genkit's `.text` getter concatenates all text parts,
 * so empty strings contribute nothing to the final text while preserving the citation
 * in the parts array for consumers who need to access citation metadata.
 */
export function citationsDeltaToPart(delta: {
  type: 'citations_delta';
  citation: AnthropicCitationInput;
}): Part | undefined {
  const citation = fromAnthropicCitation(delta.citation);
  if (citation) {
    return {
      text: '',
      metadata: { citations: [citation] },
    };
  }
  return undefined;
}

/**
 * Error for unsupported input_json_delta in streaming.
 */
export function inputJsonDeltaError(): Error {
  return new Error(
    'Anthropic streaming tool input (input_json_delta) is not yet supported. Please disable streaming or upgrade this plugin.'
  );
}

// --- Document block converters (shared between stable and beta APIs) ---

/**
 * Document block type constraint for generics.
 */
type DocumentBlockBase = {
  type: 'document';
  source: unknown;
  title?: string | null;
  context?: string | null;
  citations?: { enabled?: boolean } | null;
};

/**
 * Converts AnthropicDocumentOptions to Anthropic's document block format.
 * Works for both stable and beta APIs via generics.
 */
export function createDocumentBlock<T extends DocumentBlockBase>(
  options: AnthropicDocumentOptions,
  sourceConverter: (source: AnthropicDocumentOptions['source']) => T['source']
): T {
  return {
    type: 'document' as const,
    source: sourceConverter(options.source),
    ...(options.title && { title: options.title }),
    ...(options.context && { context: options.context }),
    ...(options.citations && { citations: options.citations }),
  } as T;
}

/**
 * Converts document source options to Anthropic's source format.
 * Works for both stable and beta APIs via a file handler callback.
 * The file handler is called for 'file' type sources, allowing different
 * behavior (error for stable, conversion for beta).
 */
export function convertDocumentSource<T>(
  source: AnthropicDocumentOptions['source'],
  fileHandler: (fileId: string) => T
): T {
  switch (source.type) {
    case 'text':
      return {
        type: 'text',
        media_type: (source.mediaType ?? 'text/plain') as 'text/plain',
        data: source.data,
      } as T;
    case 'base64':
      return {
        type: 'base64',
        media_type: source.mediaType as 'application/pdf',
        data: source.data,
      } as T;
    case 'file':
      return fileHandler(source.fileId);
    case 'content':
      return {
        type: 'content',
        content: source.content.map((item) => {
          if (item.type === 'text') {
            return item;
          }
          // Validate media type with Zod
          const mediaTypeResult = MediaTypeSchema.safeParse(
            item.source.mediaType
          );
          if (!mediaTypeResult.success) {
            throw new Error(
              `Unsupported image media type for Anthropic document content: ${item.source.mediaType}. Supported types: ${Object.values(MEDIA_TYPES).join(', ')}`
            );
          }
          return {
            type: 'image' as const,
            source: {
              type: 'base64' as const,
              media_type: mediaTypeResult.data,
              data: item.source.data,
            },
          };
        }),
      } as T;
    case 'url':
      return {
        type: 'url',
        url: source.url,
      } as T;
    default:
      throw new Error(
        `Unsupported document source type: ${(source as { type: string }).type}`
      );
  }
}
