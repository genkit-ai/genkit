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
 * Converters for beta API content blocks.
 */

import type { BetaRequestDocumentBlock } from '@anthropic-ai/sdk/resources/beta/messages';
import type { Part } from 'genkit';
import type { AnthropicDocumentOptions } from '../../types.js';
import { convertDocumentSource, createDocumentBlock } from './shared.js';

/**
 * Converts a server_tool_use block to a Genkit Part.
 * In the beta API, name may be undefined and server_name prefix is supported.
 */
export function betaServerToolUseBlockToPart(block: {
  id: string;
  name?: string;
  input: unknown;
  server_name?: string;
}): Part {
  const baseName = block.name ?? 'unknown_tool';
  const serverToolName = block.server_name
    ? `${block.server_name}/${baseName}`
    : baseName;
  return {
    text: `[Anthropic server tool ${serverToolName}] input: ${JSON.stringify(block.input)}`,
    metadata: {
      anthropicServerToolUse: {
        id: block.id,
        name: serverToolName,
        input: block.input,
      },
    },
  };
}

/**
 * Error message for unsupported server tool block types in the beta API.
 */
export function unsupportedServerToolError(blockType: string): string {
  return `Anthropic beta runner does not yet support server-managed tool block '${blockType}'. Please retry against the stable API or wait for dedicated support.`;
}

/**
 * Converts AnthropicDocumentOptions to Anthropic's beta API document block format.
 * The beta API supports file-based sources via the Files API.
 */
export function toBetaDocumentBlock(
  options: AnthropicDocumentOptions
): BetaRequestDocumentBlock {
  return createDocumentBlock<BetaRequestDocumentBlock>(options, (source) =>
    convertDocumentSource<BetaRequestDocumentBlock['source']>(
      source,
      (fileId) =>
        ({
          type: 'file',
          file_id: fileId,
        }) as BetaRequestDocumentBlock['source']
    )
  );
}
