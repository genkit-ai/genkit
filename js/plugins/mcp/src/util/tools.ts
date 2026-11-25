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

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import type {
  CallToolResult,
  Tool,
} from '@modelcontextprotocol/sdk/types.js' with {
  'resolution-mode': 'import',
};
import {
  JSONSchema7,
  tool as genkitTool,
  z,
  type Genkit,
  type MultipartToolResponse,
  type Part,
  type ToolAction,
} from 'genkit';
import { logger } from 'genkit/logging';

/**
 * How an MCP tool result (`CallToolResult`) becomes a Genkit tool response.
 *
 * - `'content'` (default): only `content` reaches the model, mapped 1:1 to
 *   tool response content parts. `structuredContent`, `isError` and `_meta` are
 *   kept for app code in the tool response `metadata.mcp`. This follows the MCP
 *   spec, where `content` stands on its own (servers should also serialize
 *   `structuredContent` into a text block) and matches most MCP clients.
 * - `'structured'`: `structuredContent` becomes the tool response `output`
 *   (the model sees it); `content` is mapped as above, minus text blocks that
 *   just repeat `structuredContent` as JSON.
 * - `'raw'`: the whole `CallToolResult` becomes the `output`.
 * - a function: custom mapping.
 */
export type McpToolResponseMode =
  | 'content'
  | 'structured'
  | 'raw'
  | ((
      result: CallToolResult,
      tool: { server: string; name: string }
    ) => MultipartToolResponse);

/** Converts an MCP tool result to a Genkit tool response. */
export function toToolResponse(
  result: CallToolResult,
  mode: McpToolResponseMode = 'content',
  tool: { server: string; name: string }
): MultipartToolResponse {
  if (typeof mode === 'function') return mode(result, tool);
  if (mode === 'raw') return { output: result };

  const mcp: Record<string, unknown> = {};
  if (result.structuredContent !== undefined) {
    mcp.structuredContent = result.structuredContent;
  }
  if (result.isError) mcp.isError = true;
  if (result._meta) mcp._meta = result._meta;
  const metadata = Object.keys(mcp).length ? { mcp } : undefined;

  let blocks = result.content ?? [];
  if (mode === 'structured' && result.structuredContent !== undefined) {
    blocks = blocks.filter(
      (b) => !(b.type === 'text' && isJsonOf(b.text, result.structuredContent))
    );
    const content = blocks.map(toPart);
    return {
      output: result.structuredContent,
      ...(content.length ? { content } : {}),
      ...(metadata ? { metadata } : {}),
    };
  }

  const content = blocks.map(toPart);
  // A server that returns only structuredContent skips the spec's "also
  // serialize it into a text block"; do it here so the model gets the result.
  if (!content.length && result.structuredContent !== undefined) {
    content.push({ text: JSON.stringify(result.structuredContent) });
  }
  return {
    ...(content.length ? { content } : {}),
    ...(metadata ? { metadata } : {}),
  };
}

/**
 * Whether `text` is the JSON serialization of `value`. Compares serialized
 * forms, so it's key-order sensitive; servers normally serialize the same
 * object they return.
 */
function isJsonOf(text: string, value: unknown): boolean {
  try {
    return JSON.stringify(JSON.parse(text)) === JSON.stringify(value);
  } catch {
    return false;
  }
}

type ContentBlock = CallToolResult['content'][number];

/** Converts an MCP content block to a Genkit part. */
function toPart(block: ContentBlock): Part {
  const annotations =
    'annotations' in block && block.annotations
      ? { annotations: block.annotations }
      : {};
  const withMcp = (part: Part, mcp: Record<string, unknown> = {}): Part => {
    const meta = { ...annotations, ...mcp };
    return Object.keys(meta).length
      ? { ...part, metadata: { mcp: meta } }
      : part;
  };
  const dataUrl = (mimeType: string | undefined, data: string) => {
    // mimeType is optional on embedded resources; providers need one.
    const type = mimeType || 'application/octet-stream';
    return { url: `data:${type};base64,${data}`, contentType: type };
  };

  switch (block.type) {
    case 'text':
      return withMcp({ text: block.text });
    case 'image':
    case 'audio':
      return withMcp({ media: dataUrl(block.mimeType, block.data) });
    case 'resource': {
      const { uri, mimeType } = block.resource;
      if ('text' in block.resource && typeof block.resource.text === 'string') {
        return withMcp(
          { text: block.resource.text },
          { uri, ...(mimeType ? { mimeType } : {}) }
        );
      }
      if ('blob' in block.resource && typeof block.resource.blob === 'string') {
        return withMcp(
          { media: dataUrl(mimeType, block.resource.blob) },
          { uri }
        );
      }
      return withMcp({ resource: { uri } });
    }
    case 'resource_link': {
      const { uri, name, title, mimeType, description, size } = block;
      return withMcp(
        { resource: { uri } },
        Object.fromEntries(
          Object.entries({ name, title, mimeType, description, size }).filter(
            ([, v]) => v !== undefined
          )
        )
      );
    }
    default:
      return { text: JSON.stringify(block) };
  }
}

/** Options for converting MCP tools to Genkit tools. */
interface McpToolParams {
  /** The Genkit MCP client/host name. */
  name: string;
  /** The MCP server name, used to namespace tool names. */
  serverName: string;
  /** How tool results are mapped, see {@link McpToolResponseMode}. */
  toolResponseMode?: McpToolResponseMode;
}

/** The Genkit tool config and implementation for an MCP tool. */
function toolDefinition(client: Client, tool: Tool, params: McpToolParams) {
  const name = `${params.serverName}/${tool.name}`;
  return {
    config: {
      name,
      description: tool.description || '',
      inputJsonSchema: tool.inputSchema as JSONSchema7,
      outputSchema: z.any(),
      metadata: { mcp: { _meta: tool._meta || {} } },
      multipart: true as const,
    },
    fn: async (
      args: unknown,
      { context }: { context?: Record<string, any> }
    ): Promise<MultipartToolResponse> => {
      logger.debug(
        `[MCP] calling tool '${name}' in host '${params.name}' with arguments`,
        JSON.stringify(args)
      );
      const result = await client.callTool({
        name: tool.name,
        arguments: args as Record<string, unknown>,
        _meta: context?.mcp?._meta,
      });
      return toToolResponse(result as CallToolResult, params.toolResponseMode, {
        server: params.serverName,
        name: tool.name,
      });
    },
  };
}

/**
 * Lookup all tools available in the server and register each as a Genkit tool.
 */
export async function registerAllTools(
  ai: Genkit,
  client: Client,
  params: McpToolParams
): Promise<void> {
  let cursor: string | undefined;
  while (true) {
    const { nextCursor, tools } = await client.listTools({ cursor });
    tools.forEach((t) => {
      logger.debug(
        `[MCP] Registering tool '${params.name}/${t.name}' from server '${params.serverName}'`
      );
      const { config, fn } = toolDefinition(client, t, params);
      ai.defineTool(config, fn);
    });
    cursor = nextCursor;
    if (!cursor) break;
  }
}

/**
 * Lookup all tools available in the server and fetches as a Genkit dynamic tool.
 */
export async function fetchDynamicTools(
  ai: Genkit,
  client: Client,
  params: McpToolParams
): Promise<ToolAction[]> {
  let cursor: string | undefined;
  const allTools: ToolAction[] = [];
  while (true) {
    const { nextCursor, tools } = await client.listTools({ cursor });
    allTools.push(
      ...tools.map((t) => {
        const { config, fn } = toolDefinition(client, t, params);
        return genkitTool(config, fn);
      })
    );
    cursor = nextCursor;
    if (!cursor) break;
  }
  return allTools;
}
