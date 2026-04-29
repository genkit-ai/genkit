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
} from '@modelcontextprotocol/sdk/types.js' with { 'resolution-mode': 'import' };
import {
  JSONSchema7,
  tool as genkitTool,
  z,
  type Genkit,
  type MultipartToolAction,
  type Part,
  type ToolAction,
} from 'genkit';
import { logger } from 'genkit/logging';

const toText = (c: CallToolResult['content']) =>
  c.map((p) => (p.type === 'text' ? p.text : '')).join('');

function processResult(result: CallToolResult) {
  if (result.isError) return { error: toText(result.content) };
  if (result.content.every((c) => c.type === 'text' && !!c.text)) {
    const text = toText(result.content);
    if (text.trim().startsWith('{') || text.trim().startsWith('[')) {
      try {
        return JSON.parse(text);
      } catch (e) {
        return text;
      }
    }
    return text;
  }
  if (result.content.length === 1) return result.content[0];
  return result;
}

function processMultipartResult(result: CallToolResult) {
  if (result.isError) {
    return {
      output: { error: toText(result.content) },
      metadata: result._meta,
    };
  }

  const content: Part[] = [];
  for (const c of result.content) {
    if (c.type === 'text') {
      if (c.text) {
        content.push({ text: c.text });
      }
    } else if (c.type === 'image') {
      if (c.data) {
        content.push({
          media: {
            url: `data:${c.mimeType};base64,${c.data}`,
            contentType: c.mimeType,
          },
        });
      }
    } else if (c.type === 'resource') {
      if (c.resource) {
        if ('text' in c.resource && c.resource.text) {
          content.push({
            text: `Resource (${c.resource.uri}):\n${c.resource.text}`,
          });
        } else if ('blob' in c.resource && c.resource.blob) {
          content.push({
            media: {
              url: `data:${c.resource.mimeType};base64,${c.resource.blob}`,
              contentType: c.resource.mimeType,
            },
          });
        }
      }
    }
  }

  return {
    content,
    metadata: result._meta,
  };
}

/**
 * Registers a single MCP tool as a Genkit tool.
 * It defines a new Genkit tool action that, when called, will
 * interact with the MCP client to execute the corresponding MCP tool.
 *
 * @param ai The Genkit instance to define the tool on.
 * @param client The MCP client instance used to interact with the MCP server.
 * @param tool The MCP Tool object to register.
 * @param params Contains the Genkit client name, MCP server name for namespacing,
 *               and a flag for raw tool responses.
 */
function registerTool(
  ai: Genkit,
  client: Client,
  tool: Tool,
  params: {
    serverName: string;
    name: string;
    rawToolResponses?: boolean;
    multipart?: boolean;
  }
) {
  logger.debug(
    `[MCP] Registering tool '${params.name}/${tool.name}'' from server '${params.serverName}'`
  );
  if (params.multipart && params.rawToolResponses) {
    logger.warn(
      `[MCP] Tool '${params.serverName}/${tool.name}' is configured with both multipart and rawToolResponses. Genkit will return the raw MCP CallToolResult in the output field, and media parts will not be natively parsed.`
    );
  }
  if (params.multipart) {
    ai.defineTool(
      {
        name: `${params.serverName}/${tool.name}`,
        description: tool.description || '',
        inputJsonSchema: tool.inputSchema as JSONSchema7,
        outputSchema: z.any(),
        metadata: { mcp: { _meta: tool._meta || {} } },
        multipart: true as const,
      },
      async (args) => {
        logger.debug(
          `[MCP] Calling MCP tool '${params.serverName}/${tool.name}' with arguments`,
          JSON.stringify(args)
        );
        const result = await client.callTool({
          name: tool.name,
          arguments: args,
        });
        if (params.rawToolResponses) return { output: result };
        return processMultipartResult(result as CallToolResult);
      }
    );
  } else {
    ai.defineTool(
      {
        name: `${params.serverName}/${tool.name}`,
        description: tool.description || '',
        inputJsonSchema: tool.inputSchema as JSONSchema7,
        outputSchema: z.any(),
        metadata: { mcp: { _meta: tool._meta || {} } },
      },
      async (args) => {
        logger.debug(
          `[MCP] Calling MCP tool '${params.serverName}/${tool.name}' with arguments`,
          JSON.stringify(args)
        );
        const result = await client.callTool({
          name: tool.name,
          arguments: args,
        });
        if (params.rawToolResponses) return result;
        return processResult(result as CallToolResult);
      }
    );
  }
}

/**
 * Creates a Genkit dynamic tool action for a given MCP tool.
 * This is similar to `registerTool` but returns the `ToolAction` directly
 * instead of defining it on the Genkit instance.
 *
 * @param ai The Genkit instance, used for creating the dynamic tool.
 * @param client The MCP client instance.
 * @param tool The MCP Tool object.
 * @param params Configuration parameters including namespacing and raw response flag.
 * @returns A Genkit `ToolAction` representing the MCP tool.
 */
function createDynamicTool<Multipart extends boolean = false>(
  ai: Genkit,
  client: Client,
  tool: Tool,
  params: {
    serverName: string;
    name: string;
    rawToolResponses?: boolean;
    multipart?: Multipart;
  }
): Multipart extends true ? MultipartToolAction : ToolAction {
  if (params.multipart && params.rawToolResponses) {
    logger.warn(
      `[MCP] Tool '${params.serverName}/${tool.name}' is configured with both multipart and rawToolResponses. Genkit will return the raw MCP CallToolResult in the output field, and media parts will not be natively parsed.`
    );
  }
  if (params.multipart) {
    return genkitTool(
      {
        name: `${params.serverName}/${tool.name}`,
        description: tool.description || '',
        inputJsonSchema: tool.inputSchema as JSONSchema7,
        outputSchema: z.any(),
        metadata: { mcp: { _meta: tool._meta || {} } },
        multipart: true as const,
      },
      async (args, { context }) => {
        logger.debug(
          `[MCP] calling tool '${params.serverName}/${tool.name}' in host '${params.name}'`
        );
        const result = await client.callTool({
          name: tool.name,
          arguments: args,
          _meta: context?.mcp?._meta,
        });
        if (params.rawToolResponses) return { output: result };
        return processMultipartResult(result as CallToolResult);
      }
    ) as unknown as Multipart extends true ? MultipartToolAction : ToolAction;
  }

  return genkitTool(
    {
      name: `${params.serverName}/${tool.name}`,
      description: tool.description || '',
      inputJsonSchema: tool.inputSchema as JSONSchema7,
      outputSchema: z.any(),
      metadata: { mcp: { _meta: tool._meta || {} } },
    },
    async (args, { context }) => {
      logger.debug(
        `[MCP] calling tool '${params.serverName}/${tool.name}' in host '${params.name}'`
      );
      const result = await client.callTool({
        name: tool.name,
        arguments: args,
        _meta: context?.mcp?._meta,
      });
      if (params.rawToolResponses) return result as CallToolResult;
      return processResult(result as CallToolResult);
    }
  ) as unknown as Multipart extends true ? MultipartToolAction : ToolAction;
}

/**
 * Lookup all tools available in the server and register each as a Genkit tool.
 */
export async function registerAllTools(
  ai: Genkit,
  client: Client,
  params: {
    name: string;
    serverName: string;
    rawToolResponses?: boolean;
    multipart?: boolean;
  }
): Promise<void> {
  let cursor: string | undefined;
  while (true) {
    const { nextCursor, tools } = await client.listTools({ cursor });
    tools.forEach((t) => registerTool(ai, client, t, params));
    cursor = nextCursor;
    if (!cursor) break;
  }
}

/**
 * Lookup all tools available in the server and fetches as a Genkit dynamic tool.
 */
export async function fetchDynamicTools<Multipart extends boolean = false>(
  ai: Genkit,
  client: Client,
  params: {
    name: string;
    serverName: string;
    rawToolResponses?: boolean;
    multipart?: Multipart;
  }
): Promise<(Multipart extends true ? MultipartToolAction : ToolAction)[]> {
  let cursor: string | undefined;
  let allTools: (Multipart extends true ? MultipartToolAction : ToolAction)[] =
    [];
  while (true) {
    const { nextCursor, tools } = await client.listTools({ cursor });
    allTools.push(
      ...tools.map((t) => createDynamicTool(ai, client, t, params))
    );
    cursor = nextCursor;
    if (!cursor) break;
  }
  return allTools;
}
