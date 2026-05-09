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

import type { Server } from '@modelcontextprotocol/sdk/server/index.js' with { 'resolution-mode': 'import' };
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js' with { 'resolution-mode': 'import' };
import type {
  CallToolRequest,
  CallToolResult,
  GetPromptRequest,
  GetPromptResult,
  ListPromptsRequest,
  ListPromptsResult,
  ListResourceTemplatesResult,
  ListToolsRequest,
  ListToolsResult,
  Prompt,
  PromptMessage,
  Tool,
} from '@modelcontextprotocol/sdk/types.js' with { 'resolution-mode': 'import' };
import {
  ListResourceTemplatesRequest,
  ListResourcesRequest,
  ListResourcesResult,
  ReadResourceRequest,
  ReadResourceResult,
  Resource,
  ResourceTemplate,
} from '@modelcontextprotocol/sdk/types.js';
import {
  GenkitError,
  Message,
  type Genkit,
  type MessageData,
  type Part,
  type PromptAction,
  type ResourceAction,
} from 'genkit';
import { logger } from 'genkit/logging';
import { toJsonSchema } from 'genkit/schema';
import { toToolDefinition, type ToolAction } from 'genkit/tool';
import type { McpServerOptions } from './index.js';

/**
 * Represents an MCP (Model Context Protocol) server that exposes Genkit tools
 * and prompts. This class wraps a Genkit instance and makes its registered
 * actions (tools, prompts) available to MCP clients. It handles the translation
 * between Genkit's action definitions and MCP's expected formats.
 */
export class GenkitMcpServer {
  ai: Genkit;
  options: McpServerOptions;
  server?: Server;
  actionsResolved = false;
  toolActions: ToolAction[] = [];
  promptActions: PromptAction[] = [];
  resourceActions: ResourceAction[] = [];

  /**
   * Creates an instance of GenkitMcpServer.
   * @param ai The Genkit instance whose actions will be exposed.
   * @param options Configuration options for the MCP server, like its name and version.
   */
  constructor(ai: Genkit, options: McpServerOptions) {
    this.ai = ai;
    this.options = options;
  }

  /**
   * Initializes the MCP server instance and registers request handlers. It
   * dynamically imports MCP SDK components and sets up handlers for listing
   * tools, calling tools, listing prompts, and getting prompts. It also
   * resolves and stores all tool and prompt actions from the Genkit instance.
   *
   * This method is called by the constructor and ensures the server is ready
   * before any requests are handled. It's idempotent.
   */
  async setup(): Promise<void> {
    if (this.actionsResolved) return;
    const { Server } = await import(
      '@modelcontextprotocol/sdk/server/index.js'
    );

    this.server = new Server(
      { name: this.options.name, version: this.options.version || '1.0.0' },
      {
        capabilities: {
          prompts: {},
          tools: {},
          resources: {},
        },
      }
    );

    const {
      CallToolRequestSchema,
      GetPromptRequestSchema,
      ListPromptsRequestSchema,
      ListToolsRequestSchema,
      ListResourcesRequestSchema,
      ListResourceTemplatesRequestSchema,
      ReadResourceRequestSchema,
    } = await import('@modelcontextprotocol/sdk/types.js');

    this.server.setRequestHandler(
      ListToolsRequestSchema,
      this.listTools.bind(this)
    );
    this.server.setRequestHandler(
      CallToolRequestSchema,
      this.callTool.bind(this)
    );
    this.server.setRequestHandler(
      ListPromptsRequestSchema,
      this.listPrompts.bind(this)
    );
    this.server.setRequestHandler(
      ListResourcesRequestSchema,
      this.listResources.bind(this)
    );
    this.server.setRequestHandler(
      ListResourceTemplatesRequestSchema,
      this.listResourceTemplates.bind(this)
    );
    this.server.setRequestHandler(
      ReadResourceRequestSchema,
      this.readResource.bind(this)
    );
    this.server.setRequestHandler(
      GetPromptRequestSchema,
      this.getPrompt.bind(this)
    );

    // TODO -- use listResolvableActions.
    const allActions = await this.ai.registry.listActions();
    const toolList: ToolAction[] = [];
    const promptList: PromptAction[] = [];
    const resourceList: ResourceAction[] = [];
    for (const k in allActions) {
      if (k.startsWith('/tool/')) {
        toolList.push(allActions[k] as ToolAction);
      } else if (k.startsWith('/prompt/')) {
        promptList.push(allActions[k] as PromptAction);
      } else if (k.startsWith('/resource/')) {
        resourceList.push(allActions[k] as ResourceAction);
      }
    }
    this.toolActions = toolList;
    this.promptActions = promptList;
    this.resourceActions = resourceList;
    this.actionsResolved = true;
  }

  /**
   * Handles MCP requests to list available tools.
   * It maps the resolved Genkit tool actions to the MCP Tool format.
   * @param req The MCP ListToolsRequest.
   * @returns A Promise resolving to an MCP ListToolsResult.
   */
  async listTools(req: ListToolsRequest): Promise<ListToolsResult> {
    await this.setup();
    return {
      tools: this.toolActions.map((t): Tool => {
        const def = toToolDefinition(t);
        return {
          name: def.name,
          inputSchema: (def.inputSchema as any) || { type: 'object' },
          description: def.description,
          _meta: t.__action.metadata?.mcp?._meta,
        };
      }),
    };
  }

  /**
   * Handles MCP requests to call a specific tool. It finds the corresponding
   * Genkit tool action and executes it with the provided arguments. The result
   * is then formatted as an MCP CallToolResult.
   * @param req The MCP CallToolRequest containing the tool name and arguments.
   * @returns A Promise resolving to an MCP CallToolResult.
   * @throws GenkitError if the requested tool is not found.
   */
  async callTool(req: CallToolRequest): Promise<CallToolResult> {
    await this.setup();
    const tool = this.toolActions.find(
      (t) => t.__action.name === req.params.name
    );
    if (!tool)
      throw new GenkitError({
        status: 'NOT_FOUND',
        message: `Tried to call tool '${req.params.name}' but it could not be found.`,
      });
    const result = await tool(req.params.arguments);
    return {
      content: [
        {
          type: 'text',
          text: typeof result === 'string' ? result : JSON.stringify(result),
        },
      ],
    };
  }

  /**
   * Handles MCP requests to list available prompts.
   * It maps the resolved Genkit prompt actions to the MCP Prompt format,
   * including converting input schemas to MCP argument definitions.
   * @param req The MCP ListPromptsRequest.
   * @returns A Promise resolving to an MCP ListPromptsResult.
   */
  async listPrompts(req: ListPromptsRequest): Promise<ListPromptsResult> {
    await this.setup();
    return {
      prompts: this.promptActions.map((p): Prompt => {
        return {
          name: p.__action.name,
          description: p.__action.description,
          arguments: toMcpPromptArguments(p),
          _meta: p.__action.metadata?.mcp?._meta,
        };
      }),
    };
  }

  /**
   * Handles MCP requests to list available resources.
   * It maps the resolved Genkit resource actions to the MCP Resource format.
   * @param req The MCP ListResourcesRequest.
   * @returns A Promise resolving to an MCP ListResourcesResult.
   */
  async listResources(req: ListResourcesRequest): Promise<ListResourcesResult> {
    await this.setup();
    return {
      resources: this.resourceActions
        .filter((r) => r.__action.metadata?.resource.uri)
        .map((r): Resource => {
          return {
            name: r.__action.name,
            description: r.__action.description,
            uri: r.__action.metadata?.resource.uri,
            _meta: r.__action.metadata?.mcp?._meta,
          };
        }),
    };
  }

  /**
   * Handles MCP requests to list available resources.
   * It maps the resolved Genkit resource actions to the MCP Resource format.
   * @param req The MCP ListResourcesRequest.
   * @returns A Promise resolving to an MCP ListResourcesResult.
   */
  async listResourceTemplates(
    req: ListResourceTemplatesRequest
  ): Promise<ListResourceTemplatesResult> {
    await this.setup();
    return {
      resourceTemplates: this.resourceActions
        .filter((r) => r.__action.metadata?.resource.template)
        .map((r): ResourceTemplate => {
          return {
            name: r.__action.name,
            description: r.__action.description,
            uriTemplate: r.__action.metadata?.resource.template,
            _meta: r.__action.metadata?.mcp?._meta,
          };
        }),
    };
  }

  /**
   * Handles MCP requests to list available resources.
   * It maps the resolved Genkit resource actions to the MCP Resource format.
   * @param req The MCP ListResourcesRequest.
   * @returns A Promise resolving to an MCP ListResourcesResult.
   */
  async readResource(req: ReadResourceRequest): Promise<ReadResourceResult> {
    await this.setup();
    const resource = this.resourceActions.find((r) =>
      r.matches({ uri: req.params.uri })
    );
    if (!resource) {
      throw new GenkitError({
        status: 'NOT_FOUND',
        message: `Tried to call resource '${req.params.uri}' but it could not be found.`,
      });
    }
    const result = await resource({ uri: req.params.uri });
    return {
      contents: toMcpResourceMessage(req.params.uri, result.content),
    };
  }

  /**
   * Handles MCP requests to get (render) a specific prompt. It finds the
   * corresponding Genkit prompt action, executes it with the provided
   * arguments, and then formats the resulting messages into the MCP
   * PromptMessage format.
   * @param req The MCP GetPromptRequest containing the prompt name and
   * arguments.
   * @returns A Promise resolving to an MCP GetPromptResult.
   * @throws GenkitError if the requested prompt is not found.
   */
  async getPrompt(req: GetPromptRequest): Promise<GetPromptResult> {
    await this.setup();
    const prompt = this.promptActions.find(
      (p) => p.__action.name === req.params.name
    );
    if (!prompt)
      throw new GenkitError({
        status: 'NOT_FOUND',
        message: `[MCP Server] Tried to call prompt '${req.params.name}' but it could not be found.`,
      });
    const result = await prompt(req.params.arguments);
    return {
      description: prompt.__action.description,
      messages: result.messages.map(toMcpPromptMessage),
    };
  }

  /**
   * Starts the MCP server with the specified transport or a default
   * StdioServerTransport. Ensures the server is set up before connecting the
   * transport.
   * @param transport Optional MCP transport instance. If not provided, a
   * StdioServerTransport will be created and used.
   */
  async start(transport?: Transport) {
    if (!transport) {
      const { StdioServerTransport } = await import(
        '@modelcontextprotocol/sdk/server/stdio.js'
      );
      transport = new StdioServerTransport();
    }
    await this.setup();
    await this.server!.connect(transport);
    logger.debug(
      `[MCP Server] MCP server '${this.options.name}' started successfully.`
    );
  }
}

/**
 * Converts a Genkit prompt's input schema to an array of MCP prompt arguments.
 * MCP prompts currently only support string arguments.
 * @param p The Genkit PromptAction.
 * @returns An array of MCP prompt arguments, or undefined if the schema is not defined.
 * @throws GenkitError if the input schema is not an object or if any property is not a string.
 */
function toMcpPromptArguments(
  p: PromptAction
): Prompt['arguments'] | undefined {
  const jsonSchema = toJsonSchema({
    schema: p.__action.inputSchema,
    jsonSchema: p.__action.inputJsonSchema,
  });

  if (!jsonSchema) return undefined;
  if (!jsonSchema.properties)
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: '[MCP Server] MCP prompts must take objects as input schema.',
    });

  const args: Prompt['arguments'] = [];
  for (const k in jsonSchema.properties) {
    const { type, description } = jsonSchema.properties[k];
    if (
      type !== 'string' &&
      (!Array.isArray(type) || !type.includes('string'))
    ) {
      throw new GenkitError({
        status: 'FAILED_PRECONDITION',
        message: `[MCP Server] MCP prompts may only take string arguments, but ${p.__action.name} has property '${k}' of type '${type}'.`,
      });
    }
    args.push({
      name: k,
      description,
      required: jsonSchema.required?.includes(k),
    });
  }
  return args;
}

const ROLE_MAP = { model: 'assistant', user: 'user' } as const;

/**
 * Converts a Genkit MessageData object to an MCP PromptMessage.
 * Handles mapping of roles and content types (text, image).
 * MCP only supports 'user' and 'assistant' (model) roles and base64 data images.
 * @param messageData The Genkit MessageData object.
 * @returns An MCP PromptMessage.
 * @throws GenkitError if the role is unsupported or if media is not a data URL.
 */
function toMcpPromptMessage(messageData: MessageData): PromptMessage {
  if (messageData.role !== 'model' && messageData.role !== 'user') {
    throw new GenkitError({
      status: 'UNIMPLEMENTED',
      message: `[MCP Server] MCP prompt messages do not support role '${messageData.role}'. Only 'user' and 'model' messages are supported.`,
    });
  }
  const message = new Message(messageData);
  const common = { role: ROLE_MAP[messageData.role] };
  if (message.media) {
    const { url, contentType } = message.media;
    if (!url.startsWith('data:'))
      throw new GenkitError({
        status: 'UNIMPLEMENTED',
        message: `[MCP Server] MCP prompt messages only support base64 data images.`,
      });
    const mimeType =
      contentType || url.substring(url.indexOf(':')! + 1, url.indexOf(';'));
    const data = url.substring(url.indexOf(',') + 1);
    return { ...common, content: { type: 'image', mimeType, data } };
  } else {
    return { ...common, content: { type: 'text', text: message.text } };
  }
}

/**
 * Converts a Genkit Parts to an MCP resource content.
 * Handles mapping of roles and content types (text, image).
 */
function toMcpResourceMessage(
  uri: string,
  content: Part[]
): ReadResourceResult['contents'] {
  return content.map((p) => {
    if (p.media) {
      const { url, contentType } = p.media;
      if (!url.startsWith('data:'))
        throw new GenkitError({
          status: 'UNIMPLEMENTED',
          message: `[MCP Server] MCP resource messages only support base64 data images.`,
        });
      const mimeType =
        contentType || url.substring(url.indexOf(':')! + 1, url.indexOf(';'));
      const data = url.substring(url.indexOf(',') + 1);
      return { uri, mimeType, blob: data };
    } else if (p.text) {
      return { uri, text: p.text };
    } else {
      throw new GenkitError({
        status: 'UNIMPLEMENTED',
        message: `[MCP Server] MCP resource messages only support media and text parts.`,
      });
    }
  });
}
