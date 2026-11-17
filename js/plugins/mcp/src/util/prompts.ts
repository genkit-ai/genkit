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
import type { Prompt } from '@modelcontextprotocol/sdk/types.js';
import {
  GenkitError,
  type ExecutablePrompt,
  type GenerateOptions,
  type GenerateResponse,
  type GenerateStreamResponse,
  type Genkit,
  type GenkitSchema,
  type JSONSchema,
  type PromptGenerateOptions,
  type ToolAction,
} from 'genkit';
import { logger } from 'genkit/logging';
import type { InferOutput } from 'genkit/schema';
import { fromMcpPromptMessage } from './message.js';

function toSchema(args: Prompt['arguments']) {
  if (!args) return {};
  const schema: JSONSchema = { type: 'object', properties: {}, required: [] };
  for (const arg of args) {
    schema.properties[arg.name] = {
      type: 'string',
      description: arg.description,
    };
    if (arg.required) schema.required.push(arg.name);
  }
  return schema;
}

/**
 * Registers a single MCP prompt as a Genkit prompt.
 * It defines a new Genkit prompt action that, when called, will
 * interact with the MCP client to fetch and render the corresponding MCP prompt.
 *
 * @param ai The Genkit instance to define the prompt on.
 * @param client The MCP client instance used to interact with the MCP server.
 * @param prompt The MCP Prompt object to register.
 * @param params Contains the Genkit client name and the MCP server name for namespacing and logging.
 */
function registerPrompt(
  ai: Genkit,
  client: Client,
  prompt: Prompt,
  params: { name: string; serverName: string }
) {
  logger.debug(`[MCP] Registering MCP prompt ${params.name}/${prompt.name}`);
  ai.definePrompt({
    name: prompt.name,
    description: prompt.description || '',
    input: { jsonSchema: toSchema(prompt.arguments) },
    output: { format: 'text' },
    metadata: { mcp: { _meta: prompt._meta || {} } },
    messages: async (args, { context }) => {
      logger.debug(
        `[MCP] Calling MCP prompt ${params.name}/${prompt.name} with arguments`,
        JSON.stringify(args)
      );
      const result = await client.getPrompt({
        name: prompt.name,
        arguments: args,
        _meta: context?.mcp?._meta,
      });
      return result.messages.map(fromMcpPromptMessage);
    },
  });
}

function createExecutablePrompt<
  I extends GenkitSchema = GenkitSchema,
  O extends GenkitSchema = GenkitSchema,
  CustomOptions extends GenkitSchema = GenkitSchema,
>(
  client: Client,
  prompt: Prompt,
  params: {
    ai: Genkit;
    name: string;
    serverName: string;
    promptName: string;
    options?: PromptGenerateOptions<any, any>;
  }
): ExecutablePrompt<InferOutput<I>, O, CustomOptions> {
  const callPrompt = (async (
    input?: InferOutput<I>,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): Promise<GenerateResponse<InferOutput<O>>> => {
    logger.debug(`[MCP] Calling MCP prompt ${params.name}/${prompt.name}`);
    return params.ai.generate(callPrompt.render(input, opts));
  }) as ExecutablePrompt<InferOutput<I>, O, CustomOptions>;

  callPrompt.ref = {
    name: prompt.name,
    metadata: {
      description: prompt.description,
      arguments: prompt.arguments,
      mcp: { _meta: prompt._meta || {} },
    },
  };

  callPrompt.stream = (
    input?: InferOutput<I>,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): GenerateStreamResponse<InferOutput<O>> => {
    logger.debug(`[MCP] Streaming MCP prompt ${params.name}/${prompt.name}`);
    return params.ai.generateStream(callPrompt.render(input, opts));
  };

  callPrompt.render = async (
    input?: I,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): Promise<GenerateOptions<O, CustomOptions>> => {
    logger.debug(`[MCP] Rendering MCP prompt ${params.name}/${prompt.name}`);
    const result = await client.getPrompt({
      name: prompt.name,
      arguments: input as any,
      _meta: opts?.context?.mcp?._meta,
    });
    const messages = result.messages.map(fromMcpPromptMessage);
    return {
      ...params.options,
      ...opts,
      messages,
    };
  };

  callPrompt.asTool = async (): Promise<ToolAction> => {
    throw new GenkitError({
      status: 'UNIMPLEMENTED',
      message: `[MCP] prompt.asTool not supported with MCP`,
    });
  };

  return callPrompt;
}

/**
 * Lookup all prompts available in the server and register each as a Genkit
 * prompt.
 */
export async function registerAllPrompts(
  ai: Genkit,
  client: Client,
  params: { name: string; serverName: string }
): Promise<void> {
  let cursor: string | undefined;
  while (true) {
    const { nextCursor, prompts } = await client.listPrompts({ cursor });
    prompts.forEach((p) => registerPrompt(ai, client, p, params));
    cursor = nextCursor;
    if (!cursor) break;
  }
}

/**
 * Lookup a specified prompt from the server and return as an ExecutablePrompt.
 */
export async function getExecutablePrompt(
  client: Client,
  params: {
    name: string;
    serverName: string;
    promptName: string;
    ai: Genkit;
    options?: PromptGenerateOptions;
  }
): Promise<ExecutablePrompt | undefined> {
  let cursor: string | undefined;

  while (true) {
    const { nextCursor, prompts } = await client.listPrompts({ cursor });
    const maybePrompt = prompts.find(
      (p: Prompt) => p.name === params.promptName
    );
    if (maybePrompt) {
      return createExecutablePrompt(client, maybePrompt, params);
    }
    cursor = nextCursor;
    if (!cursor) break;
  }
  return undefined;
}

export async function fetchAllPrompts(
  client: Client,
  params: {
    name: string;
    serverName: string;
    ai: Genkit;
    options?: PromptGenerateOptions;
  }
): Promise<ExecutablePrompt[]> {
  let cursor: string | undefined;
  const out: ExecutablePrompt[] = [];

  while (true) {
    const { nextCursor, prompts } = await client.listPrompts({ cursor });
    for (const p of prompts) {
      out.push(
        createExecutablePrompt(client, p, { ...params, promptName: p.name })
      );
    }
    cursor = nextCursor;
    if (!cursor) break;
  }
  return out;
}
