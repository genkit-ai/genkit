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

import type { Genkit } from 'genkit';
import {
  GenkitMcpClient,
  McpClientOptions,
  McpClientOptionsWithCache,
  McpServerConfig,
  McpStdioServerConfig,
} from './client/client.js';
import {
  GenkitMcpHost,
  McpHostOptions,
  McpHostOptionsWithCache,
} from './client/index.js';
import { GenkitMcpServer } from './server.js';
export {
  GenkitMcpClient,
  GenkitMcpHost,
  type McpClientOptions,
  type McpClientOptionsWithCache,
  type McpHostOptions,
  type McpHostOptionsWithCache,
  type McpServerConfig,
  type McpStdioServerConfig,
};

export interface McpServerOptions {
  /** The name you want to give your server for MCP inspection. */
  name: string;
  /** The version you want the server to advertise to clients. Defaults to
   * 1.0.0. */
  version?: string;
}

/**
 * Defines an MCP Client Host that connects to one or more MCP servers.
 * Each server is defined in the `mcpServers` option, where the key is a
 * client-side name for the server and the value is the server's configuration.
 *
 * By default, all servers in the config will be attempted to connect unless
 * their configuration includes `{disabled: true}`.
 *
 * The host is registered with `ai` as a dynamic action provider named
 * `options.name`, so its tools and resources can be referenced by name (e.g.
 * `'my-mcp-host:tool/*'`) in addition to being passed in directly.
 *
 * ```ts
 * const mcpHost = defineMcpHost(ai, {
 *   name: 'my-mcp-host',
 *   mcpServers: {
 *     // Each key becomes the namespace of that server's tools.
 *     git: { command: 'uvx', args: ['mcp-server-git'] },
 *     customApi: { url: 'http://localhost:1234/mcp' },
 *   },
 * });
 *
 * await ai.generate({
 *   prompt: '<a prompt requiring tools>',
 *   tools: ['my-mcp-host:tool/*'], // or: await mcpHost.getActiveTools(ai)
 * });
 * ```
 *
 * @param ai The Genkit instance to register the host with.
 * @param options Configuration for the MCP Client Host, including the definitions of MCP servers to connect to.
 * @returns A new instance of GenkitMcpHost.
 */
export function defineMcpHost<M extends boolean = false>(
  ai: Genkit,
  options: McpHostOptionsWithCache<M>
) {
  const mcpHost = new GenkitMcpHost<M>(options);
  const dap = ai.defineDynamicActionProvider(
    {
      name: options.name,
      cacheConfig: {
        ttlMillis: options.cacheTtlMillis,
      },
    },
    async () => ({
      tool: await mcpHost.getActiveTools(ai),
      resource: await mcpHost.getActiveResources(ai),
    })
  );
  mcpHost.dynamicActionProvider = dap;
  return mcpHost;
}

/**
 * Defines an MCP Client that connects to a single MCP server.
 * This is useful when you only need to interact with one MCP server,
 * or if you want to manage client instances individually.
 *
 * The client is registered with `ai` as a dynamic action provider named
 * `options.name`, so its tools and resources can be referenced by name (e.g.
 * `'my-mcp-client:tool/*'`) in addition to being passed in directly.
 *
 * ```ts
 * const mcpClient = defineMcpClient(ai, {
 *   name: 'my-mcp-client',
 *   mcpServer: {
 *     command: 'npx',
 *     args: ['-y', '@modelcontextprotocol/server-everything'],
 *   },
 * });
 *
 * await ai.generate({
 *   prompt: '<a prompt requiring tools>',
 *   tools: ['my-mcp-client:tool/*'], // or: await mcpClient.getActiveTools(ai)
 * });
 * ```
 *
 * @param ai The Genkit instance to register the client with.
 * @param options Configuration for the MCP Client, defining how it connects
 *                to the MCP server and its behavior.
 * @returns A new instance of GenkitMcpClient.
 */
export function defineMcpClient<M extends boolean = false>(
  ai: Genkit,
  options: McpClientOptionsWithCache<M>
) {
  const mcpClient = new GenkitMcpClient<M>(options);
  const dap = ai.defineDynamicActionProvider(
    {
      name: options.name,
      cacheConfig: {
        ttlMillis: options.cacheTtlMillis,
      },
    },
    async () => {
      return {
        tool: await mcpClient.getActiveTools(ai),
        resource: await mcpClient.getActiveResources(ai),
      };
    }
  );
  mcpClient.dynamicActionProvider = dap;
  return mcpClient;
}

/**
 * Creates an MCP server based on the supplied Genkit instance. All tools and prompts
 * will be automatically converted to MCP compatibility.
 *
 * ```ts
 * const mcpServer = createMcpServer(ai, {name: 'my-mcp-server', version: '0.1.0'});
 *
 * await mcpServer.start(); // starts a stdio transport, OR
 * await mcpServer.start(customMcpTransport); // starts server using supplied transport
 * ```
 *
 * @param ai Your Genkit instance with registered tools and prompts.
 * @param options Configuration metadata for the server.
 * @returns GenkitMcpServer instance.
 */
export function createMcpServer(
  ai: Genkit,
  options: McpServerOptions
): GenkitMcpServer {
  return new GenkitMcpServer(ai, options);
}
