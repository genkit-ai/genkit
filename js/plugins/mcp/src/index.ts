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
 * Creates an MCP Client Host that connects to one or more MCP servers.
 * Each server is defined in the `mcpClients` option, where the key is a
 * client-side name for the server and the value is the server's configuration.
 *
 * By default, all servers in the config will be attempted to connect unless
 * their configuration includes `{disabled: true}`.
 *
 * ```ts
 * const clientHost = createMcpHost({
 *   name: "my-mcp-client-host", // Name for the host itself
 *   mcpServers: {
 *     // Each key is a name for this client/server configuration
 *     // Each value is an McpServerConfig object
 *     gitToolServer: { command: "uvx", args: ["mcp-server-git"] },
 *     customApiServer: { url: "http://localhost:1234/mcp" }
 *   }
 * });
 * ```
 *
 * @param options Configuration for the MCP Client Host, including the definitions of MCP servers to connect to.
 * @returns A new instance of GenkitMcpHost.
 */
export function createMcpHost<M extends boolean = false>(
  options: McpHostOptions<M>
) {
  return new GenkitMcpHost<M>(options);
}

/**
 * Creates an MCP Client Host that connects to one or more MCP servers.
 * Each server is defined in the `mcpClients` option, where the key is a
 * client-side name for the server and the value is the server's configuration.
 *
 * By default, all servers in the config will be attempted to connect unless
 * their configuration includes `{disabled: true}`.
 *
 * ```ts
 * const clientHost = defineMcpHost(ai, {
 *   name: "my-mcp-client-host", // Name for the host itself
 *   mcpServers: {
 *     // Each key is a name for this client/server configuration
 *     // Each value is an McpServerConfig object
 *     gitToolServer: { command: "uvx", args: ["mcp-server-git"] },
 *     customApiServer: { url: "http://localhost:1234/mcp" }
 *   }
 * });
 * ```
 *
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
        ttlMillis: options.cacheTTLMillis,
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
 * Creates an MCP Client that connects to a single MCP server.
 * This is useful when you only need to interact with one MCP server,
 * or if you want to manage client instances individually.
 *
 * ```ts
 * const client = createMcpClient({
 *   name: "mySingleMcpClient", // A name for this client instance
 *   command: "npx", // Example: Launching a local server
 *   args: ["-y", "@modelcontextprotocol/server-everything", "/path/to/allowed/dir"],
 * });
 *
 * // To get tools from this client:
 * // const tools = await client.getActiveTools(ai);
 * ```
 *
 * @param options Configuration for the MCP Client, defining how it connects
 *                to the MCP server and its behavior.
 * @returns A new instance of GenkitMcpClient.
 */
export function createMcpClient<M extends boolean = false>(
  options: McpClientOptions<M>
) {
  return new GenkitMcpClient<M>(options);
}

/**
 * Defines an MCP Client that connects to a single MCP server.
 * This is useful when you only need to interact with one MCP server,
 * or if you want to manage client instances individually.
 *
 * ```ts
 * const client = defineMcpClient(ai, {
 *   name: "mySingleMcpClient", // A name for this client instance
 *   command: "npx", // Example: Launching a local server
 *   args: ["-y", "@modelcontextprotocol/server-everything", "/path/to/allowed/dir"],
 * });
 *
 * // To get tools from this client:
 * // const tools = await client.getActiveTools(ai);
 *
 * // Or in a generate call you can use:
 * ai.generate({
    prompt: `<a prompt requiring tools>`,
    tools: ['mySingleMcpClient:tool/*'],
  });
 * ```
 *
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
