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

import { McpServerConfig } from '../client/client.js';
import type { StdioServerParameters, Transport } from '../client/index.js';

function mergeHeaders(
  requestHeaders: HeadersInit | undefined,
  configHeaders: HeadersInit
): Headers {
  const headers = new Headers(requestHeaders);
  new Headers(configHeaders).forEach((value, key) => headers.set(key, value));
  return headers;
}

/**
 * Creates an MCP transport instance based on the provided server configuration.
 * It supports creating SSE, Stdio, or using a pre-configured custom transport.
 *
 * @param config The configuration for the MCP server, determining the type of transport to create.
 * @returns A Promise resolving to an object containing the created `Transport` instance
 *          (or `null` if configuration is invalid) and a string indicating the `type` of transport.
 * @throws May throw an error if essential MCP SDK components cannot be imported.
 */
export async function transportFrom(
  config: McpServerConfig,
  sessionId?: string
): Promise<{
  transport: Transport | null;
  type: string;
}> {
  // Handle pre-configured transport first
  if ('transport' in config && config.transport) {
    return { transport: config.transport, type: 'custom' };
  }
  // Handle SSE config
  if ('url' in config && config.url) {
    const { headers, url, ...httpConfig } = config;
    const { StreamableHTTPClientTransport } = await import(
      '@modelcontextprotocol/sdk/client/streamableHttp.js'
    );
    return {
      transport: new StreamableHTTPClientTransport(new URL(url), {
        ...httpConfig,
        requestInit: headers
          ? {
              ...httpConfig.requestInit,
              headers: mergeHeaders(httpConfig.requestInit?.headers, headers),
            }
          : httpConfig.requestInit,
        sessionId,
      }),
      type: 'http',
    };
  }
  // Handle Stdio config
  if ('command' in config && config.command) {
    // Create a copy and remove McpServerControls properties
    const stdioConfig = { ...config };
    const { StdioClientTransport } = await import(
      '@modelcontextprotocol/sdk/client/stdio.js'
    );
    return {
      transport: new StdioClientTransport(stdioConfig as StdioServerParameters),
      type: 'stdio',
    };
  }
  return { transport: null, type: 'unknown' };
}
