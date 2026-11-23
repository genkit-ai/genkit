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

import {
  GenkitMcpClient,
  GenkitMcpHost,
  createMcpClient,
  createMcpHost,
  createMcpServer,
  defineMcpClient,
  defineMcpHost,
  type McpClientOptions,
  type McpClientOptionsWithCache,
  type McpHostOptions,
  type McpHostOptionsWithCache,
  type McpServerConfig,
  type McpServerOptions,
  type McpStdioServerConfig,
} from '@genkit-ai/mcp';
import type { Genkit } from 'genkit';

export function verifyPublicTypes(
  ai: Genkit,
  clientOptions: McpClientOptions,
  clientOptionsWithCache: McpClientOptionsWithCache,
  hostOptions: McpHostOptions,
  hostOptionsWithCache: McpHostOptionsWithCache,
  serverOptions: McpServerOptions,
  serverConfig: McpServerConfig,
  stdioServerConfig: McpStdioServerConfig
): boolean {
  const client = createMcpClient(clientOptions);
  const definedClient = defineMcpClient(ai, clientOptionsWithCache);
  const host = createMcpHost(hostOptions);
  const definedHost = defineMcpHost(ai, hostOptionsWithCache);
  const server = createMcpServer(ai, serverOptions);
  const configs: McpServerConfig[] = [serverConfig, stdioServerConfig];
  return (
    client instanceof GenkitMcpClient &&
    definedClient instanceof GenkitMcpClient &&
    host instanceof GenkitMcpHost &&
    definedHost instanceof GenkitMcpHost &&
    typeof server.start === 'function' &&
    configs.length > 0
  );
}
