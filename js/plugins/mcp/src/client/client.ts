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

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import type { StdioServerParameters } from '@modelcontextprotocol/sdk/client/stdio.js';
import type { StreamableHTTPClientTransportOptions } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import {
  ListRootsRequestSchema,
  Root,
} from '@modelcontextprotocol/sdk/types.js';
import {
  GenkitError,
  type DynamicActionProviderAction,
  type DynamicResourceAction,
  type ExecutablePrompt,
  type Genkit,
  type MultipartToolAction,
  type PromptGenerateOptions,
  type ToolAction,
} from 'genkit';
import { logger } from 'genkit/logging';
import {
  fetchAllPrompts,
  fetchDynamicTools,
  getExecutablePrompt,
  transportFrom,
} from '../util';
import { fetchDynamicResources } from '../util/resource';

interface McpServerRef {
  client: Client;
  transport: Transport;
  error?: string;
}

export interface McpServerControls {
  /** when true, the server will be stopped and its registered components will
   * not appear in lists/plugins/etc */
  disabled?: boolean;

  // MCP roots configuration. See: https://modelcontextprotocol.io/docs/concepts/roots
  roots?: Root[];
}

export type McpStdioServerConfig = StdioServerParameters & {
  url?: never;
  transport?: never;
};

export type McpStreamableHttpConfig = {
  url: string;
  command?: never;
  transport?: never;
} & Omit<StreamableHTTPClientTransportOptions, 'sessionId'>;

export type McpTransportServerConfig = {
  transport: Transport;
  command?: never;
  url?: never;
};

/**
 * Configuration for an individual MCP server. The interface should be familiar
 * and compatible with existing tool configurations e.g. Cursor or Claude
 * Desktop.
 *
 * In addition to stdio servers, remote servers are supported via URL and
 * custom/arbitary transports are supported as well.
 */
export type McpServerConfig = (
  | McpStdioServerConfig
  | McpStreamableHttpConfig
  | McpTransportServerConfig
) &
  McpServerControls;

/**
 * Configuration options for an individual `GenkitMcpClient` instance.
 * This defines how the client connects to a single MCP server and how it behaves.
 */
export type McpClientOptions<M extends boolean = false> = {
  /** Client name to advertise to the server. */
  name: string;
  /** Name for the server, defaults to the server's advertised name. */
  serverName?: string;

  /**
   * An optional version number for this client. This is primarily for logging
   * and identification purposes. Defaults to '1.0.0'.
   */
  version?: string;
  /**
   * If true, tool responses from the MCP server will be returned in their raw
   * MCP format. Otherwise (default), they are processed and potentially
   * simplified for better compatibility with Genkit's typical data structures.
   */
  rawToolResponses?: boolean;
  /** If true, tools will be registered as multipart tool.v2 actions. */
  multipart?: M;
  /** The server configuration to connect. */
  mcpServer: McpServerConfig;
  /** Manually supply a session id for HTTP streaming clients if desired. */
  sessionId?: string;
};

export type McpClientOptionsWithCache<M extends boolean = false> =
  McpClientOptions<M> & {
    cacheTtlMillis?: number;
  };

/**
 * Represents a client connection to a single MCP (Model Context Protocol) server.
 * It handles the lifecycle of the connection (connect, disconnect, disable, re-enable, reconnect)
 * and provides methods to fetch tools from the connected server.
 *
 * An instance of `GenkitMcpClient` is typically managed by a `GenkitMcpHost`
 * when dealing with multiple MCP server connections.
 */
export class GenkitMcpClient<Multipart extends boolean = false> {
  _server?: McpServerRef;
  private _dynamicActionProvider: DynamicActionProviderAction | undefined;

  sessionId?: string;
  readonly name: string;
  readonly suppliedServerName?: string;
  private version: string;
  private serverConfig: McpServerConfig;
  private rawToolResponses?: boolean;
  private multipart?: boolean;
  private disabled: boolean;
  private roots?: Root[];

  private _readyListeners: {
    resolve: () => void;
    reject: (err: Error) => void;
  }[] = [];
  private _ready = false;

  constructor(options: McpClientOptions<Multipart>) {
    this.name = options.name;
    this.suppliedServerName = options.serverName;
    this.version = options.version || '1.0.0';
    this.serverConfig = options.mcpServer;
    this.rawToolResponses = !!options.rawToolResponses;
    this.multipart = !!options.multipart;
    this.disabled = !!options.mcpServer.disabled;
    this.roots = options.mcpServer.roots;
    this.sessionId = options.sessionId;

    this._initializeConnection();
  }

  set dynamicActionProvider(dap: DynamicActionProviderAction) {
    this._dynamicActionProvider = dap;
  }

  _invalidateDapCache(): void {
    if (this._dynamicActionProvider) {
      this._dynamicActionProvider.invalidateCache();
    }
  }

  get serverName(): string {
    return (
      this.suppliedServerName ??
      this._server?.client.getServerVersion()?.name ??
      'unknown-server'
    );
  }

  async updateRoots(roots: Root[]) {
    this.roots = roots;
    await this._server?.client.sendRootsListChanged();
    this._invalidateDapCache();
  }

  /**
   * Sets up a connection based on a provided map of server configurations.
   * - Reconnects existing servers if their configuration appears to have
   *   changed (implicitly handled by `connectServer`).
   * - Sets the client's ready state once all connection attempts are complete.
   * @param mcpServers A record mapping server names to their configurations.
   */
  private async _initializeConnection() {
    this._ready = false;
    try {
      await this._connect();
      this._ready = true;
      while (this._readyListeners.length) {
        this._readyListeners.pop()?.resolve();
      }
    } catch (err) {
      while (this._readyListeners.length) {
        this._readyListeners.pop()?.reject(err as Error);
      }
    }
    if (this.roots) {
      await this.updateRoots(this.roots);
    }
    this._invalidateDapCache();
  }

  /**
   * Returns a Promise that resolves when the client has attempted to connect
   * to all configured servers, or rejects if a critical error occurs during
   * the initial connection phase.
   */
  async ready() {
    if (this._ready) return;
    return new Promise<void>((resolve, reject) => {
      this._readyListeners.push({ resolve, reject });
    });
  }

  /**
   * Connects to a single MCP server defined by the provided configuration.
   * @param config The configuration object for the server.
   */
  private async _connect() {
    if (this._server) await this._server.transport.close();
    this._invalidateDapCache();
    logger.debug(
      `[MCP Client] Connecting MCP server '${this.serverName}' in client '${this.name}'.`
    );

    const { transport, type: transportType } = await transportFrom(
      this.serverConfig,
      this.sessionId
    );
    if (!transport) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `[MCP Client] Could not determine valid transport config from supplied options.`,
      });
    }

    let error: string | undefined;

    const client = new Client(
      { name: this.name, version: this.version },
      { capabilities: { roots: { listChanged: true } } }
    );
    client.setRequestHandler(ListRootsRequestSchema, () => {
      logger.debug(`[MCP Client] fetching roots for ${this.name}`);
      return { roots: this.roots || [] };
    });

    try {
      await client.connect(transport);
    } catch (e) {
      logger.warn(
        `[MCP Client] Error connecting server via ${transportType} transport: ${e}`
      );
      this.disabled = true;
      error = (e as Error).toString();
    }

    this._server = {
      client,
      transport,
      error,
    } as McpServerRef;
    this._invalidateDapCache();
  }

  /**
   * Disconnects the MCP server and removes its registration
   * from this client instance.
   */
  async _disconnect() {
    if (this._server) {
      logger.debug(
        `[MCP Client] Disconnecting MCP server in client '${this.name}'.`
      );
      await this._server.client.close();
      this._server = undefined;
      this._invalidateDapCache();
    }
  }

  /**
   * Disables a server. Closes the underlying transport and server's configuration. Does nothing if the server is
   * already disabled.
   */
  async disable() {
    if (!this.isEnabled()) return;
    if (this._server) {
      logger.debug(
        `[MCP Client] Disabling MCP server in client '${this.name}'`
      );
      await this._disconnect();
      this.disabled = true;
      this._invalidateDapCache();
    }
  }

  /**
   * Whether this client-server connection is enabled or not.
   */
  isEnabled() {
    return !this.disabled;
  }

  /**
   * Enables a server connection, including previously disabled ones. Does nothing if the
   * server is not disabled.
   */
  async enable() {
    if (this.isEnabled()) return;
    logger.debug(`[MCP Client] Reenabling MCP server in client '${this.name}'`);
    await this._initializeConnection();
    this.disabled = !!this._server!.error;
    this._invalidateDapCache();
  }

  /**
   * Closes and then restarts the transport connection for the specified server.
   * Useful for attempting to recover from connection issues without full
   * reconfiguration.
   */
  async restart() {
    if (this._server) {
      logger.debug(
        `[MCP Client] Restarting connection to MCP server in client '${this.name}'`
      );
      await this._disconnect();
      await this._initializeConnection();
      this._invalidateDapCache();
    }
  }

  /**
   * Fetches all tools available through this client, if the server
   * configuration is not disabled.
   */
  async getActiveTools(
    ai: Genkit
  ): Promise<(Multipart extends true ? MultipartToolAction : ToolAction)[]> {
    await this.ready();

    if (this._server) {
      const capabilities = this._server.client.getServerCapabilities();
      if (capabilities?.tools) {
        if (this.multipart) {
          const tools = await fetchDynamicTools(ai, this._server.client, {
            rawToolResponses: this.rawToolResponses,
            multipart: true,
            serverName: this.serverName,
            name: this.name,
          });
          return tools as unknown as (Multipart extends true
            ? MultipartToolAction
            : ToolAction)[];
        } else {
          const tools = await fetchDynamicTools(ai, this._server.client, {
            rawToolResponses: this.rawToolResponses,
            multipart: false,
            serverName: this.serverName,
            name: this.name,
          });
          return tools as unknown as (Multipart extends true
            ? MultipartToolAction
            : ToolAction)[];
        }
      }
    }

    return [] as unknown as (Multipart extends true
      ? MultipartToolAction
      : ToolAction)[];
  }

  /**
   * Fetches all resources available through this client, if the server
   * configuration is not disabled.
   */
  async getActiveResources(ai: Genkit): Promise<DynamicResourceAction[]> {
    await this.ready();
    let resources: DynamicResourceAction[] = [];

    if (this._server) {
      const capabilities = this._server.client.getServerCapabilities();
      if (capabilities?.resources)
        resources.push(
          ...(await fetchDynamicResources(ai, this._server.client, {
            serverName: this.serverName,
            name: this.name,
          }))
        );
    }

    return resources;
  }

  /**
   * Fetches all active prompts available through this client, if the server
   * configuration supports prompts.
   * @param ai The Genkit instance.
   * @param options Optional prompt generation options.
   * @returns A promise that resolves to an array of ExecutablePrompt.
   */
  async getActivePrompts(
    ai: Genkit,
    options?: PromptGenerateOptions
  ): Promise<ExecutablePrompt[]> {
    if (this._server?.client.getServerCapabilities()?.prompts) {
      return fetchAllPrompts(this._server.client, {
        ai,
        serverName: this.serverName,
        name: this.name,
        options,
      });
    }
    return [];
  }

  /**
   * Get the specified prompt as an `ExecutablePrompt` available through this
   * client. If no such prompt is found, return undefined.
   */
  async getPrompt(
    ai: Genkit,
    promptName: string,
    opts?: PromptGenerateOptions
  ): Promise<ExecutablePrompt | undefined> {
    await this.ready();

    if (this._server) {
      const capabilities = await this._server.client.getServerCapabilities();
      if (capabilities?.prompts) {
        return await getExecutablePrompt(this._server.client, {
          ai,
          serverName: this.name,
          promptName,
          name: this.name,
          options: opts,
        });
      }
      logger.debug(`[MCP Client] No prompts are found in this MCP server.`);
    }
    return;
  }

  /** Returns the underlying MCP SDK client if one has been initialized. */
  get mcpClient(): Client | undefined {
    return this._server?.client;
  }
}
