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

import { Root } from '@modelcontextprotocol/sdk/types.js';
import {
  type DynamicActionProviderAction,
  type DynamicResourceAction,
  type ExecutablePrompt,
  type Genkit,
  type MultipartToolAction,
  type PromptGenerateOptions,
  type ToolAction,
} from 'genkit';
import { logger } from 'genkit/logging';
import { GenkitMcpClient, McpServerConfig } from './client.js';

export interface McpHostOptions<M extends boolean = false> {
  /**
   * An optional client name for this MCP host. This name is advertised to MCP Servers
   * as the connecting client name. Defaults to 'genkit-mcp'.
   */
  name?: string;
  /**
   * An optional version for this MCP host. Primarily for
   * logging and identification within Genkit.
   * Defaults to '1.0.0'.
   */
  version?: string;
  /**
   * A record for configuring multiple MCP servers. Each server connection is
   * controlled by a `GenkitMcpClient` instance managed by `GenkitMcpHost`.
   * The key in the record is used as the identifier for the MCP server.
   */
  mcpServers?: Record<string, McpServerConfig>;

  /**
   * If true, tool responses from the MCP server will be returned in their raw
   * MCP format. Otherwise (default), they are processed and potentially
   * simplified for better compatibility with Genkit's typical data structures.
   */
  rawToolResponses?: boolean;

  /** If true, tools will be registered as multipart tool.v2 actions. */
  multipart?: M;

  /**
   * When provided, each connected MCP server will be sent the roots specified here. Overridden by any specific roots sent in the `mcpServers` config for a given server.
   */
  roots?: Root[];
}

export type McpHostOptionsWithCache<M extends boolean = false> = Omit<
  McpHostOptions<M>,
  'name'
> & {
  /**
   * A client name for this MCP host. This name is advertised to MCP Servers
   * as the connecting client name.
   */
  name: string;

  /**
   * Cache TTL. The dynamic action provider has a cache for the available actions
   * The default TTL is 3 seconds.
   * The cache will automatically be invalidated if any connections change.
   *   Negative = no caching (expect noisy traces and slower resolution)
   *   Zero or undefined = use the default 3000 millis (3 seconds)
   *   Positive: The number of milliseconds to keep the cache.
   */
  cacheTTLMillis?: number;
};

/** Internal representation of client state for logging. */
interface ClientState {
  error?: {
    message: string;
    detail?: any;
  };
}

/**
 * Manages connections to multiple MCP (Model Context Protocol) servers.
 * Each server connection is individually configured and managed by an instance of `GenkitMcpClient`.
 * This host provides a centralized way to initialize, update, and interact with these clients.
 *
 * It allows for dynamic registration of tools from all connected and enabled MCP servers
 * into a Genkit instance.
 */
export class GenkitMcpHost<Multipart extends boolean = false> {
  name: string;
  private _clients: Record<string, GenkitMcpClient<Multipart>> = {};
  private _clientStates: Record<string, ClientState> = {};
  private _readyListeners: {
    resolve: () => void;
    reject: (err: Error) => void;
  }[] = [];
  private _ready = false;
  private _dynamicActionProvider: DynamicActionProviderAction | undefined;
  private roots: Root[] | undefined;
  rawToolResponses?: boolean;
  multipart?: Multipart;

  constructor(options: McpHostOptions<Multipart>) {
    this.name = options.name || 'genkit-mcp';
    this.rawToolResponses = options.rawToolResponses;
    this.multipart = options.multipart;
    this.roots = options.roots;

    if (options.mcpServers) {
      this.updateServers(options.mcpServers);
    } else {
      this._ready = true;
    }
  }

  set dynamicActionProvider(dap: DynamicActionProviderAction) {
    this._dynamicActionProvider = dap;
  }

  _invalidateCache(): void {
    if (this._dynamicActionProvider) {
      this._dynamicActionProvider.invalidateCache();
    }
  }

  /**
   * Returns a Promise that resolves when the host has attempted to connect
   * to all configured clients, or rejects if a critical error occurs during
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
   * If a server with the same name already exists, it will be disconnected first.
   * Stores the client and transport references internally. Handles connection errors
   * by marking the server as disabled.
   * @param serverName The name to assign to this server connection.
   * @param config The configuration object for the server.
   */
  async connect(serverName: string, config: McpServerConfig) {
    const existingEntry = this._clients[serverName];
    if (existingEntry) {
      try {
        await existingEntry._disconnect();
      } catch (e) {
        existingEntry.disable();
        this.setError(serverName, {
          message: `[MCP Host] Error disconnecting from existing connection for ${serverName}`,
          detail: `Details: ${e}`,
        });
      }
    }

    logger.debug(
      `[MCP Host] Connecting to MCP server '${serverName}' in host '${this.name}'.`
    );
    try {
      const client = new GenkitMcpClient<Multipart>({
        name: this.name,
        serverName: serverName,
        mcpServer: { ...config, roots: config.roots || this.roots },
        rawToolResponses: this.rawToolResponses,
        multipart: this.multipart,
      });
      this._clients[serverName] = client;
    } catch (e) {
      this.setError(serverName, {
        message: `[MCP Host] Error connecting to ${serverName} with config ${config}`,
        detail: `Details: ${e}`,
      });
    }
    this._invalidateCache();
  }

  /**
   * Disconnects the specified MCP server and removes its registration
   * from this client instance.
   * @param serverName The name of the server to disconnect.
   */
  async disconnect(serverName: string) {
    const client = this._clients[serverName];
    if (!client) {
      logger.warn(`[MCP Host] unable to find server ${serverName}`);
      return;
    }

    logger.debug(
      `[MCP Host] Disconnecting MCP server '${serverName}' in host '${this.name}'.`
    );
    try {
      await client._disconnect();
    } catch (e) {
      client.disable();
      this.setError(serverName, {
        message: `[MCP Host] Error disconnecting from existing connection for ${serverName}`,
        detail: `Details: ${e}`,
      });
    }
    delete this._clients[serverName];
    this._invalidateCache();
  }

  /**
   * Temporarily disables a server connection. Closes the underlying transport
   * but retains the server's configuration. Does nothing if the server is
   * already disabled.
   * @param serverName The name of the server to disable.
   */
  async disable(serverName: string) {
    const client = this._clients[serverName];
    if (!client) {
      logger.warn(`[MCP Host] unable to find server ${serverName}`);
      return;
    }
    if (!client.isEnabled()) {
      logger.warn(`[MCP Host] server ${serverName} already disabled`);
      return;
    }

    logger.debug(
      `[MCP Host] Disabling MCP server '${serverName}' in host '${this.name}'`
    );
    await client.disable();
    this._invalidateCache();
  }

  /**
   * Enables a server connection, including previously disabled ones. Attempts to reconnect
   * using the stored transport. Does nothing if the server is not disabled.
   * @param serverName The name of the server to re-enable.
   */
  async enable(serverName: string) {
    const client = this._clients[serverName];
    if (!client) {
      logger.warn(`[MCP Host] unable to find server ${serverName}`);
      return;
    }

    logger.debug(
      `[MCP Host] Reenabling MCP server '${serverName}' in host '${this.name}'`
    );
    try {
      await client.enable();
    } catch (e) {
      client.disable();
      this.setError(serverName, {
        message: `[MCP Host] Error reenabling server ${serverName}`,
        detail: `Details: ${e}`,
      });
    }
    this._invalidateCache();
  }

  /**
   * Closes and then restarts the transport connection for the specified server.
   * Useful for attempting to recover from connection issues without full
   * reconfiguration.
   * @param serverName The name of the server to reconnect.
   */
  async reconnect(serverName: string) {
    const client = this._clients[serverName];
    if (!client) {
      logger.warn(`[MCP Host] unable to find server ${serverName}`);
      return;
    }

    logger.debug(
      `[MCP Host] Restarting connection to MCP server '${serverName}' in host '${this.name}'`
    );
    try {
      await client.restart();
    } catch (e) {
      client.disable();
      this.setError(serverName, {
        message: `[MCP Host] Error restarting to server ${serverName}`,
        detail: `Details: ${e}`,
      });
    }
    this._invalidateCache();
  }

  /**
   * Updates the connections based on a provided map of server configurations.
   * - Connects any new servers defined in `mcpServers`.
   * - Disconnects any servers currently connected but not present in `mcpServers`.
   * - Reconnects existing servers if their configuration appears to have changed (implicitly handled by `connectServer`).
   * Sets the client's ready state once all connection attempts are complete.
   * @param mcpServers A record mapping server names to their configurations.
   */
  updateServers(mcpServers: Record<string, McpServerConfig>) {
    this._ready = false;
    const newServerNames = new Set(Object.keys(mcpServers));
    const currentServerNames = new Set(Object.keys(this._clients));

    const promises: Promise<void>[] = [];
    for (const serverName in mcpServers) {
      promises.push(this.connect(serverName, mcpServers[serverName]));
    }

    // Disconnect servers that are no longer in the config
    for (const serverName of currentServerNames) {
      if (!newServerNames.has(serverName)) {
        this.disconnect(serverName);
      }
    }

    Promise.all(promises)
      .then(() => {
        this._ready = true;
        while (this._readyListeners.length) {
          this._readyListeners.pop()?.resolve();
        }
      })
      .catch((err) => {
        while (this._readyListeners.length) {
          this._readyListeners.pop()?.reject(err);
        }
      });

    this._invalidateCache();
  }

  /**
   * Retrieves all tools from all connected and enabled MCP clients managed by
   * this instance. This method waits for the host to be ready (all initial
   * connection attempts made) before fetching tools.
   *
   * It iterates through each managed `GenkitMcpClient`, and if the client is
   * not disabled, it calls the client's `getTools` method to fetch its
   * available tools. These are then aggregated into a single array.
   *
   * This is useful for dynamically providing a list of all available MCP tools
   * to Genkit, for example, when setting up a Genkit plugin.
   *
   * ```ts
   * const mcpHost = createMcpHost({ ... });
   * // In your Genkit configuration:
   * // const allMcpTools = await McpHost.getActiveTools(ai);
   * // Then, these tools can be used or registered with Genkit.
   * ```
   *
   * @param ai The Genkit instance, used by individual clients to define dynamic
   * tools.
   * @returns A Promise that resolves to an array of `ToolAction` from all
   * active MCP clients.
   */
  async getActiveTools(
    ai: Genkit
  ): Promise<(Multipart extends true ? MultipartToolAction : ToolAction)[]> {
    await this.ready();
    let allTools: (Multipart extends true
      ? MultipartToolAction
      : ToolAction)[] = [];

    for (const serverName in this._clients) {
      const client = this._clients[serverName];
      if (client.isEnabled() && !this.hasError(serverName)) {
        try {
          const tools = await client.getActiveTools(ai);
          allTools.push(...(tools as any));
        } catch (e) {
          logger.error(
            `Error fetching active tools from client ${serverName}.`,
            JSON.stringify(e)
          );
        }
      }
    }
    return allTools;
  }

  /**
   * Retrieves all resources from all connected and enabled MCP clients managed by
   * this instance. This method waits for the host to be ready (all initial
   * connection attempts made) before fetching resources.
   *
   * It iterates through each managed `GenkitMcpClient`, and if the client is
   * not disabled, it calls the client's `getActiveResources` method to fetch its
   * available resources. These are then aggregated into a single array.
   *
   * This is useful for dynamically providing a list of all available MCP resources
   * to Genkit, for example, when setting up a Genkit plugin.
   *
   * @param ai The Genkit instance, used by individual clients to define dynamic
   * resources.
   * @returns A Promise that resolves to an array of `DynamicResourceAction` from all
   * active MCP clients.
   */
  async getActiveResources(ai: Genkit): Promise<DynamicResourceAction[]> {
    await this.ready();
    let allResources: DynamicResourceAction[] = [];

    for (const serverName in this._clients) {
      const client = this._clients[serverName];
      if (client.isEnabled() && !this.hasError(serverName)) {
        try {
          const resources = await client.getActiveResources(ai);
          allResources.push(...resources);
        } catch (e) {
          logger.error(
            `Error fetching active resources from client ${serverName}.`,
            JSON.stringify(e)
          );
        }
      }
    }
    return allResources;
  }

  /**
   * Retrieves all prompts from all connected and enabled MCP clients managed by
   * this instance. This method waits for the host to be ready (all initial
   * connection attempts made) before fetching prompts.
   *
   * It iterates through each managed `GenkitMcpClient`, and if the client is
   * not disabled, it calls the client's `getActivePrompts` method to fetch its
   * available prompts. These are then aggregated into a single array.
   *
   * This is useful for dynamically providing a list of all available MCP prompts
   * to Genkit, for example, when setting up a Genkit plugin.
   *
   * @param ai The Genkit instance, used by individual clients to define dynamic
   * prompts.
   * @returns A Promise that resolves to an array of `ExecutablePrompt` from all
   * active MCP clients.
   */
  async getActivePrompts(ai: Genkit): Promise<ExecutablePrompt[]> {
    await this.ready();
    let allPrompts: ExecutablePrompt[] = [];

    for (const serverName in this._clients) {
      const client = this._clients[serverName];
      if (client.isEnabled() && !this.hasError(serverName)) {
        allPrompts.push(...(await client.getActivePrompts(ai)));
      }
    }
    return allPrompts;
  }

  /**
   * Get the specified prompt as an `ExecutablePrompt` available through the
   * specified server. If no such prompt is found, return undefined.
   */
  async getPrompt(
    ai: Genkit,
    serverName: string,
    promptName: string,
    opts?: PromptGenerateOptions
  ): Promise<ExecutablePrompt<any> | undefined> {
    await this.ready();
    const client = this._clients[serverName];
    if (!client) {
      logger.error(`No client found with name '${serverName}'.`);
      return;
    }
    if (this.hasError(serverName)) {
      const errorStringified = JSON.stringify(
        this._clientStates[serverName].error
      );
      logger.error(
        `Client '${serverName}' is in an error state. ${errorStringified}`
      );
    }
    if (client.isEnabled()) {
      const prompt = await client.getPrompt(ai, promptName, opts);
      if (!prompt) {
        logger.error(
          `[MCP Host] Unable to fetch the specified ${promptName} in server ${serverName}.`
        );
        return;
      }
      return prompt;
    }
    return;
  }

  async close() {
    for (const client of Object.values(this._clients)) {
      await client._disconnect();
    }
    this._invalidateCache();
  }

  /** Helper method to track and log client errors. */
  private setError(
    serverName: string,
    error: {
      message: string;
      detail?: any;
    }
  ) {
    this._clientStates[serverName] = { error };
    logger.warn(
      `An error has occured while managing your MCP client '${serverName}'. The client may be disabled to avoid further issues. Please resolve the issue and reenable the client '${serverName}' to continue using its resources.`
    );
    logger.warn(error);
  }

  private hasError(serverName: string) {
    return (
      this._clientStates[serverName] && !!this._clientStates[serverName].error
    );
  }

  /**
   * Returns an array of all active clients.
   */
  get activeClients(): GenkitMcpClient<Multipart>[] {
    return Object.values(this._clients).filter((c) => c.isEnabled());
  }

  /**
   * Returns the client by name.
   */
  getClient(name: string) {
    return this._clients[name];
  }
}
