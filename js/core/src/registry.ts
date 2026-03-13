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

import { Dotprompt } from 'dotprompt';
import type * as z from 'zod';
import {
  runOutsideActionRuntimeContext,
  type Action,
  type ActionMetadata,
} from './action.js';
import {
  BackgroundAction,
  lookupBackgroundAction,
} from './background-action.js';
import { ActionContext } from './context.js';
import { isDynamicActionProvider } from './dynamic-action-provider.js';
import { GenkitError } from './error.js';
import { logger } from './logging.js';
import type { PluginProvider } from './plugin.js';
import { toJsonSchema, type JSONSchema } from './schema.js';

export type AsyncProvider<T> = () => Promise<T>;

/**
 * Type of a runnable action.
 */
const ACTION_TYPES = [
  'custom',
  'dynamic-action-provider',
  'embedder',
  'evaluator',
  'executable-prompt',
  'flow',
  'indexer',
  'model',
  'background-model',
  'check-operation',
  'cancel-operation',
  'prompt',
  'reranker',
  'retriever',
  'tool',
  'tool.v2',
  'util',
  'resource',
] as const;
export type ActionType = (typeof ACTION_TYPES)[number];

export function isActionType(value: string): value is ActionType {
  return (ACTION_TYPES as readonly string[]).includes(value);
}

/**
 * A schema is either a Zod schema or a JSON schema.
 */
export interface Schema {
  schema?: z.ZodTypeAny;
  jsonSchema?: JSONSchema;
}

function parsePluginName(registryKey: string) {
  const tokens = registryKey.split('/');
  if (tokens.length >= 4) {
    return tokens[2];
  }
  return undefined;
}

interface ParsedRegistryKey {
  dynamicActionHost?: string;
  actionType: ActionType;
  pluginName?: string;
  actionName: string;
}

/**
 * Parses the registry key into key parts as per the key format convention. Ex:
 *  - mcp-host:tool/my-tool
 *  - /model/googleai/gemini-2.5-flash
 *  - /prompt/my-plugin/folder/my-prompt
 *  - /util/generate
 */
export function parseRegistryKey(
  registryKey: string
): ParsedRegistryKey | undefined {
  if (registryKey.startsWith('/dynamic-action-provider')) {
    // Dynamic Action Provider format: 'dynamic-action-provider/mcp-host:tool/mytool' or 'mcp-host:tool/*'
    const keyTokens = registryKey.split(':');
    const hostTokens = keyTokens[0].split('/');
    if (hostTokens.length < 3) {
      return undefined;
    }
    if (keyTokens.length < 2) {
      return {
        actionType: 'dynamic-action-provider',
        actionName: hostTokens[2],
      };
    }
    const tokens = keyTokens[1].split('/');
    if (tokens.length < 2 || !isActionType(tokens[0])) {
      return undefined;
    }
    return {
      dynamicActionHost: hostTokens[2],
      actionType: tokens[0],
      actionName: tokens.slice(1).join('/'),
    };
  }

  const tokens = registryKey.split('/');
  if (tokens.length < 3) {
    // Invalid key format
    return undefined;
  }
  // ex: /model/googleai/gemini-2.5-flash or /prompt/my-plugin/folder/my-prompt
  if (tokens.length >= 4) {
    return {
      actionType: tokens[1] as ActionType,
      pluginName: tokens[2],
      actionName: tokens.slice(3).join('/'),
    };
  }
  // ex: /util/generate
  return {
    actionType: tokens[1] as ActionType,
    actionName: tokens[2],
  };
}

export type ActionsRecord = Record<string, Action<z.ZodTypeAny, z.ZodTypeAny>>;
export type ActionMetadataRecord = Record<string, ActionMetadata>;

/**
 * The registry is used to store and lookup actions, trace stores, flow state stores, plugins, and schemas.
 */
export class Registry {
  private actionsById: Record<
    string,
    | Action<z.ZodTypeAny, z.ZodTypeAny>
    | PromiseLike<Action<z.ZodTypeAny, z.ZodTypeAny>>
  > = {};
  private pluginsByName: Record<string, PluginProvider> = {};
  private schemasByName: Record<string, Schema> = {};
  private valueByTypeAndName: Record<string, Record<string, any>> = {};
  private allPluginsInitialized = false;
  public apiStability: 'stable' | 'beta' = 'stable';

  readonly dotprompt: Dotprompt;
  readonly parent?: Registry;
  /** Additional runtime context data for flows and tools. */
  context?: ActionContext;

  constructor(parent?: Registry) {
    if (parent) {
      this.parent = parent;
      this.apiStability = parent?.apiStability;
      this.dotprompt = parent.dotprompt;
    } else {
      this.dotprompt = new Dotprompt({
        schemaResolver: async (name) => {
          const resolvedSchema = await this.lookupSchema(name);
          if (!resolvedSchema) {
            throw new GenkitError({
              message: `Schema '${name}' not found`,
              status: 'NOT_FOUND',
            });
          }
          return toJsonSchema(resolvedSchema);
        },
      });
    }
  }

  /**
   * Creates a new registry overlaid onto the provided registry.
   * @param parent The parent registry.
   * @returns The new overlaid registry.
   */
  static withParent(parent: Registry) {
    return new Registry(parent);
  }

  async resolveActionNames(key: string): Promise<string[]> {
    const parsedKey = parseRegistryKey(key);
    if (parsedKey?.dynamicActionHost) {
      const hostId = `/dynamic-action-provider/${parsedKey.dynamicActionHost}`;
      const dap = await this.actionsById[hostId];
      if (!dap || !isDynamicActionProvider(dap)) {
        return [];
      }
      return (
        await dap.listActionMetadata(parsedKey.actionType, parsedKey.actionName)
      ).map((m) => `${hostId}:${parsedKey.actionType}/${m.name}`);
    }
    if (await this.lookupAction(key)) {
      return [key];
    }
    return [];
  }

  /**
   * Looks up an action in the registry.
   * @param key The key of the action to lookup.
   * @returns The action.
   */
  async lookupAction<
    I extends z.ZodTypeAny,
    O extends z.ZodTypeAny,
    R extends Action<I, O>,
  >(key: string): Promise<R> {
    const parsedKey = parseRegistryKey(key);
    if (
      parsedKey?.dynamicActionHost &&
      this.actionsById[
        `/dynamic-action-provider/${parsedKey.dynamicActionHost}`
      ]
    ) {
      // If it's a dynamic action provider, get the dynamic action.
      const action = await this.getDynamicAction(parsedKey);
      if (action) {
        return action as R;
      }
    }

    // We always try to initialize the plugin first.
    if (parsedKey?.pluginName && this.pluginsByName[parsedKey.pluginName]) {
      await this.initializePlugin(parsedKey.pluginName);

      // If we don't see the key in the registry, we try to resolve
      // the action with the dynamic resolver. If it exists, it will
      // register the action in the registry.
      if (!this.actionsById[key]) {
        await this.resolvePluginAction(
          parsedKey.pluginName,
          parsedKey.actionType,
          parsedKey.actionName
        );
      }
    }

    return (
      ((await this.actionsById[key]) as R) || this.parent?.lookupAction(key)
    );
  }

  /**
   * Looks up a background action from the registry.
   * @param key The key of the action to lookup.
   * @returns The action.
   */
  async lookupBackgroundAction(
    key: string
  ): Promise<BackgroundAction | undefined> {
    return lookupBackgroundAction(this, key);
  }

  /**
   * Registers an action in the registry.
   * @param type The type of the action to register.
   * @param action The action to register.
   */
  registerAction<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    type: ActionType,
    action: Action<I, O>,
    opts?: { namespace?: string }
  ) {
    if (type !== action.__action.actionType) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `action type (${type}) does not match type on action (${action.__action.actionType})`,
      });
    }
    if (
      opts?.namespace &&
      !action.__action.name.startsWith(`${opts.namespace}/`)
    ) {
      action.__action.name = `${opts.namespace}/${action.__action.name}`;
    }
    const key = `/${type}/${action.__action.name}`;
    logger.debug(`registering ${key}`);
    if (this.actionsById.hasOwnProperty(key)) {
      logger.error(
        `ERROR: ${key} already has an entry in the registry. Overwriting.`
      );
    }
    this.actionsById[key] = action;
    if (action.__registry) {
      logger.error(`ERROR: ${key} already registered.`);
    }
    action.__registry = this;
  }

  /**
   * Registers an action promise in the registry.
   */
  registerActionAsync<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    type: ActionType,
    name: string,
    action: PromiseLike<Action<I, O>>,
    opts?: { namespace?: string }
  ) {
    if (opts?.namespace && !name.startsWith(`${opts.namespace}/`)) {
      name = `${opts.namespace}/${name}`;
    }
    const key = `/${type}/${name}`;
    logger.debug(`registering ${key} (async)`);
    if (this.actionsById.hasOwnProperty(key)) {
      logger.error(
        `ERROR: ${key} already has an entry in the registry. Overwriting.`
      );
    }
    this.actionsById[key] = action;
  }

  /**
   * Returns all actions that have been registered in the registry.
   * @returns All actions in the registry as a map of <key, action>.
   */
  async listActions(): Promise<ActionsRecord> {
    await this.initializeAllPlugins();
    const actions: Record<string, Action<z.ZodTypeAny, z.ZodTypeAny>> = {};
    await Promise.all(
      Object.entries(this.actionsById).map(async ([key, action]) => {
        actions[key] = await action;
      })
    );
    return {
      ...(await this.parent?.listActions()),
      ...actions,
    };
  }

  /**
   * Returns all actions that are resolvable by plugins as well as those that are already
   * in the registry.
   *
   * NOTE: this method should not be used in latency sensitive code paths.
   * It may rely on "admin" API calls such as "list models", which may cause increased cold start latency.
   *
   * @returns All resolvable action metadata as a map of <key, action metadata>.
   */
  async listResolvableActions(): Promise<ActionMetadataRecord> {
    let resolvableActions = {} as ActionMetadataRecord;
    // We listActions for all plugins in parallel.
    await Promise.all(
      Object.entries(this.pluginsByName).map(async ([pluginName, plugin]) => {
        if (plugin.listActions) {
          try {
            (await plugin.listActions()).forEach((meta) => {
              if (!meta.name) {
                throw new GenkitError({
                  status: 'INVALID_ARGUMENT',
                  message: `Invalid metadata when listing actions from ${pluginName} - name required`,
                });
              }
              if (!meta.actionType) {
                throw new GenkitError({
                  status: 'INVALID_ARGUMENT',
                  message: `Invalid metadata when listing actions from ${pluginName} - actionType required`,
                });
              }
              resolvableActions[`/${meta.actionType}/${meta.name}`] = meta;
            });
          } catch (e) {
            logger.error(`Error listing actions for ${pluginName}\n`, e);
          }
        }
      })
    );
    // Also add actions that are already registered, and expand DAP actions
    for (const [key, action] of Object.entries(await this.listActions())) {
      resolvableActions[key] = action.__action;
      if (isDynamicActionProvider(action)) {
        try {
          // Include the dynamic actions
          const dapPrefix = `/${action.__action.actionType}/${action.__action.name}`;
          const dapMetadataRecord =
            await action.getActionMetadataRecord(dapPrefix);
          resolvableActions = { ...resolvableActions, ...dapMetadataRecord };
        } catch (e) {
          logger.error(
            `Error listing actions for Dynamic Action Provider ${action.__action.name}`
          );
        }
      }
    }
    return {
      ...(await this.parent?.listResolvableActions()),
      ...resolvableActions,
    };
  }

  /**
   * Initializes all plugins in the registry.
   */
  async initializeAllPlugins() {
    if (this.allPluginsInitialized) {
      return;
    }
    for (const pluginName of Object.keys(this.pluginsByName)) {
      await this.initializePlugin(pluginName);
    }
    this.allPluginsInitialized = true;
  }

  /**
   * Registers a plugin provider. This plugin must be initialized before it can be used by calling {@link initializePlugin} or {@link initializeAllPlugins}.
   * @param name The name of the plugin to register.
   * @param provider The plugin provider.
   */
  registerPluginProvider(name: string, provider: PluginProvider) {
    if (this.pluginsByName[name]) {
      throw new Error(`Plugin ${name} already registered`);
    }
    this.allPluginsInitialized = false;
    let cached;
    let isInitialized = false;
    this.pluginsByName[name] = {
      name: provider.name,
      initializer: () => {
        if (!isInitialized) {
          cached = provider.initializer();
          isInitialized = true;
        }
        return cached;
      },
      resolver: async (actionType: ActionType, actionName: string) => {
        if (provider.resolver) {
          await provider.resolver(actionType, actionName);
        }
      },
      listActions: async () => {
        if (provider.listActions) {
          return await provider.listActions();
        }
        return [];
      },
    };
  }

  /**
   * Looks up a plugin.
   * @param name The name of the plugin to lookup.
   * @returns The plugin provider.
   */
  lookupPlugin(name: string): PluginProvider | undefined {
    return this.pluginsByName[name] || this.parent?.lookupPlugin(name);
  }

  /**
   * Resolves a new Action dynamically by registering it.
   * @param pluginName The name of the plugin
   * @param actionType The type of the action
   * @param actionName The name of the action
   * @returns
   */
  async resolvePluginAction(
    pluginName: string,
    actionType: ActionType,
    actionName: string
  ) {
    const plugin = this.pluginsByName[pluginName];
    if (plugin) {
      return await runOutsideActionRuntimeContext(async () => {
        if (plugin.resolver) {
          await plugin.resolver(actionType, actionName);
        }
      });
    }
  }

  async getDynamicAction(
    key: ParsedRegistryKey
  ): Promise<Action<z.ZodTypeAny, z.ZodTypeAny> | undefined> {
    if (key.actionName.includes('*')) {
      // * means multiple actions, this returns exactly one.
      return undefined;
    }
    const actionId = `/dynamic-action-provider/${key.dynamicActionHost}`;
    const dap = await this.actionsById[actionId];
    if (!dap || !isDynamicActionProvider(dap)) {
      return undefined;
    }
    return await dap.getAction(key.actionType, key.actionName);
  }

  /**
   * Initializes a plugin already registered with {@link registerPluginProvider}.
   * @param name The name of the plugin to initialize.
   * @returns The plugin.
   */
  async initializePlugin(name: string) {
    if (this.pluginsByName[name]) {
      return await runOutsideActionRuntimeContext(() =>
        this.pluginsByName[name].initializer()
      );
    }
  }

  /**
   * Registers a schema.
   * @param name The name of the schema to register.
   * @param data The schema to register (either a Zod schema or a JSON schema).
   */
  registerSchema(name: string, data: Schema) {
    if (this.schemasByName[name]) {
      throw new Error(`Schema ${name} already registered`);
    }
    this.schemasByName[name] = data;
  }

  registerValue(type: string, name: string, value: any) {
    if (!this.valueByTypeAndName[type]) {
      this.valueByTypeAndName[type] = {};
    }
    this.valueByTypeAndName[type][name] = value;
  }

  async lookupValue<T = unknown>(
    type: string,
    key: string
  ): Promise<T | undefined> {
    const pluginName = parsePluginName(key);
    if (!this.valueByTypeAndName[type]?.[key] && pluginName) {
      await this.initializePlugin(pluginName);
    }
    return (
      (this.valueByTypeAndName[type]?.[key] as T) ||
      this.parent?.lookupValue(type, key)
    );
  }

  async listValues<T>(type: string): Promise<Record<string, T>> {
    await this.initializeAllPlugins();
    return {
      ...((await this.parent?.listValues(type)) || {}),
      ...(this.valueByTypeAndName[type] || {}),
    } as Record<string, T>;
  }

  /**
   * Looks up a schema.
   * @param name The name of the schema to lookup.
   * @returns The schema.
   */
  lookupSchema(name: string): Schema | undefined {
    return this.schemasByName[name] || this.parent?.lookupSchema(name);
  }
}

/**
 * An object that has a reference to Genkit Registry.
 */
export interface HasRegistry {
  get registry(): Registry;
}
