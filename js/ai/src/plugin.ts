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

import type { ActionMetadata, ResolvableAction } from '@genkit-ai/core';
import type { ActionType } from '@genkit-ai/core/registry';
import type { GenerateMiddleware } from './generate/middleware.js';
import type { GenkitAI } from './genkit-ai.js';

/**
 * Passed to {@link GenkitPlugin.init} and {@link GenkitPlugin.resolve}.
 */
export interface PluginContext {
  /**
   * Bound to the registry the plugin is installed in. Lets plugin actions call
   * `generate`, `embed` etc. (e.g. a judge model inside an evaluator), but not
   * register anything.
   *
   * Use it inside the actions the plugin returns, not while `init` or
   * `resolve` themselves run: plugin init is cached as a single in-flight
   * promise, so calling one of the plugin's own actions during `init` waits on
   * itself and never resolves.
   */
  ai: GenkitAI;
}

/**
 * A Genkit plugin. Plugins return actions (models, embedders, evaluators,
 * tools, ...) and Genkit registers them, namespaced with the plugin name: a
 * model named `fast` from plugin `my` is registered as `my/fast`.
 *
 * Usually created with `genkitPlugin({...})` from `genkit/plugin`.
 */
export interface GenkitPlugin {
  name: string;
  /**
   * Returns actions to register eagerly. Called once, lazily, on the first
   * lookup that needs this plugin.
   */
  init?: (
    ctx: PluginContext
  ) => ResolvableAction[] | Promise<ResolvableAction[]>;
  /**
   * Returns an action that wasn't returned by `init`, or `undefined` if the
   * plugin doesn't provide it. `name` is never namespaced: for `my/fast` the
   * plugin gets `fast`.
   */
  resolve?: (
    actionType: ActionType,
    name: string,
    ctx: PluginContext
  ) => ResolvableAction | undefined | Promise<ResolvableAction | undefined>;
  /**
   * Lists actions this plugin can resolve, for developer tooling. Names may
   * be returned with or without the plugin namespace.
   */
  list?: () => ActionMetadata[] | Promise<ActionMetadata[]>;
  /** Generate middleware usable in `generate({ use: [...] })`. */
  middleware?: () => GenerateMiddleware<any, any>[];
}
