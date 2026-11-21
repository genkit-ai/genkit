/**
 * @license
 *
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

/**
 * Plugin authoring utilities: the {@link genkitPlugin} helper, plugin types,
 * and action factory functions (`model`, `embedder`, `evaluator`, etc.).
 *
 * ```ts
 * import { genkitPlugin, model } from 'genkit/plugin';
 *
 * export const myPlugin = () =>
 *   genkitPlugin({
 *     name: 'my',
 *     init: () => [model({ name: 'fast' }, runFast)],
 *   });
 * ```
 *
 * @module plugin
 */

import type {
  GenerateMiddleware,
  GenkitAI,
  GenkitPlugin,
  PluginContext,
} from '@genkit-ai/ai';
import { GenkitError, type ResolvableAction } from '@genkit-ai/core';

export { embedder, embedderActionMetadata } from '@genkit-ai/ai/embedder';
export { evaluator } from '@genkit-ai/ai/evaluator';
export {
  backgroundModel,
  model,
  modelActionMetadata,
} from '@genkit-ai/ai/model';
export type {
  GenerateMiddleware,
  GenkitAI,
  GenkitPlugin,
  PluginContext,
  ResolvableAction,
};

/**
 * Defines a Genkit plugin. Actions returned from `init` and `resolve` are
 * registered by Genkit under the plugin's namespace (`<name>/<action>`).
 *
 * ```ts
 * export function myPlugin(opts: MyOptions): GenkitPlugin {
 *   return genkitPlugin({
 *     name: 'my',
 *     init: ({ ai }) => [
 *       model({ name: 'fast' }, runFast),
 *       evaluator({ name: 'judge', ... }, async (dataPoint) => {
 *         const { text } = await ai.generate({ model: opts.judge, prompt: ... });
 *         return toScore(dataPoint, text);
 *       }),
 *     ],
 *     resolve: (actionType, name) =>
 *       actionType === 'model' ? model({ name }, runAny(name)) : undefined,
 *     list: () => [modelActionMetadata({ name: 'fast' })],
 *   });
 * }
 * ```
 */
export function genkitPlugin(options: GenkitPlugin): GenkitPlugin {
  // Typed callers get a compile error for the removed v1 call shape
  // `genkitPlugin(name, initFn, ...)`; this catches plain JS callers.
  const arg: unknown = options;
  if (typeof arg === 'string') {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        `genkitPlugin(name, initFn) was removed in Genkit 2.0 (plugin '${arg}'). ` +
        'Return actions from genkitPlugin({ name, init }) instead of calling ai.define* in initFn.',
    });
  }
  if (typeof arg !== 'object' || arg === null || !options.name) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'genkitPlugin() expects an options object with a `name`.',
    });
  }
  return options;
}
