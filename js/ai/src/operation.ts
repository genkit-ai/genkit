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

import type { BackgroundActionRunOptions } from '@genkit-ai/core';

/**
 * The reserved context key under which per-call `config` overrides are carried
 * when invoking a background operation's `check`/`cancel`. This is an internal
 * transport detail: callers set a top-level `config` on {@link OperationOptions}
 * and plugins read it back from `options.context.config`.
 */
export const OPERATION_CONFIG_CONTEXT_KEY = 'config';

/**
 * Options for {@link Genkit.checkOperation} and {@link Genkit.cancelOperation}.
 *
 * Extends the generic {@link BackgroundActionRunOptions} with a top-level
 * `config` field so callers can supply per-call configuration overrides (e.g.
 * `baseUrl`, `location`) for calls that don't otherwise carry a request config.
 */
export interface OperationOptions extends BackgroundActionRunOptions {
  /**
   * Per-call configuration overrides. Because operation `check`/`cancel` calls
   * don't include the original request's `config`, this is the mechanism for
   * supplying critical config options (e.g. `baseUrl`, `location`) at call
   * time.
   *
   * Under the hood this is delivered to plugins via
   * `options.context.config`.
   */
  config?: Record<string, any>;
}

/**
 * Folds the top-level `config` from {@link OperationOptions} into the run
 * options' `context` (under a reserved key), producing a
 * {@link BackgroundActionRunOptions} suitable for dispatching to a core
 * background action. `config` is an AI-layer concept, so it never leaks into
 * core's generic options; it rides through `context`, which is the only side
 * channel plumbed all the way through to a plugin's `check`/`cancel`.
 */
export function toRunOptions(
  options?: OperationOptions
): BackgroundActionRunOptions | undefined {
  if (!options) {
    return undefined;
  }
  const { config, context, ...rest } = options;
  if (config === undefined) {
    return { ...rest, context };
  }
  return {
    ...rest,
    context: {
      ...context,
      [OPERATION_CONFIG_CONTEXT_KEY]: config,
    },
  };
}
