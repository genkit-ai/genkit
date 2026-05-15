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

import { logger } from './logging';

const CONFIG_KEY = '__GENKIT_RUNTIME_CONFIG__';

/**
 * Runtime configuration for Genkit.
 */
export interface GenkitRuntimeConfig {
  /**
   * JSON schema validation mode.
   * - 'compile': Uses `ajv` to compile schema validation functions. This is faster but requires `eval` or `new Function` support.
   * - 'interpret': Uses `@cfworker/json-schema` to interpret schemas. This is slower but works in environments that restrict code generation (e.g. Cloudflare Workers).
   */
  jsonSchemaMode?: 'compile' | 'interpret';

  /**
   * Whether the runtime is sandboxed.
   * If true, features that require access to the file system or spawning processes (like the Reflection API) will be disabled.
   */
  sandboxedRuntime?: boolean;
}

function getConfig(): GenkitRuntimeConfig {
  if (!global[CONFIG_KEY]) {
    global[CONFIG_KEY] = {};
  }
  return global[CONFIG_KEY];
}

/**
 * Sets the runtime configuration for Genkit.
 */
export function setGenkitRuntimeConfig(config: GenkitRuntimeConfig) {
  if (config.jsonSchemaMode === 'interpret') {
    logger.warn(
      "It looks like you're trying to disable schema code generation. Please ensure that the '@cfworker/json-schema' package is installed: `npm i --save @cfworker/json-schema`"
    );
  }
  const current = getConfig();
  global[CONFIG_KEY] = { ...current, ...config };
}

/**
 * Gets the current runtime configuration for Genkit.
 */
export function getGenkitRuntimeConfig(): GenkitRuntimeConfig {
  const config = getConfig();
  return {
    jsonSchemaMode: config.jsonSchemaMode ?? 'compile',
    sandboxedRuntime: config.sandboxedRuntime ?? false,
  };
}

/**
 * Resets the current runtime configuration for Genkit.
 */
export function resetGenkitRuntimeConfig() {
  global[CONFIG_KEY] = {};
}
