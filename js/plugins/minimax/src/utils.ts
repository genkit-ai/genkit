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

import { GenkitError } from 'genkit';

/**
 * Namespace this plugin registers its actions under.
 */
export const PLUGIN_NAME = 'minimax';

/**
 * Environment variable read when no API key is configured.
 */
export const API_KEY_ENV_VAR = 'MINIMAX_API_KEY';

/**
 * Reads the API key from the environment.
 */
export function getApiKeyFromEnvVar(): string | undefined {
  return process.env[API_KEY_ENV_VAR];
}

const MISSING_API_KEY_ERROR = new GenkitError({
  status: 'FAILED_PRECONDITION',
  message: `Please pass in the API key or set the ${API_KEY_ENV_VAR} environment variable.`,
});

const API_KEY_FALSE_ERROR = new GenkitError({
  status: 'INVALID_ARGUMENT',
  message:
    'MiniMax plugin was initialized with {apiKey: false} but no apiKey configuration was passed at call time.',
});

/**
 * Fails fast at plugin initialization when no API key can be resolved.
 *
 * Passing `false` defers the check to request time.
 */
export function checkApiKey(pluginApiKey: string | false | undefined): void {
  if (pluginApiKey === false) {
    return;
  }
  if (!pluginApiKey && !getApiKeyFromEnvVar()) {
    throw MISSING_API_KEY_ERROR;
  }
}

/**
 * Resolves the API key to use for a single request.
 *
 * Precedence is request config, then plugin options, then the environment.
 */
export function calculateApiKey(
  pluginApiKey: string | false | undefined,
  requestApiKey: string | undefined
): string {
  if (requestApiKey) {
    return requestApiKey;
  }
  if (pluginApiKey === false) {
    throw API_KEY_FALSE_ERROR;
  }
  const apiKey = pluginApiKey || getApiKeyFromEnvVar();
  if (!apiKey) {
    throw MISSING_API_KEY_ERROR;
  }
  return apiKey;
}

/**
 * Strips the `minimax/` namespace prefix from a model name.
 */
export function modelName(name?: string): string | undefined {
  if (!name) {
    return name;
  }
  return name.replace(new RegExp(`^${PLUGIN_NAME}/`), '');
}

/**
 * Returns the bare model name, rejecting empty values.
 */
export function checkModelName(name?: string): string {
  const version = modelName(name);
  if (!version) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'Model name is required.',
    });
  }
  return version;
}
