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

import { GenerateRequest } from 'genkit';
import { EmbedRequest } from 'genkit/embedder';
import OpenAI from 'openai';
import { PluginOptions } from '.';

/**
 * Inspects the request and if apiKey is provided in config, creates a new client.
 * Otherwise falls back on the `defaultClient`.
 */
export function maybeCreateRequestScopedOpenAIClient(
  pluginOptions: PluginOptions | undefined,
  request: GenerateRequest | EmbedRequest,
  defaultClient: OpenAI
): OpenAI {
  const requestApiKey =
    (request as GenerateRequest)?.config?.apiKey ??
    (request as EmbedRequest)?.options?.apiKey;
  if (!requestApiKey) {
    return defaultClient;
  }
  return new OpenAI({
    // if pluginOptions are not passed in we attempt to get options from the default client.
    ...(pluginOptions ?? defaultClient['_options']),
    apiKey: requestApiKey,
  });
}

/**
 * Gets the model name without certain prefixes.
 */
export function toModelName(name: string, prefix?: string): string {
  const pattern = '^/(background-model|model|models|embedder|embedders)/';
  const refPrefixes = new RegExp(pattern);
  const maybePluginRef = name.replace(refPrefixes, '');
  if (prefix) {
    const pluginPrefix = new RegExp(`^${prefix}/`, 'g');
    return maybePluginRef.replace(pluginPrefix, '');
  } else {
    return maybePluginRef;
  }
}
