/**
 * Copyright 2024 Bloom Labs Inc
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

import { APIError } from '@anthropic-ai/sdk';
import type {
  GenerateRequest,
  GenerateResponseData,
  ModelReference,
  StreamingCallback,
} from 'genkit';
import {
  GenkitError,
  z,
  type ErrorResponseMetadata,
  type StatusName,
} from 'genkit';
import type { GenerateResponseChunkData, ModelAction } from 'genkit/model';
import { modelRef } from 'genkit/model';
import { model } from 'genkit/plugin';

import type { ModelInfo } from 'genkit/model';
import { BetaRunner, Runner } from './runner/index.js';
import {
  AnthropicConfigSchema,
  resolveBetaEnabled,
  type AnthropicConfigSchemaType,
  type ClaudeModelParams,
  type ClaudeRunnerParams,
} from './types.js';
import { checkModelName, isKnownKey } from './utils.js';

/**
 * Parses a `Retry-After` header value into milliseconds.
 * Supports delay-seconds and HTTP-date formats (RFC 7231 §7.1.3).
 */
function parseRetryAfterMs(value: string): number | undefined {
  if (!value || !value.trim()) return undefined;
  const seconds = Number(value);
  if (!isNaN(seconds) && seconds >= 0) return seconds * 1000;
  const date = new Date(value);
  if (!isNaN(date.getTime())) return Math.max(0, date.getTime() - Date.now());
  return undefined;
}

// This contains all the Anthropic config schema types
type ConfigSchemaType = AnthropicConfigSchemaType;

/**
 * Generic Claude model info for unknown/unsupported models.
 * Used when a model name is not in KNOWN_CLAUDE_MODELS.
 */
export const GENERIC_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: true,
    systemRole: true,
    output: ['text'],
  },
};

/**
 * Advanced Claude model info for models that support JSON output.
 */
export const ADVANCED_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: true,
    systemRole: true,
    output: ['text', 'json'],
    constrained: 'all',
  },
};

/**
 * Creates a model reference for a Claude model.
 */
function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = AnthropicConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `anthropic/${name}`,
    configSchema,
    info: info ?? GENERIC_MODEL_INFO,
  });
}

const KNOWN_MODELS = {
  'claude-fable-5': commonRef('claude-fable-5', ADVANCED_MODEL_INFO),
  'claude-opus-4-8': commonRef('claude-opus-4-8', ADVANCED_MODEL_INFO),
  'claude-opus-4-7': commonRef('claude-opus-4-7', ADVANCED_MODEL_INFO),
  'claude-opus-4-6': commonRef('claude-opus-4-6', ADVANCED_MODEL_INFO),
  'claude-opus-4-5': commonRef('claude-opus-4-5', ADVANCED_MODEL_INFO),
  'claude-haiku-4-5': commonRef('claude-haiku-4-5', ADVANCED_MODEL_INFO),
  'claude-sonnet-5': commonRef('claude-sonnet-5', ADVANCED_MODEL_INFO),
  'claude-sonnet-4-6': commonRef('claude-sonnet-4-6', ADVANCED_MODEL_INFO),
  'claude-sonnet-4-5': commonRef('claude-sonnet-4-5', ADVANCED_MODEL_INFO),
  'claude-opus-4-1': commonRef('claude-opus-4-1', ADVANCED_MODEL_INFO),
  'claude-sonnet-4': commonRef('claude-sonnet-4'),
  'claude-opus-4': commonRef('claude-opus-4'),
} as const;
export type KnownClaudeModels = keyof typeof KNOWN_MODELS;
export type ClaudeModelName = `claude-${string}`;

export function listKnownModels(
  client: any,
  defaultApiVersion?: 'stable' | 'beta'
): ModelAction<ConfigSchemaType>[] {
  return Object.keys(KNOWN_MODELS).map((name: string) =>
    claudeModel({ name, client, defaultApiVersion })
  );
}

/**
 * Creates the runner used by Genkit to interact with the Claude model.
 * @param params Configuration for the Claude runner.
 * @param configSchema The config schema for this model (used for type inference).
 * @returns The runner that Genkit will call when the model is invoked.
 */
export function claudeRunner<TConfigSchema extends z.ZodTypeAny>(
  params: ClaudeRunnerParams,
  configSchema: TConfigSchema
) {
  const { defaultApiVersion, ...runnerParams } = params;

  if (!runnerParams.client) {
    throw new Error('Anthropic client is required to create a runner');
  }

  let stableRunner: Runner | null = null;
  let betaRunner: BetaRunner | null = null;

  return async (
    request: GenerateRequest<TConfigSchema>,
    {
      streamingRequested,
      sendChunk,
      abortSignal,
    }: {
      streamingRequested: boolean;
      sendChunk: StreamingCallback<GenerateResponseChunkData>;
      abortSignal: AbortSignal;
    }
  ): Promise<GenerateResponseData> => {
    // Cast to AnthropicConfigSchema for internal runner which expects the full schema
    const normalizedRequest = request as unknown as GenerateRequest<
      typeof AnthropicConfigSchema
    >;
    const isBeta = resolveBetaEnabled(
      normalizedRequest.config,
      defaultApiVersion
    );
    const runner = isBeta
      ? (betaRunner ??= new BetaRunner(runnerParams))
      : (stableRunner ??= new Runner(runnerParams));
    try {
      return await runner.run(normalizedRequest, {
        streamingRequested,
        sendChunk,
        abortSignal,
      });
    } catch (e) {
      if (e instanceof APIError) {
        let status: StatusName = 'UNKNOWN';
        switch (e.status) {
          case 429:
            status = 'RESOURCE_EXHAUSTED';
            break;
          case 401:
            status = 'UNAUTHENTICATED';
            break;
          case 403:
            status = 'PERMISSION_DENIED';
            break;
          case 400:
            status = 'INVALID_ARGUMENT';
            break;
          case 500:
            status = 'INTERNAL';
            break;
          case 503:
          case 529:
            status = 'UNAVAILABLE';
            break;
        }
        const retryAfterHeader = e.headers?.get?.('retry-after');
        const retryAfterMs = retryAfterHeader
          ? parseRetryAfterMs(retryAfterHeader)
          : undefined;
        const responseMetadata: ErrorResponseMetadata | undefined =
          retryAfterMs !== undefined ? { retryAfterMs } : undefined;
        throw new GenkitError({
          status,
          message: e.message,
          responseMetadata,
        });
      }
      throw e;
    }
  };
}

/**
 * Creates a model reference for a Claude model.
 * This allows referencing models without initializing the plugin.
 */
export function claudeModelReference(
  name: string,
  config: z.infer<typeof AnthropicConfigSchema> = {}
): ModelReference<ConfigSchemaType> {
  const modelName = checkModelName(name);

  if (isKnownKey(modelName, KNOWN_MODELS)) {
    return KNOWN_MODELS[modelName].withConfig(config);
  }

  return modelRef({
    name: `anthropic/${modelName}`,
    config: config,
    configSchema: AnthropicConfigSchema,
    info: {
      ...GENERIC_MODEL_INFO,
    },
  });
}

/**
 * Defines a Claude model with the given name and Anthropic client.
 * Accepts any model name and lets the API validate it. If the model is in KNOWN_CLAUDE_MODELS, uses that modelRef
 * for better defaults; otherwise creates a generic model reference.
 */
export function claudeModel(
  params: ClaudeModelParams
): ModelAction<ConfigSchemaType> {
  const { name, client: runnerClient, defaultApiVersion: apiVersion } = params;

  const ref = claudeModelReference(name);

  return model<ConfigSchemaType>(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema!,
    },
    claudeRunner(
      {
        name,
        client: runnerClient,
        defaultApiVersion: apiVersion,
      },
      ref.configSchema!
    )
  );
}

export const TEST_ONLY = { KNOWN_MODELS };
