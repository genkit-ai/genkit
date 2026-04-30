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

import { ActionRunOptions, GenkitError, z } from '@genkit-ai/core';
import type { Registry } from '@genkit-ai/core/registry';
import { toJsonSchema } from '@genkit-ai/core/schema';
import { GenkitAI } from '../genkit-ai.js';
import type { GenerateActionOptions } from '../model-types.js';
import { type MiddlewareRef } from '../model-types.js';
import {
  GenerateRequest,
  GenerateResponseChunkData,
  GenerateResponseData,
  ToolRequestPart,
  ToolResponsePart,
} from '../model.js';
import { type GenkitPluginV2 } from '../plugin.js';
import { ToolAction } from '../tool.js';

/** Descriptor for a registered middleware, returned by reflection API. */
export const MiddlewareDescSchema = z.object({
  /** Unique name of the middleware. */
  name: z.string(),
  /** Human-readable description of what the middleware does. */
  description: z.string().optional(),
  /** JSON Schema for the middleware's configuration. */
  configSchema: z.record(z.any()).nullish(),
  /** User defined metadata for the middleware. */
  metadata: z.record(z.any()).nullish(),
});
export type MiddlewareDesc = z.infer<typeof MiddlewareDescSchema>;

/**
 * Defines a Genkit Generate Middleware instance, which can be configured and registered.
 * When invoked with an optional configuration, it returns a reference suitable for
 * inclusion in a `GenerateOptions.use` array.
 */
export interface GenerateMiddleware<
  ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny,
  PluginOptions = void,
> extends MiddlewareDesc {
  /** Configures the middleware, returning a MiddlewareRef for usage in `generate({use: [...]})`. */
  (
    config?: z.infer<ConfigSchema>
  ): MiddlewareRef & { __def: GenerateMiddleware<ConfigSchema, PluginOptions> };
  /** The unique name of this middleware. */
  name: string;
  /** An optional description of what the middleware does. */
  description?: string;
  /** An optional Zod schema for validating the middleware's configuration. */
  configSchema?: ConfigSchema;
  /** Metadata describing this middleware. */
  metadata?: Record<string, any>;
  /**
   * Factory function that receives the validated configuration and creates
   * a `GenerateMiddlewareDef` holding the active hooks.
   */
  instantiate: (
    options: MiddlewareFnOptions<ConfigSchema, PluginOptions>
  ) => GenerateMiddlewareDef;
  /**
   * Optional plugin wrapper exposing this middleware for framework-level registration.
   */
  plugin: (pluginOptions: PluginOptions) => GenkitPluginV2;
  /** Generates a JSON-compatible representation of the middleware metadata. */
  toJson: () => MiddlewareDesc;
}

interface GenerateMiddlewarePluginInstance<
  ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny,
  PluginOptions = void,
> extends GenerateMiddleware<ConfigSchema, PluginOptions> {
  /** Options passed to the middleware plugin function. */
  pluginOptions: PluginOptions;
}

/**
 * An instantiated implementation of a Generate Middleware.
 * Provides optional hooks to intercept the high-level `generate` action,
 * the underlying `model` execution, or individual `tool` calls, as well as
 * tools to inject into the execution.
 */
export interface GenerateMiddlewareDef {
  /**
   * Hook for intercepting the top-level generate action.
   * Can be used to inject request parameters, modify the response, or catch errors.
   */
  generate?: (
    envelope: {
      request: GenerateActionOptions;
      currentTurn: number;
      messageIndex: number;
    },
    ctx: ActionRunOptions<GenerateResponseChunkData>,
    next: (
      envelope: {
        request: GenerateActionOptions;
        currentTurn: number;
        messageIndex: number;
      },
      ctx: ActionRunOptions<GenerateResponseChunkData>
    ) => Promise<GenerateResponseData>
  ) => Promise<GenerateResponseData>;
  /**
   * Hook for intercepting the underlying model execution.
   * Ideal for model-level caching, retry logic, or prompt/response parsing.
   */
  model?: (
    req: GenerateRequest<any>,
    ctx: ActionRunOptions<GenerateResponseChunkData>,
    next: (
      req: GenerateRequest<any>,
      ctx: ActionRunOptions<GenerateResponseChunkData>
    ) => Promise<GenerateResponseData>
  ) => Promise<GenerateResponseData>;
  /**
   * Hook for intercepting individual tool calls.
   * Enables caching tool responses, validating inputs, or overriding tool execution.
   */
  tool?: (
    req: ToolRequestPart,
    ctx: ActionRunOptions<any>,
    next: (
      req: ToolRequestPart,
      ctx: ActionRunOptions<any>
    ) => Promise<ToolResponsePart | undefined>
  ) => Promise<ToolResponsePart | undefined>;
  /**
   * Tools to statically inject into the generation request whenever this middleware is active.
   */
  tools?: ToolAction[];
}

/**
 * Options passed to the middleware factory function.
 */
export interface MiddlewareFnOptions<
  ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny,
  PluginOptions = void,
> {
  config: z.infer<ConfigSchema> | undefined;
  pluginConfig: PluginOptions;
  ai: GenkitAI;
}

export function generateMiddleware<
  PluginOptions = void,
  ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: {
    name: string;
    configSchema?: ConfigSchema;
    description?: string;
    metadata?: Record<string, any>;
  },
  middlewareFn: (
    options: MiddlewareFnOptions<ConfigSchema, PluginOptions>
  ) => GenerateMiddlewareDef
): GenerateMiddleware<ConfigSchema, PluginOptions> {
  const def = function (config?: z.infer<ConfigSchema>) {
    return {
      name: options.name,
      config,
      __def: def,
    };
  } as GenerateMiddleware<ConfigSchema, PluginOptions>;
  Object.defineProperty(def, 'name', { value: options.name });
  def.configSchema = options.configSchema;
  def.description = options.description;
  def.metadata = options.metadata;
  def.instantiate = middlewareFn;
  def.plugin = (pluginConfig: PluginOptions) => ({
    name: `middleware:${options.name}`,
    version: 'v2',
    middleware: () => {
      const wrappedDef = Object.assign(
        function (config?: z.infer<ConfigSchema>) {
          return def(config);
        },
        def,
        { pluginOptions: pluginConfig }
      ) as GenerateMiddlewarePluginInstance<ConfigSchema, PluginOptions>;
      Object.defineProperty(wrappedDef, 'name', { value: options.name });
      return [wrappedDef];
    },
    model: (_) => {
      throw new Error('Not supported for middleware plugins');
    },
  });

  def.toJson = () => ({
    name: options.name,
    description: options.description,
    configSchema: options.configSchema
      ? toJsonSchema({ schema: options.configSchema })
      : undefined,
    metadata: options.metadata,
  });

  return def;
}

export async function resolveMiddleware(
  registry: Registry,
  refs?: MiddlewareRef[]
): Promise<GenerateMiddlewareDef[]> {
  const result: GenerateMiddlewareDef[] = [];
  if (!refs) return result;
  const ai = new GenkitAI(registry);
  for (const ref of refs) {
    const def = await registry.lookupValue<GenerateMiddleware>(
      'middleware',
      ref.name
    );
    if (!def) {
      throw new GenkitError({
        status: 'NOT_FOUND',
        message: `Middleware ${ref.name} not found in registry.`,
      });
    }
    result.push(
      def.instantiate({
        config: ref.config,
        ai,
        pluginConfig: (def as GenerateMiddlewarePluginInstance).pluginOptions,
      })
    );
  }
  return result;
}
