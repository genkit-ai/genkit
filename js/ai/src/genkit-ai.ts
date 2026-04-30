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

import {
  getContext,
  run,
  z,
  type ActionContext,
  type Operation,
} from '@genkit-ai/core';
import { type Registry } from '@genkit-ai/core/registry';
import { cancelOperation } from './cancel-operation.js';
import { checkOperation } from './check-operation.js';
import { type DocumentData } from './document.js';
import {
  embed,
  embedMany,
  type EmbedderArgument,
  type EmbedderParams,
  type Embedding,
  type EmbeddingBatch,
} from './embedder.js';
import {
  generate,
  generateStream,
  type GenerateOptions,
  type GenerateResponse,
  type GenerateStreamOptions,
  type GenerateStreamResponse,
} from './generate.js';
import { GenerationCommonConfigSchema, type Part } from './model-types.js';

/**
 * `GenkitAI` encapsulates Genkit's AI APIs.
 */
export class GenkitAI {
  readonly registry: Registry;

  constructor(registry: Registry) {
    this.registry = registry;
  }

  /**
   * Embeds the given `content` using the specified `embedder`.
   */
  embed<CustomOptions extends z.ZodTypeAny>(
    params: EmbedderParams<CustomOptions>
  ): Promise<Embedding[]> {
    return embed(this.registry, params);
  }

  /**
   * A veneer for interacting with embedder models in bulk.
   */
  embedMany<ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny>(params: {
    embedder: EmbedderArgument<ConfigSchema>;
    content: string[] | DocumentData[];
    metadata?: Record<string, unknown>;
    options?: z.infer<ConfigSchema>;
  }): Promise<EmbeddingBatch> {
    return embedMany(this.registry, params);
  }

  /**
   * Make a generate call to the default model with a simple text prompt.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [googleAI()],
   *   model: googleAI.model('gemini-flash-latest'), // default model
   * })
   *
   * const { text } = await ai.generate('hi');
   * ```
   */
  generate<O extends z.ZodTypeAny = z.ZodTypeAny>(
    strPrompt: string
  ): Promise<GenerateResponse<z.infer<O>>>;

  /**
   * Make a generate call to the default model with a multipart request.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [googleAI()],
   *   model: googleAI.model('gemini-flash-latest'), // default model
   * })
   *
   * const { text } = await ai.generate([
   *   { media: {url: 'http://....'} },
   *   { text: 'describe this image' }
   * ]);
   * ```
   */
  generate<O extends z.ZodTypeAny = z.ZodTypeAny>(
    parts: Part[]
  ): Promise<GenerateResponse<z.infer<O>>>;

  /**
   * Generate calls a generative model based on the provided prompt and configuration. If
   * `messages` is provided, the generation will include a conversation history in its
   * request. If `tools` are provided, the generate method will automatically resolve
   * tool calls returned from the model unless `returnToolRequests` is set to `true`.
   *
   * See {@link GenerateOptions} for detailed information about available options.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [googleAI()],
   * })
   *
   * const { text } = await ai.generate({
   *   system: 'talk like a pirate',
   *   prompt: [
   *     { media: { url: 'http://....' } },
   *     { text: 'describe this image' }
   *   ],
   *   messages: conversationHistory,
   *   tools: [ userInfoLookup ],
   *   model: googleAI.model('gemini-flash-latest'),
   * });
   * ```
   */
  generate<
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
  >(
    opts:
      | GenerateOptions<O, CustomOptions>
      | PromiseLike<GenerateOptions<O, CustomOptions>>
  ): Promise<GenerateResponse<z.infer<O>>>;

  async generate<
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
  >(
    options:
      | string
      | Part[]
      | GenerateOptions<O, CustomOptions>
      | PromiseLike<GenerateOptions<O, CustomOptions>>
  ): Promise<GenerateResponse<z.infer<O>>> {
    let resolvedOptions: GenerateOptions<O, CustomOptions>;
    if (options instanceof Promise) {
      resolvedOptions = await options;
    } else if (typeof options === 'string' || Array.isArray(options)) {
      resolvedOptions = {
        prompt: options,
      };
    } else {
      resolvedOptions = options as GenerateOptions<O, CustomOptions>;
    }
    return generate(this.registry, resolvedOptions);
  }

  /**
   * Make a streaming generate call to the default model with a simple text prompt.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [googleAI()],
   *   model: googleAI.model('gemini-flash-latest'), // default model
   * })
   *
   * const { response, stream } = ai.generateStream('hi');
   * for await (const chunk of stream) {
   *   console.log(chunk.text);
   * }
   * console.log((await response).text);
   * ```
   */
  generateStream<O extends z.ZodTypeAny = z.ZodTypeAny>(
    strPrompt: string
  ): GenerateStreamResponse<z.infer<O>>;

  /**
   * Make a streaming generate call to the default model with a multipart request.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [googleAI()],
   *   model: googleAI.model('gemini-flash-latest'), // default model
   * })
   *
   * const { response, stream } = ai.generateStream([
   *   { media: {url: 'http://....'} },
   *   { text: 'describe this image' }
   * ]);
   * for await (const chunk of stream) {
   *   console.log(chunk.text);
   * }
   * console.log((await response).text);
   * ```
   */
  generateStream<O extends z.ZodTypeAny = z.ZodTypeAny>(
    parts: Part[]
  ): GenerateStreamResponse<z.infer<O>>;

  /**
   * Streaming generate calls a generative model based on the provided prompt and configuration. If
   * `messages` is provided, the generation will include a conversation history in its
   * request. If `tools` are provided, the generate method will automatically resolve
   * tool calls returned from the model unless `returnToolRequests` is set to `true`.
   *
   * See {@link GenerateOptions} for detailed information about available options.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [googleAI()],
   * })
   *
   * const { response, stream } = ai.generateStream({
   *   system: 'talk like a pirate',
   *   prompt: [
   *     { media: { url: 'http://....' } },
   *     { text: 'describe this image' }
   *   ],
   *   messages: conversationHistory,
   *   tools: [ userInfoLookup ],
   *   model: googleAI.model('gemini-flash-latest'),
   * });
   * for await (const chunk of stream) {
   *   console.log(chunk.text);
   * }
   * console.log((await response).text);
   * ```
   */
  generateStream<
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
  >(
    opts:
      | GenerateStreamOptions<O, CustomOptions>
      | PromiseLike<GenerateStreamOptions<O, CustomOptions>>
  ): GenerateStreamResponse<z.infer<O>>;

  generateStream<
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
  >(
    options:
      | string
      | Part[]
      | GenerateStreamOptions<O, CustomOptions>
      | PromiseLike<GenerateStreamOptions<O, CustomOptions>>
  ): GenerateStreamResponse<z.infer<O>> {
    if (typeof options === 'string' || Array.isArray(options)) {
      options = { prompt: options };
    }
    return generateStream(this.registry, options);
  }

  /**
   * Checks the status of of a given operation. Returns a new operation which will contain the updated status.
   *
   * ```ts
   * let operation = await ai.generateOperation({
   *   model: googleAI.model('veo-2.0-generate-001'),
   *   prompt: 'A banana riding a bicycle.',
   * });
   *
   * while (!operation.done) {
   *   operation = await ai.checkOperation(operation!);
   *   await new Promise((resolve) => setTimeout(resolve, 5000));
   * }
   * ```
   *
   * @param operation
   * @returns
   */
  checkOperation<T>(operation: Operation<T>): Promise<Operation<T>> {
    return checkOperation(this.registry, operation);
  }

  /**
   * Cancels a given operation. Returns a new operation which will contain the updated status.
   *
   * @param operation
   * @returns
   */
  cancelOperation<T>(operation: Operation<T>): Promise<Operation<T>> {
    return cancelOperation(this.registry, operation);
  }

  /**
   * A flow step that executes the provided function. Each run step is recorded separately in the trace.
   *
   * ```ts
   * ai.defineFlow('hello', async() => {
   *   await ai.run('step1', async () => {
   *     // ... step 1
   *   });
   *   await ai.run('step2', async () => {
   *     // ... step 2
   *   });
   *   return result;
   * })
   * ```
   */
  run<T>(name: string, func: () => Promise<T>): Promise<T>;

  /**
   * A flow step that executes the provided function. Each run step is recorded separately in the trace.
   *
   * ```ts
   * ai.defineFlow('hello', async(name) => {
   *   const greeting = await ai.run('step1', name, async (input) => {
   *     return `Hello, ${input}!`;
   *   });
   *   const result = await ai.run('step2', greeting, async (input) => {
   *     // ... step 2
   *   });
   *   return result;
   * })
   */
  run<T>(
    name: string,
    input: any,
    func: (input?: any) => Promise<T>
  ): Promise<T>;

  run<T>(
    name: string,
    funcOrInput: () => Promise<T> | any,
    maybeFunc?: (input?: any) => Promise<T>
  ): Promise<T> {
    if (maybeFunc) {
      return run(name, funcOrInput, maybeFunc, this.registry);
    }
    return run(name, funcOrInput, this.registry);
  }

  /**
   * Returns current action (or flow) invocation context. Can be used to access things like auth
   * data set by HTTP server frameworks. If invoked outside of an action (e.g. flow or tool) will
   * return `undefined`.
   */
  currentContext(): ActionContext | undefined {
    return getContext();
  }
}
