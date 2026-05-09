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

import {
  defineInterrupt,
  defineResource,
  generateOperation,
  GenerateOptions,
  GenerateResponseData,
  GenerationCommonConfigSchema,
  isExecutablePrompt,
  ResourceAction,
  ResourceFn,
  ResourceOptions,
  type ExecutablePrompt,
  type InterruptConfig,
  type ToolAction,
} from '@genkit-ai/ai';
import type { Chat, ChatOptions } from '@genkit-ai/ai/chat';
import { defineFormat } from '@genkit-ai/ai/formats';
import {
  getCurrentSession,
  Session,
  SessionError,
  type SessionData,
  type SessionOptions,
} from '@genkit-ai/ai/session';
import type { Operation, z } from '@genkit-ai/core';
import { v4 as uuidv4 } from 'uuid';
import type { Formatter } from './formats';
import { Genkit, type GenkitOptions } from './genkit';

export type { GenkitOptions as GenkitBetaOptions }; // in case they drift later

/**
 * WARNING: these APIs are considered unstable and subject to frequent breaking changes that may not honor semver.
 *
 * Initializes Genkit BETA APIs with a set of options.
 *
 * This will create a new Genkit registry, register the provided plugins, stores, and other configuration. This
 * should be called before any flows are registered.
 *
 * @beta
 */
export function genkit(options: GenkitOptions): GenkitBeta {
  return new GenkitBeta(options);
}

/**
 * Genkit BETA APIs.
 *
 * @beta
 */
export class GenkitBeta extends Genkit {
  constructor(options?: GenkitOptions) {
    super(options);
    this.registry.apiStability = 'beta';
  }

  /**
   * Create a chat session with the provided options.
   *
   * ```ts
   * const chat = ai.chat({
   *   system: 'talk like a pirate',
   * })
   * let response = await chat.send('tell me a joke')
   * response = await chat.send('another one')
   * ```
   *
   * @beta
   */
  chat<I>(options?: ChatOptions<I>): Chat;

  /**
   * Create a chat session with the provided preamble.
   *
   * ```ts
   * const triageAgent = ai.definePrompt({
   *   system: 'help the user triage a problem',
   * })
   * const chat = ai.chat(triageAgent)
   * const { text } = await chat.send('my phone feels hot');
   * ```
   *
   * @beta
   */
  chat<I>(preamble: ExecutablePrompt<I>, options?: ChatOptions<I>): Chat;

  /**
   * Create a chat session with the provided options.
   *
   * ```ts
   * const chat = ai.chat({
   *   system: 'talk like a pirate',
   * })
   * let response = await chat.send('tell me a joke')
   * response = await chat.send('another one')
   * ```
   *
   * @beta
   */
  chat<I>(
    preambleOrOptions?: ChatOptions<I> | ExecutablePrompt<I>,
    maybeOptions?: ChatOptions<I>
  ): Chat {
    let options: ChatOptions<I> | undefined;
    let preamble: ExecutablePrompt<I> | undefined;
    if (maybeOptions) {
      options = maybeOptions;
    }
    if (preambleOrOptions) {
      if (isExecutablePrompt(preambleOrOptions)) {
        preamble = preambleOrOptions as ExecutablePrompt<I>;
      } else {
        options = preambleOrOptions as ChatOptions<I>;
      }
    }

    const session = this.createSession();
    if (preamble) {
      return session.chat(preamble, options);
    }
    return session.chat(options);
  }

  /**
   * Create a session for this environment.
   */
  createSession<S = any>(options?: SessionOptions<S>): Session<S> {
    const sessionId = options?.sessionId?.trim() || uuidv4();
    const sessionData: SessionData = {
      id: sessionId,
      state: options?.initialState,
    };
    return new Session(this.registry, {
      id: sessionId,
      sessionData,
      store: options?.store,
    });
  }

  /**
   * Loads a session from the store.
   *
   * @beta
   */
  async loadSession(
    sessionId: string,
    options: SessionOptions
  ): Promise<Session> {
    if (!options.store) {
      throw new Error('options.store is required');
    }
    const sessionData = await options.store.get(sessionId);

    return new Session(this.registry, {
      id: sessionId,
      sessionData,
      store: options.store,
    });
  }

  /**
   * Gets the current session from async local storage.
   *
   * @beta
   */
  currentSession<S = any>(): Session<S> {
    const currentSession = getCurrentSession(this.registry);
    if (!currentSession) {
      throw new SessionError('not running within a session');
    }
    return currentSession as Session;
  }

  /**
   * Defines and registers a custom model output formatter.
   *
   * Here's an example of a custom JSON output formatter:
   *
   * ```ts
   * import { extractJson } from 'genkit/extract';
   *
   * ai.defineFormat(
   *   { name: 'customJson' },
   *   (schema) => {
   *     let instructions: string | undefined;
   *     if (schema) {
   *       instructions = `Output should be in JSON format and conform to the following schema:
   * \`\`\`
   * ${JSON.stringify(schema)}
   * \`\`\`
   * `;
   *     }
   *     return {
   *       parseChunk: (chunk) => extractJson(chunk.accumulatedText),
   *       parseMessage: (message) => extractJson(message.text),
   *       instructions,
   *     };
   *   }
   * );
   *
   * const { output } = await ai.generate({
   *   prompt: 'Invent a menu item for a pirate themed restaurant.',
   *   output: { format: 'customJson', schema: MenuItemSchema },
   * });
   * ```
   *
   * @beta
   */
  defineFormat(
    options: {
      name: string;
    } & Formatter['config'],
    handler: Formatter['handler']
  ): { config: Formatter['config']; handler: Formatter['handler'] } {
    return defineFormat(this.registry, options, handler);
  }

  /**
   * Defines and registers an interrupt.
   *
   * Interrupts are special tools that halt model processing and return control back to the caller. Interrupts make it simpler to implement
   * "human-in-the-loop" and out-of-band processing patterns that require waiting on external actions to complete.
   *
   * @beta
   */
  defineInterrupt<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    config: InterruptConfig<I, O>
  ): ToolAction<I, O> {
    return defineInterrupt(this.registry, config);
  }

  /**
   * Starts a generate operation for long running generation models, typically for
   * video and complex audio generation.
   *
   * See {@link GenerateOptions} for detailed information about available options.
   *
   * ```ts
   * const operation = await ai.generateOperation({
   *   model: googleAI.model('veo-2.0-generate-001'),
   *   prompt: 'A banana riding a bicycle.',
   * });
   * ```
   *
   * The status of the operation and final result can be obtained using {@link Genkit.checkOperation}.
   */
  generateOperation<
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
  >(
    opts:
      | GenerateOptions<O, CustomOptions>
      | PromiseLike<GenerateOptions<O, CustomOptions>>
  ): Promise<Operation<GenerateResponseData>> {
    return generateOperation(this.registry, opts);
  }

  /**
   * Defines a resource. Resources can then be accessed from a generate call.
   *
   * ```ts
   * ai.defineResource({
   *   uri: 'my://resource/{param}',
   *   description: 'provides my resource',
   * }, async ({param}) => {
   *   return [{ text: `resource ${param}` }]
   * });
   *
   * await ai.generate({
   *   prompt: [{ resource: 'my://resource/value' }]
   * })
   */
  defineResource(opts: ResourceOptions, fn: ResourceFn): ResourceAction {
    return defineResource(this.registry, opts, fn);
  }
}
