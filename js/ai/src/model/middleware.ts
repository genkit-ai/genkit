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

import { GenkitError, StatusName } from '@genkit-ai/core';
import { HasRegistry } from '@genkit-ai/core/registry';
import { Document } from '../document.js';
import { injectInstructions } from '../formats/index.js';
import { ModelArgument } from '../index.js';
import type {
  MediaPart,
  MessageData,
  ModelInfo,
  ModelMiddleware,
  Part,
} from '../model.js';
import { resolveModel } from '../model.js';

/**
 * Preprocess a GenerateRequest to download referenced http(s) media URLs and
 * inline them as data URIs.
 */
export function downloadRequestMedia(options?: {
  maxBytes?: number;
  filter?: (part: MediaPart) => boolean;
}): ModelMiddleware {
  return async (req, next) => {
    const { default: fetch } = await import('node-fetch');

    const newReq = {
      ...req,
      messages: await Promise.all(
        req.messages.map(async (message) => {
          const content: Part[] = await Promise.all(
            message.content.map(async (part) => {
              // skip non-media parts and non-http urls, or parts that have been
              // filtered out by user config
              if (
                !part.media ||
                !part.media.url.startsWith('http') ||
                (options?.filter && !options?.filter(part))
              ) {
                return part;
              }

              const response = await fetch(part.media.url, {
                size: options?.maxBytes,
              });
              if (response.status !== 200)
                throw new Error(
                  `HTTP error downloading media '${
                    part.media.url
                  }': ${await response.text()}`
                );

              // use provided contentType or sniff from response
              const contentType =
                part.media.contentType ||
                response.headers.get('content-type') ||
                '';

              return {
                media: {
                  contentType,
                  url: `data:${contentType};base64,${Buffer.from(
                    await response.arrayBuffer()
                  ).toString('base64')}`,
                },
              };
            })
          );

          return {
            ...message,
            content,
          };
        })
      ),
    };

    return next(newReq);
  };
}

/**
 * Validates that a GenerateRequest does not include unsupported features.
 */
export function validateSupport(options: {
  name: string;
  supports?: ModelInfo['supports'];
}): ModelMiddleware {
  const supports = options.supports || {};
  return async (req, next) => {
    function invalid(message: string): never {
      throw new Error(
        `Model '${
          options.name
        }' does not support ${message}. Request: ${JSON.stringify(
          req,
          null,
          2
        )}`
      );
    }

    if (
      supports.media === false &&
      req.messages.some((message) => message.content.some((part) => part.media))
    )
      invalid('media, but media was provided');
    if (supports.tools === false && req.tools?.length)
      invalid('tool use, but tools were provided');
    if (supports.multiturn === false && req.messages.length > 1)
      invalid(`multiple messages, but ${req.messages.length} were provided`);
    // if (
    //   typeof supports.output !== 'undefined' &&
    //   req.output?.format &&
    //   !supports.output.includes(req.output?.format)
    // )
    //   invalid(`requested output format '${req.output?.format}'`);
    return next();
  };
}

// N.B. Figure out why array.findLast isn't available despite setting target
// to ES2022 (Node 16.14.0)
function lastUserMessage(messages: MessageData[]) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') {
      return messages[i];
    }
  }
  return undefined;
}

/**
 * Provide a simulated system prompt for models that don't support it natively.
 */
export function simulateSystemPrompt(options?: {
  preface: string;
  acknowledgement: string;
}): ModelMiddleware {
  const preface = options?.preface || 'SYSTEM INSTRUCTIONS:\n';
  const acknowledgement = options?.acknowledgement || 'Understood.';

  return (req, next) => {
    const messages = [...req.messages];
    for (let i = 0; i < messages.length; i++) {
      if (req.messages[i].role === 'system') {
        const systemPrompt = messages[i].content;
        messages.splice(
          i,
          1,
          { role: 'user', content: [{ text: preface }, ...systemPrompt] },
          { role: 'model', content: [{ text: acknowledgement }] }
        );
        break;
      }
    }
    return next({ ...req, messages });
  };
}

export interface AugmentWithContextOptions {
  /** Preceding text to place before the rendered context documents. */
  preface?: string | null;
  /** A function to render a document into a text part to be included in the message. */
  itemTemplate?: (d: Document, options?: AugmentWithContextOptions) => string;
  /** The metadata key to use for citation reference. Pass `null` to provide no citations. */
  citationKey?: string | null;
}

export const CONTEXT_PREFACE =
  '\n\nUse the following information to complete your task:\n\n';
const CONTEXT_ITEM_TEMPLATE = (
  d: Document,
  index: number,
  options?: AugmentWithContextOptions
) => {
  let out = '- ';
  if (options?.citationKey) {
    out += `[${d.metadata![options.citationKey]}]: `;
  } else if (options?.citationKey === undefined) {
    out += `[${d.metadata?.['ref'] || d.metadata?.['id'] || index}]: `;
  }
  out += d.text + '\n';
  return out;
};

export function augmentWithContext(
  options?: AugmentWithContextOptions
): ModelMiddleware {
  const preface =
    typeof options?.preface === 'undefined' ? CONTEXT_PREFACE : options.preface;
  const itemTemplate = options?.itemTemplate || CONTEXT_ITEM_TEMPLATE;
  return (req, next) => {
    // if there is no context in the request, no-op
    if (!req.docs?.length) return next(req);
    const userMessage = lastUserMessage(req.messages);
    // if there are no messages, no-op
    if (!userMessage) return next(req);
    // if there is already a context part, no-op
    const contextPartIndex = userMessage?.content.findIndex(
      (p) => p.metadata?.purpose === 'context'
    );
    const contextPart =
      contextPartIndex >= 0 && userMessage.content[contextPartIndex];

    if (contextPart && !contextPart.metadata?.pending) {
      return next(req);
    }
    let out = `${preface || ''}`;
    req.docs?.forEach((d, i) => {
      out += itemTemplate(new Document(d), i, options);
    });
    out += '\n';
    if (contextPartIndex >= 0) {
      userMessage.content[contextPartIndex] = {
        ...contextPart,
        text: out,
        metadata: { purpose: 'context' },
      } as Part;
    } else {
      userMessage.content.push({ text: out, metadata: { purpose: 'context' } });
    }

    return next(req);
  };
}

/**
 * Options for the `retry` middleware.
 */
export interface RetryOptions {
  /**
   * The maximum number of times to retry a failed request.
   * @default 3
   */
  maxRetries?: number;
  /**
   * An array of `StatusName` values that should trigger a retry.
   * @default ['UNAVAILABLE', 'DEADLINE_EXCEEDED', 'RESOURCE_EXHAUSTED', 'ABORTED', 'INTERNAL']
   */
  statuses?: StatusName[];
  /**
   * The initial delay between retries, in milliseconds.
   * @default 1000
   */
  initialDelayMs?: number;
  /**
   * The maximum delay between retries, in milliseconds.
   * @default 60000
   */
  maxDelayMs?: number;
  /**
   * The factor by which the delay increases after each retry (exponential backoff).
   * @default 2
   */
  backoffFactor?: number;
  /**
   * Whether to disable jitter on the delay. Jitter adds a random factor to the
   * delay to help prevent a "thundering herd" of clients all retrying at the
   * same time.
   * @default false
   */
  noJitter?: boolean;
  /**
   * A callback to be executed on each retry attempt.
   */
  onError?: (error: Error, attempt: number) => void;
}

let __setTimeout: (
  callback: (...args: any[]) => void,
  ms?: number
) => NodeJS.Timeout = setTimeout;

/**
 * FOR TESTING ONLY.
 * @internal
 */
export const TEST_ONLY = {
  setRetryTimeout(
    impl: (callback: (...args: any[]) => void, ms?: number) => NodeJS.Timeout
  ) {
    __setTimeout = impl;
  },
};

const DEFAULT_RETRY_STATUSES: StatusName[] = [
  'UNAVAILABLE',
  'DEADLINE_EXCEEDED',
  'RESOURCE_EXHAUSTED',
  'ABORTED',
  'INTERNAL',
];

const DEFAULT_FALLBACK_STATUSES: StatusName[] = [
  'UNAVAILABLE',
  'DEADLINE_EXCEEDED',
  'RESOURCE_EXHAUSTED',
  'ABORTED',
  'INTERNAL',
  'NOT_FOUND',
  'UNIMPLEMENTED',
];

/**
 * Creates a middleware that retries requests on specific error statuses.
 *
 * ```ts
 * const { text } = await ai.generate({
 *   model: googleAI.model('gemini-pro-latest'),
 *   prompt: 'You are a helpful AI assistant named Walt, say hello',
 *   use: [
 *     retry({
 *       maxRetries: 2,
 *       initialDelayMs: 1000,
 *       backoffFactor: 2,
 *     }),
 *   ],
 * });
 * ```
 */
export function retry(options: RetryOptions = {}): ModelMiddleware {
  const {
    maxRetries = 3,
    statuses = DEFAULT_RETRY_STATUSES,
    initialDelayMs = 1000,
    maxDelayMs = 60000,
    backoffFactor = 2,
    noJitter = false,
    onError,
  } = options;

  return async (req, next) => {
    let lastError: any;
    let currentDelay = initialDelayMs;
    for (let i = 0; i <= maxRetries; i++) {
      try {
        return await next(req);
      } catch (e) {
        lastError = e;
        const error = e as Error;
        if (i < maxRetries) {
          let shouldRetry = false;
          if (error instanceof GenkitError) {
            if (statuses.includes(error.status)) {
              shouldRetry = true;
            }
          } else {
            shouldRetry = true;
          }

          if (shouldRetry) {
            onError?.(error, i + 1);
            let delay = currentDelay;
            if (!noJitter) {
              delay = delay + 1000 * Math.pow(2, i) * Math.random();
            }
            await new Promise((resolve) => __setTimeout(resolve, delay));
            currentDelay = Math.min(currentDelay * backoffFactor, maxDelayMs);
            continue;
          }
        }
        throw error;
      }
    }
    throw lastError;
  };
}

/**
 * Options for the `fallback` middleware.
 */
export interface FallbackOptions {
  /**
   * An array of models to try in order.
   */
  models: ModelArgument[];
  /**
   * An array of `StatusName` values that should trigger a fallback.
   * @default ['UNAVAILABLE', 'DEADLINE_EXCEEDED', 'RESOURCE_EXHAUSTED', 'ABORTED', 'INTERNAL', 'NOT_FOUND', 'UNIMPLEMENTED']
   */
  statuses?: StatusName[];
  /**
   * A callback to be executed on each fallback attempt.
   */
  onError?: (error: Error) => void;
}

/**
 * Creates a middleware that falls back to a different model on specific error statuses.
 *
 * ```ts
 * const { text } = await ai.generate({
 *   model: googleAI.model('gemini-pro-latest'),
 *   prompt: 'You are a helpful AI assistant named Walt, say hello',
 *   use: [
 *     fallback(ai, {
 *       models: [googleAI.model('gemini-flash-latest')],
 *       statuses: ['RESOURCE_EXHAUSTED'],
 *     }),
 *   ],
 * });
 * ```
 */
export function fallback(
  ai: HasRegistry,
  options: FallbackOptions
): ModelMiddleware {
  const { models, statuses = DEFAULT_FALLBACK_STATUSES, onError } = options;

  return async (req, next) => {
    try {
      return await next(req);
    } catch (e) {
      if (e instanceof GenkitError && statuses.includes(e.status)) {
        onError?.(e);
        let lastError: any = e;
        for (const model of models) {
          try {
            const resolved = await resolveModel(ai.registry, model);
            return await resolved.modelAction(req);
          } catch (e2) {
            lastError = e2;
            if (e2 instanceof GenkitError && statuses.includes(e2.status)) {
              onError?.(e2);
              continue;
            }
            throw e2;
          }
        }
        throw lastError;
      }
      throw e;
    }
  };
}

export interface SimulatedConstrainedGenerationOptions {
  instructionsRenderer?: (schema: Record<string, any>) => string;
}

const DEFAULT_CONSTRAINED_GENERATION_INSTRUCTIONS = (
  schema: Record<string, any>
) => `Output should be in JSON format and conform to the following schema:

\`\`\`
${JSON.stringify(schema)}
\`\`\`
`;

/**
 * Model middleware that simulates constrained generation by injecting generation
 * instructions into the user message.
 */
export function simulateConstrainedGeneration(
  options?: SimulatedConstrainedGenerationOptions
): ModelMiddleware {
  return (req, next) => {
    let instructions: string | undefined;
    if (req.output?.constrained && req.output?.schema) {
      instructions = (
        options?.instructionsRenderer ??
        DEFAULT_CONSTRAINED_GENERATION_INSTRUCTIONS
      )(req.output?.schema);

      req = {
        ...req,
        messages: injectInstructions(req.messages, instructions),
        output: {
          ...req.output,
          // we're simulating it, so to the underlying model it's unconstrained.
          constrained: false,
          format: undefined,
          contentType: undefined,
          schema: undefined,
        },
      };
    }

    return next(req);
  };
}
