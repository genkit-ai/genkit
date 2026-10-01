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

import { annotateSchema, generateMiddleware, z } from 'genkit';
import { lazyToolNames } from '../common/util.js';
import { ai } from '../genkit.js';

export const randomFail = generateMiddleware(
  {
    name: 'test/randomFail',
    description: 'Randomly throws an error for experimentation.',
    configSchema: z.object({
      probability: z
        .number()
        .min(0)
        .max(1)
        .default(0.5)
        .optional()
        .describe('Probability of failure between 0 and 1 (default 0.5).'),
      message: z
        .string()
        .default('Intentional random failure: weatherAgent experiment')
        .optional()
        .describe('Custom error message when failing.'),
    }),
  },
  ({ config }) => ({
    model: async (req, ctx, next) => {
      const probability = config?.probability ?? 0.5;
      if (Math.random() < probability) {
        throw new Error(
          config?.message ??
            'Intentional random failure: weatherAgent experiment'
        );
      }
      return next(req, ctx);
    },
  })
);

export const alwaysFail = generateMiddleware(
  {
    name: 'test/alwaysFail',
    description: 'Always throws an error on model or tool execution.',
    configSchema: z.object({
      target: z
        .enum(['model', 'tool'])
        .default('model')
        .optional()
        .describe('Whether to fail on model execution or tool execution.'),
      message: z
        .string()
        .default('Intentional failure from alwaysFail')
        .optional()
        .describe('Custom error message when failing.'),
    }),
  },
  ({ config }) => {
    const target = config?.target ?? 'model';
    const message = config?.message ?? 'Intentional failure from alwaysFail';
    return {
      model:
        target === 'model'
          ? async () => {
              throw new Error(message);
            }
          : undefined,
      tool:
        target === 'tool'
          ? async (req) => {
              throw new Error(`${message} (tool: ${req.toolRequest.name})`);
            }
          : undefined,
    };
  }
);

export const failTool = generateMiddleware(
  {
    name: 'test/failTool',
    description:
      'Fails a specific tool (or any tool if toolName is omitted), either by throwing an exception or returning an error payload to the model.',
    configSchema: z.object({
      toolName: annotateSchema(
        z
          .string()
          .optional()
          .describe(
            'Name of the tool to fail (e.g. "getWeather"). If omitted, applies to all tools.'
          ),
        {
          // Lazily evaluated via `toJSON()` when the reflection server
          // serializes the middleware schema for the Dev UI, after `ai`
          // and all tools have been registered.
          enum: lazyToolNames(() => ai),
        }
      ),
      probability: z
        .number()
        .min(0)
        .max(1)
        .default(1)
        .optional()
        .describe('Probability of failing the matching tool (default 1).'),
      mode: z
        .enum(['throw', 'returnError'])
        .default('throw')
        .optional()
        .describe(
          '"throw" throws an exception; "returnError" returns an { error } object as the tool output so the model can react.'
        ),
      message: z
        .string()
        .default('Intentional tool failure')
        .optional()
        .describe('Custom error message when the tool fails.'),
    }),
  },
  ({ config }) => ({
    tool: async (req, ctx, next) => {
      const targetTool = config?.toolName?.trim();
      const currentTool = req.toolRequest.name;
      const matches =
        !targetTool ||
        currentTool === targetTool ||
        currentTool.endsWith(`/${targetTool}`);

      if (matches) {
        const probability = config?.probability ?? 1;
        if (Math.random() < probability) {
          const errorMsg = `${config?.message ?? 'Intentional tool failure'} (tool: ${currentTool})`;
          if (config?.mode === 'returnError') {
            return {
              toolResponse: {
                name: currentTool,
                ref: req.toolRequest.ref,
                output: { error: errorMsg },
              },
            };
          }
          throw new Error(errorMsg);
        }
      }
      return next(req, ctx);
    },
  })
);

export const failFirstN = generateMiddleware(
  {
    name: 'test/failFirstN',
    description:
      'Fails the first N model attempts per generation, then succeeds (useful for deterministic retry testing).',
    configSchema: z.object({
      count: z
        .number()
        .int()
        .min(1)
        .default(1)
        .optional()
        .describe(
          'Number of initial model attempts to fail before succeeding.'
        ),
      message: z
        .string()
        .default('Intentional transient failure')
        .optional()
        .describe('Custom error message for transient failures.'),
    }),
  },
  ({ config }) => {
    let attempts = 0;
    const maxFailures = config?.count ?? 1;
    return {
      generate: async (envelope, ctx, next) => {
        attempts = 0;
        return next(envelope, ctx);
      },
      model: async (req, ctx, next) => {
        if (attempts < maxFailures) {
          attempts++;
          throw new Error(
            `${config?.message ?? 'Intentional transient failure'} (attempt ${attempts}/${maxFailures})`
          );
        }
        return next(req, ctx);
      },
    };
  }
);

export const delay = generateMiddleware(
  {
    name: 'test/delay',
    description: 'Adds artificial delay to model and/or tool execution.',
    configSchema: z.object({
      delayMs: z
        .number()
        .min(0)
        .default(2000)
        .optional()
        .describe('Delay in milliseconds before execution (default 2000).'),
      chunkDelayMs: z
        .number()
        .min(0)
        .optional()
        .describe('Optional delay in milliseconds between streaming chunks.'),
      target: z
        .enum(['model', 'tool', 'both'])
        .default('model')
        .optional()
        .describe('Whether to delay model calls, tool calls, or both.'),
    }),
  },
  ({ config }) => {
    const delayMs = config?.delayMs ?? 2000;
    const chunkDelayMs = config?.chunkDelayMs;
    const target = config?.target ?? 'model';
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

    return {
      model:
        target === 'model' || target === 'both'
          ? async (req, ctx, next) => {
              if (delayMs > 0) await sleep(delayMs);
              const wrappedCtx =
                chunkDelayMs && ctx.onChunk
                  ? {
                      ...ctx,
                      onChunk: async (chunk: any) => {
                        await sleep(chunkDelayMs);
                        return ctx.onChunk!(chunk);
                      },
                    }
                  : ctx;
              return next(req, wrappedCtx);
            }
          : undefined,
      tool:
        target === 'tool' || target === 'both'
          ? async (req, ctx, next) => {
              if (delayMs > 0) await sleep(delayMs);
              return next(req, ctx);
            }
          : undefined,
    };
  }
);

export const mockResponse = generateMiddleware(
  {
    name: 'test/mockResponse',
    description:
      'Short-circuits the LLM call and returns a canned or echoed text response.',
    configSchema: z.object({
      text: z
        .string()
        .default('This is a mocked response from middleware.')
        .optional()
        .describe('Canned text response to return when echoPrompt is false.'),
      echoPrompt: z
        .boolean()
        .default(false)
        .optional()
        .describe(
          'If true, echoes back the user prompt instead of fixed text.'
        ),
    }),
  },
  ({ config }) => ({
    model: async (req) => {
      const lastText = req.messages
        .at(-1)
        ?.content.map((p) => p.text ?? '')
        .join('');
      const outputText = config?.echoPrompt
        ? `Echo: ${lastText}`
        : (config?.text ?? 'This is a mocked response from middleware.');
      return {
        message: {
          role: 'model',
          content: [{ text: outputText }],
        },
        finishReason: 'stop',
      };
    },
  })
);
