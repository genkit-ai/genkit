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
  GenkitError,
  ModelReferenceSchema,
  generateMiddleware,
  z,
  type GenerateMiddleware,
  type StatusName,
} from 'genkit';
import { ModelAction } from 'genkit/model';
import { Registry } from 'genkit/registry';

const DEFAULT_FALLBACK_STATUSES: StatusName[] = [
  'UNAVAILABLE',
  'DEADLINE_EXCEEDED',
  'RESOURCE_EXHAUSTED',
  'ABORTED',
  'INTERNAL',
  'NOT_FOUND',
  'UNIMPLEMENTED',
];

export const FallbackOptionsSchema = z
  .object({
    /**
     * An array of models to try in order.
     */
    models: z
      .array(ModelReferenceSchema)
      .describe('An array of models to try in order.'),
    /**
     * An array of `StatusName` values that should trigger a fallback.
     * @default ['UNAVAILABLE', 'DEADLINE_EXCEEDED', 'RESOURCE_EXHAUSTED', 'ABORTED', 'INTERNAL', 'NOT_FOUND', 'UNIMPLEMENTED']
     */
    statuses: z
      .array(z.string())
      .optional()
      .describe(
        'An array of StatusName values that should trigger a fallback.'
      ),
    /**
     * If true, the fallback model will not inherit the original request's configuration.
     * @default false
     */
    isolateConfig: z
      .boolean()
      .optional()
      .describe(
        "If true, the fallback model will not inherit the original request's configuration."
      ),
  })
  .passthrough();

export type FallbackOptions = z.infer<typeof FallbackOptionsSchema>;

/**
 * Creates a middleware that falls back to a different model on specific error statuses.
 *
 * ```ts
 * const { text } = await ai.generate({
 *   model: googleAI.model('gemini-2.5-pro'),
 *   prompt: 'You are a helpful AI assistant named Walt, say hello',
 *   use: [
 *     fallback({
 *       models: [googleAI.model('gemini-2.5-flash')],
 *       statuses: ['RESOURCE_EXHAUSTED'],
 *     }),
 *   ],
 * });
 * ```
 */
export const fallback: GenerateMiddleware<typeof FallbackOptionsSchema> =
  generateMiddleware(
    {
      name: 'fallback',
      description: 'Fallback to a different model on specific error statuses.',
      configSchema: FallbackOptionsSchema,
    },
    (options) => {
      const {
        models = [],
        statuses = DEFAULT_FALLBACK_STATUSES,
        isolateConfig = false,
      } = options.config || {};

      return {
        model: async (req, ctx, next) => {
          try {
            return await next(req, ctx);
          } catch (e) {
            if (
              e instanceof GenkitError &&
              statuses.includes(e.status as StatusName)
            ) {
              let lastError: any = e;
              for (const model of models) {
                const normalizedModel = await resolveModel(
                  options.ai.registry,
                  model
                );
                try {
                  return await normalizedModel.model(
                    {
                      ...req,
                      config: isolateConfig
                        ? normalizedModel.config
                        : (normalizedModel.config ?? req.config),
                    },
                    ctx
                  );
                } catch (e2) {
                  lastError = e2;
                  if (
                    e2 instanceof GenkitError &&
                    statuses.includes(e2.status as StatusName)
                  ) {
                    continue;
                  }
                  throw e2;
                }
              }
              throw lastError;
            }
            throw e;
          }
        },
      };
    }
  );

async function resolveModel(
  registry: Registry,
  model: z.infer<typeof ModelReferenceSchema>
): Promise<{ model: ModelAction; config?: any }> {
  return {
    model: await registry.lookupAction(`/model/${model.name}`),
    config: model.config,
  };
}
