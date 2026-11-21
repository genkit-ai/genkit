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

import {
  generateMiddleware,
  z,
  type GenerateMiddleware,
  type GenerateResponseData,
  type Part,
} from 'genkit';
import type { GoogleAuthOptions } from 'google-auth-library';
import { initializeAuth } from './auth.js';
import { Guardrails } from './guardrails.js';
import {
  ChecksEvaluationMetricType,
  type ChecksEvaluationMetric,
} from './metrics.js';

export const ChecksMiddlewareOptionsSchema = z.object({
  /** The Checks policies to classify input and output text against. */
  metrics: z
    .array(
      z.union([
        z.enum(ChecksEvaluationMetricType),
        z.object({
          type: z.enum(ChecksEvaluationMetricType),
          threshold: z.number().optional(),
        }),
      ])
    )
    .describe('Checks policies to classify input and output text against.'),
});

export type ChecksMiddlewareOptions = z.infer<
  typeof ChecksMiddlewareOptionsSchema
>;

/**
 * Instance-level options for {@link checksMiddleware}, set once with
 * `checksMiddleware.plugin(...)`. Kept out of the per-call config so
 * credentials never end up in traces.
 */
export interface ChecksMiddlewarePluginOptions {
  /**
   * Google auth configuration for the Checks API. `projectId` is the project
   * charged for quota. Defaults to Application Default Credentials.
   */
  authOptions?: GoogleAuthOptions;
}

// One Guardrails client per plugin instance, and a shared default for when
// the middleware is used without `.plugin(...)`, so auth is not set up again
// on every generate call.
const guardrailsByPluginOptions = new WeakMap<
  ChecksMiddlewarePluginOptions,
  Guardrails
>();
let defaultGuardrails: Guardrails | undefined;

function guardrailsFor(
  pluginOptions: ChecksMiddlewarePluginOptions | undefined
): Guardrails {
  if (!pluginOptions) {
    defaultGuardrails ??= new Guardrails(initializeAuth());
    return defaultGuardrails;
  }
  let guardrails = guardrailsByPluginOptions.get(pluginOptions);
  if (!guardrails) {
    guardrails = new Guardrails(
      initializeAuth(pluginOptions.authOptions),
      pluginOptions.authOptions?.projectId
    );
    guardrailsByPluginOptions.set(pluginOptions, guardrails);
  }
  return guardrails;
}

/**
 * Generate middleware that classifies model input and output with Google
 * Checks, and blocks the response when a policy is violated.
 *
 * ```ts
 * const ai = genkit({
 *   plugins: [
 *     googleAI(),
 *     // Optional: only needed to configure auth.
 *     checksMiddleware.plugin({ authOptions: { projectId: 'my-project' } }),
 *   ],
 * });
 *
 * const { text } = await ai.generate({
 *   prompt,
 *   use: [
 *     checksMiddleware({
 *       metrics: [ChecksEvaluationMetricType.DANGEROUS_CONTENT],
 *     }),
 *   ],
 * });
 * ```
 */
export const checksMiddleware: GenerateMiddleware<
  typeof ChecksMiddlewareOptionsSchema,
  ChecksMiddlewarePluginOptions | undefined
> = generateMiddleware(
  {
    name: 'checks',
    description:
      'Blocks model input or output that violates Google Checks policies.',
    configSchema: ChecksMiddlewareOptionsSchema,
  },
  ({ config, pluginConfig }) => {
    const guardrails = guardrailsFor(pluginConfig);
    const metrics: ChecksEvaluationMetric[] = config?.metrics ?? [];

    // Returns the violated policy types of the first text part that has any.
    const findViolations = async (parts: Part[]): Promise<string[]> => {
      for (const part of parts) {
        if (!part.text) continue;
        const response = await guardrails.classifyContent(part.text, metrics);
        const violated = response.policyResults
          .filter((policy) => policy.violationResult === 'VIOLATIVE')
          .map((policy) => policy.policyType);
        if (violated.length > 0) return violated;
      }
      return [];
    };

    const blocked = (message: string): GenerateResponseData => ({
      finishReason: 'blocked',
      finishMessage: message,
    });

    return {
      model: async (req, ctx, next) => {
        for (const message of req.messages) {
          const violated = await findViolations(message.content);
          if (violated.length > 0) {
            return blocked(
              `Model input violated Checks policies: [${violated.join(' ')}], further processing blocked.`
            );
          }
        }

        const response = await next(req, ctx);

        const violated = await findViolations(response.message?.content ?? []);
        if (violated.length > 0) {
          return blocked(
            `Model output violated Checks policies: [${violated.join(' ')}], output blocked.`
          );
        }
        return response;
      },
    };
  }
);
