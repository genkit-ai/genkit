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

import type { Genkit } from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import type { GoogleAuthOptions } from 'google-auth-library';
import { initializeAuth } from './auth.js';
import { checksEvaluators } from './evaluation.js';
import {
  ChecksEvaluationMetricType,
  type ChecksEvaluationMetric,
  type ChecksEvaluationMetricConfig,
} from './metrics.js';

export {
  ChecksMiddlewareOptionsSchema,
  checksMiddleware,
  type ChecksMiddlewareOptions,
  type ChecksMiddlewarePluginOptions,
} from './middleware.js';
export {
  ChecksEvaluationMetricType,
  type ChecksEvaluationMetric,
  type ChecksEvaluationMetricConfig,
};

export interface PluginOptions {
  /** The Google Cloud project id to call. Must have quota for the Checks API. */
  projectId?: string;
  /** Provide custom authentication configuration for connecting to Checks API. */
  googleAuthOptions?: GoogleAuthOptions;
  /** Configure Checks evaluators. */
  evaluation?: {
    metrics: ChecksEvaluationMetric[];
  };
}

/**
 * Add Google Checks evaluators.
 */
export function checks(options?: PluginOptions): GenkitPlugin {
  return genkitPlugin('checks', async (ai: Genkit) => {
    const googleAuth = initializeAuth(options?.googleAuthOptions);

    const projectId = options?.projectId || (await googleAuth.getProjectId());

    if (!projectId) {
      throw new Error(
        `Checks Plugin is missing the 'projectId' configuration. Please set the 'GCLOUD_PROJECT' environment variable or explicitly pass 'projectId' into genkit config.`
      );
    }

    const metrics =
      options?.evaluation && options.evaluation.metrics.length > 0
        ? options.evaluation.metrics
        : [];
    checksEvaluators(ai, googleAuth, metrics, projectId);
  });
}

export default checks;
