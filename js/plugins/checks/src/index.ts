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

import { logger } from 'genkit/logging';
import type { ModelMiddleware } from 'genkit/model';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import { GoogleAuth, type GoogleAuthOptions } from 'google-auth-library';
import { checksEvaluators } from './evaluation.js';
import {
  ChecksEvaluationMetricType,
  type ChecksEvaluationMetric,
} from './metrics.js';
import { checksMiddleware as authorizedMiddleware } from './middleware.js';

export { ChecksEvaluationMetricType };

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

const CLOUD_PLATFROM_OAUTH_SCOPE =
  'https://www.googleapis.com/auth/cloud-platform';

const CHECKS_OAUTH_SCOPE = 'https://www.googleapis.com/auth/checks';

/**
 * Add Google Checks evaluators.
 */
export function checks(options?: PluginOptions): GenkitPlugin {
  return genkitPlugin({
    name: 'checks',
    init: async () => {
      const googleAuth = inititializeAuth(options?.googleAuthOptions);

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
      return [checksEvaluators(googleAuth, metrics, projectId)];
    },
  });
}

export function checksMiddleware(options: {
  authOptions: GoogleAuthOptions;
  metrics: ChecksEvaluationMetric[];
}): ModelMiddleware {
  const googleAuth = inititializeAuth(options.authOptions);

  return authorizedMiddleware({
    auth: googleAuth,
    metrics: options.metrics,
    projectId: options.authOptions.projectId,
  });
}

/**
 * Helper function for initializing an instance of GoogleAuth.
 *
 * @param options Options for initializing a GoogleAuth instance.
 * @returns GoogleAuth
 */
function inititializeAuth(options?: GoogleAuthOptions): GoogleAuth {
  let googleAuth: GoogleAuth;

  // Allow customers to pass in cloud credentials from environment variables
  // following: https://github.com/googleapis/google-auth-library-nodejs?tab=readme-ov-file#loading-credentials-from-environment-variables
  if (process.env.GCLOUD_SERVICE_ACCOUNT_CREDS) {
    const serviceAccountCreds = JSON.parse(
      process.env.GCLOUD_SERVICE_ACCOUNT_CREDS
    );
    options = {
      credentials: serviceAccountCreds,
      scopes: [CLOUD_PLATFROM_OAUTH_SCOPE, CHECKS_OAUTH_SCOPE],
    };
    googleAuth = new GoogleAuth(options);
  } else {
    googleAuth = new GoogleAuth(
      options ?? {
        scopes: [CLOUD_PLATFROM_OAUTH_SCOPE, CHECKS_OAUTH_SCOPE],
      }
    );
  }

  googleAuth.getClient().then((client) => {
    if (client.quotaProjectId && options?.projectId) {
      logger.warn(
        `Checks Evaluator: Your Google cloud authentication has a default quota project(${client.quotaProjectId}) associated with it which will overrid the projectId in your Checks plugin config(${options?.projectId}).`
      );
    }
  });

  return googleAuth;
}

export default checks;
