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
import { GoogleAuth, type GoogleAuthOptions } from 'google-auth-library';

const CLOUD_PLATFROM_OAUTH_SCOPE =
  'https://www.googleapis.com/auth/cloud-platform';

const CHECKS_OAUTH_SCOPE = 'https://www.googleapis.com/auth/checks';

/**
 * Helper function for initializing an instance of GoogleAuth.
 *
 * @param options Options for initializing a GoogleAuth instance.
 * @returns GoogleAuth
 */
export function initializeAuth(options?: GoogleAuthOptions): GoogleAuth {
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

  googleAuth
    .getClient()
    .then((client) => {
      if (client.quotaProjectId && options?.projectId) {
        logger.warn(
          `Checks: Your Google cloud authentication has a default quota project(${client.quotaProjectId}) associated with it which will override the projectId in your Checks config(${options?.projectId}).`
        );
      }
    })
    // Only feeds the warning above; auth errors surface on the first request.
    .catch(() => {});

  return googleAuth;
}
