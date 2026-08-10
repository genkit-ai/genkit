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

import { GenkitError } from 'genkit';
import { GoogleAuth, GoogleAuthOptions } from 'google-auth-library';
import { CLOUD_PLATFORM_OAUTH_SCOPE } from './constants.js';

/**
 * Safely extracts the error message from the error.
 * @param e The error
 * @returns The error message
 */
export function extractErrMsg(e: unknown): string {
  let errorMessage = 'An unknown error occurred';
  if (e instanceof Error) {
    errorMessage = e.message;
  } else if (typeof e === 'string') {
    errorMessage = e;
  } else {
    // Fallback for other types
    try {
      errorMessage = JSON.stringify(e);
    } catch (stringifyError) {
      errorMessage = 'Failed to stringify error object';
    }
  }
  return errorMessage;
}

/**
 * Gets the model name without certain prefixes..
 * e.g. for "models/vertexai/gemini-2.5-pro" it returns just 'gemini-2.5-pro'
 * @param name A string containing the model string with possible prefixes
 * @returns the model string stripped of certain prefixes
 */
export function modelName(name?: string): string | undefined {
  if (!name) return name;

  // Remove any of these prefixes:
  const escapedPrefixes = [
    'background-model/',
    'model/',
    'models/',
    'embedders/',
    'vertex-model-garden/',
    'vertex-rerankers/',
    'vertexai/',
  ];
  const prefixesToRemove = new RegExp(escapedPrefixes.join('|'), 'g');
  return name.replace(prefixesToRemove, '');
}

/**
 * Gets the suffix of a model string.
 * Throws if the string is empty.
 * @param name A string containing the model string
 * @returns the model string stripped of prefixes and guaranteed not empty.
 */
export function checkModelName(name?: string): string {
  const version = modelName(name);
  if (!version) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'Model name is required.',
    });
  }
  return version;
}

export const TEST_ONLY = { setFakeDerivedOptions };

function setFakeDerivedOptions(options: any) {
  __fake_getDerivedOptions = options;
}
let __fake_getDerivedOptions: any;

function parseFirebaseProjectId(): string | undefined {
  if (!process.env.FIREBASE_CONFIG) return undefined;
  try {
    return JSON.parse(process.env.FIREBASE_CONFIG).projectId as string;
  } catch {
    return undefined;
  }
}

interface CommonOptions {
  location?: string;
  projectId?: string;
  googleAuth?: GoogleAuthOptions;
}

interface DerivedOptions {
  location: string;
  projectId: string;
  authClient: GoogleAuth;
}

export async function getDerivedOptions(
  pluginName: string,
  options?: CommonOptions
): Promise<DerivedOptions> {
  if (__fake_getDerivedOptions) {
    return __fake_getDerivedOptions;
  }

  let authOptions = options?.googleAuth;
  let authClient: GoogleAuth;
  const providedProjectId =
    options?.projectId ||
    process.env.GCLOUD_PROJECT ||
    parseFirebaseProjectId();
  if (process.env.GCLOUD_SERVICE_ACCOUNT_CREDS) {
    const serviceAccountCreds = JSON.parse(
      process.env.GCLOUD_SERVICE_ACCOUNT_CREDS
    );
    authOptions = {
      credentials: serviceAccountCreds,
      scopes: [CLOUD_PLATFORM_OAUTH_SCOPE],
      projectId: providedProjectId,
    };
    authClient = new GoogleAuth(authOptions);
  } else {
    authClient = new GoogleAuth(
      authOptions ?? {
        scopes: [CLOUD_PLATFORM_OAUTH_SCOPE],
        projectId: providedProjectId,
      }
    );
  }

  const projectId = options?.projectId || (await authClient.getProjectId());
  const location = options?.location || 'us-central1';

  if (!location) {
    throw new Error(
      `${pluginName} Plugin is missing the 'location' configuration. Please set the 'GCLOUD_LOCATION' environment variable or explicitly pass 'location' into genkit config.`
    );
  }
  if (!projectId) {
    throw new Error(
      `${pluginName} Plugin is missing the 'project' configuration. Please set the 'GCLOUD_PROJECT' environment variable or explicitly pass 'project' into genkit config.`
    );
  }

  return {
    location,
    projectId,
    authClient,
  };
}
