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

import { GenkitError, z } from 'genkit';
import { GoogleAuth } from 'google-auth-library';
import type {
  ClientOptions,
  ExpressClientOptions,
  GlobalClientOptions,
  RegionalClientOptions,
  VertexPluginOptions,
} from './types.js';

export {
  checkModelName,
  checkSupportedMimeType,
  cleanSchema,
  extractMedia,
  extractMimeType,
  extractText,
  extractVersion,
  modelName,
} from '../common/utils.js';

const CLOUD_PLATFORM_OAUTH_SCOPE =
  'https://www.googleapis.com/auth/cloud-platform';

function parseFirebaseProjectId(): string | undefined {
  if (!process.env.FIREBASE_CONFIG) return undefined;
  try {
    return JSON.parse(process.env.FIREBASE_CONFIG).projectId as string;
  } catch {
    return undefined;
  }
}

export function shouldUseLegacyEndpoint(model: string) {
  // This only covers lyria models so far
  return model === 'lyria-002';
}

let __mockDerivedOptions: ClientOptions | undefined = undefined;
function setMockDerivedOptions(options: ClientOptions | undefined): void {
  __mockDerivedOptions = options;
}
export const TEST_ONLY = { setMockDerivedOptions };

export async function getDerivedOptions(
  options?: VertexPluginOptions,
  AuthClass: typeof GoogleAuth = GoogleAuth // Injectable testing
): Promise<ClientOptions> {
  if (__mockDerivedOptions) {
    return Promise.resolve(__mockDerivedOptions);
  }

  // Figure out the type of preferred options if possible
  // The order of the if statements is important.
  if (options?.location == 'global') {
    return await getGlobalDerivedOptions(AuthClass, options);
  } else if (options?.location) {
    return await getRegionalDerivedOptions(AuthClass, options);
  } else if (options?.apiKey !== undefined) {
    // apiKey = false still indicates apiKey expectation
    return getExpressDerivedOptions(options);
  }

  // If we got here then we're relying on environment variables.
  // Try regional first, it's the most common usage.
  try {
    const regionalOptions = await getRegionalDerivedOptions(AuthClass, options);
    return regionalOptions;
  } catch (e: unknown) {
    /* no-op - try global next */
  }
  try {
    const globalOptions = await getGlobalDerivedOptions(AuthClass, options);
    return globalOptions;
  } catch (e: unknown) {
    /* no-op - try express last */
  }
  try {
    const expressOptions = getExpressDerivedOptions(options);
    return expressOptions;
  } catch (e: unknown) {
    /* no-op */
  }

  // We did not have enough information in the options or in environment variables
  // to properly determine client options.
  throw new GenkitError({
    status: 'INVALID_ARGUMENT',
    message:
      'Unable to determine client options. Please set either apiKey or projectId and location',
  });
}

async function getGlobalDerivedOptions(
  AuthClass: typeof GoogleAuth,
  options?: VertexPluginOptions
): Promise<GlobalClientOptions> {
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
    authClient = new AuthClass(authOptions);
  } else {
    authClient = new AuthClass(
      authOptions ?? {
        scopes: [CLOUD_PLATFORM_OAUTH_SCOPE],
        projectId: providedProjectId,
      }
    );
  }

  const projectId =
    options?.projectId ||
    process.env.GCLOUD_PROJECT ||
    (await authClient.getProjectId());

  if (!projectId) {
    throw new Error(
      `VertexAI Plugin is missing the 'project' configuration. Please set the 'GCLOUD_PROJECT' environment variable or explicitly pass 'project' into genkit config.`
    );
  }

  const clientOpt: GlobalClientOptions = {
    kind: 'global',
    location: 'global',
    projectId,
    authClient,
  };
  if (options?.apiKey) {
    clientOpt.apiKey = options.apiKey;
  }

  return clientOpt;
}

function getExpressDerivedOptions(
  options?: VertexPluginOptions
): ExpressClientOptions {
  const apiKey = checkApiKey(options?.apiKey);
  return {
    kind: 'express',
    apiKey,
  };
}

async function getRegionalDerivedOptions(
  AuthClass: typeof GoogleAuth,
  options?: VertexPluginOptions
): Promise<RegionalClientOptions> {
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
    authClient = new AuthClass(authOptions);
  } else {
    authClient = new AuthClass(
      authOptions ?? {
        scopes: [CLOUD_PLATFORM_OAUTH_SCOPE],
        projectId: providedProjectId,
      }
    );
  }

  const projectId =
    options?.projectId ||
    process.env.GCLOUD_PROJECT ||
    (await authClient.getProjectId());
  const location =
    options?.location || process.env.GCLOUD_LOCATION || 'us-central1';

  if (!location) {
    throw new Error(
      `VertexAI Plugin is missing the 'location' configuration. Please set the 'GCLOUD_LOCATION' environment variable or explicitly pass 'location' into genkit config.`
    );
  }
  if (!projectId) {
    throw new Error(
      `VertexAI Plugin is missing the 'project' configuration. Please set the 'GCLOUD_PROJECT' environment variable or explicitly pass 'project' into genkit config.`
    );
  }

  const clientOpt: RegionalClientOptions = {
    kind: 'regional',
    location,
    projectId,
    authClient,
  };
  if (options?.apiKey) {
    clientOpt.apiKey = options.apiKey;
  }
  return clientOpt;
}

export type RequestClientOptions = ClientOptions & {
  signal: AbortSignal;
};

/**
 * If location or apiKey are present in reqConfig, they will
 * override the values in the clientOptions. The newOptions will
 * contain the clientOptions with those overrides.
 * @param clientOptions The client options
 * @param reqConfig The request config
 */
export function calculateRequestOptions<T extends z.ZodObject<any, any, any>>(
  clientOptions: RequestClientOptions,
  reqConfig?: z.infer<T>
): RequestClientOptions;
export function calculateRequestOptions<T extends z.ZodObject<any, any, any>>(
  clientOptions: ClientOptions,
  reqConfig?: z.infer<T>
): ClientOptions;
export function calculateRequestOptions<T extends z.ZodObject<any, any, any>>(
  clientOptions: RequestClientOptions | ClientOptions,
  reqConfig?: z.infer<T>
): RequestClientOptions | ClientOptions {
  let newOptions = { ...clientOptions };
  if (
    reqConfig?.location &&
    typeof reqConfig.location == 'string' &&
    newOptions.kind != 'express' &&
    newOptions.location != reqConfig.location
  ) {
    // Override the location if it's specified in the request
    if (reqConfig.location == 'global') {
      newOptions.location = 'global';
      newOptions.kind = 'global';
    } else {
      newOptions.kind = 'regional';
      newOptions.location = reqConfig.location;
    }
  }
  if (
    clientOptions.kind == 'express' &&
    reqConfig?.apiKey &&
    typeof reqConfig.apiKey == 'string'
  ) {
    newOptions.apiKey = calculateApiKey(clientOptions.apiKey, reqConfig.apiKey);
  } else if (reqConfig?.apiKey && typeof reqConfig.apiKey == 'string') {
    // Regional or Global can still use APIKey for billing (not auth)
    newOptions.apiKey = reqConfig.apiKey;
  }
  return newOptions;
}

/**
 * Retrieves an API key from environment variables.
 *
 * @returns The API key as a string, or `undefined` if none of the specified
 *          environment variables are set.
 */
export function getApiKeyFromEnvVar(): string | undefined {
  return (
    process.env.VERTEX_API_KEY ||
    process.env.GOOGLE_API_KEY ||
    process.env.GOOGLE_GENAI_API_KEY
  );
}

export const MISSING_API_KEY_ERROR = new GenkitError({
  status: 'FAILED_PRECONDITION',
  message:
    'Please pass in the API key or set the VERTEX_API_KEY or GOOGLE_API_KEY environment variable.\n' +
    'For more details see https://genkit.dev/docs/integrations/google-genai',
});

export const API_KEY_FALSE_ERROR = new GenkitError({
  status: 'INVALID_ARGUMENT',
  message:
    'VertexAI plugin was initialized with {apiKey: false} but no apiKey configuration was passed at call time.',
});

export const NOT_SUPPORTED_IN_EXPRESS_ERROR = new GenkitError({
  status: 'PERMISSION_DENIED',
  message:
    'This method or model is not supported in Vertex AI Express Mode.\n' +
    'For more details see https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/vertex-ai-express-mode-api-reference',
});

/**
 * Checks and retrieves an API key based on the provided argument and environment variables.
 *
 * - If `pluginApiKey` is a non-empty string, it's used as the API key.
 * - If `pluginApiKey` is `undefined` or an empty string, it attempts to fetch the API key from environment
 * - If `pluginApiKey` is `false`, key retrieval from the environment is skipped, and the function
 *   will return `undefined`. This mode indicates that the API key is expected to be provided
 *   at a later stage or in a different context.
 *
 * @param pluginApiKey - An optional API key string, `undefined` to check the environment, or `false` to bypass all checks in this function.
 * @returns The resolved API key as a string, or `undefined` if `pluginApiKey` is `false`.
 * @throws {Error} MISSING_API_KEY_ERROR - Thrown if `pluginApiKey` is not `false` and no API key
 *   can be found either in the `pluginApiKey` argument or from the environment.
 */
export function checkApiKey(
  pluginApiKey: string | false | undefined
): string | undefined {
  let apiKey: string | undefined;

  // Don't get the key from the environment if pluginApiKey is false
  if (pluginApiKey !== false) {
    apiKey = pluginApiKey || getApiKeyFromEnvVar();
  }

  // If pluginApiKey is false, then we don't throw because we are waiting for
  // the apiKey passed into the individual call
  if (pluginApiKey !== false && !apiKey) {
    throw MISSING_API_KEY_ERROR;
  }
  return apiKey;
}

/**
 * Calculates and returns the effective API key based on multiple potential sources.
 * The order of precedence for determining the API key is:
 * 1. `requestApiKey` (if provided)
 * 2. `pluginApiKey` (if provided and not `false`)
 * 3. Environment variable (if `pluginApiKey` is not `false` and `pluginApiKey` is not provided)
 *
 * @param pluginApiKey - The apiKey value provided during plugin initialization.
 * @param requestApiKey - The apiKey provided to an individual generate call.
 * @returns The resolved API key as a string.
 * @throws {Error} API_KEY_FALSE_ERROR - Thrown if `pluginApiKey` is `false` and `requestApiKey` is not provided
 * @throws {Error} MISSING_API_KEY_ERROR - Thrown if no API key can be resolved from any source
 */
export function calculateApiKey(
  pluginApiKey: string | false | undefined,
  requestApiKey: string | undefined
): string {
  let apiKey: string | undefined;

  // Don't get the key from the environment if pluginApiKey is false
  if (pluginApiKey !== false) {
    apiKey = pluginApiKey || getApiKeyFromEnvVar();
  }

  apiKey = requestApiKey || apiKey;

  if (pluginApiKey === false && !requestApiKey) {
    throw API_KEY_FALSE_ERROR;
  }

  if (!apiKey) {
    throw MISSING_API_KEY_ERROR;
  }
  return apiKey;
}

/** Vertex Express Mode lets you try a *subset* of Vertex AI features */
export function checkSupportedResourceMethod(params: {
  clientOptions: ClientOptions;
  resourcePath?: string;
  resourceMethod?: string;
}) {
  if (params.resourcePath == '') {
    // This is how we get a base url for metadata
    return;
  }

  const supportedExpressMethods = [
    'countTokens',
    'generateContent',
    'streamGenerateContent',
  ];

  if (
    params.clientOptions.kind === 'express' &&
    (!supportedExpressMethods.includes(params.resourceMethod ?? '') ||
      params.resourcePath?.includes('endpoints/'))
  ) {
    throw NOT_SUPPORTED_IN_EXPRESS_ERROR;
  }
}
