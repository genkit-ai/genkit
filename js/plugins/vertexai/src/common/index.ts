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

import { VertexAI } from '@google-cloud/vertexai';
import { getClientHeader as defaultGetClientHeader } from 'genkit';
import type { GenerateRequest } from 'genkit/model';
import { GoogleAuth } from 'google-auth-library';
import type { GeminiConfigSchema } from '../gemini.js';
import { CLOUD_PLATFORM_OAUTH_SCOPE } from './constants.js';
import type { PluginOptions } from './types.js';
export type { PluginOptions };

interface DerivedParams {
  location: string;
  projectId: string;
  vertexClientFactory: (
    request: GenerateRequest<typeof GeminiConfigSchema>
  ) => VertexAI;
  authClient: GoogleAuth;
}

function parseFirebaseProjectId(): string | undefined {
  if (!process.env.FIREBASE_CONFIG) return undefined;
  try {
    return JSON.parse(process.env.FIREBASE_CONFIG).projectId as string;
  } catch {
    return undefined;
  }
}

/** @hidden */
export function __setFakeDerivedParams(params: any) {
  __fake_getDerivedParams = params;
}
let __fake_getDerivedParams: any;

export async function getDerivedParams(
  options?: PluginOptions
): Promise<DerivedParams> {
  if (__fake_getDerivedParams) {
    return __fake_getDerivedParams;
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
      `VertexAI Plugin is missing the 'location' configuration. Please set the 'GCLOUD_LOCATION' environment variable or explicitly pass 'location' into genkit config.`
    );
  }
  if (!projectId) {
    throw new Error(
      `VertexAI Plugin is missing the 'project' configuration. Please set the 'GCLOUD_PROJECT' environment variable or explicitly pass 'project' into genkit config.`
    );
  }

  const vertexClientFactoryCache: Record<string, VertexAI> = {};
  const vertexClientFactory = (
    request: GenerateRequest<typeof GeminiConfigSchema>
  ): VertexAI => {
    const requestLocation = request.config?.location || location;
    if (!vertexClientFactoryCache[requestLocation]) {
      vertexClientFactoryCache[requestLocation] = new VertexAI({
        project: projectId,
        location: requestLocation,
        googleAuthOptions: { projectId: providedProjectId, ...authOptions },
      });
    }
    return vertexClientFactoryCache[requestLocation];
  };

  return {
    location,
    projectId,
    vertexClientFactory,
    authClient,
  };
}

export function getGenkitClientHeader() {
  if (process.env.MONOSPACE_ENV == 'true') {
    return defaultGetClientHeader() + ' firebase-studio-vm';
  }
  return defaultGetClientHeader();
}
