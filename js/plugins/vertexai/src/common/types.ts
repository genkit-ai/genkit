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

import type { ModelReference } from 'genkit';
import type { GoogleAuthOptions } from 'google-auth-library';
import type { GeminiConfigSchema } from '../gemini.js';

/** Common options for Vertex AI plugin configuration */
export interface CommonPluginOptions {
  /** The Google Cloud project id to call. */
  projectId?: string;
  /** The Google Cloud region to call. */
  location: string;
  /** Provide custom authentication configuration for connecting to Vertex AI. */
  googleAuth?: GoogleAuthOptions;
  /** Enables additional debug traces (e.g. raw model API call details). */
  experimental_debugTraces?: boolean;
}

/** Combined plugin options, extending common options with subplugin-specific options */
export interface PluginOptions extends CommonPluginOptions {
  models?: (
    | ModelReference</** @ignore */ typeof GeminiConfigSchema>
    | string
  )[];
}
