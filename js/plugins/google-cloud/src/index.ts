/**
 * @license
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

/**
 * Google Cloud telemetry plugin for Genkit.
 *
 * @module
 */

import { getCurrentEnv } from 'genkit';
import { logger } from 'genkit/logging';
import { enableTelemetry } from 'genkit/tracing';
import { credentialsFromEnvironment } from './auth.js';
import { GcpLogger } from './gcpLogger.js';
import { GcpOpenTelemetry } from './gcpOpenTelemetry.js';
import { TelemetryConfigs } from './telemetry/defaults.js';
import type { GcpTelemetryConfig, GcpTelemetryConfigOptions } from './types.js';

/**
 * Enables telemetry export to the Google Cloud Observability suite.
 *
 * @param options configuration options
 */
export function enableGoogleCloudTelemetry(
  options?: GcpTelemetryConfigOptions
) {
  return enableTelemetry(
    configureGcpPlugin(options).then(async (pluginConfig) => {
      logger.init(await new GcpLogger(pluginConfig).getLogger(getCurrentEnv()));
      return new GcpOpenTelemetry(pluginConfig).getConfig();
    })
  );
}

/**
 * Create a configuration object for the plugin.
 * Not normally needed, but exposed for use by the Firebase plugin.
 */
async function configureGcpPlugin(
  options?: GcpTelemetryConfigOptions
): Promise<GcpTelemetryConfig> {
  const envOptions = await credentialsFromEnvironment();
  return {
    projectId: options?.projectId || envOptions.projectId,
    credentials: options?.credentials || envOptions.credentials,
    ...TelemetryConfigs.defaults(options),
  };
}

export * from './gcpLogger.js';
export * from './gcpOpenTelemetry.js';
export type { GcpTelemetryConfigOptions } from './types.js';
