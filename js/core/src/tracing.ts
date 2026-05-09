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

import { GenkitError } from './error.js';
import { logger } from './logging.js';
import type { TelemetryConfig } from './telemetryTypes.js';

export * from './tracing/exporter.js';
export * from './tracing/instrumentation.js';
export * from './tracing/types.js';

const oTelInitializationKey = '__GENKIT_DISABLE_GENKIT_OTEL_INITIALIZATION';
const instrumentationKey = '__GENKIT_TELEMETRY_INSTRUMENTED';
const telemetryProviderKey = '__GENKIT_TELEMETRY_PROVIDER';

/**
 * @hidden
 */
export async function ensureBasicTelemetryInstrumentation() {
  await checkFirebaseMonitoringAutoInit();

  if (global[instrumentationKey]) {
    return await global[instrumentationKey];
  }

  await enableTelemetry({});
}

/**
 * Checks to see if the customer is using Firebase Genkit Monitoring
 * auto initialization via environment variable by attempting to resolve
 * the firebase plugin.
 *
 * Enables Firebase Genkit Monitoring if the plugin is installed and warns
 * if it hasn't been installed.
 */
async function checkFirebaseMonitoringAutoInit() {
  if (
    !global[instrumentationKey] &&
    process.env.ENABLE_FIREBASE_MONITORING === 'true'
  ) {
    try {
      const importModule = new Function(
        'moduleName',
        'return import(moduleName)'
      );
      const firebaseModule = await importModule('@genkit-ai/firebase');

      firebaseModule.enableFirebaseTelemetry();
    } catch (e) {
      logger.warn(
        "It looks like you're trying to enable firebase monitoring, but " +
          "haven't installed the firebase plugin. Please run " +
          '`npm i --save @genkit-ai/firebase` and redeploy.'
      );
    }
  }
}

export interface TelemetryProvider {
  enableTelemetry(
    telemetryConfig: TelemetryConfig | Promise<TelemetryConfig>
  ): Promise<void>;
  flushTracing(): Promise<void>;
}

function getTelemetryProvider(): TelemetryProvider {
  if (global[telemetryProviderKey]) {
    return global[telemetryProviderKey];
  }
  throw new GenkitError({
    status: 'FAILED_PRECONDITION',
    message: 'TelemetryProvider is not initialized.',
  });
}
export function setTelemetryProvider(provider: TelemetryProvider) {
  if (global[telemetryProviderKey]) return;
  global[telemetryProviderKey] = provider;
}

/**
 * Enables tracing and metrics open telemetry configuration.
 */
export async function enableTelemetry(
  telemetryConfig: TelemetryConfig | Promise<TelemetryConfig>
) {
  if (isOTelInitializationDisabled()) {
    return;
  }
  global[instrumentationKey] =
    telemetryConfig instanceof Promise ? telemetryConfig : Promise.resolve();
  return getTelemetryProvider().enableTelemetry(telemetryConfig);
}

/**
 * Flushes all configured span processors.
 *
 * @hidden
 */
export async function flushTracing() {
  return getTelemetryProvider().flushTracing();
}

function isOTelInitializationDisabled(): boolean {
  return global[oTelInitializationKey] === true;
}

/**
 * Disables Genkit's OTel initialization. This is useful when you want to
 * control the OTel initialization yourself.
 *
 * This function attempts to control Genkit's internal OTel instrumentation behaviour,
 * since internal implementation details are subject to change at any time consider
 * this function "unstable" and subject to breaking changes as well.
 *
 * @hidden
 */
export function disableGenkitOTelInitialization() {
  global[oTelInitializationKey] = true;
}
