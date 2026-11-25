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
import { flushTelemetryServerPosts } from './tracing/exporter.js';

export * from './tracing/exporter.js';
export * from './tracing/instrumentation.js';
export * from './tracing/types.js';

const oTelInitializationKey = '__GENKIT_DISABLE_GENKIT_OTEL_INITIALIZATION';
const instrumentationKey = '__GENKIT_TELEMETRY_INSTRUMENTED';
const telemetryProviderKey = '__GENKIT_TELEMETRY_PROVIDER';

/**
 * Ensures telemetry setup has settled before a span is opened: runs the
 * flag-gated Firebase auto-init and waits for any pending `enableTelemetry`
 * config.
 *
 * The wait matters because plugins like `enableFirebaseTelemetry()` resolve
 * their config asynchronously (credentials lookup) and are typically called
 * without `await`. Without it, spans opened before the SDK starts (e.g. the
 * first request after a cold start) go to the no-op tracer and are dropped.
 *
 * Historically this also booted a NodeSDK (`enableTelemetry({})`) when nothing
 * was configured. That no longer happens: the Developer UI is fed by
 * `DirectTelemetryInstrumentation`, so collection stays opt-in.
 *
 * @hidden
 */
export async function ensureBasicTelemetryInstrumentation() {
  await checkFirebaseMonitoringAutoInit();
  if (global[instrumentationKey]) {
    await global[instrumentationKey];
  }
}

const FIREBASE_PLUGIN_MODULE = '@genkit-ai/firebase';

function isModuleNotFoundError(err: unknown): boolean {
  if (!err || typeof err !== 'object') {
    return false;
  }
  const code = (err as { code?: unknown }).code;
  return code === 'ERR_MODULE_NOT_FOUND' || code === 'MODULE_NOT_FOUND';
}

async function importFirebasePlugin(): Promise<{
  enableFirebaseTelemetry?: () => unknown;
  default?: { enableFirebaseTelemetry?: () => unknown };
}> {
  // Keep the specifier typed as `string` so TypeScript does not statically
  // require the optional peer dependency and bundlers leave it dynamic.
  const moduleName: string = FIREBASE_PLUGIN_MODULE;
  try {
    return await import(moduleName);
  } catch (err) {
    if (
      !isModuleNotFoundError(err) ||
      typeof process === 'undefined' ||
      typeof process.cwd !== 'function'
    ) {
      throw err;
    }
    // Fallback for layouts (e.g., workspace symlinks or strict package managers
    // where an intermediate dependency did not forward the optional peer) where
    // @genkit-ai/firebase is installed under the app rather than adjacent to
    // @genkit-ai/core.
    const [{ createRequire }, path, { pathToFileURL }] = await Promise.all([
      import('node:module'),
      import('node:path'),
      import('node:url'),
    ]);
    const searchAnchors = [path.join(process.cwd(), 'package.json')];
    if (typeof process.argv?.[1] === 'string' && process.argv[1]) {
      searchAnchors.push(path.resolve(process.argv[1]));
    }
    for (const anchor of searchAnchors) {
      try {
        const resolved = createRequire(anchor).resolve(moduleName);
        return await import(pathToFileURL(resolved).href);
      } catch (fallbackErr) {
        if (!isModuleNotFoundError(fallbackErr)) {
          throw fallbackErr;
        }
      }
    }
    throw err;
  }
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
    typeof process !== 'undefined' &&
    process.env?.ENABLE_FIREBASE_MONITORING === 'true'
  ) {
    let firebaseModule: Awaited<ReturnType<typeof importFirebasePlugin>>;
    try {
      firebaseModule = await importFirebasePlugin();
    } catch (e) {
      logger.warn(
        "It looks like you're trying to enable firebase monitoring, but " +
          "haven't installed the firebase plugin. Please run " +
          '`npm i --save @genkit-ai/firebase` and redeploy.'
      );
      return;
    }
    try {
      const enableFirebaseTelemetry =
        firebaseModule?.enableFirebaseTelemetry ??
        firebaseModule?.default?.enableFirebaseTelemetry;
      if (typeof enableFirebaseTelemetry === 'function') {
        enableFirebaseTelemetry();
      } else {
        logger.warn(
          'Unable to enable firebase monitoring: `@genkit-ai/firebase` does not export `enableFirebaseTelemetry`.'
        );
      }
    } catch (e) {
      logger.warn('Failed to enable firebase monitoring.', e);
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
  // Also await DirectTelemetryInstrumentation's posts, which bypass the OTel
  // processors. The reflection server relies on this so a trace is saved
  // before its traceId is returned to the Dev UI.
  await Promise.all([
    getTelemetryProvider().flushTracing(),
    flushTelemetryServerPosts(),
  ]);
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
