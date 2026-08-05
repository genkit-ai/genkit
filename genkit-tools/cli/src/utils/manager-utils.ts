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

import {
  LocalFileLogStore,
  LocalFileTraceStore,
  startTelemetryServer,
} from '@genkit-ai/telemetry-server';
import type { Status } from '@genkit-ai/tools-common';
import {
  BaseRuntimeManager,
  ProcessManager,
  RuntimeEvent,
  RuntimeManager,
  type GenkitToolsError,
} from '@genkit-ai/tools-common/manager';
import { logger } from '@genkit-ai/tools-common/utils';
import getPort, { makeRange } from 'get-port';

/**
 * Returns the telemetry server address either based on environment setup or starts one.
 *
 * This function is not idempotent. Typically you want to make sure it's called only once per cli instance.
 */
export async function resolveTelemetryServer(options: {
  projectRoot: string;
  corsOrigin?: string;
}): Promise<string> {
  let telemetryServerUrl = process.env.GENKIT_TELEMETRY_SERVER;
  if (!telemetryServerUrl) {
    const telemetryPort = await getPort({ port: makeRange(4033, 4999) });
    telemetryServerUrl = `http://localhost:${telemetryPort}`;
    await startTelemetryServer({
      port: telemetryPort,
      traceStore: new LocalFileTraceStore({
        storeRoot: options.projectRoot,
        indexRoot: options.projectRoot,
      }),
      logStore: new LocalFileLogStore({
        storeRoot: options.projectRoot,
        indexRoot: options.projectRoot,
      }),
      corsOrigin: options.corsOrigin,
    });
  }
  return telemetryServerUrl;
}

/**
 * Starts the runtime manager and its dependencies.
 */
export async function startManager(options: {
  projectRoot: string;
  manageHealth?: boolean;
  corsOrigin?: string;
  experimentalReflectionV2?: boolean;
  reflectionV2Port?: number;
  telemetryServerUrl?: string;
}): Promise<BaseRuntimeManager> {
  const telemetryServerUrl =
    options.telemetryServerUrl ?? (await resolveTelemetryServer(options));
  const manager = RuntimeManager.create({
    telemetryServerUrl,
    manageHealth: options.manageHealth,
    projectRoot: options.projectRoot,
    experimentalReflectionV2: options.experimentalReflectionV2,
    reflectionV2Port: options.reflectionV2Port,
  });
  return manager;
}

export interface DevProcessManagerOptions {
  disableRealtimeTelemetry?: boolean;
  nonInteractive?: boolean;
  healthCheck?: boolean;
  timeout?: number;
  cwd?: string;
  corsOrigin?: string;
  experimentalReflectionV2?: boolean;
  envVars?: Record<string, string>;
  reflectionV2Port?: number;
  telemetryServerUrl?: string;
}

export async function getDevEnvVars(
  projectRoot: string,
  options?: DevProcessManagerOptions
): Promise<{
  envVars: Record<string, string>;
  reflectionV2Port?: number;
  telemetryServerUrl: string;
}> {
  const telemetryServerUrl = await resolveTelemetryServer({
    projectRoot,
    corsOrigin: options?.corsOrigin,
  });
  const disableRealtimeTelemetry = options?.disableRealtimeTelemetry ?? false;
  const experimentalReflectionV2 = options?.experimentalReflectionV2 ?? false;

  let reflectionV2Port: number | undefined;
  const envVars: Record<string, string> = {
    GENKIT_TELEMETRY_SERVER: telemetryServerUrl,
    GENKIT_ENV: 'dev',
  };

  if (experimentalReflectionV2) {
    reflectionV2Port = await getPort({ port: makeRange(3200, 3400) });
    envVars.GENKIT_REFLECTION_V2_SERVER = `ws://localhost:${reflectionV2Port}`;
  }

  if (!disableRealtimeTelemetry) {
    envVars.GENKIT_ENABLE_REALTIME_TELEMETRY = 'true';
  }

  return { envVars, reflectionV2Port, telemetryServerUrl };
}

export async function startDevProcessManager(
  projectRoot: string,
  command: string,
  args: string[],
  options?: DevProcessManagerOptions
): Promise<{ manager: BaseRuntimeManager; processPromise: Promise<void> }> {
  const { envVars, reflectionV2Port, telemetryServerUrl } =
    options?.envVars &&
    options?.telemetryServerUrl &&
    (!options?.experimentalReflectionV2 || options?.reflectionV2Port)
      ? {
          envVars: options.envVars,
          reflectionV2Port: options.reflectionV2Port,
          telemetryServerUrl: options.telemetryServerUrl,
        }
      : await getDevEnvVars(projectRoot, options);

  const disableRealtimeTelemetry = options?.disableRealtimeTelemetry ?? false;
  const experimentalReflectionV2 = options?.experimentalReflectionV2 ?? false;

  const processManager = new ProcessManager(command, args, envVars);
  const manager = await RuntimeManager.create({
    telemetryServerUrl,
    manageHealth: true,
    projectRoot,
    processManager,
    disableRealtimeTelemetry,
    experimentalReflectionV2,
    reflectionV2Port,
  });
  const processPromise = processManager.start({ ...options });

  if (options?.healthCheck) {
    await waitForRuntime(manager, processPromise, options?.timeout);
  }

  return { manager, processPromise };
}

/**
 * Waits for a new runtime to register itself.
 * Rejects if the process exits or if the timeout is reached.
 */
export async function waitForRuntime(
  manager: BaseRuntimeManager,
  processPromise: Promise<void>,
  timeoutMs: number = 30000
): Promise<void> {
  let unsubscribe: (() => void) | undefined;
  let timeoutId: NodeJS.Timeout | undefined;

  if (manager.listRuntimes().length > 0) {
    return;
  }

  try {
    const runtimeAddedPromise = new Promise<void>((resolve) => {
      unsubscribe = manager.onRuntimeEvent((event) => {
        // Just listen for a new runtime, not for a specific ID.
        if (event === RuntimeEvent.ADD) {
          resolve();
        }
      });
      if (manager.listRuntimes().length > 0) {
        resolve();
      }
    });

    const timeoutPromise = new Promise<void>((_, reject) => {
      timeoutId = setTimeout(
        () => reject(new Error('Timeout waiting for runtime to be ready')),
        timeoutMs
      );
    });

    const processExitedPromise = processPromise.then(
      () =>
        Promise.reject(new Error('Process exited before runtime was ready')),
      (err) => Promise.reject(err)
    );

    await Promise.race([
      runtimeAddedPromise,
      timeoutPromise,
      processExitedPromise,
    ]);
  } finally {
    if (unsubscribe) unsubscribe();
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export interface WaitForActionKeysOptions {
  /** How often to poll the runtime for its action list. */
  pollIntervalMs?: number;
  /**
   * If the set of registered actions stops changing for this long and the
   * target action(s) still haven't appeared, stop waiting. This keeps a
   * mistyped action name from blocking for the full timeout.
   */
  stableForMs?: number;
  /** Hard upper bound as a safety net. */
  timeoutMs?: number;
}

/**
 * Waits until all of the given action keys are registered with the runtime.
 *
 * Ephemeral runtimes (used by the `-- <cmd>` commands) register their actions
 * asynchronously after startup. Some SDKs (notably Go) register the runtime
 * with the CLI during initialization but define their actions slightly later.
 * If we dispatch a runAction before the target action is registered, the
 * runtime returns an "action not found" error. Polling until the actions
 * appear closes that race.
 *
 * To avoid making a mistyped action name block for the full timeout, we watch
 * the number of registered actions: while it keeps changing, registration is
 * still in progress; once it settles (stops changing for `stableForMs`) without
 * the target appearing, we stop waiting and let the subsequent runAction
 * surface the real "not found" error.
 */
export async function waitForActionKeys(
  manager: BaseRuntimeManager,
  keys: string[],
  {
    pollIntervalMs = 500,
    stableForMs = 5000,
    timeoutMs = 30000,
  }: WaitForActionKeysOptions = {}
): Promise<void> {
  const requiredKeys = keys.filter((k) => !!k);
  if (requiredKeys.length === 0) return;

  const deadline = Date.now() + timeoutMs;

  let hasSeenRuntime = manager.listRuntimes().length > 0;
  let lastCount = -1;
  let lastChange = Date.now();

  while (true) {
    // If the runtime process crashed or exited after registering but before
    // registering its actions, stop waiting instead of hanging. A subsequent
    // runAction will surface the real error.
    if (manager.listRuntimes().length > 0) {
      hasSeenRuntime = true;
    } else if (hasSeenRuntime) {
      logger.debug(
        'Runtime disconnected while waiting for actions. Stopping wait.'
      );
      return;
    }

    try {
      const actions = await manager.listActions();
      const registered = Object.keys(actions);
      const missing = requiredKeys.filter((k) => !registered.includes(k));
      if (missing.length === 0) return;

      if (registered.length !== lastCount) {
        // Still registering actions; reset the stability window.
        lastCount = registered.length;
        lastChange = Date.now();
      } else if (Date.now() - lastChange >= stableForMs) {
        logger.debug(
          `Action list stabilized without registering: ${missing.join(
            ', '
          )}. Stopping wait.`
        );
        return;
      }
    } catch (e) {
      // The actions endpoint may not be ready yet; keep polling.
      logger.debug(`Polling for actions failed, will retry: ${e}`);
    }

    if (Date.now() >= deadline) {
      logger.debug('Timed out waiting for actions to register. Proceeding.');
      return;
    }
    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }
}

/**
 * Runs the given function with a runtime manager.
 */
export interface RunWithManagerOptions {
  /** Command to start the runtime process. If provided, an ephemeral manager is used. */
  runtimeCommand?: string[];
  /**
   * Action keys to wait for before invoking the function. Only used for
   * ephemeral runtimes to avoid dispatching before the runtime has finished
   * registering the target action(s).
   */
  waitForActionKeys?: string[];
}

export async function runWithManager(
  projectRoot: string,
  fn: (manager: BaseRuntimeManager) => Promise<void>,
  options?: RunWithManagerOptions
) {
  const useEphemeral =
    options?.runtimeCommand && options.runtimeCommand.length > 0;
  let manager: BaseRuntimeManager;
  const oldLevel = logger.level;

  try {
    if (useEphemeral) {
      const devEnv = await getDevEnvVars(projectRoot, {
        experimentalReflectionV2: true,
      });
      const { envVars, telemetryServerUrl, reflectionV2Port } = devEnv;

      logger.level = 'warn';

      const result = await startDevProcessManager(
        projectRoot,
        options!.runtimeCommand![0],
        options!.runtimeCommand!.slice(1),
        {
          experimentalReflectionV2: true,
          healthCheck: true,
          envVars,
          telemetryServerUrl,
          reflectionV2Port,
          nonInteractive: true,
        }
      );
      manager = result.manager;
    } else {
      manager = await startManager({
        projectRoot,
        manageHealth: false,
      });
    }
  } catch (e) {
    logger.error('Failed to start manager', e);
    process.exit(1);
  } finally {
    if (useEphemeral) {
      logger.level = oldLevel;
    }
  }

  try {
    if (useEphemeral && options?.waitForActionKeys?.length) {
      await waitForActionKeys(manager, options.waitForActionKeys);
    }
    await fn(manager);
  } catch (err) {
    logger.error('Command exited with an Error:');
    const error = err as GenkitToolsError;
    if (typeof error.data === 'object') {
      const errorStatus = error.data as Status;
      const { code, details, message } = errorStatus;
      logger.error(`\tCode: ${code}`);
      logger.error(`\tMessage: ${message}`);
      if (details?.traceId) {
        logger.error(`\tTrace ID: ${details.traceId}\n`);
      }
    } else {
      logger.error(`\tMessage: ${error.data}\n`);
    }
    logger.error('Stack trace:');
    logger.error(`${error.stack}`);
  } finally {
    if (manager) {
      await manager.stop();
    }
  }
}
