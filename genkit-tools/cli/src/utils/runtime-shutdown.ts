/**
 * Copyright 2026 Google LLC
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

import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import { logger } from '@genkit-ai/tools-common/utils';

const signals = ['SIGINT', 'SIGTERM', 'SIGHUP'] as const;

/** Stops the owned runtime on normal completion or CLI termination. */
export async function waitForRuntimeShutdown(
  manager: BaseRuntimeManager,
  processPromise: Promise<void>
): Promise<void> {
  let stopPromise: Promise<void> | undefined;
  let exiting = false;
  const stop = () => (stopPromise ??= manager.stopOnce());
  const onSignal = () => {
    if (exiting) return;
    exiting = true;
    void stop().then(
      () => process.exit(0),
      (error) => {
        logger.error(`Failed to stop runtime: ${error}`);
        process.exit(1);
      }
    );
  };
  for (const signal of signals) {
    process.on(signal, onSignal);
  }
  let processFailed = false;
  let processError: unknown;
  try {
    await processPromise;
  } catch (error) {
    processFailed = true;
    processError = error;
  }
  try {
    await stop();
  } catch (error) {
    if (!processFailed) throw error;
    logger.error(`Failed to stop runtime after process failure: ${error}`);
  } finally {
    for (const signal of signals) {
      process.off(signal, onSignal);
    }
  }
  if (processFailed) throw processError;
}
