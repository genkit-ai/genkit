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

import { GenkitError, Operation } from '@genkit-ai/core';
import { Registry } from '@genkit-ai/core/registry';

export async function cancelOperation<T = unknown>(
  registry: Registry,
  operation: Operation<T>
): Promise<Operation<T>> {
  if (!operation.action) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'Provided operation is missing original request information',
    });
  }
  const backgroundAction = await registry.lookupBackgroundAction(
    operation.action
  );
  if (!backgroundAction) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Failed to resolve background action from original request: ${operation.action}`,
    });
  }
  if (!backgroundAction.supportsCancel) {
    throw new GenkitError({
      status: 'UNIMPLEMENTED',
      message: `Background action ${operation.action} does not support cancellation.`,
    });
  }
  return await backgroundAction.cancel(operation);
}
