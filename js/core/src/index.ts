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

import { version } from './__codegen/version.js';

/**
 * Genkit library version.
 */
export const GENKIT_VERSION = version;

/**
 * Genkit client header for API calls.
 */
export const GENKIT_CLIENT_HEADER = `genkit-node/${GENKIT_VERSION} gl-node/${process.versions.node}`;
export const GENKIT_REFLECTION_API_SPEC_VERSION = 1;

export { z } from 'zod';
export * from './action.js';
export { getAsyncContext } from './async-context.js';
export {
  OperationSchema,
  backgroundAction,
  defineBackgroundAction,
  isBackgroundAction,
  registerBackgroundAction,
  type BackgroundAction,
  type BackgroundActionFnArg,
  type BackgroundActionParams,
  type BackgroundActionRunOptions,
  type Operation,
} from './background-action.js';
export {
  getGenkitRuntimeConfig,
  resetGenkitRuntimeConfig,
  setGenkitRuntimeConfig,
  type GenkitRuntimeConfig,
} from './config.js';
export {
  apiKey,
  getContext,
  runWithContext,
  type ActionContext,
  type ApiKeyContext,
  type ContextProvider,
  type RequestData,
} from './context.js';
export {
  defineDynamicActionProvider,
  type DapConfig,
  type DapFn,
  type DynamicActionProviderAction,
} from './dynamic-action-provider.js';
export {
  GenkitError,
  UnstableApiError,
  UserFacingError,
  assertUnstable,
  getCallableJSON,
  getHttpStatus,
  type StatusName,
} from './error.js';
export {
  defineFlow,
  flow,
  run,
  type Flow,
  type FlowConfig,
  type FlowFn,
  type FlowSideChannel,
} from './flow.js';
export * from './plugin.js';
export * from './reflection.js';
export { defineJsonSchema, defineSchema, type JSONSchema } from './schema.js';
export * from './telemetryTypes.js';
export * from './utils.js';

const clientHeaderGlobalKey = '__genkit_ClientHeader';

/** Additional attribution information to include in the x-goog-api-client header. */
export function getClientHeader() {
  if (global[clientHeaderGlobalKey]) {
    return GENKIT_CLIENT_HEADER + ' ' + global[clientHeaderGlobalKey];
  }
  return GENKIT_CLIENT_HEADER;
}

/** Sets additional attribution information to include in the x-goog-api-client header. */
export function setClientHeader(header: string) {
  global[clientHeaderGlobalKey] = header;
}
