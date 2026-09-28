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

export { createAgentProxy } from './agent-proxy.js';
export { Box, box, type BoxCreateOptions, type ProxySpec } from './box.js';
export { BOX_SELF_ID_ENV } from './env.js';
export {
  actionKeyFor,
  createProxyAction,
  indexActionsByName,
  type ProxyDispatcher,
  type ProxyKind,
} from './proxy.js';
export {
  SINGLETON_KEY,
  perRequest,
  resolveRetention,
  singleton,
} from './route.js';
export type {
  BoxConnection,
  BoxOptions,
  BoxRunner,
  Retention,
  RouteFn,
  RunActionRequest,
  RunActionResult,
  RunOptions,
} from './types.js';
