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

import { flushTelemetryServerPosts } from './tracing/telemetry-server.js';

export * from './tracing/instrumentation.js';
export * from './tracing/telemetry-server.js';
export * from './tracing/types.js';

/**
 * Awaits in-flight Developer UI telemetry posts. The reflection server relies
 * on this so a trace is saved before its traceId is returned to the Dev UI.
 *
 * Instrumentations that buffer (e.g. an OTel batch span processor) own their
 * own flushing.
 *
 * @hidden
 */
export async function flushTracing() {
  await flushTelemetryServerPosts();
}
