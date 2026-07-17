/**
 * @license
 *
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
 * Testing utilities for Genkit apps and model plugins.
 *
 * - {@link mockModel} — a programmable mock model with typed call inspection,
 *   for testing flows, prompts, and tools deterministically.
 * - {@link echoModel} — a zero-config model that echoes the rendered request,
 *   for asserting prompt/message assembly.
 * - `testModels` — a conformance harness for model *plugin* authors.
 *
 * ```ts
 * import { mockModel, echoModel, testModels } from 'genkit/testing';
 * ```
 *
 * @module testing
 */

export {
  echoModel,
  mockModel,
  testModels,
  type EchoModelOptions,
  type MockChunk,
  type MockContext,
  type MockModel,
  type MockModelOptions,
  type MockRespond,
  type MockRespondFn,
  type MockResponse,
} from '@genkit-ai/ai/testing';
