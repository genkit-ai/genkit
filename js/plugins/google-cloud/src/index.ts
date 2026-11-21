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
 * Google Cloud integrations for Genkit.
 *
 * Model Armor is also available from `@genkit-ai/google-cloud/model-armor`,
 * and the Firestore session store from `@genkit-ai/google-cloud/beta`.
 *
 * For telemetry, use `@genkit-ai/otel` with a Google Cloud OpenTelemetry
 * exporter.
 *
 * @module
 */

export {
  ModelArmorOptionsSchema,
  modelArmor,
  type ApplyDeidentificationResultsFn,
  type ModelArmorOptions,
  type ModelArmorPluginOptions,
} from './model-armor.js';
