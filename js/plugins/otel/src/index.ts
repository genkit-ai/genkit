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

/**
 * OpenTelemetry GenAI semantic-conventions instrumentation for Genkit.
 *
 * @module @genkit-ai/otel
 */

export {
  GenAiInstrumentation,
  type GenAiContentMode,
  type GenAiInstrumentationOptions,
} from './gen-ai-instrumentation.js';
export {
  GenAiAttr,
  GenAiMetric,
  GenAiOperation,
  GenkitAttr,
  captureContentEnvVar,
  genAiOperationDetailsEvent,
  genAiSemConvVersion,
} from './genai/gen-ai-attributes.js';
