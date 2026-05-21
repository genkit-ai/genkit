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
 * Schema utilities — JSON Schema conversion, validation helpers, and related
 * types for working with Zod schemas in Genkit.
 *
 * ```ts
 * import { toJsonSchema, parseSchema, ValidationError } from 'genkit/schema';
 * ```
 *
 * @module schema
 */

export {
  ValidationError,
  parseSchema,
  toJsonSchema,
  validateSchema,
  type JSONSchema,
  type ProvidedSchema,
  type ValidationErrorDetail,
  type ValidationResponse,
} from '@genkit-ai/core/schema';
