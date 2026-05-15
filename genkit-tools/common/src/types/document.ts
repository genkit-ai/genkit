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
import z from 'zod';
import { MediaPartSchema, TextPartSchema } from './parts';

//
// IMPORTANT: Keep this file in sync with genkit/ai/src/document.ts!
//

// Disclaimer: genkit/js/ai/document.ts defines the following schema, type pair
// as PartSchema and Part, respectively. genkit-tools cannot retain those names
// due to it clashing with similar schema in model.ts, and genkit-tools
// exporting all types at root. We use a different name here and updated
// coresponding the imports.
export const DocumentPartSchema = z.union([TextPartSchema, MediaPartSchema]);
export type DocumentPart = z.infer<typeof DocumentPartSchema>;

export const DocumentDataSchema = z.object({
  content: z.array(DocumentPartSchema),
  metadata: z.record(z.string(), z.any()).optional(),
});
export type DocumentData = z.infer<typeof DocumentDataSchema>;
