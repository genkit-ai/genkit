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

/**
 * `@genkit-ai/a2ui` — A2UI (Agent-to-UI) streaming UI protocol support for
 * Genkit agents.
 *
 * The whole server-side integration is the {@link a2ui} model middleware; add it
 * to `defineAgent({ use: [...] })` or `ai.generate({ use: [...] })`. Pair it with
 * {@link basicCatalog} (or your own catalog) and render on the client with the
 * `@a2ui/lit` renderer plus the helpers in `@genkit-ai/a2ui/client`.
 *
 * @module
 */

export {
  A2UI_CATALOG_VALUE_TYPE,
  DEFAULT_CATALOG_ID,
  SURFACE_ID_PLACEHOLDER,
  basicCatalog,
  renderCatalogInstructions,
  type A2uiCatalog,
  type A2uiCatalogComponent,
} from './catalog.js';
export { loadCatalog, type LoadCatalogOptions } from './loader.js';
export { A2uiOptionsSchema, a2ui, type A2uiOptions } from './middleware.js';
export {
  A2uiStreamParser,
  type A2uiParserOptions,
  type ParseResult,
} from './parser.js';
export { a2uiEnvelopesFromParts, a2uiPart, isA2uiPart } from './part.js';
export {
  A2UI_MIME_TYPE,
  A2UI_VERSION,
  BASIC_CATALOG_ID,
  type A2uiClientAction,
  type A2uiComponent,
  type A2uiEnvelope,
  type A2uiPart,
  type CreateSurfaceEnvelope,
  type DeleteSurfaceEnvelope,
  type UpdateComponentsEnvelope,
  type UpdateDataModelEnvelope,
} from './types.js';
