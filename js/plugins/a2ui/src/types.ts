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
 * Shared A2UI protocol types and constants.
 *
 * This module is intentionally free of any Node-only dependencies so it can be
 * imported both on the server (the {@link a2ui} middleware) and in the browser
 * (the client transport helpers). It mirrors the wire shapes of the A2UI v0.9
 * specification without depending on `@a2ui/web_core` (the renderer packages are
 * only needed on the client).
 *
 * @module
 */

/**
 * The MIME type that identifies an A2UI payload. This is stamped onto the
 * `metadata.mimeType` of the Genkit `data` part that carries A2UI envelopes,
 * matching the A2A binding of the A2UI spec exactly.
 */
export const A2UI_MIME_TYPE = 'application/a2ui+json';

/** The default A2UI protocol version stamped on emitted envelopes. */
export const A2UI_VERSION = 'v0.9';

/**
 * The catalog id of the A2UI "Basic Catalog" (v0.9). Surfaces created with the
 * basic catalog reference this id, and the client renderer registers a catalog
 * under the same id.
 */
export const BASIC_CATALOG_ID =
  'https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json';

export type SupportedVersion = 'v0.9' | `v0.9.1`;

/**
 * A single component entry in an A2UI adjacency list. UI is expressed as a flat
 * list of components; the tree is reconstructed via `id` references. Exactly one
 * component has `id: "root"`.
 *
 * Beyond `component`/`id`/`weight`, every component carries catalog-specific
 * props (e.g. `text`, `children`, `child`, `action`), so this is intentionally
 * open-ended.
 */
export interface A2uiComponent {
  /** The catalog component type name, e.g. `Text`, `Column`, `Button`. */
  component: string;
  /** The unique id of this component. Exactly one component must be `"root"`. */
  id?: string;
  /** Relative flex weight when nested directly in a Row/Column. */
  weight?: number;
  /** Catalog-specific component props. */
  [prop: string]: unknown;
}

/** Creates (or re-initializes) a UI surface. */
export interface CreateSurfaceEnvelope {
  version: SupportedVersion;
  createSurface: {
    surfaceId: string;
    catalogId: string;
    theme?: unknown;
    sendDataModel?: boolean;
  };
}

/** Adds or replaces components on an existing surface. */
export interface UpdateComponentsEnvelope {
  version: SupportedVersion;
  updateComponents: {
    surfaceId: string;
    components: A2uiComponent[];
  };
}

/** Mutates a value in a surface's per-surface data model. */
export interface UpdateDataModelEnvelope {
  version: SupportedVersion;
  updateDataModel: {
    surfaceId: string;
    /** JSON-Pointer path. When omitted, `value` replaces the whole model. */
    path?: string;
    value?: unknown;
  };
}

/** Removes a surface. */
export interface DeleteSurfaceEnvelope {
  version: SupportedVersion;
  deleteSurface: {
    surfaceId: string;
  };
}

/** A single server → client A2UI envelope message. */
export type A2uiEnvelope =
  | CreateSurfaceEnvelope
  | UpdateComponentsEnvelope
  | UpdateDataModelEnvelope
  | DeleteSurfaceEnvelope;

/**
 * The canonical "a2ui part": a Genkit `data` part whose `data` is an object
 * `{ envelopes }` wrapping the array of A2UI envelopes, and whose
 * `metadata.mimeType` is {@link A2UI_MIME_TYPE}.
 *
 * The array is wrapped in an object (rather than being the `data` value
 * directly) so the payload is a `Map<String, dynamic>`-shaped object on every
 * runtime — some (e.g. Dart) expect a data part's `data` to be an object, not a
 * bare array.
 */
export interface A2uiPart {
  data: { envelopes: A2uiEnvelope[] };
  metadata: {
    mimeType: typeof A2UI_MIME_TYPE;
    [key: string]: unknown;
  };
}

/**
 * A client → server user action reported by a rendered surface (e.g. a button
 * press). Sent back to the agent as the next turn's input.
 */
export interface A2uiClientAction {
  name: string;
  surfaceId: string;
  sourceComponentId: string;
  timestamp: string;
  context: Record<string, unknown>;
}
