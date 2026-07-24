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
 * Catalog loading helpers.
 *
 * The {@link a2ui} middleware references catalogs by id and looks them up in the
 * Genkit registry (value type {@link A2UI_CATALOG_VALUE_TYPE}). {@link loadCatalog}
 * registers a catalog under an id so the middleware — and, in the future, Genkit
 * tooling — can find it. Provide the catalog inline, or load it from a JSON file.
 *
 * This module uses Node's `fs` for the `file` variant, so it is server-only (it
 * is exported from the main `@genkit-ai/a2ui` entry, not `@genkit-ai/a2ui/client`).
 *
 * @module
 */

import type { HasRegistry } from 'genkit/registry';

import {
  A2UI_CATALOG_VALUE_TYPE,
  DEFAULT_CATALOG_ID,
  basicCatalog,
  type A2uiCatalog,
} from './catalog.js';

/** Options for {@link loadCatalog}. Provide either `catalog` or `file`. */
export type LoadCatalogOptions = {
  /** The id to register the catalog under (referenced by `a2ui({ catalog })`). */
  id: string;
} & (
  | {
      /** An in-memory catalog to register. */
      catalog: A2uiCatalog;
      file?: never;
    }
  | {
      /** Path to a JSON file containing a catalog. */
      file: string;
      catalog?: never;
    }
);

/**
 * Registers an A2UI catalog under an id so the {@link a2ui} middleware can look
 * it up (via `a2ui({ catalog: id })`).
 *
 * @example
 * ```ts
 * // From an in-memory object:
 * await loadCatalog(ai, { id: 'my-catalog', catalog: myCatalog });
 *
 * // From a JSON file:
 * await loadCatalog(ai, { id: 'my-catalog', file: './my-catalog.json' });
 * ```
 *
 * @returns The registered catalog (with its `id` set to the given id).
 */
export async function loadCatalog(
  ai: HasRegistry,
  options: LoadCatalogOptions
): Promise<A2uiCatalog> {
  const { id } = options;
  if (!id) {
    throw new Error('loadCatalog(): `id` is required.');
  }

  let catalog: A2uiCatalog;
  if (options.file !== undefined) {
    catalog = await readCatalogFile(options.file);
  } else if (options.catalog !== undefined) {
    catalog = options.catalog;
  } else {
    throw new Error('loadCatalog(): provide either `catalog` or `file`.');
  }

  if (!catalog || !Array.isArray(catalog.components)) {
    throw new Error(
      `loadCatalog(): catalog "${id}" must have a "components" array.`
    );
  }

  // Register under the requested id. Keep the catalog's own `id` (used as the
  // `catalogId` on surfaces) intact if present; otherwise default it to the id.
  const registered: A2uiCatalog = { ...catalog, id: catalog.id ?? id };
  ai.registry.registerValue(A2UI_CATALOG_VALUE_TYPE, id, registered);
  return registered;
}

/** Reads and parses a catalog from a JSON file. */
async function readCatalogFile(file: string): Promise<A2uiCatalog> {
  // Import lazily so this module stays importable in non-Node contexts until
  // the `file` variant is actually used.
  const { readFile } = await import('node:fs/promises');
  let raw: string;
  try {
    raw = await readFile(file, 'utf8');
  } catch (e) {
    throw new Error(
      `loadCatalog(): failed to read catalog file "${file}": ${(e as Error).message}`
    );
  }
  try {
    return JSON.parse(raw) as A2uiCatalog;
  } catch (e) {
    throw new Error(
      `loadCatalog(): catalog file "${file}" is not valid JSON: ${(e as Error).message}`
    );
  }
}

/**
 * Resolves a catalog by id from the registry, falling back to the bundled
 * {@link basicCatalog} for the default id. Used by the {@link a2ui} middleware.
 *
 * @internal
 */
export async function resolveCatalog(
  ai: HasRegistry,
  id: string
): Promise<A2uiCatalog> {
  const found = await ai.registry.lookupValue<A2uiCatalog>(
    A2UI_CATALOG_VALUE_TYPE,
    id
  );
  if (found) return found;
  if (id === DEFAULT_CATALOG_ID) return basicCatalog;
  throw new Error(
    `a2ui(): no catalog registered under id "${id}". ` +
      `Register one with loadCatalog(ai, { id: "${id}", catalog }) or use the ` +
      `default "${DEFAULT_CATALOG_ID}" catalog.`
  );
}
