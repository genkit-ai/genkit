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

import assert from 'node:assert';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, it } from 'node:test';
import {
  A2UI_CATALOG_VALUE_TYPE,
  DEFAULT_CATALOG_ID,
  basicCatalog,
  type A2uiCatalog,
} from '../src/catalog.js';
import { loadCatalog, resolveCatalog } from '../src/loader.js';

/** A minimal fake registry supporting the value APIs the loader uses. */
function fakeAi() {
  const store: Record<string, Record<string, unknown>> = {};
  return {
    registry: {
      registerValue(type: string, name: string, value: unknown) {
        (store[type] ??= {})[name] = value;
      },
      async lookupValue<T = unknown>(
        type: string,
        name: string
      ): Promise<T | undefined> {
        return store[type]?.[name] as T | undefined;
      },
    },
    store,
  };
}

const CUSTOM: A2uiCatalog = {
  id: 'my-catalog',
  components: [
    { name: 'Widget', description: 'A widget.', props: 'label: string.' },
  ],
};

describe('loadCatalog', () => {
  it('registers an in-memory catalog under its id', async () => {
    const ai = fakeAi();
    const result = await loadCatalog(ai as any, {
      id: 'my-catalog',
      catalog: CUSTOM,
    });
    assert.strictEqual(result.id, 'my-catalog');
    assert.strictEqual(ai.store[A2UI_CATALOG_VALUE_TYPE]['my-catalog'], result);
  });

  it('defaults the catalog id to the registration id when absent', async () => {
    const ai = fakeAi();
    const result = await loadCatalog(ai as any, {
      id: 'anon',
      catalog: { components: CUSTOM.components } as A2uiCatalog,
    });
    assert.strictEqual(result.id, 'anon');
  });

  it('throws when neither catalog nor file is provided', async () => {
    const ai = fakeAi();
    await assert.rejects(
      () => loadCatalog(ai as any, { id: 'x' } as any),
      /provide either `catalog` or `file`/
    );
  });

  it('throws when the catalog has no components array', async () => {
    const ai = fakeAi();
    await assert.rejects(
      () => loadCatalog(ai as any, { id: 'x', catalog: {} as A2uiCatalog }),
      /components/
    );
  });

  describe('from a file', () => {
    let dir: string;
    beforeEach(async () => {
      dir = await mkdtemp(join(tmpdir(), 'a2ui-loader-'));
    });
    afterEach(async () => {
      await rm(dir, { recursive: true, force: true });
    });

    it('loads and registers a catalog from a JSON file', async () => {
      const file = join(dir, 'catalog.json');
      await writeFile(file, JSON.stringify(CUSTOM), 'utf8');
      const ai = fakeAi();
      const result = await loadCatalog(ai as any, { id: 'my-catalog', file });
      assert.deepStrictEqual(result.components, CUSTOM.components);
    });

    it('throws on a missing file', async () => {
      const ai = fakeAi();
      await assert.rejects(
        () =>
          loadCatalog(ai as any, {
            id: 'x',
            file: join(dir, 'nope.json'),
          }),
        /failed to read catalog file/
      );
    });

    it('throws on invalid JSON', async () => {
      const file = join(dir, 'bad.json');
      await writeFile(file, '{not json}', 'utf8');
      const ai = fakeAi();
      await assert.rejects(
        () => loadCatalog(ai as any, { id: 'x', file }),
        /not valid JSON/
      );
    });
  });
});

describe('resolveCatalog', () => {
  it('resolves a registered catalog by id', async () => {
    const ai = fakeAi();
    await loadCatalog(ai as any, { id: 'my-catalog', catalog: CUSTOM });
    const resolved = await resolveCatalog(ai as any, 'my-catalog');
    assert.strictEqual(resolved.id, 'my-catalog');
  });

  it('falls back to the bundled basic catalog for the default id', async () => {
    const ai = fakeAi();
    const resolved = await resolveCatalog(ai as any, DEFAULT_CATALOG_ID);
    assert.strictEqual(resolved.id, basicCatalog.id);
  });

  it('throws for an unknown id', async () => {
    const ai = fakeAi();
    await assert.rejects(
      () => resolveCatalog(ai as any, 'nope'),
      /no catalog registered under id "nope"/
    );
  });
});
