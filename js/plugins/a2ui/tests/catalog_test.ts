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
import { describe, it } from 'node:test';
import {
  BASIC_ICON_NAMES,
  basicCatalog,
  renderCatalogInstructions,
} from '../src/catalog.js';

describe('basicCatalog', () => {
  it('has a stable id and uniquely-named components', () => {
    assert.ok(basicCatalog.id.length > 0);
    const names = basicCatalog.components.map((c) => c.name);
    assert.strictEqual(new Set(names).size, names.length);
  });
});

describe('renderCatalogInstructions', () => {
  const text = renderCatalogInstructions(basicCatalog);

  it('includes the catalog id', () => {
    assert.ok(text.includes(basicCatalog.id));
  });

  it('lists every catalog component name', () => {
    for (const c of basicCatalog.components) {
      assert.ok(
        text.includes(`- ${c.name}:`),
        `expected instructions to document component ${c.name}`
      );
    }
  });

  it('tells the model to use the SURFACE_ID placeholder', () => {
    assert.ok(text.includes('SURFACE_ID'));
  });

  it('lists the basic icon allow-list', () => {
    assert.ok(text.includes(BASIC_ICON_NAMES[0]));
  });
});
