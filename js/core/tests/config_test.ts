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

import * as assert from 'assert';
import { afterEach, describe, it } from 'node:test';
import {
  getGenkitRuntimeConfig,
  resetGenkitRuntimeConfig,
  setGenkitRuntimeConfig,
} from '../src/config.js';

describe('config', () => {
  afterEach(() => {
    // Reset config
    setGenkitRuntimeConfig({
      jsonSchemaMode: undefined,
      sandboxedRuntime: undefined,
    });
  });

  it('should have default values', () => {
    const config = getGenkitRuntimeConfig();
    assert.strictEqual(config.jsonSchemaMode, 'compile');
    assert.strictEqual(config.sandboxedRuntime, false);
  });

  it('should update config values', () => {
    setGenkitRuntimeConfig({
      jsonSchemaMode: 'interpret',
      sandboxedRuntime: true,
    });
    const config = getGenkitRuntimeConfig();
    assert.strictEqual(config.jsonSchemaMode, 'interpret');
    assert.strictEqual(config.sandboxedRuntime, true);
  });

  it('should unset config values', () => {
    setGenkitRuntimeConfig({
      jsonSchemaMode: 'interpret',
      sandboxedRuntime: true,
    });
    let config = getGenkitRuntimeConfig();
    assert.strictEqual(config.jsonSchemaMode, 'interpret');
    assert.strictEqual(config.sandboxedRuntime, true);

    resetGenkitRuntimeConfig();

    config = getGenkitRuntimeConfig();
    assert.strictEqual(config.jsonSchemaMode, 'compile');
    assert.strictEqual(config.sandboxedRuntime, false);
  });
});
