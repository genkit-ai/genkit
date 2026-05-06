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

/**
 * Non-regression isolation tests.
 *
 * The Responses-API code path lives under `src/openai/responses/` and is
 * an OpenAI-only feature (no other compat provider supports the
 * `/v1/responses` endpoint). These tests guarantee that:
 *
 *  1. Importing a non-OpenAI compat provider (DeepSeek, XAI) does NOT
 *     transitively load any Responses API source files. This is enforced
 *     by inspecting `require.cache` after a clean module-cache reset and
 *     loading only the provider entry point.
 *  2. The OpenAI plugin itself does load the Responses files (sanity
 *     check that the new path is wired in).
 *
 * If a future refactor pulls a `responses/*` import into `src/model.ts`
 * or `src/index.ts`, these tests fail fast — protecting downstream
 * compat providers from accidental schema or runtime leakage.
 */

import { beforeEach, describe, expect, it } from '@jest/globals';

function loadedResponsesFiles(): string[] {
  return Object.keys(require.cache).filter((p) =>
    // Match the responses/ fragment regardless of workspace layout
    // (./lib vs ./src vs symlinked node_modules).
    p.includes('openai/responses')
  );
}

function clearCompatOaiFromCache() {
  for (const key of Object.keys(require.cache)) {
    if (
      key.includes('@genkit-ai/compat-oai') ||
      key.includes('plugins/compat-oai/lib') ||
      key.includes('plugins/compat-oai/src')
    ) {
      delete require.cache[key];
    }
  }
}

describe('compat-oai isolation: Responses API code path is OpenAI-only', () => {
  beforeEach(() => {
    clearCompatOaiFromCache();
  });

  it('importing the deepseek subpackage does not load responses/*', () => {
    require('../src/deepseek');
    expect(loadedResponsesFiles()).toEqual([]);
  });

  it('importing the xai subpackage does not load responses/*', () => {
    require('../src/xai');
    expect(loadedResponsesFiles()).toEqual([]);
  });

  it('importing the openai subpackage DOES load responses/* (sanity)', () => {
    require('../src/openai');
    const loaded = loadedResponsesFiles();
    expect(loaded.length).toBeGreaterThan(0);
    expect(
      loaded.some((p) => p.includes('responses') && p.includes('types'))
    ).toBe(true);
  });
});
