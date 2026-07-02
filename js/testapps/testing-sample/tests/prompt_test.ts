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

import { echoModel } from 'genkit/testing';
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { ai, recommendPrompt } from '../src/menu.js';

// This file registers `echoModel` under the app's default model name. It lives
// in its own test file because menu_test.ts already claims 'menuModel' with a
// `mockModel` — and since the test runner isolates each file (`node --test`
// spawns a child process per file, so each gets a fresh registry), the two
// registrations never meet.
echoModel(ai, { name: 'menuModel', info: { supports: { tools: true } } });

describe('prompt assembly with echoModel', () => {
  it('shows the full rendered request — system + interpolated template', async () => {
    // echoModel echoes the whole assembled conversation, so we can assert on
    // the system instruction *and* the Handlebars-rendered user message — what
    // the model would have seen — without a live model.
    const res = await recommendPrompt({
      restaurant: 'Lumen',
      mood: 'tired',
      budgetUSD: 40,
    });

    assert.match(res.text, /system: You are a concise restaurant concierge/);
    assert.match(
      res.text,
      /Recommend a dish at Lumen for someone feeling tired\. Their budget is 40 USD/
    );
  });
});
