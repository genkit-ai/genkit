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
import { describe, expect, it } from 'vitest';
import { ai, recommendPrompt } from '../src/menu.js';

// `echoModel` gets its own test file because menu.vitest.test.ts already
// claims 'menuModel' with a `mockModel` — vitest gives each file its own
// worker (and so a fresh registry), so the two registrations never meet.
echoModel(ai, { name: 'menuModel', info: { supports: { tools: true } } });

describe('prompt assembly with echoModel (vitest)', () => {
  it('shows the full rendered request — system + interpolated template', async () => {
    const res = await recommendPrompt({
      restaurant: 'Lumen',
      mood: 'tired',
      budgetUSD: 40,
    });

    expect(res.text).toMatch(/system: You are a concise restaurant concierge/);
    expect(res.text).toMatch(
      /Recommend a dish at Lumen for someone feeling tired\. Their budget is 40 USD/
    );
  });
});
