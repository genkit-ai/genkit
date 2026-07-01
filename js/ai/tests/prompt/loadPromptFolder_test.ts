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

import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import { mkdirSync, mkdtempSync, writeFileSync } from 'fs';
import assert from 'node:assert';
import { beforeEach, describe, it } from 'node:test';
import { tmpdir } from 'os';
import { join } from 'path';
import { loadPromptFolder, prompt } from '../../src/prompt.js';
import { defineEchoModel } from '../helpers.js';

initNodeFeatures();

describe('loadPromptFolder', () => {
  let registry: Registry;
  let promptDir: string;

  beforeEach(() => {
    registry = new Registry();
    defineEchoModel(registry);
    promptDir = mkdtempSync(join(tmpdir(), 'genkit-prompts-'));
  });

  it('registers nested partials with subdirectory namespace', async () => {
    mkdirSync(join(promptDir, 'flowA'), { recursive: true });
    mkdirSync(join(promptDir, 'flowB'), { recursive: true });
    writeFileSync(join(promptDir, 'flowA', '_system.prompt'), 'system A');
    writeFileSync(join(promptDir, 'flowB', '_system.prompt'), 'system B');
    writeFileSync(
      join(promptDir, 'flowA', 'main.prompt'),
      `---
model: echoModel
---
{{> flowA/system}}`
    );
    writeFileSync(
      join(promptDir, 'flowB', 'main.prompt'),
      `---
model: echoModel
---
{{> flowB/system}}`
    );

    loadPromptFolder(registry, promptDir, '');

    const flowA = await prompt(registry, 'flowA/main');
    const flowB = await prompt(registry, 'flowB/main');

    const renderedA = await flowA.render({});
    const renderedB = await flowB.render({});

    assert.strictEqual(renderedA.messages[0].content[0].text, 'system A');
    assert.strictEqual(renderedB.messages[0].content[0].text, 'system B');
  });

  it('registers root-level partials without subdirectory prefix', async () => {
    writeFileSync(join(promptDir, '_greeting.prompt'), 'hello');
    writeFileSync(
      join(promptDir, 'test.prompt'),
      `---
model: echoModel
---
{{> greeting}}`
    );

    loadPromptFolder(registry, promptDir, '');

    const testPrompt = await prompt(registry, 'test');
    const rendered = await testPrompt.render({});
    assert.strictEqual(rendered.messages[0].content[0].text, 'hello');
  });
});
