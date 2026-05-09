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

import * as assert from 'assert';
import * as fs from 'fs/promises';
import { genkit } from 'genkit';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as os from 'os';
import * as path from 'path';
import { skills } from '../src/skills.js';

describe('skills middleware', () => {
  let tempDir: string;
  let skillsDir: string;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'genkit-skills-test-'));
    skillsDir = path.join(tempDir, 'skills');
    await fs.mkdir(skillsDir);

    // Create a dummy skill
    const pythonSkillDir = path.join(skillsDir, 'python');
    await fs.mkdir(pythonSkillDir);
    await fs.writeFile(
      path.join(pythonSkillDir, 'SKILL.md'),
      '---\nname: python\ndescription: A python expert skill\n---\nPython prompt content'
    );

    // Create another skill without description
    const jsSkillDir = path.join(skillsDir, 'javascript');
    await fs.mkdir(jsSkillDir);
    await fs.writeFile(
      path.join(jsSkillDir, 'SKILL.md'),
      'Just javascript content'
    );
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  function createToolModel(ai: any, toolName: string, input: any) {
    let turn = 0;
    return ai.defineModel(
      { name: `pm-${toolName}-${Math.random()}` },
      async () => {
        turn++;
        if (turn === 1) {
          return {
            message: {
              role: 'model',
              content: [{ toolRequest: { name: toolName, input } }],
            },
          };
        }
        return { message: { role: 'model', content: [{ text: 'done' }] } };
      }
    );
  }

  it('injects system prompt with available skills', async () => {
    const ai = genkit({});

    // We want to see the messages passed to the model, so we can define a mock model
    // that captures the messages it receives.
    let capturedMessages: any[] = [];
    const mockModel = ai.defineModel({ name: 'capture-model' }, async (req) => {
      capturedMessages = req.messages;
      return {
        message: { role: 'model', content: [{ text: 'mock response' }] },
      };
    });

    await ai.generate({
      model: mockModel,
      prompt: 'hello',
      use: [skills({ skillPaths: [skillsDir] })],
    });

    // Verify system message exists and contains skills
    const sysMsg = capturedMessages.find((m) => m.role === 'system');
    assert.ok(sysMsg);
    assert.match(sysMsg.content[0].text, /python - A python expert skill/);
    assert.match(sysMsg.content[0].text, /javascript/);
  });

  it('grants access to use_skill tool', async () => {
    const ai = genkit({});
    const pm = createToolModel(ai, 'use_skill', { skillName: 'python' });

    const result = (await ai.generate({
      model: pm,
      prompt: 'use python skill',
      use: [skills({ skillPaths: [skillsDir] })],
    })) as any;

    const toolMsg = result.messages.find((m: any) => m.role === 'tool');
    assert.ok(toolMsg);
    assert.match(
      toolMsg.content[0].toolResponse.output,
      /Python prompt content/
    );
  });

  it('rejects access to unknown skills', async () => {
    const ai = genkit({});
    const pm = createToolModel(ai, 'use_skill', { skillName: 'nonexistent' });

    await assert.rejects(async () => {
      await ai.generate({
        model: pm,
        prompt: 'use skill',
        use: [skills({ skillPaths: [skillsDir] })],
      });
    }, /not found/);
  });

  it('is idempotent when injecting prompt', async () => {
    const ai = genkit({});

    let capturedMessages: any[] = [];
    const mockModel = ai.defineModel(
      { name: 'capture-model-' + Math.random() },
      async (req) => {
        capturedMessages = req.messages;
        return {
          message: { role: 'model', content: [{ text: 'mock response' }] },
        };
      }
    );

    // First call
    const response = await ai.generate({
      model: mockModel,
      prompt: 'hello',
      use: [skills({ skillPaths: [skillsDir] })],
    });

    const firstSysMsg = capturedMessages.find((m) => m.role === 'system');
    assert.ok(firstSysMsg);

    // Count occurrences of "<skills>" in the first system message
    const firstCount = (firstSysMsg.content[0].text.match(/<skills>/g) || [])
      .length;
    assert.strictEqual(firstCount, 1);

    // Second call (simulating multi-turn scenario by passing messages back)
    await ai.generate({
      model: mockModel,
      messages: response.messages, // pass history back
      use: [skills({ skillPaths: [skillsDir] })],
    });

    const secondSysMsg = capturedMessages.find((m) => m.role === 'system');
    assert.ok(secondSysMsg);

    // Count occurrences of "<skills>" in the second system message
    const secondCount = (secondSysMsg.content[0].text.match(/<skills>/g) || [])
      .length;
    assert.strictEqual(secondCount, 1, 'Should not duplicate skills block');
  });
});
