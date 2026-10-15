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
import { genkit, type MessageData } from 'genkit';
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
    assert.match(
      sysMsg.content[0].text,
      /Call the use_skill tool with a skill's name to load its instructions\./
    );
    assert.deepStrictEqual(sysMsg.content[0].metadata, {
      'skills-instructions': true,
      skillsActivationTool: 'use_skill',
    });
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

  it('returns available names for an unknown skill without aborting generation', async () => {
    const ai = genkit({});
    const pm = createToolModel(ai, 'use_skill', { skillName: 'nonexistent' });

    const result = await ai.generate({
      model: pm,
      prompt: 'use skill',
      use: [skills({ skillPaths: [skillsDir] })],
    });
    const response = result.messages.find((m) => m.role === 'tool');
    assert.strictEqual(
      response?.content[0].toolResponse?.output,
      'Unknown skill "nonexistent". Available skills: "javascript", "python".'
    );
    assert.strictEqual(result.text, 'done');
  });

  it('lets the model retry an unknown skill with an available name', async () => {
    const ai = genkit({});
    let turn = 0;
    const model = ai.defineModel({ name: 'retry-skill' }, async () => ({
      message: {
        role: 'model',
        content:
          turn++ < 2
            ? [
                {
                  toolRequest: {
                    name: 'use_skill',
                    input: { skillName: turn === 1 ? 'missing' : 'python' },
                  },
                },
              ]
            : [{ text: 'recovered' }],
      },
    }));
    const result = await ai.generate({
      model,
      prompt: 'use a skill',
      use: [skills({ skillPaths: [skillsDir] })],
    });
    const outputs = result.messages
      .flatMap((m) => m.content)
      .filter((p) => p.toolResponse)
      .map((p) => p.toolResponse!.output);
    assert.ok(typeof outputs[0] === 'string');
    assert.ok(typeof outputs[1] === 'string');
    assert.match(outputs[0], /Available skills: "javascript", "python"/);
    assert.match(outputs[1], /Python prompt content/);
    assert.strictEqual(result.text, 'recovered');
  });

  it('returns a useful response when no skills are available', async () => {
    const ai = genkit({});
    const model = createToolModel(ai, 'use_skill', { skillName: 'python' });
    const result = await ai.generate({
      model,
      prompt: 'use a skill',
      use: [skills({ skillPaths: [] })],
    });
    const response = result.messages.find((m) => m.role === 'tool');
    assert.strictEqual(
      response?.content[0].toolResponse?.output,
      'Unknown skill "python". No skills available.'
    );
    assert.strictEqual(result.text, 'done');
  });

  it('still reports a file that becomes unreadable after discovery', async () => {
    const ai = genkit({});
    const model = ai.defineModel({ name: 'removed-skill' }, async () => {
      await fs.unlink(path.join(skillsDir, 'python', 'SKILL.md'));
      return {
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'use_skill',
                input: { skillName: 'python' },
              },
            },
          ],
        },
      };
    });
    await assert.rejects(
      ai.generate({
        model,
        prompt: 'use a skill',
        use: [skills({ skillPaths: [skillsDir] })],
      }),
      /Failed to read skill "python"/
    );
  });

  for (const owner of [undefined, 123, 'use_skill', 'other_use_skill', '']) {
    it(`refreshes only its own catalog when the previous owner is ${owner}`, async () => {
      const ai = genkit({});
      let capturedMessages: MessageData[] = [];
      const model = ai.defineModel({ name: 'catalog-owner' }, async (req) => {
        capturedMessages = req.messages;
        return { message: { role: 'model', content: [{ text: 'done' }] } };
      });
      const messages: MessageData[] = [
        {
          role: 'system',
          content: [
            {
              text: '<skills>old catalog</skills>',
              metadata: {
                'skills-instructions': true,
                ...(owner !== undefined ? { skillsActivationTool: owner } : {}),
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'hello' }] },
      ];
      const original = structuredClone(messages);
      await ai.generate({
        model,
        messages,
        use: [skills({ skillPaths: [skillsDir] })],
      });
      const catalogs = capturedMessages
        .flatMap((m) => m.content)
        .filter((p) => p.metadata?.['skills-instructions']);
      const isForeign = typeof owner === 'string' && owner !== 'use_skill';
      assert.strictEqual(catalogs.length, isForeign ? 2 : 1);
      const ours = catalogs.find(
        (p) => p.metadata?.skillsActivationTool === 'use_skill'
      );
      assert.ok(ours);
      assert.match(ours.text!, /python - A python expert skill/);
      if (isForeign) {
        assert.deepStrictEqual(catalogs[0], original[0].content[0]);
      }
      assert.deepStrictEqual(messages, original);
    });
  }

  it('adds ownership to an unchanged legacy catalog without duplicating it', async () => {
    const ai = genkit({});
    const model = ai.defineModel({ name: 'legacy-catalog' }, async () => ({
      message: { role: 'model', content: [{ text: 'done' }] },
    }));
    const response = await ai.generate({
      model,
      prompt: 'hello',
      use: [skills({ skillPaths: [skillsDir] })],
    });
    const messages = structuredClone(response.messages);
    const catalog = messages
      .flatMap((m) => m.content)
      .find((p) => p.metadata?.['skills-instructions']);
    assert.ok(catalog);
    delete catalog.metadata!.skillsActivationTool;

    const nextResponse = await ai.generate({
      model,
      messages,
      use: [skills({ skillPaths: [skillsDir] })],
    });
    const catalogs = nextResponse.messages
      .flatMap((m) => m.content)
      .filter((p) => p.metadata?.['skills-instructions']);
    assert.strictEqual(catalogs.length, 1);
    assert.strictEqual(catalogs[0].text, catalog.text);
    assert.strictEqual(catalogs[0].metadata?.skillsActivationTool, 'use_skill');
    assert.strictEqual(catalog.metadata?.skillsActivationTool, undefined);
  });

  it('discovers .agents/skills by default and lets skills override duplicates', async () => {
    const agentSkillsDir = path.join(tempDir, '.agents', 'skills');
    await fs.mkdir(path.join(agentSkillsDir, 'python'), { recursive: true });
    await fs.writeFile(
      path.join(agentSkillsDir, 'python', 'SKILL.md'),
      '---\ndescription: Shadowed description\n---\nShadowed instructions'
    );
    await fs.mkdir(path.join(agentSkillsDir, 'typescript'));
    await fs.writeFile(
      path.join(agentSkillsDir, 'typescript', 'SKILL.md'),
      '---\ndescription: TypeScript expert\n---\nTypeScript instructions'
    );
    const previousCwd = process.cwd();
    try {
      process.chdir(tempDir);
      const ai = genkit({});
      const model = createToolModel(ai, 'use_skill', { skillName: 'python' });
      const result = await ai.generate({
        model,
        prompt: 'hello',
        use: [skills()],
      });
      const catalog = result.messages.find((m) => m.role === 'system');
      assert.match(catalog!.content[0].text!, /typescript - TypeScript expert/);
      assert.doesNotMatch(catalog!.content[0].text!, /Shadowed description/);
      const response = result.messages.find((m) => m.role === 'tool');
      const output = response!.content[0].toolResponse!.output;
      assert.ok(typeof output === 'string');
      assert.match(output, /Python prompt content/);

      const explicitPaths = await ai.generate({
        model,
        prompt: 'hello',
        use: [skills({ skillPaths: [skillsDir] })],
      });
      const explicitCatalog = explicitPaths.messages.find(
        (m) => m.role === 'system'
      );
      assert.match(
        explicitCatalog!.content[0].text!,
        /python - A python expert skill/
      );
      assert.doesNotMatch(explicitCatalog!.content[0].text!, /typescript/);
    } finally {
      process.chdir(previousCwd);
    }
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
