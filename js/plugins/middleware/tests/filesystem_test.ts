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
import { filesystem } from '../src/filesystem.js';

describe('filesystem middleware', () => {
  let tempDir: string;
  let fakeGenerateAPI: any = {};

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(
      path.join(os.tmpdir(), 'genkit-filesystem-test-')
    );
    await fs.mkdir(path.join(tempDir, 'sub'));
    await fs.writeFile(path.join(tempDir, 'file1.txt'), 'hello world');
    await fs.writeFile(path.join(tempDir, 'sub', 'file2.txt'), 'sub file');
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  function createToolModel(ai: any, toolName: string, input: any) {
    let turn = 0;
    return ai.defineModel(
      { name: `pm-${toolName}-${Math.random()}` },
      async (_, sendChunk) => {
        turn++;
        if (turn === 1) {
          const content = [{ toolRequest: { name: toolName, input } }];
          if (sendChunk) {
            sendChunk({ content });
          }

          return {
            message: {
              role: 'model',
              content,
            },
          };
        }
        const content = [{ text: 'done' }];
        if (sendChunk) {
          sendChunk({ content });
        }
        return { message: { role: 'model', content } };
      }
    );
  }

  it('fails if rootDirectory is not provided', () => {
    assert.throws(
      () =>
        filesystem.instantiate({
          config: {} as any,
          ai: fakeGenerateAPI,
          pluginConfig: undefined,
        }),
      /requires a rootDirectory option/
    );
  });

  it('injects all tools when allowWriteAccess is true', () => {
    const mw = filesystem.instantiate({
      config: { rootDirectory: tempDir, allowWriteAccess: true },
      ai: fakeGenerateAPI,
      pluginConfig: undefined,
    });
    assert.ok(mw.tools);
    assert.strictEqual(mw.tools.length, 4);
    assert.strictEqual(mw.tools[0].__action.name, 'list_files');
    assert.strictEqual(mw.tools[1].__action.name, 'read_file');
    assert.strictEqual(mw.tools[2].__action.name, 'write_file');
    assert.strictEqual(mw.tools[3].__action.name, 'search_and_replace');
  });

  it('injects only readonly tools by default', () => {
    const mw = filesystem.instantiate({
      config: { rootDirectory: tempDir },
      ai: fakeGenerateAPI,
      pluginConfig: undefined,
    });
    assert.ok(mw.tools);
    assert.strictEqual(mw.tools.length, 2);
    assert.strictEqual(mw.tools[0].__action.name, 'list_files');
    assert.strictEqual(mw.tools[1].__action.name, 'read_file');
  });

  it('injects tools with prefix when toolNamePrefix is provided', () => {
    const mw = filesystem.instantiate({
      config: {
        rootDirectory: tempDir,
        toolNamePrefix: 'my_',
        allowWriteAccess: true,
      },
      ai: fakeGenerateAPI,
      pluginConfig: undefined,
    });
    assert.ok(mw.tools);
    assert.strictEqual(mw.tools.length, 4);
    assert.strictEqual(mw.tools[0].__action.name, 'my_list_files');
    assert.strictEqual(mw.tools[1].__action.name, 'my_read_file');
    assert.strictEqual(mw.tools[2].__action.name, 'my_write_file');
    assert.strictEqual(mw.tools[3].__action.name, 'my_search_and_replace');
  });

  describe('list_files', () => {
    it('lists files in root directory', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'list_files', { dirPath: '' });
      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const toolMsg = result.messages.find((m: any) => m.role === 'tool');
      assert.ok(toolMsg);
      const output = toolMsg.content[0].toolResponse.output;
      assert.ok(
        output.find((r: any) => r.path === 'file1.txt' && !r.isDirectory)
      );
      assert.ok(output.find((r: any) => r.path === 'sub' && r.isDirectory));
      assert.ok(
        !output.find((r: any) => r.path === path.join('sub', 'file2.txt'))
      );
    });

    it('lists files recursively', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'list_files', {
        dirPath: '',
        recursive: true,
      });
      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const toolMsg = result.messages.find((m: any) => m.role === 'tool');
      assert.ok(toolMsg);
      const output = toolMsg.content[0].toolResponse.output;
      assert.ok(
        output.find((r: any) => r.path === 'file1.txt' && !r.isDirectory)
      );
      assert.ok(output.find((r: any) => r.path === 'sub' && r.isDirectory));
      assert.ok(
        output.find(
          (r: any) => r.path === path.join('sub', 'file2.txt') && !r.isDirectory
        )
      );
    });

    it('rejects listing outside root directory', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'list_files', { dirPath: '../' });

      // The middleware catches errors and injects user message.
      // So verify that user message contains access denied error.
      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const userMsg = result.messages.find(
        (m: any) =>
          m.role === 'user' && m.content[0].text.includes('Access denied')
      );
      assert.ok(userMsg);
    });

    it('allows listing when root is /', async () => {
      if (os.platform() === 'win32') return;
      const ai = genkit({});
      const pm = createToolModel(ai, 'list_files', { dirPath: 'tmp' });
      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: '/' })],
      })) as any;

      const userMsg = result.messages.find(
        (m: any) =>
          m.role === 'user' && m.content[0].text.includes('Access denied')
      );
      assert.ok(!userMsg, 'Should not fail with Access denied when root is /');
    });
  });

  describe('read_file', () => {
    it('reads a file in root directory', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'read_file', { filePath: 'file1.txt' });
      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const toolMsg = result.messages.find((m: any) => m.role === 'tool');
      assert.ok(toolMsg);
      assert.match(toolMsg.content[0].toolResponse.output, /read successfully/);

      const userMsg = result.messages.find(
        (m: any) =>
          m.role === 'user' && m.content[0].text.includes('<read_file')
      );
      assert.ok(userMsg);
      assert.ok(userMsg.content[0].text.includes('hello world'));
    });

    it('reads a file in sub directory', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'read_file', {
        filePath: 'sub/file2.txt',
      });
      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const toolMsg = result.messages.find((m: any) => m.role === 'tool');
      assert.ok(toolMsg);
      assert.match(toolMsg.content[0].toolResponse.output, /read successfully/);

      const userMsg = result.messages.find(
        (m: any) =>
          m.role === 'user' && m.content[0].text.includes('<read_file')
      );
      assert.ok(userMsg);
      assert.ok(userMsg.content[0].text.includes('sub file'));
    });

    it('streams file contents when reading a file', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'read_file', { filePath: 'file1.txt' });

      const { stream, response } = ai.generateStream({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir })],
      });

      const chunks: any[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      await response;

      const indices = chunks.map((c) => c.index);
      const uniqueIndices = [...new Set(indices)];

      // We expect indices 0, 1, 2, 3
      // 0: Model tool request
      // 1: Tool response string
      // 2: Injected file content <--- IMPORTANT
      // 3: Final model response
      assert.deepStrictEqual(
        uniqueIndices,
        [0, 1, 2, 3],
        'Should have sequential message indices'
      );

      const fileContentChunk = chunks.find(
        (c) => c.text && c.text.includes('hello world')
      );
      assert.ok(fileContentChunk, 'Stream should contain file content');
      assert.strictEqual(
        fileContentChunk.index,
        2,
        'File content should have index 2'
      );
    });

    it('rejects reading outside root directory', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'read_file', {
        filePath: '../etc/passwd',
      });

      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const userMsg = result.messages.find(
        (m: any) =>
          m.role === 'user' && m.content[0].text.includes('Access denied')
      );
      assert.ok(userMsg);
    });
  });

  describe('write_file', () => {
    it('writes a new file', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'write_file', {
        filePath: 'new.txt',
        content: 'new content',
      });
      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir, allowWriteAccess: true })],
      })) as any;

      const toolMsg = result.messages.find((m: any) => m.role === 'tool');
      assert.ok(toolMsg);
      assert.match(
        toolMsg.content[0].toolResponse.output,
        /written successfully/
      );

      const content = await fs.readFile(path.join(tempDir, 'new.txt'), 'utf8');
      assert.strictEqual(content, 'new content');
    });

    it('creates directories if needed', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'write_file', {
        filePath: 'deep/nested/file.txt',
        content: 'nested content',
      });
      await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir, allowWriteAccess: true })],
      });

      const content = await fs.readFile(
        path.join(tempDir, 'deep/nested/file.txt'),
        'utf8'
      );
      assert.strictEqual(content, 'nested content');
    });
  });

  describe('search_and_replace', () => {
    it('replaces content', async () => {
      const ai = genkit({});
      const editBlock = `<<<<<<< SEARCH
hello world
=======
hello universe
>>>>>>> REPLACE`;
      const pm = createToolModel(ai, 'search_and_replace', {
        filePath: 'file1.txt',
        edits: [editBlock],
      });

      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir, allowWriteAccess: true })],
      })) as any;

      const toolMsg = result.messages.find((m: any) => m.role === 'tool');
      if (!toolMsg) {
        const errorMsg = result.messages.find(
          (m: any) => m.role === 'user' && m.content[0].text.includes('failed')
        );
        if (errorMsg) {
          throw new Error(
            `Tool failed unexpectedly: ${errorMsg.content[0].text}`
          );
        }
      }
      assert.ok(toolMsg);
      assert.match(
        toolMsg.content[0].toolResponse.output,
        /Successfully applied/
      );

      const content = await fs.readFile(
        path.join(tempDir, 'file1.txt'),
        'utf8'
      );
      assert.strictEqual(content, 'hello universe');
    });

    it('fails if search content not found', async () => {
      const ai = genkit({});
      const editBlock = `<<<<<<< SEARCH
nonexistent
=======
replace
>>>>>>> REPLACE`;
      const pm = createToolModel(ai, 'search_and_replace', {
        filePath: 'file1.txt',
        edits: [editBlock],
      });

      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir, allowWriteAccess: true })],
      })) as any;

      const userMsg = result.messages.find(
        (m: any) =>
          m.role === 'user' &&
          m.content[0].text.includes('Search content not found')
      );
      if (!userMsg) {
        console.log(
          'Messages received:',
          JSON.stringify(result.messages, null, 2)
        );
      }
      assert.ok(userMsg);
    });

    it('handles tricky search/replace cases', async () => {
      const cases = [
        {
          name: 'marker in search',
          initial: 'line1\n=======\nline2',
          block: `<<<<<<< SEARCH
line1
=======
line2
=======
replacement
>>>>>>> REPLACE`,
          expected: 'replacement',
        },
        {
          name: 'marker in replace',
          initial: 'original',
          block: `<<<<<<< SEARCH
original
=======
new
=======
line
>>>>>>> REPLACE`,
          expected: 'new\n=======\nline',
        },
        {
          name: 'start marker in search',
          initial: '<<<<<<< SEARCH\ncontent',
          block: `<<<<<<< SEARCH
<<<<<<< SEARCH
content
=======
replaced
>>>>>>> REPLACE`,
          expected: 'replaced',
        },
        {
          name: 'start marker in replace',
          initial: 'content',
          block: `<<<<<<< SEARCH
content
=======
<<<<<<< SEARCH
new
>>>>>>> REPLACE`,
          expected: '<<<<<<< SEARCH\nnew',
        },
        {
          name: 'end marker in search',
          initial: 'content\n>>>>>>> REPLACE',
          block: `<<<<<<< SEARCH
content
>>>>>>> REPLACE
=======
replaced
>>>>>>> REPLACE`,
          expected: 'replaced',
        },
        {
          name: 'end marker in replace',
          initial: 'content',
          block: `<<<<<<< SEARCH
content
=======
new
>>>>>>> REPLACE
>>>>>>> REPLACE`,
          expected: 'new\n>>>>>>> REPLACE',
        },
        {
          name: 'multiple markers greedy search',
          initial: 'part1\n=======\npart2',
          block: `<<<<<<< SEARCH
part1
=======
part2
=======
replacement
>>>>>>> REPLACE`,
          expected: 'replacement',
        },
        {
          name: 'ambiguous separators preferring longest match',
          initial: 'A\n=======\nB',
          // search: A\n=======\nB -> replace: C\n=======\nD
          // block structure: A = B = C = D (where = is separator)
          // splits:
          // 1. S=A, R=B=C=D. (Match A? Yes)
          // 2. S=A=B, R=C=D. (Match A=B? Yes)
          // 3. S=A=B=C, R=D. (Match A=B=C? No)
          // Winner: 2.
          block: `<<<<<<< SEARCH
A
=======
B
=======
C
=======
D
>>>>>>> REPLACE`,
          expected: 'C\n=======\nD',
        },
      ];

      for (const c of cases) {
        const ai = genkit({});
        const filename = `tricky-${c.name.replace(/\s+/g, '-')}.txt`;
        await fs.writeFile(path.join(tempDir, filename), c.initial);

        const pm = createToolModel(ai, 'search_and_replace', {
          filePath: filename,
          edits: [c.block],
        });

        const result = (await ai.generate({
          model: pm,
          prompt: 'test',
          use: [filesystem({ rootDirectory: tempDir, allowWriteAccess: true })],
        })) as any;

        const toolMsg = result.messages.find((m: any) => m.role === 'tool');
        assert.ok(toolMsg, `Tool execution failed for case: ${c.name}`);
        assert.match(
          toolMsg.content[0].toolResponse.output,
          /Successfully applied/,
          `Tool output mismatch for case: ${c.name}`
        );

        const newContent = await fs.readFile(
          path.join(tempDir, filename),
          'utf8'
        );
        assert.strictEqual(
          newContent,
          c.expected,
          `Content mismatch for case: ${c.name}`
        );
      }
    });
  });

  describe('robustness', () => {
    it('should handle tool errors gracefully by injecting user message', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'read_file', { filePath: 'nonexistent' });

      const result = (await ai.generate({
        model: pm,
        prompt: 'start',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const messages = result.messages;
      const lastModelIndex = messages.findLastIndex(
        (m: any) => m.role === 'model' && m.content[0].toolRequest
      );
      const injectedUserIndex = messages.findIndex(
        (m: any) =>
          m.role === 'user' &&
          m.content[0].text.includes("Tool 'read_file' failed")
      );

      assert.ok(
        injectedUserIndex > lastModelIndex,
        'User message should appear after tool request'
      );

      const userMsg = messages[injectedUserIndex];
      assert.match(
        userMsg.content[0].text,
        /Tool 'read_file' failed: .*ENOENT.*/,
        'Error message should contain underlying error details'
      );

      const roles = messages.map((m: any) => m.role);
      assert.deepStrictEqual(roles, ['user', 'model', 'user', 'model']);

      const toolMsg = messages.find((m: any) => m.role === 'tool');
      assert.strictEqual(
        toolMsg,
        undefined,
        'Tool message should not be present'
      );
    });
  });
});

describe('filesystem middleware image support', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(
      path.join(os.tmpdir(), 'genkit-filesystem-test-')
    );
    await fs.writeFile(path.join(tempDir, 'image.png'), 'fake image content');
    await fs.writeFile(path.join(tempDir, 'unknown.xyz'), 'unknown content');
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

  it('reads an image file as media', async () => {
    const ai = genkit({});
    const pm = createToolModel(ai, 'read_file', { filePath: 'image.png' });
    const result = (await ai.generate({
      model: pm,
      prompt: 'test',
      use: [filesystem({ rootDirectory: tempDir })],
    })) as any;

    const toolMsg = result.messages.find((m: any) => m.role === 'tool');
    assert.ok(toolMsg);
    assert.match(toolMsg.content[0].toolResponse.output, /read successfully/);

    const userMsg = result.messages.find(
      (m: any) => m.role === 'user' && m.content.some((c: any) => c.media)
    );
    assert.ok(userMsg);
    const mediaPart = userMsg.content.find((c: any) => c.media);
    assert.ok(mediaPart);
    assert.strictEqual(mediaPart.media.contentType, 'image/png');
    assert.ok(mediaPart.media.url.startsWith('data:image/png;base64,'));
  });

  it('reads unknown file as text', async () => {
    const ai = genkit({});
    const pm = createToolModel(ai, 'read_file', { filePath: 'unknown.xyz' });
    const result = (await ai.generate({
      model: pm,
      prompt: 'test',
      use: [filesystem({ rootDirectory: tempDir })],
    })) as any;

    const userMsg = result.messages.find(
      (m: any) =>
        m.role === 'user' &&
        m.content.some((c: any) => c.text && c.text.includes('<read_file'))
    );
    assert.ok(userMsg);
    assert.ok(userMsg.content[0].text.includes('unknown content'));
    // Should NOT have media
    assert.ok(!userMsg.content.some((c: any) => c.media));
  });
});
