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
import { toolApproval } from '../src/tool-approval.js';

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

      // The middleware catches errors and returns a tool response with the error.
      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const toolMsg = result.messages.find(
        (m: any) =>
          m.role === 'tool' &&
          m.content.some((c: any) =>
            c.toolResponse?.output?.toString()?.includes('Access denied')
          )
      );
      assert.ok(toolMsg);
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

      const toolMsg = result.messages.find(
        (m: any) =>
          m.role === 'tool' &&
          m.content.some((c: any) =>
            c.toolResponse?.output?.toString()?.includes('Access denied')
          )
      );
      assert.ok(toolMsg);
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
      const SEARCH_MARKER = '<<<<<<< SEARCH';
      const SEP_MARKER = '=======';
      const REPLACE_MARKER = '>>>>>>> REPLACE';
      const editBlock = `${SEARCH_MARKER}\nhello world\n${SEP_MARKER}\nhello universe\n${REPLACE_MARKER}`;
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
        console.log(
          'Messages received:',
          JSON.stringify(result.messages, null, 2)
        );
      }
      assert.ok(toolMsg, 'Expected a tool response message');
      // If the tool returned an error response, fail with a clear message.
      const output = toolMsg.content[0].toolResponse.output;
      if (typeof output === 'string' && output.includes('failed')) {
        throw new Error(`Tool failed unexpectedly: ${output}`);
      }
      assert.match(output, /Successfully applied/);

      const content = await fs.readFile(
        path.join(tempDir, 'file1.txt'),
        'utf8'
      );
      assert.strictEqual(content, 'hello universe');
    });

    it('fails if search content not found', async () => {
      const ai = genkit({});
      const SEARCH_MARKER = '<<<<<<< SEARCH';
      const SEP_MARKER = '=======';
      const REPLACE_MARKER = '>>>>>>> REPLACE';
      const editBlock = `${SEARCH_MARKER}\nnonexistent\n${SEP_MARKER}\nreplace\n${REPLACE_MARKER}`;
      const pm = createToolModel(ai, 'search_and_replace', {
        filePath: 'file1.txt',
        edits: [editBlock],
      });

      const result = (await ai.generate({
        model: pm,
        prompt: 'test',
        use: [filesystem({ rootDirectory: tempDir, allowWriteAccess: true })],
      })) as any;

      const toolMsg = result.messages.find(
        (m: any) =>
          m.role === 'tool' &&
          m.content.some((c: any) =>
            c.toolResponse?.output
              ?.toString()
              ?.includes('Search content not found')
          )
      );
      if (!toolMsg) {
        console.log(
          'Messages received:',
          JSON.stringify(result.messages, null, 2)
        );
      }
      assert.ok(toolMsg);
    });

    it('handles tricky search/replace cases', async () => {
      const SEARCH_MARKER = '<<<<<<< SEARCH';
      const SEP_MARKER = '=======';
      const REPLACE_MARKER = '>>>>>>> REPLACE';

      const cases = [
        {
          name: 'marker in search',
          initial: 'line1\n=======\nline2',
          block: `${SEARCH_MARKER}\nline1\n${SEP_MARKER}\nline2\n${SEP_MARKER}\nreplacement\n${REPLACE_MARKER}`,
          expected: 'replacement',
        },
        {
          name: 'marker in replace',
          initial: 'original',
          block: `${SEARCH_MARKER}\noriginal\n${SEP_MARKER}\nnew\n${SEP_MARKER}\nline\n${REPLACE_MARKER}`,
          expected: 'new\n=======\nline',
        },
        {
          name: 'start marker in search',
          initial: `${SEARCH_MARKER}\ncontent`,
          block: `${SEARCH_MARKER}\n${SEARCH_MARKER}\ncontent\n${SEP_MARKER}\nreplaced\n${REPLACE_MARKER}`,
          expected: 'replaced',
        },
        {
          name: 'start marker in replace',
          initial: 'content',
          block: `${SEARCH_MARKER}\ncontent\n${SEP_MARKER}\n${SEARCH_MARKER}\nnew\n${REPLACE_MARKER}`,
          expected: `${SEARCH_MARKER}\nnew`,
        },
        {
          name: 'end marker in search',
          initial: `content\n${REPLACE_MARKER}`,
          block: `${SEARCH_MARKER}\ncontent\n${REPLACE_MARKER}\n${SEP_MARKER}\nreplaced\n${REPLACE_MARKER}`,
          expected: 'replaced',
        },
        {
          name: 'end marker in replace',
          initial: 'content',
          block: `${SEARCH_MARKER}\ncontent\n${SEP_MARKER}\nnew\n${REPLACE_MARKER}\n${REPLACE_MARKER}`,
          expected: `new\n${REPLACE_MARKER}`,
        },
        {
          name: 'multiple markers greedy search',
          initial: 'part1\n=======\npart2',
          block: `${SEARCH_MARKER}\npart1\n${SEP_MARKER}\npart2\n${SEP_MARKER}\nreplacement\n${REPLACE_MARKER}`,
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
          block: `${SEARCH_MARKER}\nA\n${SEP_MARKER}\nB\n${SEP_MARKER}\nC\n${SEP_MARKER}\nD\n${REPLACE_MARKER}`,
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
    it('should handle tool errors gracefully by returning error tool response', async () => {
      const ai = genkit({});
      const pm = createToolModel(ai, 'read_file', { filePath: 'nonexistent' });

      const result = (await ai.generate({
        model: pm,
        prompt: 'start',
        use: [filesystem({ rootDirectory: tempDir })],
      })) as any;

      const messages = result.messages;

      // Find the tool response containing the error
      const toolMsg = messages.find(
        (m: any) =>
          m.role === 'tool' &&
          m.content.some((c: any) =>
            c.toolResponse?.output
              ?.toString()
              ?.includes("Tool 'read_file' failed")
          )
      );
      assert.ok(toolMsg, 'Tool response with error should be present');

      const errorOutput = toolMsg.content
        .find((c: any) =>
          c.toolResponse?.output
            ?.toString()
            ?.includes("Tool 'read_file' failed")
        )
        ?.toolResponse.output.toString();
      assert.match(
        errorOutput,
        /Tool 'read_file' failed: .*ENOENT.*/,
        'Error message should contain underlying error details'
      );

      // Message ordering: user, model (tool request), tool (error response), model (final)
      const roles = messages.map((m: any) => m.role);
      assert.deepStrictEqual(roles, ['user', 'model', 'tool', 'model']);
    });

    it('should let ToolInterruptError propagate when toolApproval is after filesystem', async () => {
      const ai = genkit({});
      // Model requests write_file which is a filesystem tool but NOT in the approved list
      const pm = createToolModel(ai, 'write_file', {
        filePath: 'test.txt',
        content: 'hello',
      });

      const result = (await ai.generate({
        model: pm,
        prompt: 'write a file',
        use: [
          // filesystem is FIRST — its tool hook wraps toolApproval's tool hook.
          // Without the fix, filesystem's catch block would swallow the ToolInterruptError.
          filesystem({ rootDirectory: tempDir, allowWriteAccess: true }),
          toolApproval({ approved: ['read_file', 'list_files'] }),
        ],
      })) as any;

      // The ToolInterruptError should propagate through filesystem's error handler
      // and result in an interrupted finish reason, NOT a swallowed error message.
      assert.strictEqual(
        result.finishReason,
        'interrupted',
        'Should be interrupted, not swallowed by filesystem error handler'
      );

      // Verify there's no user message with a swallowed error
      const swallowedErrorMsg = result.messages.find(
        (m: any) =>
          m.role === 'user' &&
          m.content.some(
            (c: any) => c.text && c.text.includes("Tool 'write_file' failed")
          )
      );
      assert.strictEqual(
        swallowedErrorMsg,
        undefined,
        'ToolInterruptError should not be swallowed into a user error message'
      );

      // Verify the interrupt metadata is present
      const interruptPart = result.message?.content.find(
        (p: any) => p.metadata?.interrupt
      );
      assert.ok(interruptPart, 'Should have interrupt metadata');
      assert.match(
        interruptPart.metadata.interrupt.message,
        /Tool not in approved list/
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
