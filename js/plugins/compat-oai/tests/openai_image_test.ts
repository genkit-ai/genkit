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

import { afterEach, beforeEach, describe, expect, it } from '@jest/globals';
import { genkit } from 'genkit';
import * as path from 'path';
import * as ts from 'typescript';
import { openAI } from '../src/openai';
import { SUPPORTED_IMAGE_MODELS } from '../src/openai/dalle';
import { FakeOpenAIServer } from './fake_openai_server';

const imageModels = [
  'gpt-image-1',
  'gpt-image-1.5',
  'gpt-image-1-mini',
  'gpt-image-2',
  'gpt-image-2.5-sunburst',
  'gpt-image-2.5-flare',
  'gpt-image-2-2026-04-21',
  'chatgpt-image-latest',
] as const;
const imageData = 'aW1hZ2UtYnl0ZXM=';

describe('OpenAI image models', () => {
  let server: FakeOpenAIServer;
  let ai: ReturnType<typeof genkit>;
  let previousBaseUrl: string | undefined;

  beforeEach(async () => {
    server = new FakeOpenAIServer('image-test-key');
    await server.start();
    previousBaseUrl = process.env.OPENAI_BASE_URL;
    process.env.OPENAI_BASE_URL = server.baseUrl;
    ai = genkit({
      plugins: [openAI({ apiKey: 'image-test-key', maxRetries: 0 })],
    });
  });

  afterEach(() => {
    server.stop();
    if (previousBaseUrl === undefined) {
      delete process.env.OPENAI_BASE_URL;
    } else {
      process.env.OPENAI_BASE_URL = previousBaseUrl;
    }
  });

  async function generateImage(
    model: string | ReturnType<typeof openAI.model>,
    config?: Record<string, unknown>,
    contentType = 'image/png'
  ) {
    server.setNextResponse({
      body: { created: 0, data: [{ b64_json: imageData }] },
    });
    let response: Awaited<ReturnType<typeof ai.generate>> | undefined;
    let failure: unknown;
    try {
      response = await ai.generate({ model, prompt: 'a cat', config });
    } catch (error) {
      failure = error;
    }
    expect(server.requests).toHaveLength(1);
    expect(server.requests[0].path).toBe('/v1/images/generations');
    if (failure) throw failure;
    expect(response?.message?.content).toEqual([
      {
        media: { contentType, url: `data:${contentType};base64,${imageData}` },
      },
    ]);
    return server.requests[0].body;
  }

  describe.each(['reference', 'string'] as const)('%s lookup', (lookup) => {
    it.each(imageModels)(
      'routes %s to Images without response_format',
      async (name) => {
        const model =
          lookup === 'reference' ? openAI.model(name) : `openai/${name}`;
        const body = await generateImage(model);
        expect(body).toEqual({ model: name, prompt: 'a cat' });
      }
    );
  });

  it('pre-registers the current GPT image models', () => {
    expect(Object.keys(SUPPORTED_IMAGE_MODELS)).toEqual(
      expect.arrayContaining([
        'gpt-image-2',
        'gpt-image-2.5-sunburst',
        'gpt-image-2.5-flare',
      ])
    );
  });

  it('discovers GPT image models with image metadata and config', async () => {
    server.setNextResponse({
      body: {
        object: 'list',
        data: imageModels.map((id) => ({
          id,
          object: 'model',
          created: 0,
          owned_by: 'openai',
        })),
      },
    });
    const plugin = openAI({ apiKey: 'image-test-key' });
    const actions = await plugin.list!();
    for (const name of imageModels) {
      expect(
        actions.find((action) => action.name === `openai/${name}`)
      ).toMatchObject({
        actionType: 'model',
        metadata: {
          model: {
            supports: { output: ['media'], tools: false },
            customOptions: {
              properties: {
                quality: { enum: expect.arrayContaining(['high', 'auto']) },
              },
            },
          },
        },
      });
    }
    expect(server.requests[0].path).toBe('/v1/models');
  });

  it('forwards GPT image options, including size and zero compression', async () => {
    const config = {
      size: '1536x1024',
      quality: 'high',
      background: 'transparent',
      output_format: 'webp',
      output_compression: 0,
      moderation: 'auto',
      n: 2,
    };
    const body = await generateImage(
      'openai/gpt-image-2.5-sunburst',
      config,
      'image/webp'
    );
    expect(body).toEqual({
      model: 'gpt-image-2.5-sunburst',
      prompt: 'a cat',
      ...config,
    });
  });

  it.each([
    { format: 'png', contentType: 'image/png' },
    { format: 'jpeg', contentType: 'image/jpeg' },
    { format: 'webp', contentType: 'image/webp' },
  ])(
    'returns $contentType for $format output',
    async ({ format, contentType }) => {
      const body = await generateImage(
        'openai/gpt-image-1.5',
        { output_format: format },
        contentType
      );
      expect(body.output_format).toBe(format);
      expect(body).not.toHaveProperty('response_format');
    }
  );

  it('preserves the DALL-E request format and config', async () => {
    const config = { quality: 'hd', style: 'vivid', size: '1792x1024', n: 1 };
    expect(await generateImage(openAI.model('dall-e-3'), config)).toEqual({
      model: 'dall-e-3',
      prompt: 'a cat',
      response_format: 'b64_json',
      ...config,
    });
  });

  it('keeps text models on Chat Completions', async () => {
    server.setNextResponse({
      body: {
        choices: [
          {
            message: { role: 'assistant', content: 'hello' },
            finish_reason: 'stop',
          },
        ],
      },
    });
    const response = await ai.generate({
      model: openAI.model('gpt-4o'),
      prompt: 'hello',
    });
    expect(server.requests[0].path).toBe('/v1/chat/completions');
    expect(response.text).toBe('hello');
  });
});

it('type-checks a strict consumer of the published GPT image references', () => {
  const configPath = path.resolve(__dirname, '../tsconfig.json');
  const config = ts.readConfigFile(configPath, ts.sys.readFile);
  expect(config.error).toBeUndefined();
  const parsed = ts.parseJsonConfigFileContent(
    config.config,
    ts.sys,
    path.dirname(configPath)
  );
  const fixture = path.join(__dirname, 'fixtures/openai-image-types.ts');
  const program = ts.createProgram([fixture], {
    ...parsed.options,
    noEmit: true,
    skipLibCheck: true,
  });
  const diagnostics = ts.getPreEmitDiagnostics(program);
  expect(
    diagnostics.map((diagnostic) =>
      ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')
    )
  ).toEqual([]);
  const service = ts.createLanguageService({
    getCompilationSettings: () => parsed.options,
    getScriptFileNames: () => [fixture],
    getScriptVersion: () => '0',
    getScriptSnapshot: (filename) => {
      const text = ts.sys.readFile(filename);
      return text === undefined
        ? undefined
        : ts.ScriptSnapshot.fromString(text);
    },
    getCurrentDirectory: () => path.dirname(configPath),
    getDefaultLibFileName: (options) => ts.getDefaultLibFilePath(options),
    fileExists: ts.sys.fileExists,
    readFile: ts.sys.readFile,
    readDirectory: ts.sys.readDirectory,
    directoryExists: ts.sys.directoryExists,
    getDirectories: ts.sys.getDirectories,
    realpath: ts.sys.realpath,
  });
  try {
    const source = ts.sys.readFile(fixture)!;
    const completions = service.getCompletionsAtPosition(
      fixture,
      source.lastIndexOf("''") + 1,
      {}
    );
    expect(completions?.entries.map((entry) => entry.name)).toEqual(
      expect.arrayContaining([
        'gpt-image-1',
        'gpt-image-2',
        'gpt-image-2.5-sunburst',
        'gpt-image-2.5-flare',
      ])
    );
  } finally {
    service.dispose();
  }
});
