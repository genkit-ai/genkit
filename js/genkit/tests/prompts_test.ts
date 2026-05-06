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

import { modelRef, type ModelMiddleware } from '@genkit-ai/ai/model';
import { stripUndefinedProps } from '@genkit-ai/core';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit, z, type GenkitBeta } from '../src/beta';
import type { PromptAction } from '../src/index';
import {
  defineEchoModel,
  defineProgrammableModel,
  defineStaticResponseModel,
  type ProgrammableModel,
} from './helpers';

const wrapRequest: ModelMiddleware = async (req, next) => {
  return next({
    ...req,
    messages: [
      {
        role: 'user',
        content: [
          {
            text:
              '(' +
              req.messages
                .map((m) => m.content.map((c) => c.text).join())
                .join() +
              ')',
          },
        ],
      },
    ],
  });
};
const wrapResponse: ModelMiddleware = async (req, next) => {
  const res = await next(req);
  return {
    message: {
      role: 'model',
      content: [
        {
          text: '[' + res.message!.content.map((c) => c.text).join() + ']',
        },
      ],
    },
    finishReason: res.finishReason,
  };
};

describe('definePrompt', () => {
  let ai: GenkitBeta;

  beforeEach(() => {
    ai = genkit({
      model: 'echoModel',
    });
    defineEchoModel(ai);
  });

  it('should define the prompt', async () => {
    const prompt = ai.definePrompt({
      name: 'hi',
      metadata: { foo: 'bar' },
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      messages: async (input) => {
        return [
          {
            role: 'user',
            content: [{ text: `hi ${input.name}` }],
          },
        ];
      },
    });

    assert.deepStrictEqual(prompt.ref, {
      name: 'hi',
      metadata: { foo: 'bar' },
    });

    const lookedUpPrompt = ai.prompt('hi');
    // This is a known limitation -- prompt lookup is async under the hood,
    // so we can't actually get the metadata...
    assert.deepStrictEqual(lookedUpPrompt.ref, { name: 'hi' }); // ideally metadata should be: { foo: 'bar' }
  });

  it('should apply middleware to a prompt call', async () => {
    const prompt = ai.definePrompt({
      name: 'hi',
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      messages: async (input) => {
        return [
          {
            role: 'user',
            content: [{ text: `hi ${input.name}` }],
          },
        ];
      },
    });

    const response = await prompt(
      { name: 'Genkit' },
      { use: [wrapRequest, wrapResponse] }
    );
    assert.strictEqual(response.text, '[Echo: (hi Genkit),; config: {}]');
  });

  it('should apply middleware configured on a prompt', async () => {
    const prompt = ai.definePrompt({
      name: 'hi',
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      use: [wrapRequest, wrapResponse],
      messages: async (input) => {
        return [
          {
            role: 'user',
            content: [{ text: `hi ${input.name}` }],
          },
        ];
      },
    });

    const response = await prompt({ name: 'Genkit' });
    assert.strictEqual(response.text, '[Echo: (hi Genkit),; config: {}]');
  });

  it('should apply middleware to a prompt call on a looked up prompt', async () => {
    ai.definePrompt({
      name: 'hi',
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      use: [wrapRequest, wrapResponse],
      messages: async (input) => {
        return [
          {
            role: 'user',
            content: [{ text: `hi ${input.name}` }],
          },
        ];
      },
    });

    const prompt = ai.prompt('hi');

    const response = await prompt({ name: 'Genkit' });
    assert.strictEqual(response.text, '[Echo: (hi Genkit),; config: {}]');
  });

  it('should apply middleware configured on a prompt on a looked up prompt', async () => {
    ai.definePrompt({
      name: 'hi',
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      messages: async (input) => {
        return [
          {
            role: 'user',
            content: [{ text: `hi ${input.name}` }],
          },
        ];
      },
    });

    const prompt = ai.prompt('hi');

    const response = await prompt(
      { name: 'Genkit' },
      { use: [wrapRequest, wrapResponse] }
    );
    assert.strictEqual(response.text, '[Echo: (hi Genkit),; config: {}]');
  });
});

describe('definePrompt', () => {
  describe('default model', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({
        model: 'echoModel',
      });
      defineEchoModel(ai);
    });

    it('calls dotprompt with default model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('calls dotprompt with default model with config', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        config: {
          temperature: 11,
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(
        response.text,
        'Echo: hi Genkit; config: {"temperature":11}'
      );
    });

    it('calls dotprompt with default model via retrieved prompt', async () => {
      ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const hi = ai.prompt('hi');

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('should apply middleware to a prompt call', async () => {
      const prompt = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const response = await prompt(
        { name: 'Genkit' },
        { use: [wrapRequest, wrapResponse] }
      );
      assert.strictEqual(response.text, '[Echo: (hi Genkit),; config: {}]');
    });

    it('should apply middleware configured on a prompt', async () => {
      const prompt = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        use: [wrapRequest, wrapResponse],
        prompt: 'hi {{ name }}',
      });

      const response = await prompt({ name: 'Genkit' });
      assert.strictEqual(response.text, '[Echo: (hi Genkit),; config: {}]');
    });

    it('should apply middleware to a prompt call on a looked up prompt', async () => {
      ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        use: [wrapRequest, wrapResponse],
        prompt: 'hi {{ name }}',
      });

      const prompt = ai.prompt('hi');

      const response = await prompt({ name: 'Genkit' });
      assert.strictEqual(response.text, '[Echo: (hi Genkit),; config: {}]');
    });

    it('should apply middleware configured on a prompt on a looked up prompt', async () => {
      ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const prompt = ai.prompt('hi');

      const response = await prompt(
        { name: 'Genkit' },
        { use: [wrapRequest, wrapResponse] }
      );
      assert.strictEqual(response.text, '[Echo: (hi Genkit),; config: {}]');
    });
  });

  describe('default model ref', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({
        model: modelRef({
          name: 'echoModel',
        }),
        promptDir: './tests/prompts',
      });
      defineEchoModel(ai);
    });

    it('calls dotprompt with default model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('infers output schema', async () => {
      const Foo = z.object({
        bar: z.string(),
      });
      const model = defineStaticResponseModel(ai, {
        role: 'model',
        content: [
          {
            text: '```json\n{bar: "baz"}\n```',
          },
        ],
      });
      const hi = ai.definePrompt({
        name: 'hi',
        model,
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        output: {
          format: 'json',
          schema: Foo,
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi({ name: 'Genkit' });
      const foo = response.output;
      assert.deepStrictEqual(foo, { bar: 'baz' });
    });

    it('defaults to json format', async () => {
      const Foo = z.object({
        bar: z.string(),
      });
      const model = defineStaticResponseModel(ai, {
        role: 'model',
        content: [
          {
            text: '```json\n{bar: "baz"}\n```',
          },
        ],
      });
      const hi = ai.definePrompt({
        name: 'hi',
        model,
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        output: {
          // no format specified
          schema: Foo,
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi({ name: 'Genkit' });
      const foo = response.output;
      assert.deepStrictEqual(foo, { bar: 'baz' });
    });

    it('defaults to json format from a loaded prompt', async () => {
      defineStaticResponseModel(ai, {
        role: 'model',
        content: [
          {
            text: '```json\n{bar: "baz"}\n```',
          },
        ],
      });
      const hi = ai.prompt('output');

      const response = await hi({ name: 'Genkit' });
      const foo = response.output;
      assert.deepStrictEqual(stripUndefinedProps(response.request), {
        config: {},
        messages: [
          {
            content: [
              {
                text: 'Hi Genkit',
              },
            ],
            role: 'user',
          },
        ],
        output: {
          constrained: true,
          contentType: 'application/json',
          format: 'json',
          schema: {
            additionalProperties: false,
            properties: {
              bar: {
                type: 'string',
              },
            },
            required: ['bar'],
            type: 'object',
          },
        },
        tools: [],
      });

      assert.deepStrictEqual(foo, { bar: 'baz' });
    });

    it('streams dotprompt with default model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        config: {
          temperature: 11,
        },
        prompt: 'hi {{ name }}',
      });

      const { response, stream } = hi.stream({ name: 'Genkit' });
      const chunks: string[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk.text);
      }
      const responseText = (await response).text;

      assert.strictEqual(
        responseText,
        'Echo: hi Genkit; config: {"temperature":11}'
      );
      assert.deepStrictEqual(chunks, ['3', '2', '1']);
    });

    it('calls dotprompt with default model via retrieved prompt', async () => {
      ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const hi = ai.prompt('hi');

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');

      const { stream } = hi.stream({ name: 'Genkit' });
      const chunks: string[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk.text);
      }
      assert.deepStrictEqual(chunks, ['3', '2', '1']);
    });
  });

  describe('explicit model', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({});
      defineEchoModel(ai);
    });

    it('calls dotprompt with default model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'echoModel',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('calls dotprompt with history and places it before user message', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'echoModel',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi(
        { name: 'Genkit' },
        {
          messages: [
            { role: 'user', content: [{ text: 'hi' }] },
            { role: 'model', content: [{ text: 'bye' }] },
          ],
        }
      );
      assert.deepStrictEqual(response.messages, [
        {
          role: 'user',
          content: [{ text: 'hi' }],
        },
        {
          role: 'model',
          content: [{ text: 'bye' }],
        },
        {
          role: 'user',
          content: [{ text: 'hi Genkit' }],
        },
        {
          role: 'model',
          content: [
            { text: 'Echo: hi,bye,hi Genkit' },
            { text: '; config: {}' },
          ],
        },
      ]);
    });

    it('streams dotprompt with history and places it before user message', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'echoModel',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi.stream(
        { name: 'Genkit' },
        {
          messages: [
            { role: 'user', content: [{ text: 'hi' }] },
            { role: 'model', content: [{ text: 'bye' }] },
          ],
        }
      );
      assert.deepStrictEqual((await response.response).messages, [
        {
          role: 'user',
          content: [{ text: 'hi' }],
        },
        {
          role: 'model',
          content: [{ text: 'bye' }],
        },
        {
          role: 'user',
          content: [{ text: 'hi Genkit' }],
        },
        {
          role: 'model',
          content: [
            { text: 'Echo: hi,bye,hi Genkit' },
            { text: '; config: {}' },
          ],
        },
      ]);
    });

    it('calls dotprompt with default model with config', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'echoModel',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        config: {
          temperature: 11,
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(
        response.text,
        'Echo: hi Genkit; config: {"temperature":11}'
      );
    });

    it('rejects on invalid model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'modelThatDoesNotExist',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const response = hi({ name: 'Genkit' });
      await assert.rejects(response, {
        message: "NOT_FOUND: Model 'modelThatDoesNotExist' not found",
      });
    });
  });

  describe('render', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({});
      defineEchoModel(ai);
    });

    it('renders dotprompt messages', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        prompt: 'hi {{ name }}',
      });

      const response = await hi.render({ name: 'Genkit' });
      delete response.model; // ignore
      assert.deepStrictEqual(response, {
        messages: [{ content: [{ text: 'hi Genkit' }], role: 'user' }],
      });
    });
  });
});

describe('definePrompt', () => {
  describe('default model', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({
        model: 'echoModel',
      });
      defineEchoModel(ai);
    });

    it('calls prompt with default model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('calls legacy prompt with default model', async () => {
      const hi = ai.definePrompt(
        {
          name: 'hi',
          input: {
            schema: z.object({
              name: z.string(),
            }),
          },
        },
        async (input) => {
          return {
            messages: [
              { role: 'user', content: [{ text: `hi ${input.name}` }] },
            ],
          };
        }
      );

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('calls legacy prompt with default model', async () => {
      const hi = ai.definePrompt(
        {
          name: 'hi',
          input: {
            schema: z.object({
              name: z.string(),
            }),
          },
        },
        'hi {{ name }}'
      );

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('thrown on prompt with both legacy template and messages', async () => {
      assert.throws(() =>
        ai.definePrompt(
          {
            name: 'hi',
            input: {
              schema: z.object({
                name: z.string(),
              }),
            },
            messages: 'something',
          },
          'hi {{ name }}'
        )
      );
    });

    it('calls prompt with default model with config', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        config: {
          temperature: 11,
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(
        response.text,
        'Echo: hi Genkit; config: {"temperature":11}'
      );
    });

    it('calls prompt with default model via retrieved prompt', async () => {
      ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const hi = ai.prompt('hi');

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });
  });

  describe('default model ref', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({
        model: modelRef({
          name: 'echoModel',
        }),
      });
      defineEchoModel(ai);
    });

    it('calls prompt with default model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('streams prompt with default model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        config: {
          temperature: 11,
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const { response, stream } = await hi.stream({ name: 'Genkit' });
      const chunks: string[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk.text);
      }
      const responseText = (await response).text;

      assert.strictEqual(
        responseText,
        'Echo: hi Genkit; config: {"temperature":11}'
      );
      assert.deepStrictEqual(chunks, ['3', '2', '1']);
    });
  });

  describe('explicit model', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({});
      defineEchoModel(ai);
    });

    it('calls prompt with default model', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'echoModel',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(response.text, 'Echo: hi Genkit; config: {}');
    });

    it('calls prompt with default model with config', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'echoModel',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        config: {
          temperature: 11,
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const response = await hi({ name: 'Genkit' });
      assert.strictEqual(
        response.text,
        'Echo: hi Genkit; config: {"temperature":11}'
      );
    });

    it('calls prompt with default model with call site config', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'echoModel',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        config: {
          temperature: 11,
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const response = await hi(
        { name: 'Genkit' },
        {
          config: {
            version: 'abc',
          },
        }
      );
      assert.strictEqual(
        response.text,
        'Echo: hi Genkit; config: {"version":"abc","temperature":11}'
      );
    });
  });

  describe('render', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({});
      defineEchoModel(ai);
    });

    it('renders prompt', async () => {
      const hi = ai.definePrompt({
        name: 'hi',
        model: 'echoModel',
        input: {
          schema: z.object({
            name: z.string(),
          }),
        },
        messages: async (input) => {
          return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
        },
      });

      const response = await hi.render({ name: 'Genkit' });
      delete response.model; // ignore
      assert.deepStrictEqual(response, {
        messages: [
          {
            content: [
              {
                text: 'hi Genkit',
              },
            ],
            role: 'user',
          },
        ],
      });
    });
  });
});

describe('prompt', () => {
  let ai: GenkitBeta;
  let pm: ProgrammableModel;

  beforeEach(() => {
    ai = genkit({
      model: 'echoModel',
      promptDir: './tests/prompts',
    });
    defineEchoModel(ai);
    pm = defineProgrammableModel(ai);
    ai.defineTool(
      {
        name: 'toolA',
        description: 'toolA it is',
      },
      async () => {}
    );
    ai.defineSchema('myInputSchema', z.object({ foo: z.string() }));
    ai.defineSchema('myOutputSchema', z.object({ output: z.string() }));
  });

  it('loads from from the folder', async () => {
    const testPrompt = ai.prompt('test'); // see tests/prompts folder

    const { text } = await testPrompt();

    assert.strictEqual(
      text,
      'Echo: Hello from the prompt file; config: {"temperature":11}'
    );
    assert.deepStrictEqual(await testPrompt.render({}), {
      config: {
        temperature: 11,
      },
      messages: [
        { content: [{ text: 'Hello from the prompt file' }], role: 'user' },
      ],
      output: {},
    });
  });

  it('does not auto-load prompts when promptDir is null', async () => {
    const aiWithoutPromptDir = genkit({
      model: 'echoModel',
      promptDir: null,
    });

    const prompt = aiWithoutPromptDir.prompt('test');
    const response = prompt();
    await assert.rejects(response, {
      message: 'NOT_FOUND: Prompt test not found',
    });
  });

  it('loads from from the sub folder', async () => {
    const testPrompt = ai.prompt('sub/test'); // see tests/prompts/sub folder

    const { text } = await testPrompt();

    assert.strictEqual(
      text,
      'Echo: Hello from the sub folder prompt file; config: {"temperature":12}'
    );
    assert.deepStrictEqual(await testPrompt.render({}), {
      config: {
        temperature: 12,
      },
      messages: [
        {
          content: [{ text: 'Hello from the sub folder prompt file' }],
          role: 'user',
        },
      ],
      output: {},
    });
  });

  it('loads from from the folder with all the options', async () => {
    const testPrompt = ai.prompt('kitchensink'); // see tests/prompts folder

    const request = await testPrompt.render({ subject: 'banana' });

    assert.deepStrictEqual(request, {
      model: 'googleai/gemini-5.0-ultimate-pro-plus',
      config: {
        temperature: 11,
      },
      output: {
        format: 'csv',
        jsonSchema: {
          additionalProperties: false,
          properties: {
            arr: {
              description: 'array of objects',
              items: {
                additionalProperties: false,
                properties: {
                  nest2: {
                    type: ['boolean', 'null'],
                  },
                },
                type: 'object',
              },
              type: 'array',
            },
            obj: {
              additionalProperties: false,
              description: 'a nested object',
              properties: {
                nest1: {
                  type: ['string', 'null'],
                },
              },
              type: ['object', 'null'],
            },
          },
          required: ['arr'],
          type: 'object',
        },
      },
      maxTurns: 77,
      messages: [
        {
          content: [
            {
              text: ' Hello ',
            },
          ],
          role: 'system',
        },
        {
          content: [
            {
              text: ' from the prompt file banana',
            },
          ],
          role: 'model',
        },
      ],
      returnToolRequests: true,
      toolChoice: 'required',
      tools: ['toolA', 'toolB'],
      metadata: {
        prompt: {
          foo: 'bar',
        },
      },
    });
  });

  it('renders loaded prompt via executable-prompt', async () => {
    ai.defineModel(
      { name: 'googleai/gemini-5.0-ultimate-pro-plus' },
      async () => ({})
    );

    ai.defineTool(
      {
        name: 'toolA',
        description: 'toolA it is',
      },
      async () => {}
    );

    ai.defineTool(
      {
        name: 'toolB',
        description: 'toolB it is',
      },
      async () => {}
    );

    const generateActionOptions = await (
      await ai.registry.lookupAction('/executable-prompt/kitchensink')
    )({ subject: 'banana' });

    assert.deepStrictEqual(stripUndefinedProps(generateActionOptions), {
      config: {
        temperature: 11,
      },
      model: 'googleai/gemini-5.0-ultimate-pro-plus',
      maxTurns: 77,
      messages: [
        { role: 'system', content: [{ text: ' Hello ' }] },
        { role: 'model', content: [{ text: ' from the prompt file banana' }] },
      ],
      output: {
        format: 'csv',
        jsonSchema: {
          additionalProperties: false,
          properties: {
            arr: {
              description: 'array of objects',
              items: {
                additionalProperties: false,
                properties: {
                  nest2: {
                    type: ['boolean', 'null'],
                  },
                },
                type: 'object',
              },
              type: 'array',
            },
            obj: {
              additionalProperties: false,
              description: 'a nested object',
              properties: {
                nest1: {
                  type: ['string', 'null'],
                },
              },
              type: ['object', 'null'],
            },
          },
          required: ['arr'],
          type: 'object',
        },
      },
      returnToolRequests: true,
      toolChoice: 'required',
      tools: ['/tool/toolA', '/tool/toolB'],
    });
  });

  it('resolved schema refs', async () => {
    const prompt = ai.prompt('schemaRef');

    const rendered = await prompt.render({ foo: 'bar' });
    assert.deepStrictEqual(rendered.output?.jsonSchema, {
      $schema: 'http://json-schema.org/draft-07/schema#',
      additionalProperties: true,
      properties: {
        output: {
          type: 'string',
        },
      },
      required: ['output'],
      type: 'object',
    });

    assert.deepStrictEqual(
      (await (await prompt.asTool())({ foo: 'bar' })).messages,
      [
        {
          role: 'user',
          content: [{ text: 'Write a poem about bar.' }],
        },
      ]
    );
  });

  it('lazily resolved schema refs', async () => {
    const prompt = ai.prompt('badSchemaRef');

    await assert.rejects(prompt.render({ foo: 'bar' }), (e: Error) =>
      e.message.includes("NOT_FOUND: Schema 'badSchemaRef1' not found")
    );
  });

  it('loads a variant from from the folder', async () => {
    const testPrompt = ai.prompt('test', { variant: 'variant' }); // see tests/prompts folder

    const { text } = await testPrompt();

    assert.strictEqual(
      text,
      'Echo: Hello from a variant of the hello prompt; config: {"temperature":13}'
    );
  });

  it('includes metadata expected by the dev ui', async () => {
    const testPrompt: PromptAction = await ai.registry.lookupAction(
      '/prompt/test.variant'
    );

    assert.deepStrictEqual(testPrompt.__action.metadata, {
      prompt: {
        config: {
          temperature: 13,
        },
        description: 'a prompt variant in a file',
        ext: {},
        input: {
          schema: null,
        },
        metadata: {},
        model: undefined,
        name: 'test',
        variant: 'variant',
        template: 'Hello from a variant of the hello prompt',
        tools: ['toolA'],
        toolChoice: undefined,
        toolDefs: [],
        use: [{ name: 'myMiddleware' }],
        raw: {
          config: {
            temperature: 13,
          },
          description: 'a prompt variant in a file',
          tools: ['toolA'],
          use: ['myMiddleware'],
        },
      },
      type: 'prompt',
    });
  });

  it('returns a ref to functional prompts', async () => {
    ai.definePrompt({
      name: 'hi',
      model: 'echoModel',
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      config: {
        temperature: 11,
      },
      messages: async (input) => {
        return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
      },
    });
    const testPrompt = ai.prompt('hi');
    const { text } = await testPrompt({ name: 'banana' });

    assert.strictEqual(text, 'Echo: hi banana; config: {"temperature":11}');
  });

  it('includes metadata for functional prompts', async () => {
    ai.definePrompt({
      name: 'hi',
      model: 'echoModel',
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      config: {
        temperature: 0.13,
      },
      tools: ['toolA'],
      toolChoice: 'auto',
      use: ['myMiddleware'],
      messages: async (input) => [],
    });
    const testPrompt: PromptAction =
      await ai.registry.lookupAction('/prompt/hi');

    assert.deepStrictEqual(testPrompt.__action.metadata, {
      type: 'prompt',
      prompt: {
        name: 'hi',
        model: 'echoModel',
        config: {
          temperature: 0.13,
        },
        tools: ['toolA'],
        toolChoice: 'auto',
        use: [{ name: 'myMiddleware' }],
        input: {
          schema: {
            type: 'object',
            properties: {
              name: {
                type: 'string',
              },
            },
            required: ['name'],
            additionalProperties: true,
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
        },
      },
    });
  });

  it('passes in output options to the model', async () => {
    const hi = ai.definePrompt({
      name: 'hi',
      model: 'programmableModel',
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      output: {
        schema: z.object({
          message: z.string(),
        }),
        format: 'json',
      },
      config: {
        temperature: 11,
      },
      messages: async (input) => {
        return [{ role: 'user', content: [{ text: `hi ${input.name}` }] }];
      },
    });

    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [{ text: '```json\n{"message": "hello"}\n```' }],
        },
      };
    };

    const { output } = await hi({
      name: 'Pavel',
    });

    assert.deepStrictEqual(output, { message: 'hello' });
  });
});

describe('asTool', () => {
  let ai: GenkitBeta;
  let pm: ProgrammableModel;

  beforeEach(() => {
    ai = genkit({
      model: 'programmableModel',
      promptDir: './tests/prompts',
    });
    pm = defineProgrammableModel(ai);
  });

  it('swaps out preamble on .prompt file tool invocation', async () => {
    const session = ai.createSession({ initialState: { name: 'Genkit' } });
    const agentA = ai.definePrompt({
      name: 'agentA',
      config: { temperature: 2 },
      description: 'Agent A description',
      tools: ['toolPrompt'], // <--- defined in a .prompt file
      messages: async () => {
        return [
          {
            role: 'system',
            content: [{ text: ' agent a' }],
          },
        ];
      },
    });

    // simple hi, nothing interesting...
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [{ text: `hi ${session.state?.name} from agent a` }],
        },
      };
    };
    const chat = session.chat(agentA);
    let { text } = await chat.send('hi');
    assert.strictEqual(text, 'hi Genkit from agent a');
    assert.deepStrictEqual(pm.lastRequest, {
      config: {
        temperature: 2,
      },
      messages: [
        {
          content: [{ text: ' agent a' }],
          metadata: { preamble: true },
          role: 'system',
        },
        {
          content: [{ text: 'hi' }],
          role: 'user',
        },
      ],
      output: {},
      tools: [
        {
          name: 'toolPrompt',
          description: 'prompt in a file',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
          outputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
        },
      ],
    });

    // transfer to toolPrompt...

    // first response is a tool call, the subsequent responses are just text response from agent b.
    let reqCounter = 0;
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [
            reqCounter++ === 0
              ? {
                  toolRequest: {
                    name: 'toolPrompt',
                    input: {},
                    ref: 'ref123',
                  },
                }
              : { text: 'hi from agent b' },
          ],
        },
      };
    };

    ({ text } = await chat.send('pls transfer to b'));

    assert.deepStrictEqual(text, 'hi from agent b');
    assert.deepStrictEqual(pm.lastRequest, {
      // Original config, toolPrompt has no config.
      config: {
        temperature: 2,
      },
      messages: [
        {
          role: 'system',
          content: [{ text: ' Genkit toolPrompt prompt' }], // <--- NOTE: swapped out the preamble
          metadata: { preamble: true },
        },
        {
          role: 'user',
          content: [{ text: 'hi' }],
        },
        {
          role: 'model',
          content: [{ text: 'hi Genkit from agent a' }],
        },
        {
          role: 'user',
          content: [{ text: 'pls transfer to b' }],
        },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                input: {},
                name: 'toolPrompt',
                ref: 'ref123',
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'toolPrompt',
                output: 'transferred to toolPrompt',
                ref: 'ref123',
              },
            },
          ],
        },
      ],
      output: {},
      tools: [
        {
          name: 'agentA',
          description: 'Agent A description',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
          outputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
        },
      ],
    });

    // transfer back to to agent A...

    // first response is a tool call, the subsequent responses are just text response from agent a.
    reqCounter = 0;
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [
            reqCounter++ === 0
              ? {
                  toolRequest: {
                    name: 'agentA',
                    input: {},
                    ref: 'ref123',
                  },
                }
              : { text: 'hi Genkit from agent a' },
          ],
        },
      };
    };

    ({ text } = await chat.send('pls transfer to a'));

    assert.deepStrictEqual(text, 'hi Genkit from agent a');
    assert.deepStrictEqual(pm.lastRequest, {
      config: {
        temperature: 2,
      },
      messages: [
        {
          role: 'system',
          content: [{ text: ' agent a' }], // <--- NOTE: swapped out the preamble
          metadata: { preamble: true },
        },
        {
          role: 'user',
          content: [{ text: 'hi' }],
        },
        {
          role: 'model',
          content: [{ text: 'hi Genkit from agent a' }],
        },
        {
          role: 'user',
          content: [{ text: 'pls transfer to b' }],
        },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                input: {},
                name: 'toolPrompt',
                ref: 'ref123',
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'toolPrompt',
                output: 'transferred to toolPrompt',
                ref: 'ref123',
              },
            },
          ],
        },
        {
          role: 'model',
          content: [{ text: 'hi from agent b' }],
        },
        {
          role: 'user',
          content: [{ text: 'pls transfer to a' }],
        },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                input: {},
                name: 'agentA',
                ref: 'ref123',
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'agentA',
                output: 'transferred to agentA',
                ref: 'ref123',
              },
            },
          ],
        },
      ],
      output: {},
      tools: [
        {
          description: 'prompt in a file',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
          name: 'toolPrompt',
          outputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
        },
      ],
    });
  });
});
