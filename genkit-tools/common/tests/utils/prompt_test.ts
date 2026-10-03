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

import { describe, expect, it } from '@jest/globals';
import type { MessageData } from '../../src/types/model';
import { PromptFrontmatter } from '../../src/types/prompt';
import {
  jsonSchemaToPicoschema,
  renderPromptFile,
  toFrontmatterInput,
  toFrontmatterOutput,
  toPromptFile,
} from '../../src/utils/prompt';

describe('renderPromptFile', () => {
  it('builds a template from messages', () => {
    const frontmatter: PromptFrontmatter = {
      name: 'my-prompt',
      model: 'googleai/gemini-pro',
      config: {
        temperature: 0.5,
      },
    };
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'Who are you?' }] },
      {
        role: 'model',
        content: [
          { text: 'I am Oz -- the Great and Powerful.' },
          { media: { url: 'https://example.com/image.jpg' } },
        ],
      },
    ];
    const expected =
      '---\n' +
      'name: my-prompt\n' +
      'model: googleai/gemini-pro\n' +
      'config:\n' +
      '  temperature: 0.5\n' +
      '---\n' +
      '\n' +
      '{{role "user"}}\n' +
      'Who are you?\n' +
      '\n' +
      '{{role "model"}}\n' +
      'I am Oz -- the Great and Powerful.{{media url:https://example.com/image.jpg}}\n';
    expect(renderPromptFile(frontmatter, messages)).toStrictEqual(expected);
  });

  it('handles toolRequest by omitting the entire message', () => {
    const frontmatter: PromptFrontmatter = {
      model: 'googleai/gemini-pro',
      use: [{ name: 'test-middleware', config: { foo: 'bar' } }],
    };
    const messages: MessageData[] = [
      {
        role: 'user',
        content: [
          { text: 'Hello' },
          { reasoning: 'Thinking...' } as any,
          { toolRequest: { name: 'myTool' } } as any,
        ],
      },
    ];

    const expected =
      '---\n' +
      'model: googleai/gemini-pro\n' +
      'use:\n' +
      '  - name: test-middleware\n' +
      '    config:\n' +
      '      foo: bar\n' +
      '---\n' +
      '\n' +
      '{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}\n' +
      '\n' +
      '{{! message with role "user" omitted (toolRequest). }}\n';

    expect(renderPromptFile(frontmatter, messages)).toStrictEqual(expected);
  });

  it('omits messages entirely composed of unsupported parts', () => {
    const frontmatter: PromptFrontmatter = { model: 'model' };
    const messages: MessageData[] = [
      {
        role: 'model',
        content: [
          { toolResponse: { name: 'myTool', output: 'result' } } as any,
        ],
      },
    ];

    const expected =
      '---\n' +
      'model: model\n' +
      '---\n' +
      '\n' +
      '{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}\n' +
      '\n' +
      '{{! message with role "model" omitted (toolResponse). }}\n';

    expect(renderPromptFile(frontmatter, messages)).toStrictEqual(expected);
  });

  it('omits messages composed of other unsupported parts with "unsupported content" reason', () => {
    const frontmatter: PromptFrontmatter = { model: 'model' };
    const messages: MessageData[] = [
      {
        role: 'model',
        content: [{ reasoning: 'Thinking...' } as any],
      },
    ];

    const expected =
      '---\n' +
      'model: model\n' +
      '---\n' +
      '\n' +
      '{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}\n' +
      '\n' +
      '{{! message with role "model" omitted (unsupported content). }}\n';

    expect(renderPromptFile(frontmatter, messages)).toStrictEqual(expected);
  });

  it('handles mixed support messages without toolRequest by commenting parts', () => {
    const frontmatter: PromptFrontmatter = { model: 'model' };
    const messages: MessageData[] = [
      {
        role: 'user',
        content: [
          { text: 'Here is data: ' },
          { data: { foo: 'bar' } } as any,
          { text: ' and more text.' },
        ],
      },
    ];

    const expected =
      '---\n' +
      'model: model\n' +
      '---\n' +
      '\n' +
      '{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}\n' +
      '\n' +
      '{{role "user"}}\n' +
      'Here is data: {{! data part omitted }} and more text.\n';

    expect(renderPromptFile(frontmatter, messages)).toStrictEqual(expected);
  });

  it('recursively cleans empty objects and arrays from frontmatter', () => {
    const frontmatter: any = {
      model: 'googleai/gemini-pro',
      use: [
        {
          name: 'fallback',
          config: {},
        },
      ],
      tools: [],
      config: {
        safetySettings: [],
      },
    };
    const messages: any[] = [];

    const expected =
      '---\n' +
      'model: googleai/gemini-pro\n' +
      'use:\n' +
      '  - name: fallback\n' +
      '---\n';

    expect(renderPromptFile(frontmatter, messages)).toStrictEqual(expected);
  });
});

describe('jsonSchemaToPicoschema', () => {
  it('converts an object schema with required, optional, and described fields', () => {
    const schema = {
      type: 'object',
      properties: {
        title: { type: 'string' },
        subtitle: { type: 'string', description: 'optional subtitle' },
        servings: { type: 'integer' },
      },
      required: ['title', 'servings'],
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      title: 'string',
      'subtitle?': 'string, optional subtitle',
      servings: 'integer',
    });
  });

  it('encodes an enum', () => {
    const schema = {
      type: 'object',
      properties: {
        status: {
          type: 'string',
          enum: ['PENDING', 'APPROVED'],
          description: 'approval status',
        },
      },
      required: ['status'],
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      'status(enum, approval status)': ['PENDING', 'APPROVED'],
    });
  });

  it('encodes an array of scalars', () => {
    const schema = {
      type: 'object',
      properties: {
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'relevant tags',
        },
      },
      required: ['tags'],
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      'tags(array, relevant tags)': 'string',
    });
  });

  it('encodes an array of objects', () => {
    const schema = {
      type: 'object',
      properties: {
        authors: {
          type: 'array',
          items: {
            type: 'object',
            properties: { name: { type: 'string' } },
            required: ['name'],
          },
        },
      },
      required: ['authors'],
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      'authors(array)': { name: 'string' },
    });
  });

  it('encodes a nested object, marking optional fields', () => {
    const schema = {
      type: 'object',
      properties: {
        metadata: {
          type: 'object',
          properties: { updatedAt: { type: 'string' } },
        },
      },
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      'metadata?(object)': { 'updatedAt?': 'string' },
    });
  });

  it('encodes additionalProperties as a wildcard field', () => {
    const schema = {
      type: 'object',
      properties: { id: { type: 'string' } },
      required: ['id'],
      additionalProperties: { type: 'number' },
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      id: 'string',
      '(*)': 'number',
    });
  });

  it('passes non-object top-level schemas through unchanged', () => {
    const arraySchema = { type: 'array', items: { type: 'string' } };
    expect(jsonSchemaToPicoschema(arraySchema)).toBe(arraySchema);
  });

  it('returns any for null or malformed properties', () => {
    const schema = {
      type: 'object',
      properties: { id: { type: 'string' }, broken: null, items: {} },
      required: ['id'],
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      id: 'string',
      'broken?': 'any',
      'items?': 'any',
    });
  });

  it('handles nullable types (union with null)', () => {
    const schema = {
      type: 'object',
      properties: {
        title: { type: ['string', 'null'] },
        tags: {
          type: ['array', 'null'],
          items: { type: ['string', 'null'] },
        },
      },
      required: ['title'],
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      title: 'string',
      'tags?(array)': 'string',
    });
  });
});

describe('toFrontmatterInput', () => {
  const SCHEMA = {
    type: 'object',
    properties: { name: { type: 'string' } },
  };

  it('returns undefined when there is no input', () => {
    expect(toFrontmatterInput(undefined)).toBeUndefined();
  });

  it('maps schema and default values as raw schema by default', () => {
    expect(
      toFrontmatterInput({ schema: SCHEMA, default: { name: 'World' } })
    ).toEqual({
      schema: SCHEMA,
      default: { name: 'World' },
    });
  });

  it('converts schema to picoschema when picoSchema is true', () => {
    expect(
      toFrontmatterInput({ schema: SCHEMA, default: { name: 'World' } }, true)
    ).toEqual({
      schema: { 'name?': 'string' },
      default: { name: 'World' },
    });
  });
});

describe('toFrontmatterOutput', () => {
  const SCHEMA = {
    type: 'object',
    properties: { title: { type: 'string' } },
    required: ['title'],
  };

  it('returns undefined when there is no output', () => {
    expect(toFrontmatterOutput(undefined)).toBeUndefined();
  });

  it('reads the schema from jsonSchema and maps json formats as raw schema by default', () => {
    expect(toFrontmatterOutput({ format: 'json', jsonSchema: SCHEMA })).toEqual(
      { format: 'json', schema: SCHEMA }
    );
  });

  it('converts schema to picoschema when picoSchema is true', () => {
    expect(
      toFrontmatterOutput({ format: 'json', jsonSchema: SCHEMA }, true)
    ).toEqual({ format: 'json', schema: { title: 'string' } });
  });

  it('reads the schema from the schema field (model request shape)', () => {
    expect(toFrontmatterOutput({ format: 'json', schema: SCHEMA })).toEqual({
      format: 'json',
      schema: SCHEMA,
    });
  });

  it('maps json-producing formats onto json', () => {
    expect(
      toFrontmatterOutput({ format: 'jsonl', jsonSchema: SCHEMA })?.format
    ).toBe('json');
  });

  it('keeps the text format', () => {
    expect(toFrontmatterOutput({ format: 'text' })).toEqual({ format: 'text' });
  });

  it('keeps the media format', () => {
    expect(toFrontmatterOutput({ format: 'media' })).toEqual({
      format: 'media',
    });
  });
});

describe('toPromptFile', () => {
  const request = {
    model: '/model/googleai/gemini-pro',
    config: { temperature: 0.7 },
    tools: [{ name: 'getWeather' }],
    messages: [{ role: 'user' as const, content: [{ text: 'Hello' }] }],
    input: {
      schema: { type: 'object', properties: { name: { type: 'string' } } },
    },
    output: {
      format: 'json',
      schema: { type: 'object', properties: { answer: { type: 'string' } } },
    },
  };

  it('converts a request object into a .prompt template string with raw schemas by default', () => {
    const expected =
      '---\n' +
      'model: googleai/gemini-pro\n' +
      'config:\n' +
      '  temperature: 0.7\n' +
      'tools:\n' +
      '  - getWeather\n' +
      'input:\n' +
      '  schema:\n' +
      '    type: object\n' +
      '    properties:\n' +
      '      name:\n' +
      '        type: string\n' +
      'output:\n' +
      '  format: json\n' +
      '  schema:\n' +
      '    type: object\n' +
      '    properties:\n' +
      '      answer:\n' +
      '        type: string\n' +
      '---\n' +
      '\n' +
      '{{role "user"}}\n' +
      'Hello\n';
    expect(toPromptFile(request)).toStrictEqual(expected);
  });

  it('formats schemas as picoschema when picoSchema is true', () => {
    const expected =
      '---\n' +
      'model: googleai/gemini-pro\n' +
      'config:\n' +
      '  temperature: 0.7\n' +
      'tools:\n' +
      '  - getWeather\n' +
      'input:\n' +
      '  schema:\n' +
      '    name?: string\n' +
      'output:\n' +
      '  format: json\n' +
      '  schema:\n' +
      '    answer?: string\n' +
      '---\n' +
      '\n' +
      '{{role "user"}}\n' +
      'Hello\n';
    expect(toPromptFile({ ...request, picoSchema: true })).toStrictEqual(
      expected
    );
  });
});
