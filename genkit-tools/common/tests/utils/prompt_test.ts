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
import type { PromptFrontmatter } from '../../src/types/prompt';
import {
  fromMessages,
  jsonSchemaToPicoschema,
  toFrontmatterOutput,
} from '../../src/utils/prompt';

describe('fromMessages', () => {
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
    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
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

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
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

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
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

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
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

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
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

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
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

  it('encodes enums, arrays of scalars, arrays of objects, and nested objects', () => {
    const schema = {
      type: 'object',
      properties: {
        status: {
          type: 'string',
          enum: ['PENDING', 'APPROVED'],
          description: 'approval status',
        },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'relevant tags',
        },
        authors: {
          type: 'array',
          items: {
            type: 'object',
            properties: { name: { type: 'string' } },
            required: ['name'],
          },
        },
        metadata: {
          type: 'object',
          properties: { updatedAt: { type: 'string' } },
        },
      },
      required: ['status', 'tags', 'authors'],
    };
    expect(jsonSchemaToPicoschema(schema)).toEqual({
      'status(enum, approval status)': ['PENDING', 'APPROVED'],
      'tags(array, relevant tags)': 'string',
      'authors(array)': { name: 'string' },
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

  it('does not crash on a null or malformed property', () => {
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

  it('reads the schema from jsonSchema and maps json formats', () => {
    expect(toFrontmatterOutput({ format: 'json', jsonSchema: SCHEMA })).toEqual(
      { format: 'json', schema: { title: 'string' } }
    );
  });

  it('reads the schema from the schema field (model request shape)', () => {
    expect(toFrontmatterOutput({ format: 'json', schema: SCHEMA })).toEqual({
      format: 'json',
      schema: { title: 'string' },
    });
  });

  it('maps json-producing formats onto json', () => {
    expect(
      toFrontmatterOutput({ format: 'jsonl', jsonSchema: SCHEMA })?.format
    ).toBe('json');
  });

  it('keeps text and media formats', () => {
    expect(toFrontmatterOutput({ format: 'text' })).toEqual({ format: 'text' });
    expect(toFrontmatterOutput({ format: 'media' })).toEqual({
      format: 'media',
    });
  });
});
