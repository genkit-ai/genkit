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

describe('jsonSchemaToPicoschema', () => {
  it('converts supported JSON Schema constructs into Picoschema', () => {
    const schema = {
      type: 'object',
      additionalProperties: false,
      properties: {
        name: { type: 'string', description: 'Display name' },
        age: { type: 'integer' },
        active: { type: 'boolean' },
        tags: {
          type: 'array',
          description: 'Search labels',
          items: { type: 'string' },
        },
        address: {
          type: 'object',
          description: 'Mailing address',
          additionalProperties: false,
          properties: { city: { type: 'string' } },
          required: ['city'],
        },
        status: {
          type: 'string',
          description: 'Account status',
          enum: ['ACTIVE', 'DISABLED'],
        },
      },
      required: ['name', 'active', 'tags', 'address', 'status'],
    };

    expect(jsonSchemaToPicoschema(schema)).toEqual({
      name: 'string, Display name',
      'age?': 'integer',
      active: 'boolean',
      'tags(array)': 'string',
      'address(object)': { city: 'string' },
      'status(enum)': ['ACTIVE', 'DISABLED'],
    });
  });

  it('omits constraints that Picoschema cannot represent', () => {
    expect(jsonSchemaToPicoschema({ type: 'string', minLength: 1 })).toBe(
      'string'
    );
  });

  it('converts additional properties into a wildcard', () => {
    const schema = {
      type: 'object',
      properties: { name: { type: 'string' } },
      required: ['name'],
      additionalProperties: { type: 'number' },
    };

    expect(jsonSchemaToPicoschema(schema)).toEqual({
      name: 'string',
      '(*)': 'number',
    });
  });

  it('uses optional notation for a nullable optional property', () => {
    const schema = {
      type: 'object',
      properties: { name: { type: ['string', 'null'] } },
      required: [],
    };

    expect(jsonSchemaToPicoschema(schema)).toEqual({ 'name?': 'string' });
  });

  it('converts top-level arrays and unconstrained schemas', () => {
    expect(
      jsonSchemaToPicoschema({ type: 'array', items: { type: 'string' } })
    ).toBe('string');
    expect(jsonSchemaToPicoschema({ description: 'Anything goes' })).toBe(
      'any, Anything goes'
    );
    expect(jsonSchemaToPicoschema({})).toBeUndefined();
  });
});

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

describe('toFrontmatterInput', () => {
  const SCHEMA = {
    type: 'object',
    properties: { name: { type: 'string' } },
  };

  it('returns undefined when there is no input', () => {
    expect(toFrontmatterInput(undefined)).toBeUndefined();
  });

  it('maps schema and default values', () => {
    expect(
      toFrontmatterInput({ schema: SCHEMA, default: { name: 'World' } })
    ).toEqual({
      schema: SCHEMA,
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

  it('reads the schema from jsonSchema and maps json formats', () => {
    expect(toFrontmatterOutput({ format: 'json', jsonSchema: SCHEMA })).toEqual(
      { format: 'json', schema: SCHEMA }
    );
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
  it('converts a request object into a .prompt template string', () => {
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

  it('uses compact Picoschema for input and output when requested', () => {
    const request = {
      model: '/model/googleai/gemini-pro',
      picoSchema: true,
      messages: [{ role: 'user' as const, content: [{ text: 'Hello' }] }],
      input: {
        schema: {
          type: 'object',
          properties: { topic: { type: 'string' } },
          required: ['topic'],
        },
      },
      output: {
        format: 'json',
        schema: {
          type: 'object',
          properties: { summary: { type: 'string' } },
          required: ['summary'],
        },
      },
    };

    expect(toPromptFile(request)).toContain(
      'input:\n  schema:\n    topic: string\noutput:\n  format: json\n  schema:\n    summary: string\n'
    );
  });
});
