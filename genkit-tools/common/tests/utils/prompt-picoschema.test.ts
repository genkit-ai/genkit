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

import { Ajv } from 'ajv';
import { Dotprompt } from 'dotprompt';
import { describe, expect, it } from 'vitest';
import { toPromptFile } from '../../src/utils/prompt.js';

function object(properties: Record<string, unknown>) {
  return {
    type: 'object',
    properties,
    required: Object.keys(properties),
    additionalProperties: false,
  };
}

const fallbackCases = [
  {
    name: 'root array',
    schema: { type: 'array', items: { type: 'string' } },
    valid: ['hello'],
    invalid: 'hello',
  },
  {
    name: 'nested array dimension',
    schema: object({
      matrix: {
        type: 'array',
        items: { type: 'array', items: { type: 'number' } },
      },
    }),
    valid: { matrix: [[1, 2]] },
    invalid: { matrix: [1, 2] },
  },
  {
    name: 'array-valued wildcard',
    schema: {
      type: 'object',
      additionalProperties: { type: 'array', items: { type: 'string' } },
    },
    valid: { extra: ['hello'] },
    invalid: { extra: 'hello' },
  },
  {
    name: 'items-only property',
    schema: object({ tags: { items: { type: 'string' } } }),
    valid: { tags: ['hello'] },
    invalid: { tags: [2] },
  },
  {
    name: 'required property named type',
    schema: object({ type: { type: 'string' }, name: { type: 'string' } }),
    valid: { type: 'widget', name: 'example' },
    invalid: 'widget',
  },
  {
    name: 'unsupported name in an array item',
    schema: object({
      rows: { type: 'array', items: object({ 'a(b)': { type: 'string' } }) },
    }),
    valid: { rows: [{ 'a(b)': 'hello' }] },
    invalid: { rows: [{}] },
  },
  {
    name: 'root enum',
    schema: { type: 'string', enum: ['RED', 'BLUE'] },
    valid: 'RED',
    invalid: 'GREEN',
  },
  {
    name: 'enum array items',
    schema: object({
      colors: {
        type: 'array',
        items: { type: 'string', enum: ['RED', 'BLUE'] },
      },
    }),
    valid: { colors: ['RED'] },
    invalid: { colors: ['GREEN'] },
  },
  ...['\n', '\r', '\u2028', '\u2029'].map((separator) => ({
    name: `scalar description containing ${JSON.stringify(separator)}`,
    schema: { type: 'string', description: `first${separator}second` },
    valid: 'hello',
    invalid: 2,
  })),
  {
    name: 'empty object',
    schema: { type: 'object' },
    valid: {},
    invalid: 'hello',
  },
  {
    name: 'nested empty object',
    schema: object({ child: { type: 'object' } }),
    valid: { child: {} },
    invalid: { child: 'hello' },
  },
  ...[
    { type: ['string', 'null'] },
    { type: ['array', 'null'], items: { type: 'string' } },
    { type: ['object', 'null'], properties: { name: { type: 'string' } } },
  ].map((schema) => ({
    name: `required nullable ${schema.type[0]}`,
    schema: object({ value: schema }),
    valid: { value: null },
    invalid: {},
  })),
  {
    name: 'null-only enum property',
    schema: object({ value: { type: 'null', enum: [null] } }),
    valid: { value: null },
    invalid: { value: 'hello' },
  },
  {
    name: 'nested type union',
    schema: object({ value: { type: ['string', 'number'] } }),
    valid: { value: 3 },
    invalid: { value: false },
  },
  {
    name: 'nested combinator',
    schema: object({
      value: { anyOf: [{ type: 'string' }, { type: 'number' }] },
    }),
    valid: { value: 'hello' },
    invalid: { value: false },
  },
  {
    name: 'nested reference',
    schema: {
      ...object({ value: { $ref: '#/$defs/name' } }),
      $defs: { name: { type: 'string' } },
    },
    valid: { value: 'hello' },
    invalid: { value: 3 },
  },
  {
    name: 'nested typed constant',
    schema: object({ value: { type: 'string', const: 'v1' } }),
    valid: { value: 'v1' },
    invalid: { value: 'v2' },
  },
];

describe('Picoschema prompt round trips', () => {
  const ajv = new Ajv({ strict: false });

  describe.each(['input', 'output'] as const)('%s schema', (channel) => {
    function request(schema: unknown, picoSchema?: boolean) {
      return {
        model: 'test/model',
        messages: [{ role: 'user' as const, content: [{ text: 'Hello' }] }],
        picoSchema,
        [channel]: { schema },
      };
    }

    it.each(fallbackCases)(
      'retains JSON Schema for $name',
      async (testCase) => {
        const original = structuredClone(testCase.schema);
        const source = toPromptFile(request(testCase.schema, true));
        const baseline = toPromptFile(request(testCase.schema, false));
        const parsed = await new Dotprompt().renderMetadata(source);
        const parsedBaseline = await new Dotprompt().renderMetadata(baseline);
        const schema = parsed[channel]?.schema;

        expect(schema).toBeDefined();
        expect(schema).toEqual(testCase.schema);
        expect(ajv.validate(schema as object, testCase.valid)).toBe(true);
        expect(ajv.validate(schema as object, testCase.invalid)).toBe(false);
        expect(
          ajv.validate(
            parsedBaseline[channel]?.schema as object,
            testCase.valid
          )
        ).toBe(true);
        expect(
          ajv.validate(
            parsedBaseline[channel]?.schema as object,
            testCase.invalid
          )
        ).toBe(false);
        expect(testCase.schema).toEqual(original);
        expect(toPromptFile(request(testCase.schema))).toBe(baseline);
      }
    );

    it.each([
      {
        name: 'a described unconstrained schema',
        schema: { description: 'Anything goes' },
      },
      {
        name: 'an unconstrained schema with metadata',
        schema: { title: 'Any value' },
      },
      { name: 'a type union', schema: { type: ['string', 'number'] } },
      {
        name: 'a type union with object-specific properties',
        schema: {
          type: ['string', 'object'],
          properties: { value: { type: 'string' } },
          anyOf: [{}, { type: 'string' }],
        },
      },
      {
        name: 'a root combinator with an empty branch',
        schema: { anyOf: [{}, { type: 'string' }] },
      },
      { name: 'a root null constant', schema: { const: null } },
    ])('matches default export for $name', ({ schema }) => {
      expect(toPromptFile(request(schema, true))).toBe(
        toPromptFile(request(schema, false))
      );
    });

    it.each([
      { name: 'scalar null type', type: 'null' },
      { name: 'single-entry null type array', type: ['null'] },
    ])(
      'preserves an optional null-only field with a $name',
      async ({ type }) => {
        const schema = {
          type: 'object',
          properties: { value: { type } },
        };
        const source = toPromptFile(request(schema, true));
        const parsed = await new Dotprompt().renderMetadata(source);
        const actual = parsed[channel]?.schema as object;

        expect(actual).toStrictEqual(schema);
        const validate = ajv.compile(actual);
        expect(
          [{}, { value: null }, { value: 'v1' }].map((value) => validate(value))
        ).toEqual([true, true, false]);
      }
    );

    it.each([
      {
        name: 'an empty subschema',
        schema: { type: 'object', not: {} },
      },
      {
        name: 'a null constant',
        schema: object({ value: { const: null } }),
      },
      {
        name: 'a tuple containing an unconstrained slot',
        schema: object({
          values: { type: 'array', items: [{}, { type: 'number' }] },
        }),
      },
      {
        name: 'an array with items forbidden',
        schema: object({ values: { type: 'array', items: false } }),
      },
    ])('preserves $name in fallback frontmatter', async ({ schema }) => {
      const source = toPromptFile(request(schema, true));
      const parsed = await new Dotprompt().renderMetadata(source);

      expect(parsed[channel]?.schema).toStrictEqual(schema);
    });

    it('preserves descriptions on parenthetical property types', async () => {
      const schema = {
        type: 'object',
        properties: {
          status: { enum: ['ACTIVE'], description: 'Account status' },
          tags: {
            type: 'array',
            items: { type: 'string' },
            description: 'Search labels, when known',
          },
          address: {
            ...object({ city: { type: 'string' } }),
            description: 'Mailing address',
          },
        },
        required: ['tags'],
      };
      const source = toPromptFile(request(schema, true));
      const parsed = await new Dotprompt().renderMetadata(source);

      expect(parsed[channel]?.schema).toMatchObject({
        properties: {
          status: { enum: ['ACTIVE', null], description: 'Account status' },
          tags: { type: 'array', description: 'Search labels, when known' },
          address: { type: ['object', 'null'], description: 'Mailing address' },
        },
        required: ['tags'],
      });
    });

    it.each(['', '  Indented description'])(
      'preserves an array description Dotprompt would trim (%j)',
      async (description) => {
        const schema = object({
          values: { type: 'array', items: { type: 'string' }, description },
        });
        const source = toPromptFile(request(schema, true));
        const parsed = await new Dotprompt().renderMetadata(source);

        expect(parsed[channel]?.schema).toStrictEqual(schema);
      }
    );

    it('still exports supported objects in compact notation', async () => {
      const schema = {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'Display name' },
          tags: { type: 'array', items: { type: ['string'] } },
          child: object({ name: { type: 'string' } }),
          rows: { type: 'array', items: object({ type: { type: 'string' } }) },
          status: { type: 'string', enum: ['ACTIVE', 'DISABLED'] },
          note: { type: 'string' },
        },
        required: ['title', 'tags', 'child', 'rows', 'status'],
        additionalProperties: { type: 'number' },
      };
      const original = structuredClone(schema);
      const source = toPromptFile(request(schema, true));
      expect(source).toContain('title: string, Display name');
      expect(source).toContain('tags(array): string');
      expect(source).toContain('status(enum):');
      expect(source).toContain('note?: string');

      const parsed = await new Dotprompt().renderMetadata(source);
      const actual = parsed[channel]?.schema as object;
      const valid = {
        title: 'hello',
        tags: ['one'],
        child: { name: 'example' },
        rows: [{ type: 'widget' }],
        status: 'ACTIVE',
        note: null,
        extra: 2,
      };
      expect(ajv.validate(actual, valid)).toBe(true);
      expect(ajv.validate(actual, { ...valid, tags: 'one' })).toBe(false);
      expect(ajv.validate(actual, { ...valid, status: 'UNKNOWN' })).toBe(false);
      expect(schema).toEqual(original);
    });
  });
});
