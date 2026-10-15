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

import { describe, expect, it } from '@jest/globals';
import Ajv from 'ajv';
import { Dotprompt } from 'dotprompt';
import { toPromptFile } from '../../src/utils/prompt';

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
        expect(schema).toEqual(parsedBaseline[channel]?.schema);
        expect(ajv.validate(schema as object, testCase.valid)).toBe(true);
        expect(ajv.validate(schema as object, testCase.invalid)).toBe(false);
        expect(testCase.schema).toEqual(original);
        expect(toPromptFile(request(testCase.schema))).toBe(baseline);
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
