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

import Ajv from 'ajv';
import * as assert from 'assert';
import { afterEach, describe, it, mock } from 'node:test';

import { setGenkitRuntimeConfig } from '../src/config.js';
import {
  ValidationError,
  annotateSchema,
  parseSchema,
  toJsonSchema,
  toZodSchema,
  validateSchema,
  z,
  type GenkitSchema,
} from '../src/schema.js';

describe('validate()', () => {
  const tests = [
    {
      it: 'should return true for a valid json schema',
      jsonSchema: {
        type: 'object',
        properties: {
          foo: {
            type: 'boolean',
          },
        },
      },
      data: { foo: true },
      valid: true,
    },
    {
      it: 'should return errors for an invalid json schema',
      jsonSchema: {
        type: 'object',
        properties: {
          foo: {
            type: 'boolean',
          },
        },
      },
      data: { foo: 123 },
      valid: false,
      errors: [{ path: 'foo', message: 'must be boolean' }],
    },
    {
      it: 'should return true for a valid zod schema',
      schema: z.object({ foo: z.boolean() }),
      data: { foo: true },
      valid: true,
    },
    {
      it: 'should return errors for an invalid zod schema',
      schema: z.object({ foo: z.boolean() }),
      data: { foo: 123 },
      valid: false,
      errors: [{ path: 'foo', message: 'must be boolean' }],
    },
    {
      it: 'should allow for date types',
      schema: z.object({ date: z.string().datetime() }),
      data: { date: '2024-05-22T17:00:00Z' },
      valid: true,
    },
    {
      it: 'should return dotted path for errors',
      schema: z.object({ foo: z.array(z.object({ bar: z.boolean() })) }),
      data: { foo: [{ bar: 123 }] },
      valid: false,
      errors: [{ path: 'foo.0.bar', message: 'must be boolean' }],
    },
    {
      it: 'should be understandable for top-level errors',
      jsonSchema: { type: 'object', additionalProperties: false },
      data: { foo: 'bar' },
      valid: false,
      errors: [
        { path: '(root)', message: 'must NOT have additional properties' },
      ],
    },
    {
      it: 'should be understandable for required fields',
      jsonSchema: {
        type: 'object',
        properties: { foo: { type: 'string' } },
        required: ['foo'],
      },
      data: {},
      valid: false,
      errors: [
        { path: '(root)', message: "must have required property 'foo'" },
      ],
    },
  ];
  for (const test of tests) {
    it(test.it, () => {
      const { valid, errors } = validateSchema(test.data, {
        jsonSchema: test.jsonSchema,
        schema: test.schema,
      });
      assert.strictEqual(valid, test.valid);
      assert.deepStrictEqual(errors, test.errors);
    });
  }
});

describe('parse()', () => {
  it('should throw a ValidationError for invalid schema', () => {
    assert.throws(() => {
      parseSchema(
        { foo: 123 },
        {
          schema: z.object({ foo: z.boolean() }),
        }
      );
    }, ValidationError);
  });

  it('should return the data if valid', () => {
    assert.deepEqual(
      parseSchema(
        { foo: true },
        {
          schema: z.object({ foo: z.boolean() }),
        }
      ),
      { foo: true }
    );
  });
});

describe('toJsonSchema', () => {
  it('converts zod to JSON schema', async () => {
    assert.deepStrictEqual(
      toJsonSchema({
        schema: z.object({
          output: z.string(),
        }),
      }),
      {
        $schema: 'http://json-schema.org/draft-07/schema#',
        properties: {
          output: {
            type: 'string',
          },
        },
        required: ['output'],
        type: 'object',
      }
    );
  });
});

describe('toJsonSchema() zod v3 output compat', () => {
  it('emits date-time strings for z.date()', () => {
    const json = toJsonSchema({ schema: z.object({ d: z.date() }) });
    assert.deepStrictEqual(json.properties.d, {
      type: 'string',
      format: 'date-time',
    });
  });

  it('emits anyOf (not oneOf) for discriminated unions', () => {
    const json = toJsonSchema({
      schema: z.discriminatedUnion('t', [
        z.object({ t: z.literal('a') }),
        z.object({ t: z.literal('b') }),
      ]),
    });
    assert.strictEqual(json.oneOf, undefined);
    assert.strictEqual(json.anyOf.length, 2);
  });

  it('emits type arrays for described nullable primitives', () => {
    const json = toJsonSchema({ schema: z.string().describe('d').nullable() });
    assert.deepStrictEqual(json.type, ['string', 'null']);
    assert.strictEqual(json.description, 'd');
    assert.strictEqual(json.anyOf, undefined);
  });

  it('emits string formats without the internal regex pattern', () => {
    const json = toJsonSchema({ schema: z.string().datetime() });
    assert.deepStrictEqual(
      { type: json.type, format: json.format, pattern: json.pattern },
      { type: 'string', format: 'date-time', pattern: undefined }
    );
  });

  it('drops implicit safe-integer bounds from .int()', () => {
    const json = toJsonSchema({ schema: z.number().int().min(0) });
    assert.deepStrictEqual(
      { type: json.type, minimum: json.minimum, maximum: json.maximum },
      { type: 'integer', minimum: 0, maximum: undefined }
    );
  });

  it('omits no-op propertyNames for string-keyed records', () => {
    const json = toJsonSchema({ schema: z.record(z.string(), z.number()) });
    assert.strictEqual(json.propertyNames, undefined);
    assert.deepStrictEqual(json.additionalProperties, { type: 'number' });
  });

  it('keeps oneOf for non-discriminated oneOf unions (z.xor)', () => {
    const json = toJsonSchema({ schema: z.xor([z.string(), z.number()]) });
    assert.strictEqual(json.anyOf, undefined);
    assert.strictEqual(json.oneOf.length, 2);
  });

  it('prefers the outer description when collapsing nullable primitives', () => {
    const json = toJsonSchema({
      schema: z.string().describe('inner').nullable().describe('outer'),
    });
    assert.deepStrictEqual(json.type, ['string', 'null']);
    assert.strictEqual(json.description, 'outer');
  });

  it('keeps only the pattern for formats the validators do not know', () => {
    for (const schema of [
      z.cuid(),
      z.ulid(),
      z.base64(),
      z.nanoid(),
      z.string().startsWith('ab'),
    ]) {
      const json = toJsonSchema({ schema });
      assert.strictEqual(typeof json.pattern, 'string', json.pattern);
      assert.strictEqual(json.format, undefined, json.pattern);
      assert.strictEqual(
        validateSchema('!!not valid!!', { schema }).valid,
        false,
        json.pattern
      );
    }
  });

  it('keeps anyOf for nullable strings with several patterns', () => {
    const schema = z.string().regex(/^a/).regex(/b$/).nullable();
    assert.strictEqual(validateSchema(null, { schema }).valid, true);
    assert.strictEqual(validateSchema('ab', { schema }).valid, true);
    assert.strictEqual(validateSchema('xb', { schema }).valid, false);
  });

  it('drops the pattern for formats the validators check', () => {
    for (const schema of [z.email(), z.uuid(), z.url(), z.iso.date()]) {
      const json = toJsonSchema({ schema });
      assert.strictEqual(json.pattern, undefined, json.format);
    }
  });
});

describe('annotateSchema()', () => {
  it('should merge annotations into the JSON schema', () => {
    const schema = annotateSchema(z.string(), {
      'x-test': 'foo',
    });

    const json = toJsonSchema({ schema });
    assert.strictEqual(json['x-test'], 'foo');
  });

  it('should merge annotations for nested fields', () => {
    const schema = z.object({
      field: annotateSchema(z.string(), {
        'x-test': 'bar',
      }),
    });

    const json = toJsonSchema({ schema });
    assert.strictEqual(json.properties.field['x-test'], 'bar');
  });

  it('should merge annotations for array items', () => {
    const schema = z.array(
      annotateSchema(z.string(), {
        'x-test': 'baz',
      })
    );

    const json = toJsonSchema({ schema });
    assert.strictEqual(json.items['x-test'], 'baz');
  });

  it('should merge annotations for optional fields', () => {
    const schema = z.object({
      field: annotateSchema(z.string(), {
        'x-test': 'qux',
      }).optional(),
    });

    const json = toJsonSchema({ schema });
    assert.strictEqual(json.properties.field['x-test'], 'qux');
  });

  it('should favor outer annotations over inner ones', () => {
    const schema = annotateSchema(
      annotateSchema(z.string(), { title: 'Inner' }).optional(),
      { title: 'Outer' }
    );

    const json = toJsonSchema({ schema });
    assert.strictEqual(json.title, 'Outer');
  });

  it('should merge annotations for ZodUnion (anyOf)', () => {
    // Use objects to force anyOf instead of simple type array optimization
    const schema = z.union([
      annotateSchema(z.object({ a: z.string() }), { 'x-hint': 'a' }),
      annotateSchema(z.object({ b: z.number() }), { 'x-hint': 'b' }),
    ]);

    const json = toJsonSchema({ schema });
    assert.ok(json.anyOf, 'JSON schema should have anyOf');
    assert.strictEqual(json.anyOf[0]['x-hint'], 'a');
    assert.strictEqual(json.anyOf[1]['x-hint'], 'b');
  });

  it('should merge annotations for ZodIntersection (allOf)', () => {
    const schema = z.intersection(
      annotateSchema(z.object({ a: z.string() }), { 'x-hint': 'a' }),
      annotateSchema(z.object({ b: z.number() }), { 'x-hint': 'b' })
    );

    const json = toJsonSchema({ schema });
    assert.ok(json.allOf, 'JSON schema should have allOf');
    assert.strictEqual(json.allOf[0]['x-hint'], 'a');
    assert.strictEqual(json.allOf[1]['x-hint'], 'b');
  });

  it('should merge annotations for nested ZodIntersection (flattened allOf)', () => {
    const schema = z.intersection(
      z.intersection(
        annotateSchema(z.object({ a: z.string() }), { 'x-hint': 'a' }),
        annotateSchema(z.object({ b: z.number() }), { 'x-hint': 'b' })
      ),
      annotateSchema(z.object({ c: z.boolean() }), { 'x-hint': 'c' })
    );

    const json = toJsonSchema({ schema });
    assert.ok(json.allOf, 'JSON schema should have allOf');
    assert.strictEqual(json.allOf.length, 3, 'Should have 3 elements in allOf');
    assert.strictEqual(json.allOf[0]['x-hint'], 'a');
    assert.strictEqual(json.allOf[1]['x-hint'], 'b');
    assert.strictEqual(json.allOf[2]['x-hint'], 'c');
  });

  it('should merge annotations for ZodRecord (additionalProperties)', () => {
    const schema = z.record(annotateSchema(z.string(), { 'x-hint': 'foo' }));

    const json = toJsonSchema({ schema });
    assert.ok(
      json.additionalProperties,
      'JSON schema should have additionalProperties'
    );
    assert.strictEqual(json.additionalProperties['x-hint'], 'foo');
  });

  it('should merge annotations for ZodTuple (items array)', () => {
    const schema = z.tuple([
      annotateSchema(z.string(), { 'x-hint': 'first' }),
      annotateSchema(z.number(), { 'x-hint': 'second' }),
    ]);

    const json = toJsonSchema({ schema });
    assert.ok(
      Array.isArray(json.items),
      'JSON schema items should be an array'
    );
    assert.strictEqual(json.items[0]['x-hint'], 'first');
    assert.strictEqual(json.items[1]['x-hint'], 'second');
  });

  it('applies annotations inside recursive (z.lazy) schemas', () => {
    interface Node {
      name: string;
      children?: Node[];
    }
    const NodeSchema: z.ZodType<Node> = z.lazy(() =>
      z.object({
        name: annotateSchema(z.string(), { 'x-hint': 'name' }),
        children: z.array(NodeSchema).optional(),
      })
    );
    const json = toJsonSchema({ schema: z.object({ root: NodeSchema }) });
    // The recursive definition is emitted once under definitions/$defs.
    const defs = json.definitions ?? json.$defs;
    const def = Object.values(defs ?? {})[0];
    assert.strictEqual(def.properties.name['x-hint'], 'name');
  });

  it('should merge annotations for ZodDiscriminatedUnion (anyOf)', () => {
    const schema = z.discriminatedUnion('type', [
      annotateSchema(z.object({ type: z.literal('a'), a: z.string() }), {
        'x-hint': 'a',
      }),
      annotateSchema(z.object({ type: z.literal('b'), b: z.number() }), {
        'x-hint': 'b',
      }),
    ]);

    const json = toJsonSchema({ schema });
    assert.ok(json.anyOf, 'JSON schema should have anyOf');
    assert.strictEqual(json.anyOf[0]['x-hint'], 'a');
    assert.strictEqual(json.anyOf[1]['x-hint'], 'b');
  });

  it('annotations are merged over generated JSON schema fields', () => {
    // zod v4 metadata semantics: registry metadata is applied on top of the
    // generated schema, so conflicting keys win.
    const schema = annotateSchema(z.string(), { 'x-ok': true, title: 'T' });

    const json = toJsonSchema({ schema });

    assert.strictEqual(json.type, 'string');
    assert.strictEqual(json['x-ok'], true);
    assert.strictEqual(json.title, 'T');
  });
});

describe('disableSchemaCodeGeneration()', () => {
  let compileMock: any;

  function disableSchemaCodeGeneration() {
    setGenkitRuntimeConfig({
      jsonSchemaMode: 'interpret',
    });
  }

  afterEach(() => {
    setGenkitRuntimeConfig({
      jsonSchemaMode: undefined,
    });
    if (compileMock) {
      compileMock.mock.restore();
      compileMock = undefined;
    }
  });

  it('should validate using cfworker validator', () => {
    compileMock = mock.method(Ajv.prototype, 'compile');

    disableSchemaCodeGeneration();
    const result = validateSchema(
      { foo: 123 },
      {
        jsonSchema: {
          type: 'object',
          properties: { foo: { type: 'boolean' } },
        },
      }
    );

    assert.strictEqual(result.valid, false);
    const errorAtFoo = result.errors?.find((e) => e.path === 'foo');
    assert.ok(errorAtFoo, 'Should have error at foo');
    assert.strictEqual(compileMock.mock.callCount(), 0);
  });

  it('should strip undefined values before validating', () => {
    disableSchemaCodeGeneration();
    const result = validateSchema(
      { foo: 'hello', bar: undefined },
      {
        jsonSchema: {
          type: 'object',
          properties: { foo: { type: 'string' }, bar: { type: 'string' } },
          required: ['foo'],
        },
      }
    );
    assert.strictEqual(result.valid, true);
  });

  it('should strip undefined values recursively', () => {
    disableSchemaCodeGeneration();
    const result = validateSchema(
      { wrapper: { inner: 'hello', ignored: undefined } },
      {
        jsonSchema: {
          type: 'object',
          properties: {
            wrapper: {
              type: 'object',
              properties: { inner: { type: 'string' } },
            },
          },
        },
      }
    );
    assert.strictEqual(result.valid, true);
  });

  it('should strip undefined values in objects inside arrays', () => {
    disableSchemaCodeGeneration();
    const result = validateSchema(
      { items: [{ name: 'item1', desc: undefined }, { name: 'item2' }] },
      {
        jsonSchema: {
          type: 'object',
          properties: {
            items: {
              type: 'array',
              items: {
                type: 'object',
                properties: { name: { type: 'string' } },
                required: ['name'],
              },
            },
          },
        },
      }
    );
    assert.strictEqual(result.valid, true);
  });
});

/**
 * Minimal hand-rolled Standard Schema + Standard JSON Schema implementation
 * (what ArkType / Valibot provide), for an object with a required numeric `k`.
 */
function fakeStandardSchema(
  opts: { async?: boolean } = {}
): GenkitSchema<{ k: number }, { k: number }> {
  const validate = (value: unknown) => {
    const ok =
      typeof value === 'object' &&
      value !== null &&
      typeof (value as { k?: unknown }).k === 'number';
    const result = ok
      ? { value: value as { k: number } }
      : { issues: [{ message: 'k must be a number', path: [{ key: 'k' }] }] };
    return opts.async ? Promise.resolve(result) : result;
  };
  return {
    '~standard': {
      version: 1,
      vendor: 'fake',
      validate,
      jsonSchema: {
        input: () => ({
          $schema: 'http://json-schema.org/draft-07/schema#',
          type: 'object',
          properties: { k: { type: 'number' } },
          required: ['k'],
        }),
        output: () => ({}),
      },
    },
  };
}

describe('Standard Schema (GenkitSchema)', () => {
  it('converts a non-zod schema to JSON schema via ~standard.jsonSchema', () => {
    assert.deepStrictEqual(toJsonSchema({ schema: fakeStandardSchema() }), {
      $schema: 'http://json-schema.org/draft-07/schema#',
      type: 'object',
      properties: { k: { type: 'number' } },
      required: ['k'],
    });
  });

  it('validates a non-zod schema', () => {
    const schema = fakeStandardSchema();
    assert.strictEqual(validateSchema({ k: 1 }, { schema }).valid, true);
    assert.deepStrictEqual(validateSchema({ k: 'x' }, { schema }).errors, [
      { path: 'k', message: 'must be number' },
    ]);
  });

  it('composes into zod schemas and keeps sync .parse() working', () => {
    const composed = z.object({
      q: z.string(),
      options: toZodSchema(fakeStandardSchema()).optional(),
    });
    assert.deepStrictEqual(composed.parse({ q: 'x', options: { k: 1 } }), {
      q: 'x',
      options: { k: 1 },
    });
    const bad = composed.safeParse({ q: 'x', options: { k: 'no' } });
    assert.strictEqual(bad.success, false);
    assert.deepStrictEqual(
      bad.error?.issues.map((i) => [i.path.join('.'), i.message]),
      [['options.k', 'k must be a number']]
    );
    assert.deepStrictEqual(
      toJsonSchema({ schema: composed }).properties.options,
      {
        type: 'object',
        properties: { k: { type: 'number' } },
        required: ['k'],
      }
    );
  });

  it('supports async validation via parseAsync', async () => {
    const adapter = toZodSchema(fakeStandardSchema({ async: true }));
    assert.deepStrictEqual(await adapter.parseAsync({ k: 2 }), { k: 2 });
    assert.strictEqual((await adapter.safeParseAsync({})).success, false);
  });

  it('returns zod schemas as is', () => {
    const schema = z.string();
    assert.strictEqual(toZodSchema(schema), schema);
  });
});
