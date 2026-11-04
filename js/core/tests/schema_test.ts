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

import * as assert from 'assert';
import { describe, it } from 'node:test';

import {
  ValidationError,
  annotateSchema,
  parseSchema,
  parseSchemaAsync,
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
      errors: [
        {
          path: 'foo',
          message: 'Instance type "number" is invalid. Expected "boolean".',
        },
      ],
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
      errors: [
        {
          path: 'foo',
          message: 'Instance type "number" is invalid. Expected "boolean".',
        },
      ],
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
      errors: [
        {
          path: 'foo.0.bar',
          message: 'Instance type "number" is invalid. Expected "boolean".',
        },
      ],
    },
    {
      it: 'should be understandable for top-level errors',
      jsonSchema: { type: 'object', additionalProperties: false },
      data: { foo: 'bar' },
      valid: false,
      errors: [
        {
          path: '(root)',
          message:
            'Property "foo" does not match additional properties schema.',
        },
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
        {
          path: '(root)',
          message: 'Instance does not have required property "foo".',
        },
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
    const def = Object.values(defs ?? {})[0] as any;
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

describe('validateSchema() with JSON schemas', () => {
  it('reports one error per problem, at the failing path', () => {
    const result = validateSchema(
      { at: 'yesterday', items: [{ n: 1 }, { n: 'x' }, {}], u: true },
      {
        jsonSchema: {
          type: 'object',
          properties: {
            q: { type: 'string' },
            at: { type: 'string', format: 'date-time' },
            items: {
              type: 'array',
              items: {
                type: 'object',
                properties: { n: { type: 'number' } },
                required: ['n'],
              },
            },
            u: { anyOf: [{ type: 'string' }, { type: 'number' }] },
          },
          required: ['q'],
        },
      }
    );
    assert.deepStrictEqual(result.errors, [
      {
        path: '(root)',
        message: 'Instance does not have required property "q".',
      },
      { path: 'at', message: 'String does not match format "date-time".' },
      {
        path: 'items.1.n',
        message: 'Instance type "string" is invalid. Expected "number".',
      },
      {
        path: 'items.2',
        message: 'Instance does not have required property "n".',
      },
      { path: 'u', message: 'Instance does not match any subschemas.' },
    ]);
  });

  it('reports disallowed additional properties', () => {
    const result = validateSchema(
      { a: 1, b: 2 },
      {
        jsonSchema: {
          type: 'object',
          properties: { a: { type: 'number' } },
          additionalProperties: false,
        },
      }
    );
    assert.deepStrictEqual(result.errors, [
      {
        path: '(root)',
        message: 'Property "b" does not match additional properties schema.',
      },
    ]);
  });

  it('ignores unknown keywords such as x-genkit-* annotations', () => {
    const jsonSchema = {
      type: 'object',
      properties: { p: { type: 'string', 'x-genkit-widget': 'textarea' } },
    };
    assert.strictEqual(validateSchema({ p: 'ok' }, { jsonSchema }).valid, true);
  });

  it('supports draft 2020-12 schemas', () => {
    const jsonSchema = {
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      type: 'array',
      prefixItems: [{ type: 'string' }, { type: 'number' }],
    };
    assert.strictEqual(validateSchema(['a', 1], { jsonSchema }).valid, true);
    assert.deepStrictEqual(validateSchema(['a', 'b'], { jsonSchema }).errors, [
      {
        path: '1',
        message: 'Instance type "string" is invalid. Expected "number".',
      },
    ]);
  });

  it('should strip undefined values before validating', () => {
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
      {
        path: 'k',
        message: 'Instance type "string" is invalid. Expected "number".',
      },
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

describe('parseSchema() / parseSchemaAsync()', () => {
  const Search = z.object({
    q: z.string().trim().min(1),
    limit: z.number().default(10),
    since: z
      .string()
      .optional()
      .transform((s) => (s ? new Date(s) : undefined)),
  });

  it('returns the schema output: defaults and transforms applied', async () => {
    const expected = { q: 'x', limit: 10, since: new Date('2026-01-01') };
    const input = { q: ' x ', since: '2026-01-01' };
    assert.deepStrictEqual(parseSchema(input, { schema: Search }), expected);
    assert.deepStrictEqual(
      await parseSchemaAsync(input, { schema: Search }),
      expected
    );
  });

  it('strips unknown keys from zod objects, keeps them for looseObject', () => {
    assert.deepStrictEqual(
      parseSchema(
        { a: 1, extra: true },
        { schema: z.object({ a: z.number() }) }
      ),
      { a: 1 }
    );
    assert.deepStrictEqual(
      parseSchema(
        { a: 1, extra: true },
        { schema: z.looseObject({ a: z.number() }) }
      ),
      { a: 1, extra: true }
    );
  });

  it('enforces refinements and reports zod issues as details', () => {
    const schema = z
      .object({ a: z.string(), b: z.number() })
      .refine((v) => v.a !== 'no', {
        message: 'a must not be no',
        path: ['a'],
      });
    assert.throws(
      () => parseSchema({ a: 'no', b: 1 }, { schema }),
      (e: ValidationError) => {
        assert.deepStrictEqual(e.detail.errors, [
          { path: 'a', message: 'a must not be no' },
        ]);
        return true;
      }
    );
    assert.throws(
      () => parseSchema({}, { schema }),
      (e: ValidationError) => {
        assert.deepStrictEqual(
          e.detail.errors.map((d: { path: string }) => d.path),
          ['a', 'b']
        );
        return true;
      }
    );
    assert.throws(
      () => parseSchema('x', { schema: z.number() }),
      (e: ValidationError) => e.detail.errors[0].path === '(root)'
    );
  });

  it('supports async refinements only in parseSchemaAsync', async () => {
    const schema = z.string().refine(async (s) => s === 'ok', 'not ok');
    assert.strictEqual(await parseSchemaAsync('ok', { schema }), 'ok');
    await assert.rejects(parseSchemaAsync('nope', { schema }), ValidationError);
    assert.throws(
      () => parseSchema('ok', { schema }),
      /requires async validation/
    );
  });

  it('parses non-zod Standard Schemas with ~standard.validate', async () => {
    const schema = fakeStandardSchema();
    assert.deepStrictEqual(parseSchema({ k: 1 }, { schema }), { k: 1 });
    assert.throws(
      () => parseSchema({ k: 'x' }, { schema }),
      (e: ValidationError) => {
        assert.deepStrictEqual(e.detail.errors, [
          { path: 'k', message: 'k must be a number' },
        ]);
        return true;
      }
    );
    const asyncSchema = fakeStandardSchema({ async: true });
    assert.deepStrictEqual(
      await parseSchemaAsync({ k: 2 }, { schema: asyncSchema }),
      { k: 2 }
    );
    assert.throws(
      () => parseSchema({ k: 2 }, { schema: asyncSchema }),
      /requires async validation/
    );
  });

  it('only validates JSON schemas and returns the data as is', () => {
    const data = { a: 1, extra: true };
    const jsonSchema = {
      type: 'object',
      properties: { a: { type: 'number' } },
    };
    assert.strictEqual(parseSchema(data, { jsonSchema }), data);
    assert.throws(
      () => parseSchema({ a: 'x' }, { jsonSchema }),
      ValidationError
    );
  });
});

describe('parseSchema() with z.void()', () => {
  it('passes values through as is, like a TS void return', async () => {
    assert.strictEqual(parseSchema('done', { schema: z.void() }), 'done');
    assert.strictEqual(
      await parseSchemaAsync(null, { schema: z.void() }),
      null
    );
  });
});
