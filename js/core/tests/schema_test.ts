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
  validateSchema,
  z,
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
        additionalProperties: true,
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

  it('should not overwrite existing JSON schema fields and log a warning', () => {
    const warnSpy = mock.method(console, 'warn', () => {});
    const schema = annotateSchema(z.string(), { type: 'number', 'x-ok': true });

    const json = toJsonSchema({ schema });

    assert.strictEqual(json.type, 'string');
    assert.strictEqual(json['x-ok'], true);
    assert.strictEqual(warnSpy.mock.callCount(), 1);
    assert.ok(
      warnSpy.mock.calls[0].arguments[0].includes(
        'Annotation key "type" conflicts'
      )
    );
    warnSpy.mock.restore();
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
