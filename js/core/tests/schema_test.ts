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
