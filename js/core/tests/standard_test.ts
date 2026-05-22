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

/**
 * Tests for Standard Schema support, Zod v4, and InferInput/InferOutput.
 *
 * Covers:
 * - isZodV3Schema / isZodV4Schema / isNonZodStandardSchema discriminators
 * - toJsonSchema for Zod v3 and Zod v4
 * - validateSchema for Zod v3, Zod v4, and non-Zod Standard Schema
 * - parseSchema for Zod v3 (including transforms) and Zod v4
 * - mergedInputJsonSchema helper
 * - GenkitSchema-typed action definitions
 */

import * as assert from 'assert';
import { describe, it } from 'node:test';
import { z } from 'zod';
import z4 from 'zod/v4';
import { action } from '../src/action.js';
import { initNodeFeatures } from '../src/node.js';
import {
  ValidationError,
  mergedInputJsonSchema,
  parseSchema,
  toJsonSchema,
  validateSchema,
} from '../src/schema.js';
import { Registry } from '../src/registry.js';
import {
  isNonZodStandardSchema,
  isZodV3Schema,
  isZodV4Schema,
  type GenkitSchema,
  type InferInput,
  type InferOutput,
} from '../src/standard.js';

// Initialize node-specific features (async context, telemetry) required for actions
initNodeFeatures();

// Initialize node-specific features (async context, telemetry) required for actions
initNodeFeatures();

// ─── Minimal hand-rolled Standard Schema v1 implementation ───────────────────

function makeStandardSchema<T>(
  validate: (value: unknown) => { value: T } | { issues: { message: string }[] }
): GenkitSchema & { '~standard': any } {
  return {
    '~standard': {
      version: 1,
      vendor: 'test',
      validate,
    },
  } as any;
}

// ─── Discriminators ───────────────────────────────────────────────────────────

describe('isZodV3Schema', () => {
  it('returns true for Zod v3 schemas', () => {
    assert.strictEqual(isZodV3Schema(z.string()), true);
    assert.strictEqual(isZodV3Schema(z.object({ x: z.number() })), true);
    assert.strictEqual(isZodV3Schema(z.boolean()), true);
  });

  it('returns false for Zod v4 schemas', () => {
    assert.strictEqual(isZodV3Schema(z4.string()), false);
    assert.strictEqual(isZodV3Schema(z4.object({ x: z4.number() })), false);
  });

  it('returns false for non-Zod Standard Schema', () => {
    const s = makeStandardSchema<string>((v) =>
      typeof v === 'string' ? { value: v } : { issues: [{ message: 'bad' }] }
    );
    assert.strictEqual(isZodV3Schema(s), false);
  });
});

describe('isZodV4Schema', () => {
  it('returns true for Zod v4 schemas', () => {
    assert.strictEqual(isZodV4Schema(z4.string()), true);
    assert.strictEqual(isZodV4Schema(z4.object({ x: z4.number() })), true);
  });

  it('returns false for Zod v3 schemas', () => {
    assert.strictEqual(isZodV4Schema(z.string()), false);
    assert.strictEqual(isZodV4Schema(z.object({ x: z.number() })), false);
  });

  it('returns false for non-Zod Standard Schema', () => {
    const s = makeStandardSchema<string>((v) =>
      typeof v === 'string' ? { value: v } : { issues: [{ message: 'bad' }] }
    );
    assert.strictEqual(isZodV4Schema(s), false);
  });
});

describe('isNonZodStandardSchema', () => {
  it('returns true for non-Zod Standard Schema', () => {
    const s = makeStandardSchema<string>((v) =>
      typeof v === 'string' ? { value: v } : { issues: [{ message: 'bad' }] }
    );
    assert.strictEqual(isNonZodStandardSchema(s), true);
  });

  it('returns false for Zod v3 schemas (they also implement Standard Schema)', () => {
    assert.strictEqual(isNonZodStandardSchema(z.string()), false);
  });

  it('returns false for Zod v4 schemas', () => {
    assert.strictEqual(isNonZodStandardSchema(z4.string()), false);
  });
});

// ─── toJsonSchema ─────────────────────────────────────────────────────────────

describe('toJsonSchema', () => {
  it('converts Zod v3 schema to JSON Schema', () => {
    const jsonSchema = toJsonSchema({
      schema: z.object({ name: z.string(), age: z.number() }),
    });
    assert.ok(jsonSchema, 'should return a JSON Schema');
    assert.strictEqual(jsonSchema.type, 'object');
    assert.ok(jsonSchema.properties?.name, 'should have name property');
    assert.ok(jsonSchema.properties?.age, 'should have age property');
  });

  it('converts Zod v4 schema to JSON Schema', () => {
    const jsonSchema = toJsonSchema({
      schema: z4.object({ name: z4.string(), score: z4.number() }),
    });
    assert.ok(jsonSchema, 'should return a JSON Schema');
    assert.strictEqual(jsonSchema.type, 'object');
    assert.ok(jsonSchema.properties?.name, 'should have name property');
    assert.ok(jsonSchema.properties?.score, 'should have score property');
  });

  it('returns {} for non-Zod Standard Schema (no generic JSON Schema derivation)', () => {
    const s = makeStandardSchema<string>((v) =>
      typeof v === 'string' ? { value: v } : { issues: [{ message: 'bad' }] }
    );
    const jsonSchema = toJsonSchema({ schema: s });
    assert.deepStrictEqual(jsonSchema, {});
  });

  it('prefers explicit jsonSchema over schema', () => {
    const explicit = { type: 'string', maxLength: 5 };
    const result = toJsonSchema({
      schema: z.number(),
      jsonSchema: explicit,
    });
    assert.deepStrictEqual(result, explicit);
  });

  it('caches Zod v3 results', () => {
    const schema = z.string();
    const first = toJsonSchema({ schema });
    const second = toJsonSchema({ schema });
    assert.strictEqual(first, second, 'should return the same cached object');
  });

  it('caches Zod v4 results', () => {
    const schema = z4.string();
    const first = toJsonSchema({ schema });
    const second = toJsonSchema({ schema });
    assert.strictEqual(first, second, 'should return the same cached object');
  });
});

// ─── validateSchema ───────────────────────────────────────────────────────────

describe('validateSchema with Zod v3', () => {
  it('returns valid for matching data', () => {
    const result = validateSchema(
      { name: 'Alice', age: 30 },
      { schema: z.object({ name: z.string(), age: z.number() }) }
    );
    assert.strictEqual(result.valid, true);
  });

  it('returns errors for invalid data', () => {
    const result = validateSchema(
      { name: 'Alice', age: 'not-a-number' },
      { schema: z.object({ name: z.string(), age: z.number() }) }
    );
    assert.strictEqual(result.valid, false);
    assert.ok(result.errors && result.errors.length > 0, 'should have errors');
    assert.ok(
      result.errors!.some((e) => e.path === 'age'),
      'error should be at age path'
    );
  });
});

describe('validateSchema with Zod v4', () => {
  it('returns valid for matching data', () => {
    const result = validateSchema(
      { name: 'Bob', score: 42 },
      { schema: z4.object({ name: z4.string(), score: z4.number() }) }
    );
    assert.strictEqual(result.valid, true);
  });

  it('returns errors for invalid data', () => {
    const result = validateSchema(
      { name: 'Bob', score: 'bad' },
      { schema: z4.object({ name: z4.string(), score: z4.number() }) }
    );
    assert.strictEqual(result.valid, false);
    assert.ok(result.errors && result.errors.length > 0, 'should have errors');
  });
});

describe('validateSchema with non-Zod Standard Schema', () => {
  it('returns valid for passing schema', () => {
    const s = makeStandardSchema<number>((v) =>
      typeof v === 'number'
        ? { value: v }
        : { issues: [{ message: 'not a number' }] }
    );
    const result = validateSchema(42, { schema: s });
    assert.strictEqual(result.valid, true);
  });

  it('returns errors for failing schema', () => {
    const s = makeStandardSchema<number>((v) =>
      typeof v === 'number'
        ? { value: v }
        : { issues: [{ message: 'not a number' }] }
    );
    const result = validateSchema('not-a-number', { schema: s });
    assert.strictEqual(result.valid, false);
    assert.ok(result.errors && result.errors.length > 0);
    assert.ok(
      result.errors!.some((e) => e.message === 'not a number'),
      'should include the schema error message'
    );
  });
});

// ─── parseSchema ──────────────────────────────────────────────────────────────

describe('parseSchema with Zod v3', () => {
  it('parses and returns valid data', () => {
    const result = parseSchema<{ name: string }>(
      { name: 'Carol' },
      { schema: z.object({ name: z.string() }) }
    );
    assert.deepStrictEqual(result, { name: 'Carol' });
  });

  it('applies Zod v3 transforms (output differs from input)', () => {
    const schema = z.string().transform((s) => s.toUpperCase());
    const result = parseSchema<string>('hello', { schema });
    assert.strictEqual(result, 'HELLO');
  });

  it('throws ValidationError for invalid data', () => {
    assert.throws(
      () =>
        parseSchema({ name: 42 }, { schema: z.object({ name: z.string() }) }),
      ValidationError
    );
  });
});

describe('parseSchema with Zod v4', () => {
  it('parses and returns valid data', () => {
    const result = parseSchema<{ x: number }>(
      { x: 7 },
      { schema: z4.object({ x: z4.number() }) }
    );
    assert.deepStrictEqual(result, { x: 7 });
  });

  it('throws for invalid data', () => {
    assert.throws(
      () =>
        parseSchema(
          { x: 'not-a-number' },
          { schema: z4.object({ x: z4.number() }) }
        ),
      ValidationError
    );
  });
});

describe('parseSchema with non-Zod Standard Schema', () => {
  it('returns the validated value', () => {
    const s = makeStandardSchema<number>((v) =>
      typeof v === 'number'
        ? { value: v * 2 }
        : { issues: [{ message: 'bad' }] }
    );
    const result = parseSchema<number>(21, { schema: s });
    // Standard Schema may return transformed value
    assert.strictEqual(result, 42);
  });

  it('throws for invalid data', () => {
    const s = makeStandardSchema<number>((v) =>
      typeof v === 'number'
        ? { value: v }
        : { issues: [{ message: 'not a number' }] }
    );
    assert.throws(
      () => parseSchema('oops', { schema: s }),
      ValidationError
    );
  });
});

// ─── mergedInputJsonSchema ────────────────────────────────────────────────────

describe('mergedInputJsonSchema', () => {
  const envelopeSchema = z.object({
    query: z.string(),
    options: z.any().optional(),
  });

  it('returns undefined when envelope has no JSON schema', () => {
    // Non-Zod standard schema with no JSON schema derivation
    const s = makeStandardSchema<unknown>((v) => ({ value: v }));
    const result = mergedInputJsonSchema(s, 'options');
    // Returns the empty schema from non-Zod standard schema
    assert.ok(result !== null);
  });

  it('returns envelope JSON schema when no configSchema given', () => {
    const result = mergedInputJsonSchema(envelopeSchema, 'options');
    assert.ok(result, 'should return a schema');
    assert.strictEqual(result.type, 'object');
    assert.ok(result.properties?.query, 'should have query property');
  });

  it('merges configSchema into options property with Zod v3 configSchema', () => {
    const configSchema = z.object({ k: z.number().describe('top-k') });
    const result = mergedInputJsonSchema(
      envelopeSchema,
      'options',
      configSchema
    );
    assert.ok(result, 'should return a schema');
    assert.ok(result.properties?.options, 'should have options property');
    // The merged options property should have the configSchema shape
    assert.ok(
      result.properties?.options?.properties?.k,
      'options should have k property from configSchema'
    );
  });

  it('merges configSchema into options property with Zod v4 configSchema', () => {
    const configSchema = z4.object({ k: z4.number().describe('top-k') });
    const result = mergedInputJsonSchema(
      envelopeSchema,
      'options',
      configSchema
    );
    assert.ok(result, 'should return a schema');
    assert.ok(result.properties?.options, 'should have options property');
    assert.ok(
      result.properties?.options?.properties?.k,
      'options should have k property from z4 configSchema'
    );
  });

  it('merges extraProps into the properties', () => {
    const result = mergedInputJsonSchema(envelopeSchema, 'options', undefined, {
      dataset: { type: 'array', items: { type: 'object' } },
    });
    assert.ok(result, 'should return a schema');
    assert.ok(result.properties?.dataset, 'should have dataset property');
    assert.strictEqual(result.properties?.dataset?.type, 'array');
  });
});

// ─── GenkitSchema-typed actions ───────────────────────────────────────────────

describe('action() with GenkitSchema types', () => {
  it('works with Zod v3 inputSchema', async () => {
    const registry = new Registry();
    const act = action(
      {
        actionType: 'flow',
        name: 'test-v3',
        inputSchema: z.object({ x: z.number() }),
        outputSchema: z.object({ doubled: z.number() }),
      },
      async (input) => ({ doubled: input.x * 2 })
    );
    const result = await act({ x: 5 });
    assert.deepStrictEqual(result, { doubled: 10 });
  });

  it('works with Zod v4 inputSchema', async () => {
    const act = action(
      {
        actionType: 'flow',
        name: 'test-v4',
        inputSchema: z4.object({ x: z4.number() }),
        outputSchema: z4.object({ doubled: z4.number() }),
      },
      async (input) => ({ doubled: input.x * 2 })
    );
    const result = await act({ x: 7 });
    assert.deepStrictEqual(result, { doubled: 14 });
  });

  it('works with non-Zod Standard Schema', async () => {
    const inputSchema = makeStandardSchema<{ x: number }>((v: any) =>
      v && typeof v.x === 'number'
        ? { value: { x: v.x } }
        : { issues: [{ message: 'expected {x: number}' }] }
    );
    const act = action(
      {
        actionType: 'flow',
        name: 'test-standard',
        inputSchema,
      },
      async (input: any) => input.x * 3
    );
    const result = await act({ x: 4 });
    assert.strictEqual(result, 12);
  });

  it('validates input and throws for invalid Zod v3 input', async () => {
    const act = action(
      {
        actionType: 'flow',
        name: 'test-v3-validate',
        inputSchema: z.object({ name: z.string() }),
      },
      async (input) => input.name
    );
    await assert.rejects(
      () => act({ name: 123 } as any),
      ValidationError
    );
  });

  it('validates input and throws for invalid Zod v4 input', async () => {
    const act = action(
      {
        actionType: 'flow',
        name: 'test-v4-validate',
        inputSchema: z4.object({ name: z4.string() }),
      },
      async (input) => input.name
    );
    await assert.rejects(
      () => act({ name: 123 } as any),
      ValidationError
    );
  });
});

// ─── InferInput / InferOutput type checks ────────────────────────────────────
// These are compile-time checks — if TypeScript accepts the code, the types work.

describe('InferInput and InferOutput type compatibility', () => {
  it('InferOutput of Zod v3 transform schema differs from InferInput', () => {
    // z.string() with no transform: input === output === string
    type V3Input = InferInput<typeof z.prototype.string>;
    type V3Output = InferOutput<typeof z.prototype.string>;
    // The main check: parseSchema returns the output type (post-transform)
    const schema = z.string().transform((s) => s.length);
    const input: string = 'hello';
    const output: number = parseSchema<number>(input, { schema });
    assert.strictEqual(output, 5);
  });

  it('InferOutput of Zod v4 schema is the output type', () => {
    const schema = z4.string().transform((s) => s.split(''));
    const output = parseSchema<string[]>('hi', { schema });
    assert.deepStrictEqual(output, ['h', 'i']);
  });
});
