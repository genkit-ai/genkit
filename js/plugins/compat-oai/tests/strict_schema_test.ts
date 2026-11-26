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
import { z } from 'genkit';
import type { GenerateRequest } from 'genkit/model';
import { toJsonSchema } from 'genkit/schema';
import { toOpenAIRequestBody } from '../src/model';
import { closeObjectSchemas } from '../src/strict-schema';

const person = {
  type: 'object',
  properties: { name: { type: 'string' }, age: { type: 'integer' } },
  required: ['name', 'age'],
  additionalProperties: true,
};

describe('closeObjectSchemas', () => {
  it('closes an object with additionalProperties true or absent', () => {
    expect(closeObjectSchemas(person).additionalProperties).toBe(false);
    const { additionalProperties, ...open } = person;
    expect(closeObjectSchemas(open).additionalProperties).toBe(false);
  });

  it('closes nested objects in properties, items and combinators', () => {
    const schema = {
      type: 'object',
      properties: {
        people: { type: 'array', items: person },
        pick: { anyOf: [person, { type: 'string' }] },
        tuple: { type: 'array', prefixItems: [person] },
      },
      required: ['people', 'pick', 'tuple'],
    };
    const closed = closeObjectSchemas(schema) as any;
    expect(closed.additionalProperties).toBe(false);
    expect(closed.properties.people.items.additionalProperties).toBe(false);
    expect(closed.properties.pick.anyOf[0].additionalProperties).toBe(false);
    expect(closed.properties.pick.anyOf[1]).toStrictEqual({ type: 'string' });
    expect(closed.properties.tuple.prefixItems[0].additionalProperties).toBe(
      false
    );
  });

  it('closes objects in a tuple', () => {
    const closed = closeObjectSchemas({
      type: 'array',
      items: [person, { type: 'string' }],
    }) as any;
    expect(closed.items[0].additionalProperties).toBe(false);
    expect(closed.items[1]).toStrictEqual({ type: 'string' });
  });

  it('leaves subschemas under allOf, not and if/then/else open', () => {
    const schema = {
      type: 'object',
      properties: { a: { type: 'string' } },
      additionalProperties: false,
      allOf: [person],
      not: person,
      if: person,
      then: person,
      else: person,
    };
    expect(closeObjectSchemas(schema)).toStrictEqual(schema);
  });

  it('closes every object in a schema converted from zod', () => {
    const schema = toJsonSchema({
      schema: z.object({
        name: z.string(),
        address: z.object({ city: z.string() }),
        tags: z.array(z.object({ label: z.string() })),
        pair: z.tuple([z.object({ x: z.number() }), z.string()]),
        pick: z.union([z.object({ a: z.string() }), z.number()]),
      }),
    }) as any;
    expect(schema.additionalProperties).toBe(true);
    const closed = closeObjectSchemas(schema) as any;
    expect(closed.additionalProperties).toBe(false);
    const p = closed.properties;
    expect(p.address.additionalProperties).toBe(false);
    expect(p.tags.items.additionalProperties).toBe(false);
    expect(p.pair.items[0].additionalProperties).toBe(false);
    expect(p.pick.anyOf[0].additionalProperties).toBe(false);
  });

  it('closes objects in $defs and definitions', () => {
    const closed = closeObjectSchemas({
      type: 'array',
      items: { $ref: '#/$defs/Person' },
      $defs: { Person: person },
      definitions: { Old: person },
    }) as any;
    expect(closed.$defs.Person.additionalProperties).toBe(false);
    expect(closed.definitions.Old.additionalProperties).toBe(false);
  });

  it('treats a nullable object type as an object', () => {
    const closed = closeObjectSchemas({
      type: ['object', 'null'],
      properties: { a: { type: 'string' } },
    });
    expect(closed.additionalProperties).toBe(false);
  });

  it('leaves non-objects, closed objects and schema-valued additionalProperties alone', () => {
    expect(closeObjectSchemas({ type: 'string' })).toStrictEqual({
      type: 'string',
    });
    const closedAlready = { ...person, additionalProperties: false };
    expect(closeObjectSchemas(closedAlready)).toStrictEqual(closedAlready);
    const record = {
      type: 'object',
      additionalProperties: { type: 'number' },
    };
    expect(closeObjectSchemas(record)).toStrictEqual(record);
  });

  it('closes an object inside a schema-valued additionalProperties', () => {
    const closed = closeObjectSchemas({
      type: 'object',
      additionalProperties: person,
    }) as any;
    expect(closed.additionalProperties.additionalProperties).toBe(false);
  });

  it('does not mutate its input', () => {
    const schema = { type: 'object', properties: { p: { ...person } } };
    const copy = JSON.parse(JSON.stringify(schema));
    closeObjectSchemas(schema);
    expect(schema).toStrictEqual(copy);
  });
});

describe('toOpenAIRequestBody strict schemas', () => {
  const request = (config?: Record<string, unknown>) =>
    ({
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      config,
      output: { format: 'json', schema: person },
    }) as unknown as GenerateRequest;

  it('closes objects when strict is true', () => {
    const body = toOpenAIRequestBody(
      'gpt-4o',
      request({ strict: true })
    ) as any;
    expect(body.response_format.json_schema).toStrictEqual({
      name: 'output',
      schema: { ...person, additionalProperties: false },
      strict: true,
    });
  });

  it('sends the schema unchanged when strict is false or unset', () => {
    for (const config of [undefined, { strict: false }]) {
      const body = toOpenAIRequestBody('gpt-4o', request(config)) as any;
      expect(body.response_format.json_schema.schema).toStrictEqual(person);
    }
  });

  it('closes objects inside a wrapped non-object root', () => {
    const body = toOpenAIRequestBody(
      'gpt-4o',
      {
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
        config: { strict: true },
        output: { format: 'json', schema: { type: 'array', items: person } },
      } as unknown as GenerateRequest,
      undefined,
      { objectSchemaRoot: true }
    ) as any;
    expect(body.response_format.json_schema.schema).toStrictEqual({
      type: 'object',
      properties: {
        value: {
          type: 'array',
          items: { ...person, additionalProperties: false },
        },
      },
      required: ['value'],
      additionalProperties: false,
    });
  });
});
