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

// Real (non-zod) Standard Schema libraries on the end-user surfaces we
// recommend them for: flows, tools and agent custom state.

import { toStandardJsonSchema } from '@valibot/to-json-schema';
import { type } from 'arktype';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import * as v from 'valibot';
import { genkit, type GenkitBeta } from '../src/beta.js';
import type { GenerateRequest } from '../src/model.js';
import { toJsonSchema } from '../src/schema.js';

const libraries = {
  arktype: {
    input: type({ city: 'string', 'days?': 'number.integer > 0' }),
    state: type({ count: 'number' }),
    withDefaults: type({
      id: 'string.trim',
      tags: type('string[]').default(() => []),
    }),
  },
  valibot: {
    input: toStandardJsonSchema(
      v.object({
        city: v.string(),
        days: v.optional(v.pipe(v.number(), v.integer(), v.minValue(1))),
      })
    ),
    state: toStandardJsonSchema(v.object({ count: v.number() })),
    withDefaults: toStandardJsonSchema(
      v.object({
        id: v.pipe(v.string(), v.trim()),
        tags: v.optional(v.array(v.string()), []),
      })
    ),
  },
};

for (const [name, schemas] of Object.entries(libraries)) {
  describe(`${name} schemas`, () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({});
    });

    it('describes and validates flow input', async () => {
      const weather = ai.defineFlow(
        { name: 'weather', inputSchema: schemas.input },
        async ({ city, days }) => `${city}: ${days ?? 3}`
      );

      assert.strictEqual(await weather({ city: 'Paris', days: 2 }), 'Paris: 2');
      await assert.rejects(
        // @ts-expect-error days must be a number
        () => weather({ city: 'Paris', days: 'two' }),
        (e: Error) => /Schema validation failed/.test(e.message)
      );

      const json = toJsonSchema({ schema: schemas.input });
      assert.deepStrictEqual(json.required, ['city']);
      assert.deepStrictEqual(json.properties.city, { type: 'string' });
    });

    it('applies defaults and transforms to flow input', async () => {
      const f = ai.defineFlow(
        { name: 'withDefaults', inputSchema: schemas.withDefaults },
        async (input) => input
      );
      assert.deepStrictEqual(await f({ id: '  a  ' }), { id: 'a', tags: [] });
    });

    it('sends tool input JSON schema to the model', async () => {
      const lookup = ai.defineTool(
        {
          name: 'lookup',
          description: 'Look up the weather',
          inputSchema: schemas.input,
        },
        async ({ city }) => city
      );
      let request: GenerateRequest | undefined;
      ai.defineModel({ name: 'capture' }, async (req) => {
        request = req;
        return {
          message: { role: 'model', content: [{ text: 'ok' }] },
          finishReason: 'stop',
        };
      });
      await ai.generate({ model: 'capture', prompt: 'hi', tools: [lookup] });

      const inputSchema = request?.tools?.[0]?.inputSchema;
      assert.strictEqual(inputSchema?.type, 'object');
      assert.deepStrictEqual(
        Object.keys(inputSchema?.properties ?? {}).sort(),
        ['city', 'days']
      );
      assert.deepStrictEqual(inputSchema?.required, ['city']);
    });

    it('validates agent custom state', async () => {
      const agent = ai.defineCustomAgent(
        { name: 'counter', stateSchema: schemas.state },
        async (sess) => {
          await sess.run(async () => ({ finishReason: 'stop' as const }));
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );
      const metadata = (await ai.registry.lookupAction('/agent/counter'))!
        .__action.metadata as { agent?: { stateSchema?: { type?: string } } };
      assert.strictEqual(metadata.agent?.stateSchema?.type, 'object');

      const run = async (custom: unknown) => {
        const session = agent.streamBidi({
          state: { custom, messages: [], artifacts: [] },
        });
        session.send({
          message: { role: 'user', content: [{ text: 'hi' }] },
        });
        session.close();
        for await (const _ of session.stream) {
        }
        return await session.output;
      };
      assert.strictEqual((await run({ count: 1 })).finishReason, 'stop');
      const bad = await run({ count: 'one' });
      assert.strictEqual(bad.finishReason, 'failed');
      assert.strictEqual(bad.error?.status, 'INVALID_ARGUMENT');
    });
  });
}
