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

import { toStandardJsonSchema } from '@valibot/to-json-schema';
import { type } from 'arktype';
import { genkit, z } from 'genkit/beta';
import * as v from 'valibot';

const ai = genkit({});

// TEST FLOW: Long-running flow for broadcast testing (2-3 minutes)
export const testLongBroadcast = ai.defineFlow(
  {
    name: 'test-long-broadcast',
    inputSchema: z.object({
      steps: z.number().default(10).describe('Number of steps to execute'),
      stepDelay: z
        .number()
        .default(15000)
        .describe('Delay in ms between steps (default 15s)'),
    }),
    outputSchema: z.object({
      totalDuration: z.number(),
      stepsCompleted: z.number(),
      timeline: z.array(
        z.object({
          step: z.number(),
          timestamp: z.string(),
          elapsed: z.number(),
        })
      ),
    }),
  },
  async ({ steps = 10, stepDelay = 15000 }) => {
    const startTime = Date.now();
    const timeline: Array<{
      step: number;
      timestamp: string;
      elapsed: number;
    }> = [];

    console.log(
      `🚀 Starting long broadcast test: ${steps} steps × ${stepDelay / 1000}s = ~${(steps * stepDelay) / 60000} minutes`
    );

    for (let i = 1; i <= steps; i++) {
      const stepStart = Date.now();

      await ai.run(`step-${i}`, async () => {
        console.log(
          `[${new Date().toISOString()}] 🔄 Step ${i}/${steps} starting...`
        );

        // Simulate some work with nested spans
        await ai.run(`step-${i}-fetch`, async () => {
          await new Promise((resolve) => setTimeout(resolve, stepDelay / 3));
          console.log(`  📡 Fetched data for step ${i}`);
          return `fetch-${i}`;
        });

        await ai.run(`step-${i}-process`, async () => {
          await new Promise((resolve) => setTimeout(resolve, stepDelay / 3));
          console.log(`  ⚙️  Processed data for step ${i}`);
          return `process-${i}`;
        });

        await ai.run(`step-${i}-save`, async () => {
          await new Promise((resolve) => setTimeout(resolve, stepDelay / 3));
          console.log(`  💾 Saved results for step ${i}`);
          return `save-${i}`;
        });

        const elapsed = Date.now() - stepStart;
        console.log(
          `[${new Date().toISOString()}] ✅ Step ${i}/${steps} completed (${elapsed}ms)`
        );

        timeline.push({
          step: i,
          timestamp: new Date().toISOString(),
          elapsed,
        });

        return `Step ${i} complete`;
      });
    }

    const totalDuration = Date.now() - startTime;
    console.log(
      `🎉 Long broadcast test completed in ${totalDuration / 1000}s (${(totalDuration / 60000).toFixed(1)} minutes)`
    );

    return {
      totalDuration,
      stepsCompleted: steps,
      timeline,
    };
  }
);

/**
 * To run this flow;
 *   genkit flow:run basic "\"hello\""
 */
export const basic = ai.defineFlow(
  { name: 'basic', inputSchema: z.string() },
  async (subject) => {
    const foo = await ai.run('call-llm', async () => {
      return `subject: ${subject}`;
    });

    return await ai.run('call-llm1', async () => {
      return `foo: ${foo}`;
    });
  }
);

export const parent = ai.defineFlow(
  { name: 'parent', outputSchema: z.string() },
  async () => {
    return JSON.stringify(await basic('foo'));
  }
);

export const withInputSchema = ai.defineFlow(
  { name: 'withInputSchema', inputSchema: z.object({ subject: z.string() }) },
  async (input) => {
    const foo = await ai.run('call-llm', async () => {
      return `subject: ${input.subject}`;
    });

    return await ai.run('call-llm1', async () => {
      return `foo: ${foo}`;
    });
  }
);

// Input and output are parsed with their schemas: defaults, transforms and
// refinements apply, whether the flow is called directly, streamed, or run
// from the Dev UI. Try `{ "query": "  genkit  " }` in the Dev UI.
export const withSchemaParsing = ai.defineFlow(
  {
    name: 'withSchemaParsing',
    inputSchema: z
      .object({
        query: z.string().trim().min(1),
        limit: z.number().int().default(3),
        since: z
          .string()
          .optional()
          .transform((s) => (s ? new Date(s) : undefined)),
      })
      .refine((i) => i.query !== 'forbidden', {
        message: 'that query is not allowed',
        path: ['query'],
      }),
    outputSchema: z.object({
      results: z.array(z.string()),
      truncated: z.boolean().default(false),
    }),
  },
  // input: { query: string; limit: number; since?: Date }
  async ({ query, limit, since }) => {
    const results = Array.from(
      { length: limit },
      (_, i) =>
        `${query} #${i + 1}${since ? ` since ${since.toISOString()}` : ''}`
    );
    // `truncated` is filled in by the output schema default.
    return { results };
  }
);

export const callsWithSchemaParsing = ai.defineFlow(
  'callsWithSchemaParsing',
  async () => {
    // Callers pass the schema's input type: `limit` and `since` are optional.
    const { results, truncated } = await withSchemaParsing({ query: ' hi ' });
    return { results, truncated };
  }
);

export const withContext = ai.defineFlow(
  {
    name: 'withContext',
    inputSchema: z.object({ subject: z.string() }),
  },
  async (input, { context }) => {
    return `subject: ${input.subject}, context: ${JSON.stringify(context)}`;
  }
);

// genkit flow:run streamy 5 -s
export const streamy = ai.defineFlow(
  {
    name: 'streamy',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: z.object({ count: z.number() }),
  },
  async (count: number, { sendChunk }: any) => {
    let i = 0;
    for (; i < count; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      sendChunk({ count: i });
    }
    return `done: ${count}, streamed: ${i} times`;
  }
);

// genkit flow:run streamyThrowy 5 -s
export const streamyThrowy = ai.defineFlow(
  {
    name: 'streamyThrowy',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: z.object({ count: z.number() }),
  },
  async (count, { sendChunk }) => {
    let i = 0;
    for (; i < count; i++) {
      if (i == 3) {
        throw new Error('whoops');
      }
      await new Promise((r) => setTimeout(r, 1000));
      sendChunk({ count: i });
    }
    return `done: ${count}, streamed: ${i} times`;
  }
);

/**
 * To run this flow;
 *   genkit flow:run throwy "\"hello\""
 */
export const throwy = ai.defineFlow(
  { name: 'throwy', inputSchema: z.string(), outputSchema: z.string() },
  async (subject) => {
    const foo = await ai.run('call-llm', async () => {
      return `subject: ${subject}`;
    });
    if (subject) {
      throw new Error(subject);
    }
    return await ai.run('call-llm', async () => {
      return `foo: ${foo}`;
    });
  }
);

/**
 * To run this flow;
 *   genkit flow:run throwy2 "\"hello\""
 */
export const throwy2 = ai.defineFlow(
  { name: 'throwy2', inputSchema: z.string(), outputSchema: z.string() },
  async (subject) => {
    const foo = await ai.run('call-llm', async () => {
      if (subject) {
        throw new Error(subject);
      }
      return `subject: ${subject}`;
    });
    return await ai.run('call-llm', async () => {
      return `foo: ${foo}`;
    });
  }
);

export const flowMultiStepCaughtError = ai.defineFlow(
  { name: 'flowMultiStepCaughtError' },
  async (input: any) => {
    let i = 1;

    const result1 = await ai.run('step1', async () => {
      return `${input} ${i++},`;
    });

    let result2 = '';
    try {
      result2 = await ai.run('step2', async () => {
        if (result1) {
          throw new Error('Got an error!');
        }
        return `${result1} ${i++},`;
      });
    } catch (e) {}

    return await ai.run('step3', async () => {
      return `${result2} ${i++}`;
    });
  }
);

export const multiSteps = ai.defineFlow(
  { name: 'multiSteps', inputSchema: z.string(), outputSchema: z.number() },
  async (input) => {
    const out1 = await ai.run('step1', async () => {
      return `Hello, ${input}! step 1`;
    });
    await ai.run('step1', async () => {
      return `Hello2222, ${input}! step 1`;
    });
    const out2 = await ai.run('step2', out1, async () => {
      return out1 + ' Faf ';
    });
    const out3 = await ai.run('step3-array', async () => {
      return [out2, out2];
    });
    const out4 = await ai.run('step4-num', async () => {
      return out3.join('-()-');
    });
    return 42;
  }
);

export const largeSteps = ai.defineFlow({ name: 'largeSteps' }, async () => {
  await ai.run('large-step1', async () => {
    return generateString(100_000);
  });
  await ai.run('large-step2', async () => {
    return generateString(800_000);
  });
  await ai.run('large-step3', async () => {
    return generateString(900_000);
  });
  await ai.run('large-step4', async () => {
    return generateString(999_000);
  });
  return 'something...';
});

// Flows, tools and agent state accept any Standard Schema library that also
// implements Standard JSON Schema, not just zod. ArkType works as is; Valibot
// via `toStandardJsonSchema()`.
export const arkWeather = ai.defineFlow(
  {
    name: 'arkWeather',
    inputSchema: type({ city: 'string', 'days?': 'number.integer > 0' }),
    outputSchema: type({ city: 'string', days: 'number' }),
  },
  async ({ city, days = 3 }) => ({ city, days })
);

export const valibotLookup = ai.defineTool(
  {
    name: 'valibotLookup',
    description: 'Looks up a record by id',
    inputSchema: toStandardJsonSchema(v.object({ id: v.string() })),
    outputSchema: toStandardJsonSchema(
      v.object({ id: v.string(), found: v.boolean() })
    ),
  },
  async ({ id }) => ({ id, found: id.length > 0 })
);

export const valibotFlow = ai.defineFlow(
  {
    name: 'valibotFlow',
    inputSchema: toStandardJsonSchema(
      v.object({
        id: v.pipe(v.string(), v.minLength(1)),
        tags: v.optional(v.array(v.string())),
      })
    ),
  },
  async ({ id, tags }) => {
    const record = await valibotLookup({ id });
    return { ...record, tags: tags ?? [] };
  }
);

export const arkCounterAgent = ai.defineCustomAgent(
  { name: 'arkCounterAgent', stateSchema: type({ count: 'number' }) },
  async (sess) => {
    await sess.run(async () => {
      sess.updateCustom((s) => ({ count: (s?.count ?? 0) + 1 }));
      return { finishReason: 'stop' as const };
    });
    return {
      message: {
        role: 'model',
        content: [{ text: `count: ${sess.getCustom()?.count}` }],
      },
    };
  }
);

const loremIpsum = [
  'lorem',
  'ipsum',
  'dolor',
  'sit',
  'amet',
  'consectetur',
  'adipiscing',
  'elit',
];
function generateString(length: number) {
  let str = '';
  while (str.length < length) {
    str += loremIpsum[Math.floor(Math.random() * loremIpsum.length)] + ' ';
  }
  return str.substring(0, length);
}
