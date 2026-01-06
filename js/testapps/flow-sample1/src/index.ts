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

import { genkit, z } from 'genkit';

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

export const chatFlow = ai.defineBidiFlow(
  {
    name: 'chatFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async function* ({ inputStream }) {
    for await (const chunk of inputStream) {
      yield `echo ${chunk}`;
    }
    return 'done';
  }
);

export const chatFlowWithInit = ai.defineBidiFlow(
  {
    name: 'chatFlowWithInit',
    inputSchema: z.string(),
    outputSchema: z.string(),
    initSchema: z.object({ prefix: z.string() }),
  },
  async function* ({ inputStream, init }) {
    const prefix = init?.prefix || '';
    for await (const chunk of inputStream) {
      yield `${prefix}${chunk}`;
    }
    return 'done';
  }
);
