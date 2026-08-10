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

import { z } from '@genkit-ai/core';
import type { Registry } from '@genkit-ai/core/registry';
import { runInNewSpan } from '@genkit-ai/core/tracing';
import * as assert from 'assert';
import { generate } from '../generate.js';
import type { ModelAction } from '../model.js';
import { defineTool } from '../tool.js';

const tests: Record<string, TestCase> = {
  'basic hi': async (registry: Registry, model: string) => {
    const response = await generate(registry, {
      model,
      prompt: 'just say "Hi", literally',
    });

    const got = response.text.trim();
    assert.match(got, /Hi/i);
  },
  multimodal: async (registry: Registry, model: string) => {
    const resolvedModel = (await registry.lookupAction(
      `/model/${model}`
    )) as ModelAction;
    if (!resolvedModel.__action.metadata?.model.supports?.media) {
      skip();
    }
    const response = await generate(registry, {
      model,
      prompt: [
        {
          media: {
            url: 'data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9TpSoVETOIOGSoulgQFXHUKhShQqgVWnUwufRDaNKQtLg4Cq4FBz8Wqw4uzro6uAqC4AeIs4OToouU+L+k0CLGg+N+vLv3uHsHCLUi0+22MUA3ylYyHpPSmRUp9IpOhCCiFyMKs81ZWU7Ad3zdI8DXuyjP8j/35+jWsjYDAhLxDDOtMvE68dRm2eS8TyyygqIRnxOPWnRB4keuqx6/cc67LPBM0Uol54hFYinfwmoLs4KlE08SRzTdoHwh7bHGeYuzXqywxj35C8NZY3mJ6zQHEccCFiFDgooKNlBEGVFaDVJsJGk/5uMfcP0yuVRybYCRYx4l6FBcP/gf/O7Wzk2Me0nhGND+4jgfQ0BoF6hXHef72HHqJ0DwGbgymv5SDZj+JL3a1CJHQM82cHHd1NQ94HIH6H8yFUtxpSBNIZcD3s/omzJA3y3Qter11tjH6QOQoq4SN8DBITCcp+w1n3d3tPb275lGfz9aC3Kd0jYiSQAAAAlwSFlzAAAuIwAALiMBeKU/dgAAAAd0SU1FB+gJBxQRO1/5qB8AAAAZdEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIEdJTVBXgQ4XAAAAsUlEQVQoz61SMQqEMBDcO5SYToUE/IBPyRMCftAH+INUviApUwYjNkKCVcTiQK7IHSw45czODrMswCOQUkopEQZjzDiOWemdZfu+b5oGYYgx1nWNMPwB2vACAK01Y4wQ8qGqqirL8jzPlNI9t64r55wQUgBA27be+xDCfaJhGJxzSqnv3UKIn7ne+2VZEB2stZRSRLN93+d5RiRs28Y5RySEEI7jyEpFlp2mqeu6Zx75ApQwPdsIcq0ZAAAAAElFTkSuQmCC',
          },
        },
        {
          text: 'what math operation is this? plus, minus, multiply or divide?',
        },
      ],
    });

    const want = /plus/i;
    const got = response.text.trim();
    assert.match(got, want);
  },
  history: async (registry: Registry, model: string) => {
    const resolvedModel = (await registry.lookupAction(
      `/model/${model}`
    )) as ModelAction;
    if (!resolvedModel.__action.metadata?.model.supports?.multiturn) {
      skip();
    }
    const response1 = await generate(registry, {
      model,
      prompt: 'My name is Glorb',
    });
    const response = await generate(registry, {
      model,
      prompt: "What's my name?",
      messages: response1.messages,
    });

    const got = response.text.trim();
    assert.match(got, /Glorb/);
  },
  'system prompt': async (registry: Registry, model: string) => {
    const { text } = await generate(registry, {
      model,
      prompt: 'Hi',
      messages: [
        {
          role: 'system',
          content: [
            {
              text: 'If the user says "Hi", just say "Bye" ',
            },
          ],
        },
      ],
    });

    const want = 'Bye';
    const got = text.trim();
    assert.equal(got, want);
  },
  'structured output': async (registry: Registry, model: string) => {
    const response = await generate(registry, {
      model,
      prompt: 'extract data as json from: Jack was a Lumberjack',
      output: {
        format: 'json',
        schema: z.object({
          name: z.string(),
          occupation: z.string(),
        }),
      },
    });

    const want = {
      name: 'Jack',
      occupation: 'Lumberjack',
    };
    const got = response.output;
    assert.deepEqual(want, got);
  },
  'tool calling': async (registry: Registry, model: string) => {
    const resolvedModel = (await registry.lookupAction(
      `/model/${model}`
    )) as ModelAction;
    if (!resolvedModel.__action.metadata?.model.supports?.tools) {
      skip();
    }

    const { text } = await generate(registry, {
      model,
      prompt: 'what is a gablorken of 2? use provided tool',
      tools: ['gablorkenTool'],
    });

    const got = text.trim();
    assert.match(got, /9.407/);
  },
};

type TestReport = {
  description: string;
  models: {
    name: string;
    passed: boolean;
    skipped?: boolean;
    error?: {
      message: string;
      stack?: string;
    };
  }[];
}[];

type TestCase = (ai: Registry, model: string) => Promise<void>;

export async function testModels(
  registry: Registry,
  models: string[]
): Promise<TestReport> {
  defineTool(
    registry,
    {
      name: 'gablorkenTool',
      description: 'use when need to calculate a gablorken',
      inputSchema: z.object({
        value: z.number(),
      }),
      outputSchema: z.number(),
    },
    async (input) => {
      return Math.pow(input.value, 3) + 1.407;
    }
  );

  return await runInNewSpan(
    registry,
    { metadata: { name: 'testModels' } },
    async () => {
      const report: TestReport = [];
      for (const test of Object.keys(tests)) {
        await runInNewSpan(registry, { metadata: { name: test } }, async () => {
          report.push({
            description: test,
            models: [],
          });
          const caseReport = report[report.length - 1];
          for (const model of models) {
            caseReport.models.push({
              name: model,
              passed: true, // optimistically
            });
            const modelReport = caseReport.models[caseReport.models.length - 1];
            try {
              await tests[test](registry, model);
            } catch (e) {
              modelReport.passed = false;
              if (e instanceof SkipTestError) {
                modelReport.skipped = true;
              } else if (e instanceof Error) {
                modelReport.error = {
                  message: e.message,
                  stack: e.stack,
                };
              } else {
                modelReport.error = {
                  message: `${e}`,
                };
              }
            }
          }
        });
      }

      return report;
    }
  );
}

class SkipTestError extends Error {}

function skip() {
  throw new SkipTestError();
}
