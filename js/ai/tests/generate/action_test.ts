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

import { stripUndefinedProps, z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { readFileSync } from 'fs';
import { beforeEach, describe, it } from 'node:test';
import { parse } from 'yaml';
import {
  defineGenerateAction,
  type GenerateAction,
} from '../../src/generate/action.js';
import { generateMiddleware } from '../../src/generate/middleware.js';
import {
  GenerateActionOptionsSchema,
  GenerateResponseChunkSchema,
  GenerateResponseSchema,
  type GenerateResponseChunkData,
} from '../../src/model.js';
import { defineTool, tool } from '../../src/tool.js';
import { defineProgrammableModel, type ProgrammableModel } from '../helpers.js';

initNodeFeatures();

const SpecSuiteSchema = z
  .object({
    tests: z.array(
      z
        .object({
          name: z.string(),
          input: GenerateActionOptionsSchema,
          streamChunks: z
            .array(z.array(GenerateResponseChunkSchema))
            .optional(),
          modelResponses: z.array(GenerateResponseSchema),
          expectResponse: GenerateResponseSchema.optional(),
          stream: z.boolean().optional(),
          expectChunks: z.array(GenerateResponseChunkSchema).optional(),
        })
        .strict()
    ),
  })
  .strict();

describe('spec', () => {
  let registry: Registry;
  let pm: ProgrammableModel;

  beforeEach(() => {
    registry = new Registry();
    defineGenerateAction(registry);
    pm = defineProgrammableModel(registry);
    defineTool(
      registry,
      { name: 'testTool', description: 'description' },
      async () => 'tool called'
    );
  });

  SpecSuiteSchema.parse(
    parse(readFileSync('../../tests/specs/generate.yaml', 'utf-8'))
  ).tests.forEach((test) => {
    it(test.name, async () => {
      if (test.modelResponses || test.streamChunks) {
        let reqCounter = 0;
        pm.handleResponse = async (req, sc) => {
          if (test.streamChunks && sc) {
            test.streamChunks[reqCounter].forEach(sc);
          }
          return test.modelResponses?.[reqCounter++]!;
        };
      }
      const action = (await registry.lookupAction(
        '/util/generate'
      )) as GenerateAction;

      if (test.stream) {
        const { output, stream } = action.stream(test.input);

        const chunks = [] as GenerateResponseChunkData[];
        for await (const chunk of stream) {
          chunks.push(stripUndefinedProps(chunk));
        }

        assert.deepStrictEqual(chunks, test.expectChunks);

        assert.deepStrictEqual(
          stripUndefinedProps(await output),
          test.expectResponse
        );
      } else {
        const response = await action(test.input);

        assert.deepStrictEqual(
          stripUndefinedProps(response),
          test.expectResponse
        );
      }
    });
  });
});

describe('generateAction middleware injection', () => {
  let registry: Registry;
  let pm: ProgrammableModel;

  beforeEach(() => {
    registry = new Registry();
    defineGenerateAction(registry);
    pm = defineProgrammableModel(registry);
  });

  it('supports injecting tools through middleware definitions directly via action route', async () => {
    const injectedTool = tool(
      {
        name: 'injectedTool',
        description: 'desc',
        inputSchema: z.object({ arg: z.string() }),
      },
      async (input) => `Result: ${input.arg}`
    );

    let toolsSeen = false;
    pm.handleResponse = async (req) => {
      if (req.tools?.find((t) => t.name === 'injectedTool')) {
        toolsSeen = true;
      }
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        finishReason: 'stop',
      } as any;
    };

    const dummyMw = generateMiddleware({ name: 'dummyMw' }, () => ({
      tools: [injectedTool],
    }));

    const action = await registry.lookupAction('/util/generate');
    await action({
      model: 'programmableModel',
      messages: [{ role: 'user', content: [{ text: 'test' }] }],
      use: [dummyMw()],
    } as any);

    assert.ok(
      toolsSeen,
      'Tool was not successfully passed to the model from action generated route.'
    );
  });
});
