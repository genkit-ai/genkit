/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { describe, it } from 'node:test';
import { generate } from '../../src/generate.js';
import { defineModel, type ToolResponsePart } from '../../src/model.js';
import { defineTool, tool } from '../../src/tool.js';

initNodeFeatures();

describe('multipart tools in generate options', () => {
  for (const dynamic of [true, false]) {
    it(`preserves output, content and metadata from a ${dynamic ? 'dynamic' : 'registered'} tool`, async () => {
      const registry = new Registry();
      const config = {
        name: 'multipartTool',
        description: 'Returns a result with additional context',
        multipart: true as const,
        inputSchema: z.string(),
        outputSchema: z.string(),
      };
      const implementation = async (input: string) => ({
        output: input,
        content: [{ text: 'Additional context' }],
        metadata: { source: 'options' },
      });
      const suppliedTool = dynamic
        ? tool(config, implementation)
        : defineTool(registry, config, implementation);
      const expectedPart: ToolResponsePart = {
        toolResponse: {
          name: 'multipartTool',
          ref: 'call-1',
          output: 'hello',
          content: [{ text: 'Additional context' }],
        },
        metadata: { source: 'options' },
      };
      let calls = 0;
      const model = defineModel(
        registry,
        { name: 'testModel' },
        async (req) => {
          assert.strictEqual(req.tools?.[0].key, '/tool.v2/multipartTool');
          if (++calls === 1) {
            return {
              message: {
                role: 'model',
                content: [
                  {
                    toolRequest: {
                      name: 'multipartTool',
                      ref: 'call-1',
                      input: 'hello',
                    },
                  },
                ],
              },
            };
          }
          assert.deepStrictEqual(req.messages.at(-1)?.content, [expectedPart]);
          return { message: { role: 'model', content: [{ text: 'done' }] } };
        }
      );

      const response = await generate(registry, {
        model,
        prompt: 'hello',
        tools: [suppliedTool],
      });

      assert.strictEqual(response.text, 'done');
      assert.deepStrictEqual(
        response.messages.find((message) => message.role === 'tool')?.content,
        [expectedPart]
      );
      assert.strictEqual(calls, 2);
    });
  }
});
