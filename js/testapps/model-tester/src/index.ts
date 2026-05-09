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

import { googleAI, vertexAI } from '@genkit-ai/google-genai';
import { vertexModelGarden } from '@genkit-ai/vertexai/modelgarden';
import * as clc from 'colorette';
import { genkit } from 'genkit';
import { testModels } from 'genkit/testing';
import { ollama } from 'genkitx-ollama';
//import { openAI } from 'genkitx-openai';

export const ai = genkit({
  plugins: [
    googleAI(),
    vertexAI({
      location: 'us-central1',
    }),
    vertexModelGarden({
      location: 'us-central1',
    }),
    ollama({
      models: [
        {
          name: 'gemma2',
          type: 'chat',
        },
      ],
      serverAddress: 'http://127.0.0.1:11434', // default local address
    }),
    //openAI(),
  ],
});

testModels(ai.registry, [
  'googleai/gemini-2.5-pro',
  'googleai/gemini-2.5-flash',
  'vertexai/gemini-2.5-pro',
  'vertexai/gemini-2.5-flash',
  'vertex-model-garden/claude-sonnet-4@20250514',
  'vertex-model-garden/meta/llama-4-maverick-17b-128e-instruct-maas',
  'ollama/gemma2',
  // 'openai/gpt-4o',
  // 'openai/gpt-4o-mini',
]).then((r) => {
  let failed = false;
  for (const test of r) {
    console.log(`${clc.bold('Test:')} ${test.description}`);
    for (const model of test.models) {
      if (model.passed) {
        console.log(
          `  - ${clc.bold(clc.green('PASSED'))}: ${clc.bold(model.name)}`
        );
      } else if (model.skipped) {
        console.log(
          `  - ${clc.bold(clc.yellow('SKIPPED'))}: ${clc.bold(model.name)}`
        );
      } else {
        failed = true;
        console.log(
          `  - ${clc.bold(clc.red('FAILED'))}: ${clc.bold(model.name)}`
        );
        if (model.error?.message) {
          console.log(prefixText(model.error?.message, '    '));
        }
        if (model.error?.stack) {
          console.log(prefixText(model.error?.stack, '    '));
        }
      }
    }
  }
  if (failed) {
    process.exitCode = 1;
  }
});

function prefixText(txt: string, prefix: string) {
  return txt
    .split('\n')
    .map((t) => prefix + t)
    .join('\n');
}
