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
import {
  claude35Sonnet,
  claude35SonnetV2,
  vertexAIModelGarden,
} from '@genkit-ai/vertexai/modelgarden';
import { genkit, z, type Flow, type GenerateOptions } from 'genkit';
import { logger } from 'genkit/logging';

logger.setLogLevel('debug');

const ai = genkit({
  plugins: [
    googleAI(),
    vertexAI({
      location: 'us-east5',
    }),
    vertexAIModelGarden({
      location: 'us-east5',
      models: [claude35Sonnet, claude35SonnetV2],
    }),
  ],
});

function formatFlow(name: string, options: GenerateOptions) {
  return ai.defineFlow(
    {
      name,
      inputSchema: z.string(),
    },
    async (model) => {
      console.log('\n===', name, 'with model', model);
      options.model = model;
      if (model.includes('gemini')) {
        options.config = {
          safetySettings: [
            {
              category: 'HARM_CATEGORY_HATE_SPEECH',
              threshold: 'BLOCK_NONE',
            },
            {
              category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
              threshold: 'BLOCK_NONE',
            },
            {
              category: 'HARM_CATEGORY_HARASSMENT',
              threshold: 'BLOCK_NONE',
            },
            {
              category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
              threshold: 'BLOCK_NONE',
            },
          ],
        };
      }
      const { stream, response } = await ai.generateStream(options);
      for await (const chunk of stream) {
        console.log('text:', chunk.text);
        console.log('output:', chunk.output);
      }
      console.log();
      console.log('final output:', (await response).output);
    }
  );
}

const prompts: Record<string, GenerateOptions> = {
  text: {
    prompt: 'tell me a short story about pirates',
    output: { format: 'text' },
  },
  json: {
    prompt: 'generate a creature for an RPG game',
    output: {
      format: 'json',
      schema: z.object({
        name: z.string().describe('the name of the creature'),
        backstory: z
          .string()
          .describe('a one-paragraph backstory for the creature'),
        hitPoints: z.number(),
        attacks: z.array(z.string()).describe('named attacks'),
      }),
    },
  },
  array: {
    prompt: 'generate a list of characters from Futurama',
    output: {
      // @ts-ignore
      format: 'array',
      schema: z.array(
        z.object({
          name: z.string(),
          description: z.string(),
          friends: z.array(z.string()),
          enemies: z.array(z.string()),
        })
      ),
    },
  },
  jsonl: {
    prompt: 'generate fake products for an online pet store',
    output: {
      // @ts-ignore
      format: 'array',
      schema: z.array(
        z.object({
          name: z.string(),
          description: z.string(),
          price: z.number(),
          stock: z.number(),
          color: z.string(),
          tags: z.array(z.string()),
        })
      ),
    },
  },
  enum: {
    prompt: 'how risky is skydiving?',
    output: {
      // @ts-ignore
      format: 'enum',
      schema: z.enum(['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']),
    },
  },
};

const flows: Flow<z.ZodString, z.ZodTypeAny>[] = [];
for (const format in prompts) {
  flows.push(formatFlow(format, prompts[format]));
}

let models = process.argv.slice(2);
if (!models.length) {
  models = [
    'vertexai/gemini-2.5-pro',
    'vertexai/gemini-2.5-flash',
    'googleai/gemini-2.5-pro',
    'googleai/gemini-2.5-flash',
  ];
}

async function main() {
  const fails: { model: string; flow: string; error: string }[] = [];
  for (const model of models) {
    for (const flow of flows) {
      try {
        await flow(model);
      } catch (e: any) {
        console.error('ERROR:', e.stack);
        fails.push({ model, flow: flow.__action.name, error: e.message });
      }
    }
  }

  console.log('!!!', fails.length, 'errors');
  for (const fail of fails) {
    console.log(`${fail.model}: ${fail.flow}: ${fail.error}`);
  }
}
main();
