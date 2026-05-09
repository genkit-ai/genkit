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

import { googleAI } from '@genkit-ai/google-genai';
import { genkit, z } from 'genkit';

const ai = genkit({
  plugins: [googleAI()],
});

/*
title: string, recipe title
    ingredients(array): 
      name: string
      quantity: string
    steps(array, the steps required to complete the recipe): string
    */
const RecipeSchema = ai.defineSchema(
  'Recipe',
  z.object({
    title: z.string().describe('recipe title'),
    ingredients: z.array(z.object({ name: z.string(), quantity: z.string() })),
    steps: z
      .array(z.string())
      .describe('the steps required to complete the recipe'),
  })
);

// This example demonstrates using prompt files in a flow
// Load the prompt file during initialization.
// If it fails, due to the prompt file being invalid, the process will crash,
// instead of us getting a more mysterious failure later when the flow runs.

ai.defineHelper('list', (data: any) => {
  if (!Array.isArray(data)) {
    return '';
  }
  return data.map((item) => `- ${item}`).join('\n');
});

ai.defineFlow(
  {
    name: 'chefFlow',
    inputSchema: z.object({
      food: z.string(),
    }),
    outputSchema: RecipeSchema,
  },
  async (input) =>
    (await ai.prompt<any, typeof RecipeSchema>('recipe')(input)).output!
);

ai.defineFlow(
  {
    name: 'robotChefFlow',
    inputSchema: z.object({
      food: z.string(),
    }),
    outputSchema: z.any(),
  },
  async (input) =>
    (await ai.prompt('recipe', { variant: 'robot' })(input)).output
);

// A variation that supports streaming, optionally

ai.defineFlow(
  {
    name: 'tellStory',
    inputSchema: z.object({
      subject: z.string(),
      personality: z.string().optional(),
    }),
    outputSchema: z.string(),
    streamSchema: z.string(),
  },
  async ({ subject, personality }, { sendChunk }) => {
    const storyPrompt = ai.prompt('story');
    const { response, stream } = storyPrompt.stream({
      subject,
      personality,
    });
    for await (const chunk of stream) {
      sendChunk(chunk.content[0]?.text!);
    }
    return (await response).text;
  }
);
