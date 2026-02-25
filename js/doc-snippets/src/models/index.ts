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
import { genkit } from 'genkit';

const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-2.5-flash'),
});

async function fn01() {
  // [START ex01]
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: 'Invent a menu item for a pirate themed restaurant.',
  });
  // [END ex01]
}

async function fn02() {
  // [START ex02]
  const { text } = await ai.generate({
    model: 'googleai/gemini-2.5-flash',
    prompt: 'Invent a menu item for a pirate themed restaurant.',
  });
  // [END ex02]
}

async function fn03() {
  // [START ex04]
  const { text } = await ai.generate({
    prompt: 'Invent a menu item for a pirate themed restaurant.',
    config: {
      maxOutputTokens: 400,
      stopSequences: ['<end>', '<fin>'],
      temperature: 1.2,
      topP: 0.4,
      topK: 50,
    },
  });
  // [END ex04]
}

// [START importZod]
import { z } from 'genkit'; // Import Zod, which is re-exported by Genkit.
// [END importZod]

async function fn04() {
  // [START ex05]
  const MenuItemSchema = z.object({
    name: z.string(),
    description: z.string(),
    calories: z.number(),
    allergens: z.array(z.string()),
  });

  const { output } = await ai.generate({
    prompt: 'Invent a menu item for a pirate themed restaurant.',
    output: { schema: MenuItemSchema },
  });
  // [END ex05]

  // [START ex06]
  if (output) {
    const { name, description, calories, allergens } = output;
  }
  // [END ex06]
}

function fn05() {
  // [START ex07]
  const MenuItemSchema = z.object({
    name: z.string(),
    description: z.string(),
    calories: z.coerce.number(),
    allergens: z.array(z.string()),
  });
  // [END ex07]
}

async function fn06() {
  // [START ex08]
  const { response, stream } = ai.generateStream(
    'Suggest a complete menu for a pirate themed restaurant.'
  );
  // [END ex08]

  // [START ex09]
  for await (const chunk of stream) {
    console.log(chunk.text);
  }
  // [END ex09]

  // [START ex10]
  const completeText = (await response).text;
  // [END ex10]
}

async function fn07() {
  const MenuItemSchema = z.object({
    name: z.string(),
    description: z.string(),
    calories: z.coerce.number(),
    allergens: z.array(z.string()),
  });

  // [START ex11]
  const MenuSchema = z.object({
    starters: z.array(MenuItemSchema),
    mains: z.array(MenuItemSchema),
    desserts: z.array(MenuItemSchema),
  });

  const { response, stream } = await ai.generateStream({
    prompt: 'Suggest a complete menu for a pirate themed restaurant.',
    output: { schema: MenuSchema },
  });

  for await (const chunk of stream) {
    // `output` is an object representing the entire output so far.
    console.log(chunk.output);
  }

  // Get the completed output.
  const { output } = await response;
  // [END ex11]
}

async function fn08() {
  // [START ex12]
  const { text } = await ai.generate([
    { media: { url: 'https://example.com/photo.jpg' } },
    { text: 'Compose a poem about this image.' },
  ]);
  // [END ex12]
}

// [START importReadFileAsync]
import { readFile } from 'node:fs/promises';
// [END importReadFileAsync]

async function fn09() {
  // [START ex13]
  const b64Data = await readFile('photo.jpg', { encoding: 'base64url' });
  const dataUrl = `data:image/jpeg;base64,${b64Data}`;

  const { text } = await ai.generate([
    { media: { url: dataUrl } },
    { text: 'Compose a poem about this image.' },
  ]);
  // [END ex13]
}

async function fn10() {
  // [START ex03]
  const { text } = await ai.generate({
    system: 'You are a food industry marketing consultant.',
    prompt: 'Invent a menu item for a pirate themed restaurant.',
  });
  // [END ex03]
}
