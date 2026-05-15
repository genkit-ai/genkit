/**
 * Copyright 2025 Google LLC
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

import { GoogleGenAI } from '@google/genai';

const BLOB = new Blob(
  [
    'The Whispering Woods In the heart of Eldergrove, there stood a forest whispered about by the villagers. They spoke of trees that could talk and streams that sang. Young Elara, curious and adventurous, decided to explore the woods one crisp autumn morning. As she wandered deeper, the leaves rustled with excitement, revealing hidden paths. Elara noticed the trees bending slightly as if beckoning her to come closer. When she paused to listen, she heard soft murmurs—stories of lost treasures and forgotten dreams. Drawn by the enchanting sounds, she followed a narrow trail until she stumbled upon a shimmering pond. At its edge, a wise old willow tree spoke, “Child of the village, what do you seek?” “I seek adventure,” Elara replied, her heart racing. “Adventure lies not in faraway lands but within your spirit,” the willow said, swaying gently. “Every choice you make is a step into the unknown.” With newfound courage, Elara left the woods, her mind buzzing with possibilities. The villagers would say the woods were magical, but to Elara, it was the spark of her imagination that had transformed her ordinary world into a realm of endless adventures. She smiled, knowing her journey was just beginning',
  ],
  { type: 'text/plain' }
);

async function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function createFileSearchStore(): Promise<string> {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  const fileSearchStore = await ai.fileSearchStores.create({});
  console.log('create file search store: ', fileSearchStore);
  return fileSearchStore.name!;
}

export async function deleteFileSearchStore(fileSearchStoreName: string) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  await ai.fileSearchStores.delete({
    name: fileSearchStoreName,
    config: { force: true },
  });
}

export async function uploadBlobToFileSearchStore(fileSearchStoreName: string) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  let op = await ai.fileSearchStores.uploadToFileSearchStore({
    fileSearchStoreName,
    file: BLOB,
    config: {
      customMetadata: [{ key: 'author', stringValue: 'foo' }],
    },
  });
  while (!op.done) {
    await delay(5000);
    op = await ai.operations.get({ operation: op });
    console.log('operation: ', op);
  }
}
