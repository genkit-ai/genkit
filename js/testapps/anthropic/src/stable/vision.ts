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

import { anthropic } from '@genkit-ai/anthropic';
import * as fs from 'fs';
import { genkit } from 'genkit';
import * as path from 'path';

const ai = genkit({
  plugins: [anthropic()],
});

/**
 * This flow demonstrates image analysis using a publicly accessible URL.
 * Claude will describe what it sees in the image.
 */
ai.defineFlow('stable-vision-url', async () => {
  // Using a Wikipedia Commons image (public domain)
  const imageUrl =
    'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg';

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          { text: 'What do you see in this image? Describe it in detail.' },
          {
            media: {
              url: imageUrl,
            },
          },
        ],
      },
    ],
  });

  return text;
});

/**
 * This flow demonstrates image analysis using a local file.
 * The image is read from disk and sent as a base64 data URL.
 */
ai.defineFlow('stable-vision-base64', async () => {
  // Read image file from the same directory as this source file
  const imagePath = path.join(__dirname, 'sample-image.png');
  const imageBuffer = fs.readFileSync(imagePath);
  const imageBase64 = imageBuffer.toString('base64');

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'Describe this image. What objects, colors, and scenes do you observe?',
          },
          {
            media: {
              url: `data:image/png;base64,${imageBase64}`,
              contentType: 'image/png',
            },
          },
        ],
      },
    ],
  });

  return text;
});

/**
 * This flow demonstrates multi-turn conversation about an image.
 * Claude can answer follow-up questions about images it has seen.
 */
ai.defineFlow('stable-vision-conversation', async () => {
  const imagePath = path.join(__dirname, 'sample-image.png');
  const imageBuffer = fs.readFileSync(imagePath);
  const imageBase64 = imageBuffer.toString('base64');

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          { text: 'What do you see in this image?' },
          {
            media: {
              url: `data:image/png;base64,${imageBase64}`,
              contentType: 'image/png',
            },
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            text: 'I see a beautiful mountain landscape with a fjord or lake, green hills, and dramatic peaks under a blue sky with wispy clouds.',
          },
        ],
      },
      {
        role: 'user',
        content: [
          {
            text: 'What time of day do you think this photo was taken, and what season might it be?',
          },
        ],
      },
    ],
  });

  return text;
});
