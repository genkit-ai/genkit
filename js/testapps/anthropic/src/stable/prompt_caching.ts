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

import { anthropic, cacheControl } from '@genkit-ai/anthropic';
import { promises as fs } from 'fs';
import { genkit } from 'genkit';
import path from 'path';

const ai = genkit({
  plugins: [
    // Configure the plugin with environment-driven API key
    anthropic(),
  ],
});

const longTextPath = path.join(__dirname, '../long-text.txt');

ai.defineFlow('caching system prompt', async () => {
  const longTextBuffer = await fs.readFile(longTextPath);
  const longText = longTextBuffer.toString('utf-8');

  const response = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    system: {
      text: `You are a friendly Claude assistant. Greet the user briefly. You will be given a long text to read and answer questions about it.
      ${longText}`,
      metadata: { ...cacheControl({ ttl: '5m' }) },
    },
    messages: [
      {
        role: 'user',
        content: [{ text: 'What is the main idea of the text?' }],
      },
    ],
  });

  return {
    text: response.text,
    usage: response.usage,
  };
});

ai.defineFlow('caching user prompt', async () => {
  const longTextBuffer = await fs.readFile(longTextPath);
  const longText = longTextBuffer.toString('utf-8');

  const response = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    system: {
      text: 'You are a friendly Claude assistant. Greet the user briefly. You will be given a long text to read and answer questions about it.',
    },
    messages: [
      {
        role: 'user',
        content: [
          {
            text: longText,
            metadata: { ...cacheControl() }, // uses default ephemeral type
          },
        ],
      },
    ],
  });

  return {
    text: response.text,
    usage: response.usage,
  };
});

ai.defineFlow('caching image prompt', async () => {
  const imagePath = path.join(__dirname, 'sample-image.png');
  const imageBuffer = await fs.readFile(imagePath);
  const imageBase64 = imageBuffer.toString('base64');

  const longTextBuffer = await fs.readFile(longTextPath);
  const longText = longTextBuffer.toString('utf-8');

  const response = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    system: {
      text: 'Does the following text match the image?',
    },
    messages: [
      {
        role: 'user',
        content: [
          {
            text: longText,
          },
          {
            media: {
              url: `data:image/png;base64,${imageBase64}`,
              contentType: 'image/png',
            },
            metadata: {
              cache_control: {
                type: 'ephemeral',
                ttl: '5m',
              },
            },
          },
        ],
      },
    ],
  });

  return {
    text: response.text,
    usage: response.usage,
  };
});

ai.defineFlow('caching pdf prompt', async () => {
  const pdfPath = path.join(__dirname, '../attention-first-page.pdf');
  const pdfBuffer = await fs.readFile(pdfPath);
  const pdfBase64 = pdfBuffer.toString('base64');

  const response = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    system: {
      text: 'You are a Claude assistant. Analyze the following PDF document and describe what you see briefly.',
    },
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'Are the contents of these PDF documents the same?',
          },
          {
            media: {
              url: `data:application/pdf;base64,${pdfBase64}`,
              contentType: 'application/pdf',
            },
            metadata: {
              cache_control: {
                type: 'ephemeral',
                ttl: '5m',
              },
            },
          },
        ],
      },
    ],
  });

  return {
    text: response.text,
    usage: response.usage,
  };
});

ai.defineFlow('caching with tool call', async () => {
  const longTextBuffer = await fs.readFile(longTextPath);
  const longText = longTextBuffer.toString('utf-8');

  const response = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    system: {
      text: `You are a friendly Claude assistant. Greet the user briefly. You will be given a long text to read and answer questions about it.
      ${longText}`,
      metadata: {
        cache_control: {
          type: 'ephemeral',
          ttl: '5m',
        },
      },
    },
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'Search the web for the definition of the word longest word in the text that begins with the letter "P"',
          },
        ],
      },
    ],
    config: {
      tools: [
        {
          type: 'web_search_20250305',
          name: 'web_search',
          max_uses: 5,
        },
      ],
    },
  });

  return {
    text: response.text,
    usage: response.usage,
  };
});
