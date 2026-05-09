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

import { anthropic, anthropicDocument } from '@genkit-ai/anthropic';
import { genkit } from 'genkit';

const ai = genkit({
  plugins: [
    anthropic({
      apiVersion: 'beta',
      apiKey: process.env.ANTHROPIC_API_KEY,
    }),
  ],
});

/**
 * This flow demonstrates citations with a plain text document.
 * The response will include citations pointing back to the source text.
 */
ai.defineFlow('citations-text', async () => {
  const response = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          anthropicDocument({
            source: {
              type: 'text',
              data: 'The grass is green. The sky is blue. Water is wet. Fire is hot.',
            },
            title: 'Basic Facts',
            citations: { enabled: true },
          }),
          {
            text: 'What color is the grass and sky? Please cite your sources.',
          },
        ],
      },
    ],
  });

  // Log the response with citations
  console.log('Response text:', response.text);
  console.log(
    'Response content:',
    JSON.stringify(response.message?.content, null, 2)
  );

  // Extract citations from the response
  const citations =
    response.message?.content?.flatMap(
      (part) => part.metadata?.citations || []
    ) ?? [];

  console.log('Citations:', JSON.stringify(citations, null, 2));

  return {
    text: response.text,
    citations,
  };
});

/**
 * This flow demonstrates citations with custom content blocks
 * for more granular citation control.
 */
ai.defineFlow('citations-custom-content', async () => {
  const response = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          anthropicDocument({
            source: {
              type: 'content',
              content: [
                { type: 'text', text: 'Fact 1: Dogs are mammals.' },
                { type: 'text', text: 'Fact 2: Cats are also mammals.' },
                { type: 'text', text: 'Fact 3: Birds have feathers.' },
              ],
            },
            title: 'Animal Facts',
            citations: { enabled: true },
          }),
          {
            text: 'What do dogs and cats have in common? Cite your source.',
          },
        ],
      },
    ],
  });

  console.log('Response text:', response.text);
  console.log(
    'Response content:',
    JSON.stringify(response.message?.content, null, 2)
  );

  return response.text;
});
