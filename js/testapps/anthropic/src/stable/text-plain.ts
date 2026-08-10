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
import { genkit } from 'genkit';

const ai = genkit({
  plugins: [anthropic()],
});

/**
 * This flow demonstrates the error that occurs when trying to use text/plain
 * files as media. The plugin will throw a helpful error message guiding users
 * to use text content instead.
 *
 * Error message: "Unsupported media type: text/plain. Text files should be sent
 * as text content in the message, not as media. For example, use { text: '...' }
 * instead of { media: { url: '...', contentType: 'text/plain' } }"
 */
ai.defineFlow('stable-text-plain-error', async () => {
  try {
    await ai.generate({
      model: anthropic.model('claude-sonnet-4-5'),
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                url: 'data:text/plain;base64,SGVsbG8gV29ybGQ=',
                contentType: 'text/plain',
              },
            },
          ],
        },
      ],
    });
    return 'Unexpected: Should have thrown an error';
  } catch (error: any) {
    return {
      error: error.message,
      note: 'This demonstrates the helpful error message for text/plain files',
    };
  }
});

/**
 * This flow demonstrates the correct way to send text content.
 * Instead of using media with text/plain, use the text field directly.
 */
ai.defineFlow('stable-text-plain-correct', async () => {
  // Read the text content (in a real app, you'd read from a file)
  const textContent = 'Hello World\n\nThis is a text file content.';

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          {
            text: `Please summarize this text file content:\n\n${textContent}`,
          },
        ],
      },
    ],
  });

  return text;
});
