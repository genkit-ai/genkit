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
 * This flow demonstrates WEBP image handling with matching contentType.
 * Both the data URL and the contentType field specify image/webp.
 */
ai.defineFlow('stable-webp-matching', async () => {
  // Minimal valid WEBP image (1x1 pixel, transparent)
  // In a real app, you'd load an actual WEBP image file
  const webpImageData =
    'data:image/webp;base64,UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAwA0JaQAA3AA/vuUAAA=';

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          { text: 'Describe this image:' },
          {
            media: {
              url: webpImageData,
              contentType: 'image/webp',
            },
          },
        ],
      },
    ],
  });

  return text;
});

/**
 * This flow demonstrates the fix for WEBP images with mismatched contentType.
 * Even if contentType says 'image/png', the plugin will use 'image/webp' from
 * the data URL, preventing API validation errors.
 *
 * This fix ensures that the media_type sent to Anthropic matches the actual
 * image data, which is critical for WEBP images that were previously causing
 * "Image does not match the provided media type" errors.
 */
ai.defineFlow('stable-webp-mismatched', async () => {
  // Minimal valid WEBP image (1x1 pixel, transparent)
  const webpImageData =
    'data:image/webp;base64,UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAwA0JaQAA3AA/vuUAAA=';

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'Describe this image (note: contentType is wrong but data URL is correct):',
          },
          {
            media: {
              // Data URL says WEBP, but contentType says PNG
              // The plugin will use WEBP from the data URL (the fix)
              url: webpImageData,
              contentType: 'image/png', // This mismatch is handled correctly
            },
          },
        ],
      },
    ],
  });

  return {
    result: text,
    note: 'The plugin correctly used image/webp from the data URL, not image/png from contentType',
  };
});
