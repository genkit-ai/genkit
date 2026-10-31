/**
 * Copyright 2026 Google LLC
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

import { openAI } from '@genkit-ai/compat-oai/openai';
import { genkit } from 'genkit';

/** Compile-only consumer examples; this function is never invoked. */
export function checkImageReferences() {
  const ai = genkit({});
  const current = openAI.model('gpt-image-2').withConfig({
    size: '1024x1024',
    quality: 'high',
  });
  void ai.generate({
    model: current,
    prompt: 'a cat',
    config: { background: 'transparent', output_format: 'webp' },
  });
  for (const name of [
    'gpt-image-1.5',
    'gpt-image-2',
    'gpt-image-2.5-sunburst',
    'gpt-image-2.5-flare',
    'chatgpt-image-latest',
  ] as const) {
    const model = openAI
      .model(name)
      .withConfig({ size: '1024x1024', quality: 'high' });
    void ai.generate({
      model,
      prompt: 'a cat',
      config: { background: 'transparent', output_format: 'webp' },
    });
  }
  const model = openAI.model('gpt-image-2');
  // @ts-expect-error GPT images do not use DALL-E quality names.
  model.withConfig({ quality: 'hd', n: 1 });
  // @ts-expect-error GPT images always return base64; response_format is unsupported.
  model.withConfig({ response_format: 'b64_json', n: 1 });
  openAI.model('dall-e-3').withConfig({ quality: 'hd', style: 'vivid', n: 1 });
  // Editor completion cursor; this compile-only function is never invoked.
  openAI.model('');
}
