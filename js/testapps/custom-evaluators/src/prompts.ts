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

import { Dotprompt, type PromptFunction } from 'dotprompt';
import { readFileSync } from 'fs';
import path from 'path';

// The plugin renders its own judge prompts rather than using the app's
// `ai.prompt()`, so it doesn't depend on the app's Genkit instance.
const dp = new Dotprompt();
const compiled = new Map<string, Promise<PromptFunction>>();

/** Renders `prompts/<name>.prompt` to plain text. */
export async function renderPrompt(
  name: string,
  input: Record<string, unknown>
): Promise<string> {
  let prompt = compiled.get(name);
  if (!prompt) {
    const source = readFileSync(
      path.join(__dirname, '..', 'prompts', `${name}.prompt`),
      'utf8'
    );
    prompt = dp.compile(dp.parse(source));
    compiled.set(name, prompt);
  }
  const rendered = await (await prompt)({ input });
  return rendered.messages
    .map((m) => m.content.map((c) => c.text).join())
    .join();
}
