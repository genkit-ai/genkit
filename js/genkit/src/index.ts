/**
 * @license
 *
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

/**
 * Main genkit import.
 *
 * @example
 * ```ts
 * import { genkit, z } from 'genkit';
 * import { googleAI } from '@genkit-ai/google-genai';
 *
 * const ai = genkit({
 *   plugins: [googleAI()],
 * });
 *
 * export const myFlow = ai.defineFlow(
 *   {
 *     name: 'myFlow',
 *     inputSchema: z.object({ subject: z.string() }),
 *     outputSchema: z.string(),
 *   },
 *   async ({ subject }) => {
 *     const { text } = await ai.generate({
 *       model: googleAI.model('gemini-2.5-flash'),
 *       prompt: `Say hello to ${subject}`,
 *     });
 *     return text;
 *   }
 * );
 * ```
 *
 * @module /
 */
export * from './common.js';
export { Genkit, genkit, type GenkitOptions } from './genkit.js';
