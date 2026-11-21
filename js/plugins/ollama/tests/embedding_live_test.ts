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
import * as assert from 'assert';
import { genkit } from 'genkit';
import { describe, it } from 'node:test';
import { ollama } from '../src/index.js';
import type { OllamaPluginParams } from '../src/types.js'; // Adjust the import path as necessary
// Utility function to parse command-line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const serverAddress =
    args.find((arg) => arg.startsWith('--server-address='))?.split('=')[1] ||
    'http://localhost:11434';
  const modelName =
    args.find((arg) => arg.startsWith('--model-name='))?.split('=')[1] ||
    'nomic-embed-text';
  return { serverAddress, modelName };
}
const { serverAddress, modelName } = parseArgs();
describe('ollama embedders - Live Tests', () => {
  const options: OllamaPluginParams = {
    embedders: [{ name: modelName, dimensions: 768 }],
    serverAddress,
  };
  it('should successfully return embeddings', async () => {
    const ai = genkit({
      plugins: [ollama(options)],
    });
    const result = (
      await ai.embed({
        embedder: `ollama/${modelName}`,
        content: 'Hello, world!',
      })
    )[0].embedding;
    assert.strictEqual(result.length, 768);
  });
});
