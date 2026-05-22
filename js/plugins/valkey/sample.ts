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

/**
 * sample.ts — Full RAG demo: Valkey vector search + Ollama generation via Genkit.
 *
 * Prerequisites:
 *   1. Valkey 8+ with valkey-search module:
 *      docker run -d --name valkey -p 6379:6379 valkey/valkey:8-alpine
 *   2. Ollama running locally with models pulled:
 *      ollama pull nomic-embed-text
 *      ollama pull gemma4:e2b
 *   3. From this directory: pnpm install
 *
 * Run:
 *   npx tsx sample.ts
 */

import { Document, genkit } from 'genkit';
import { ollama } from 'genkitx-ollama';
import { valkeyIndexerRef, valkeyPlugin, valkeyRetrieverRef } from './src';

const INDEX_NAME = 'coffee-menu';
const DIMENSION = 768; // nomic-embed-text output dimension

const valkey = valkeyPlugin([
  {
    indexName: INDEX_NAME,
    embedder: ollama.embedder('nomic-embed-text'),
    dimension: DIMENSION,
    clientConfig: {
      addresses: [{ host: process.env.VALKEY_HOST ?? 'localhost', port: 6379 }],
    },
  },
]);

const ai = genkit({
  plugins: [
    ollama({
      serverAddress: process.env.OLLAMA_HOST ?? 'http://localhost:11434',
      models: [{ name: 'gemma4:e2b' }],
      embedders: [{ name: 'nomic-embed-text', dimensions: DIMENSION }],
    }),
    valkey.plugin,
  ],
});

async function main() {
  // --- Indexing flow ---
  const indexer = valkeyIndexerRef({ indexName: INDEX_NAME });

  const documents = [
    Document.fromText(
      'Espresso: a concentrated coffee brewed by forcing hot water through finely-ground beans. $3.50',
      { category: 'drinks' }
    ),
    Document.fromText(
      'Latte: espresso with steamed milk and a thin layer of foam. $4.75',
      { category: 'drinks' }
    ),
    Document.fromText(
      'Cold Brew: coffee steeped in cold water for 12-24 hours, served chilled. $4.25',
      { category: 'drinks' }
    ),
    Document.fromText(
      'Croissant: a buttery, flaky pastry of French origin. $3.00',
      { category: 'food' }
    ),
  ];

  await ai.index({ indexer, documents });
  console.log(`Indexed ${documents.length} documents into "${INDEX_NAME}".`);

  // --- Retrieval flow ---
  const retriever = valkeyRetrieverRef({ indexName: INDEX_NAME });
  const query = 'What cold coffee drinks do you have?';

  const results = await ai.retrieve({
    retriever,
    query,
    options: { k: 2 },
  });

  console.log(`\nRetrieved ${results.length} documents for: "${query}"\n`);
  for (const doc of results) {
    console.log(`  - ${doc.text}`);
  }

  // --- RAG generation with gemma4:e2b ---
  const { text } = await ai.generate({
    model: ollama.model('gemma4:e2b'),
    prompt: `Answer the customer's question concisely: ${query}`,
    docs: results,
  });

  console.log(`\n--- Generated answer ---\n${text}\n`);
}

main()
  .catch((err) => {
    console.error(err);
    process.exit(1);
  })
  .finally(() => valkey.close());
