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

import { googleAI } from '@genkit-ai/google-genai';
import { Document, z } from 'genkit';
import { ai } from './genkit.js';
import { augmentedPrompt } from './prompt.js';
import { vectraStore } from './vector-store.js';

// Local vectra indexes (.vectra/cat-facts, .vectra/dog-facts), see
// vector-store.ts. Swap in your vector database of choice; the flows only
// need Documents.
export const catFacts = vectraStore({
  ai,
  indexName: 'cat-facts',
  embedder: googleAI.embedder('gemini-embedding-001'),
});

export const dogFacts = vectraStore({
  ai,
  indexName: 'dog-facts',
  embedder: googleAI.embedder('gemini-embedding-001'),
});

// Define a simple RAG flow, we will evaluate this flow
// genkit flow:run askQuestionsAboutCats '"How much do cats sleep?"'
export const askQuestionsAboutCatsFlow = ai.defineFlow(
  {
    name: 'askQuestionsAboutCats',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (query) => {
    const docs = await ai.run('retrieve', () =>
      catFacts.query(query, { k: 3 })
    );
    const { text } = await augmentedPrompt({
      question: query,
      context: docs.map((d) => d.text),
    });
    return text;
  }
);

// Define a simple RAG flow, we will evaluate this flow
// genkit flow:run askQuestionsAboutDogs '"How many dog breeds are there?"'
export const askQuestionsAboutDogsFlow = ai.defineFlow(
  {
    name: 'askQuestionsAboutDogs',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (query) => {
    const docs = await ai.run('retrieve', () =>
      dogFacts.query(query, { k: 3 })
    );
    const { text } = await augmentedPrompt({
      question: query,
      context: docs.map((d) => d.text),
    });
    return text;
  }
);

// genkit flow:run indexCatFactsDocuments '["Cats sleep 12-16 hours a day."]'
export const indexCatFactsDocumentsFlow = ai.defineFlow(
  {
    name: 'indexCatFactsDocuments',
    inputSchema: z.array(z.string()),
    outputSchema: z.void(),
  },
  async (facts) => {
    await catFacts.index(
      facts.map((text) => Document.fromText(text, { type: 'animal' }))
    );
  }
);

// genkit flow:run indexDogFactsDocuments '["There are over 400 distinct dog breeds."]'
export const indexDogFactsDocumentsFlow = ai.defineFlow(
  {
    name: 'indexDogFactsDocuments',
    inputSchema: z.array(z.string()),
    outputSchema: z.void(),
  },
  async (facts) => {
    await dogFacts.index(
      facts.map((text) => Document.fromText(text, { type: 'animal' }))
    );
  }
);
