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

//  Sample app for using the proposed Vertex AI plugin retriever and indexer with a local file (just as a demo).

// important imports for this sample:
import { vertexAI } from '@genkit-ai/vertexai';
import {
  vertexAIRerankers,
  vertexRerankers,
} from '@genkit-ai/vertexai/rerankers';
import { Document, RankedDocument, genkit, z } from 'genkit';
import { LOCATION, PROJECT_ID } from './config';

// Configure Genkit with Vertex AI plugin
const ai = genkit({
  plugins: [
    vertexAI({
      projectId: PROJECT_ID,
      location: LOCATION,
      googleAuth: {
        scopes: ['https://www.googleapis.com/auth/cloud-platform'],
      },
    }),
    vertexAIRerankers({
      projectId: PROJECT_ID,
      location: LOCATION,
      rerankers: ['semantic-ranker-default@latest'],
    }),
    vertexRerankers(),
  ],
});

const FAKE_DOCUMENT_CONTENT = [
  'pythagorean theorem',
  'e=mc^2',
  'pi',
  'dinosaurs',
  "euler's identity",
  'prime numbers',
  'fourier transform',
  'ABC conjecture',
  'riemann hypothesis',
  'triangles',
  "schrodinger's cat",
  'quantum mechanics',
  'the avengers',
  "harry potter and the philosopher's stone",
  'movies',
];

export const legacyRerankFlow = ai.defineFlow(
  {
    name: 'legacyRerankFlow',
    inputSchema: z.object({ query: z.string().default('geometry') }),
    outputSchema: z.array(
      z.object({
        text: z.string(),
        score: z.number(),
      })
    ),
  },
  async ({ query }) => {
    const documents = FAKE_DOCUMENT_CONTENT.map((text) =>
      Document.fromText(text)
    );

    const rerankedDocuments = await ai.rerank({
      reranker: 'vertexai/semantic-ranker-default@latest',
      query: Document.fromText(query),
      documents,
    });

    return rerankedDocuments.map((doc) => ({
      text: doc.text,
      score: doc.metadata.score,
    }));
  }
);

export const v2RerankFlow = ai.defineFlow(
  {
    name: 'v2RerankFlow',
    inputSchema: z.object({ query: z.string().default('geometry') }),
    outputSchema: z.array(
      z.object({
        text: z.string(),
        score: z.number(),
      })
    ),
  },
  async ({ query }) => {
    const documents = FAKE_DOCUMENT_CONTENT.map((text) =>
      Document.fromText(text)
    );
    const response = await ai.rerank({
      reranker: vertexRerankers.reranker('semantic-ranker-fast-004'),
      documents,
      query,
      options: {
        topN: 3,
        ignoreRecordDetailsInResponse: true,
      },
    });
    return response.map((doc: RankedDocument) => ({
      text: doc.text,
      score: doc.metadata.score,
    }));
  }
);
