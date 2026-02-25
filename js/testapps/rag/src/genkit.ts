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

import { devLocalVectorstore } from '@genkit-ai/dev-local-vectorstore';
import { GenkitMetric, genkitEval } from '@genkit-ai/evaluator';
import { googleAI } from '@genkit-ai/google-genai';
import {
  claude3Sonnet,
  llama31,
  vertexAIModelGarden,
} from '@genkit-ai/vertexai/modelgarden';
import { genkit } from 'genkit';
import { chroma } from 'genkitx-chromadb';
import { pinecone } from 'genkitx-pinecone';
import { GoogleAuth, type IdTokenClient } from 'google-auth-library';

const auth = new GoogleAuth();
let authClient: IdTokenClient | undefined = undefined;

/** Helper method to cache {@link IdTokenClient} instance */
async function getCloudRunAuthClient(aud: string) {
  if (!authClient) {
    authClient = await auth.getIdTokenClient(aud);
  }
  return authClient;
}

export const ai = genkit({
  plugins: [
    googleAI(),
    vertexAIModelGarden({
      location: 'us-central1',
      models: [claude3Sonnet, llama31],
    }),
    pinecone([
      {
        indexId: 'cat-facts',
        embedder: googleAI.embedder('gemini-embedding-001'),
      },
      {
        indexId: 'pdf-chat',
        embedder: googleAI.embedder('gemini-embedding-001'),
      },
    ]),
    chroma([
      {
        collectionName: 'dogfacts_collection',
        embedder: googleAI.embedder('gemini-embedding-001'),
        createCollectionIfMissing: true,
        clientParams: async () => {
          // Replace this with your Cloud Run Instance URL
          const host = 'https://<my-cloud-run-url>.run.app';
          const client = await getCloudRunAuthClient(host);
          const idToken = await client.idTokenProvider.fetchIdToken(host);
          return {
            path: host,
            fetchOptions: {
              headers: {
                Authorization: 'Bearer ' + idToken,
              },
            },
          };
        },
      },
    ]),
    devLocalVectorstore([
      {
        indexName: 'dog-facts',
        embedder: googleAI.embedder('gemini-embedding-001'),
      },
      {
        indexName: 'pdfQA',
        embedder: googleAI.embedder('gemini-embedding-001'),
      },
    ]),
    genkitEval({
      judge: googleAI.model('gemini-2.5-flash'),
      judgeConfig: {
        safetySettings: [
          {
            category: 'HARM_CATEGORY_HATE_SPEECH',
            threshold: 'BLOCK_NONE',
          },
          {
            category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
            threshold: 'BLOCK_NONE',
          },
          {
            category: 'HARM_CATEGORY_HARASSMENT',
            threshold: 'BLOCK_NONE',
          },
          {
            category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
            threshold: 'BLOCK_NONE',
          },
        ],
      } as any,
      metrics: [GenkitMetric.FAITHFULNESS, GenkitMetric.MALICIOUSNESS],
    }),
  ],
  model: googleAI.model('gemini-2.5-flash'),
});
