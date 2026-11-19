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
import { initializeApp } from 'firebase-admin/app';
import { FieldValue, getFirestore } from 'firebase-admin/firestore';
import { readFile } from 'fs/promises';
import { Document, z } from 'genkit';
import { chunk } from 'llm-chunk';
import path from 'path';
import pdf from 'pdf-parse';
import { ai } from './genkit.js';

const app = initializeApp();
const firestore = getFirestore(app);

// There's a race condition in initializing the Firestore singleton.
// To avoid that, explicitly create an instance using the service account
// from the environment variable.
if (process.env.GCLOUD_SERVICE_ACCOUNT_CREDS) {
  console.log(`Using service account credentials.`);
  const serviceAccountCreds = JSON.parse(
    process.env.GCLOUD_SERVICE_ACCOUNT_CREDS
  );
  const authOptions = { credentials: serviceAccountCreds };
  firestore.settings(authOptions);
}

function ragTemplate({
  context,
  question,
}: {
  context: string;
  question: string;
}) {
  return `Use the following pieces of context to answer the question at the end.
 If you don't know the answer, just say that you don't know, don't try to make up an answer.

${context}
Question: ${question}
Helpful Answer:`;
}

// Firebase index config for pdfQA flow
const indexConfig = {
  collection: 'pdf-qa',
  contentField: 'facts',
  vectorField: 'embedding',
  distanceResultField: 'vector_distance',
  embedder: googleAI.embedder('gemini-embedding-001'),
};

// Define a simple RAG flow, we will evaluate this flow
export const pdfQAFirebase = ai.defineFlow(
  {
    name: 'pdfQAFirebase',
    inputSchema: z.object({
      distanceMeasure: z
        .enum(['COSINE', 'DOT_PRODUCT', 'EUCLIDEAN'])
        .default('COSINE'),
      distanceThreshold: z
        .number()
        .describe(
          'The numeric distance threshold. The significance depends on distanceMeasure'
        )
        .default(0.56)
        .optional(),
      query: z
        .string()
        .describe('Ask questions about the pdf')
        .default('Summarize the pdf'),
    }),
    outputSchema: z.object({
      documentCount: z.string(),
      distances: z.string(),
      response: z.string(),
    }),
  },
  async ({ distanceMeasure, distanceThreshold, query }) => {
    const docs = await ai.run('retrieve', async () => {
      const [{ embedding }] = await ai.embed({
        embedder: indexConfig.embedder,
        content: query,
      });
      const snapshot = await firestore
        .collection(indexConfig.collection)
        .findNearest({
          vectorField: indexConfig.vectorField,
          queryVector: embedding,
          limit: 10,
          distanceMeasure,
          distanceThreshold,
          // Firestore writes the computed distance into this field of each
          // result snapshot.
          distanceResultField: indexConfig.distanceResultField,
        })
        .get();
      return snapshot.docs.map((d) => {
        const data = d.data();
        const text: string = data[indexConfig.contentField];
        const distance: number = data[indexConfig.distanceResultField];
        return Document.fromText(text, { id: d.id, distance });
      });
    });

    const augmentedPrompt = ragTemplate({
      question: query,
      context: docs.map((d) => d.text).join('\n\n'),
    });
    const llmResponse = await ai.generate({
      model: googleAI.model('gemini-flash-latest'),
      prompt: augmentedPrompt,
    });

    const distances = docs
      .map((d) => d.metadata?.distance)
      .filter((n): n is number => typeof n === 'number');

    return {
      documentCount: `${docs.length} of 10`,
      distances: `min: ${Math.min(...distances)}, max: ${Math.max(...distances)}`,
      response: llmResponse.text,
    };
  }
);

const chunkingConfig: Parameters<typeof chunk>[1] = {
  minLength: 1000, // number of minimum characters into chunk
  maxLength: 2000, // number of maximum characters into chunk
  splitter: 'sentence', // paragraph | sentence
  overlap: 100, // number of overlap chracters
  delimiters: '', // regex for base split method
};

// Define a flow to index documents into the "vector store"
// genkit flow:run indexPdfFirestore '"./docs/flume-java.pdf"'
export const indexPdfFirebase = ai.defineFlow(
  {
    name: 'indexPdfFirestore',
    inputSchema: z
      .string()
      .describe('PDF file path')
      .default('./docs/flume-java.pdf'),
  },
  async (filePath) => {
    filePath = path.resolve(filePath);
    const pdfTxt = await ai.run('extract-text', () =>
      extractTextFromPdf(filePath)
    );

    const chunks = await ai.run('chunk-it', async () =>
      chunk(pdfTxt, chunkingConfig)
    );

    // Add chunks to the index.
    await ai.run('index-chunks', async () => indexToFirestore(chunks));
  }
);

async function indexToFirestore(data: string[]) {
  for (const text of data) {
    const embedding = (
      await ai.embed({
        embedder: indexConfig.embedder,
        content: text,
      })
    )[0].embedding;
    await firestore.collection(indexConfig.collection).add({
      [indexConfig.vectorField]: FieldValue.vector(embedding),
      [indexConfig.contentField]: text,
    });
  }
}

async function extractTextFromPdf(filePath: string) {
  const pdfFile = path.resolve(filePath);
  const dataBuffer = await readFile(pdfFile);
  const data = await pdf(dataBuffer);
  return data.text;
}
