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

import {
  devLocalIndexerRef,
  devLocalRetrieverRef,
} from '@genkit-ai/dev-local-vectorstore';
import { googleAI } from '@genkit-ai/google-genai';
import fs from 'fs';
import { Document, z } from 'genkit';
import { chunk } from 'llm-chunk';
import path from 'path';
import pdf from 'pdf-parse';
import { ai } from './genkit.js';

export const pdfChatRetriever = devLocalRetrieverRef('pdfQA');

export const pdfChatIndexer = devLocalIndexerRef('pdfQA');

// Define a simple RAG flow, we will evaluate this flow
export const pdfQA = ai.defineFlow(
  {
    name: 'pdfQA',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (query, { sendChunk }) => {
    const docs = await ai.retrieve({
      retriever: pdfChatRetriever,
      query,
      options: { k: 3 },
    });

    const { text } = await ai.generate({
      prompt: query,
      docs,
      onChunk: (c) => sendChunk(c.text),
    });
    return text;
  }
);

const chunkingConfig = {
  minLength: 1000, // number of minimum characters into chunk
  maxLength: 2000, // number of maximum characters into chunk
  splitter: 'sentence', // paragraph | sentence
  overlap: 100, // number of overlap chracters
  delimiters: '', // regex for base split method
} as any;

// Define a flow to index documents into the "vector store"
// genkit flow:run indexPdf '"./docs/flume-java.pdf"'
export const indexPdf = ai.defineFlow(
  {
    name: 'indexPdf',
    inputSchema: z.string().describe('PDF file path'),
  },
  async (filePath) => {
    filePath = path.resolve(filePath);
    const pdfTxt = await ai.run('extract-text', () => extractText(filePath));

    const chunks = await ai.run('chunk-it', async () =>
      chunk(pdfTxt, chunkingConfig)
    );

    const documents: Document[] = chunks.map((text) => {
      return Document.fromText(text, { filePath });
    });

    await ai.index({
      indexer: pdfChatIndexer,
      documents,
    });
  }
);

async function extractText(filePath: string) {
  const pdfFile = path.resolve(filePath);
  const dataBuffer = fs.readFileSync(pdfFile);
  const data = await pdf(dataBuffer);
  return data.text;
}

// genkit flow:run synthesizeQuestions '"./docs/flume-java.pdf"' --output synthesizedQuestions.json
// genkit flow:batchRun pdfQA synthesizedQuestions.json --output batchinput_small_out.json
export const synthesizeQuestions = ai.defineFlow(
  {
    name: 'synthesizeQuestions',
    inputSchema: z.string().describe('PDF file path'),
    outputSchema: z.array(z.string()),
  },
  async (filePath) => {
    filePath = path.resolve(filePath);
    const pdfTxt = await ai.run('extract-text', () => extractText(filePath));

    const chunks = await ai.run('chunk-it', async () =>
      chunk(pdfTxt, chunkingConfig)
    );

    const questions: string[] = [];
    for (let i = 0; i < chunks.length; i++) {
      const qResponse = await ai.generate({
        model: googleAI.model('gemini-flash-latest'),
        prompt: {
          text: `Generate one question about the text below: ${chunks[i]}`,
        },
      });
      questions.push(qResponse.text);
    }
    return questions;
  }
);
