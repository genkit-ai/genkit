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

import { vertexAI } from '@genkit-ai/google-genai';
import fileTypeChecker from 'file-type-checker';
import fs from 'fs';
import { Document, z } from 'genkit';
import { chunk } from 'llm-chunk';
import path from 'path';
import { PDFDocument, PDFRawStream } from 'pdf-lib';
import pdf from 'pdf-parse';
import { vectraStore } from './vector-store.js';

import { ai } from './genkit.js';
import { augmentedMultimodalPrompt } from './prompt.js';

// Text chunks and images are embedded into the same vector space by the
// multimodal embedder, so a text query can match either.
export const pdfStore = vectraStore({
  ai,
  indexName: 'multiModalIndex',
  embedder: vertexAI.embedder('multimodalembedding@001'),
});

// Define a multimodal PDF QA flow
// (Index a PDF first)
export const multimodalPdfQAFlow = ai.defineFlow(
  {
    name: 'multimodalPdfQuestions',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (query: string, { sendChunk }) => {
    const docs = await ai.run('retrieve', () =>
      pdfStore.query(query, { k: 3 })
    );

    return augmentedMultimodalPrompt(
      {
        question: query,
        text: docs.filter((d) => d.text?.length).map((d) => d.text),
        media: docs
          .filter(
            (d) => d.media[0]?.url?.length && d.media[0]?.contentType?.length
          )
          .map((d) => {
            if (
              d.media[0].url?.startsWith('gs://') ||
              d.media[0].url?.startsWith('http')
            ) {
              return {
                gcsUrl: d.media[0]?.url,
                contentType: d.media[0]?.contentType,
              };
            }
            return {
              dataUrl: makeDataUrl(d.media[0]),
            };
          }),
      },
      {
        onChunk: (c) => sendChunk(c.text),
      }
    ).then((r) => r.text);
  }
);

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function makeDataUrl(media: unknown) {
  if (isObject(media)) {
    if (
      typeof media.contentType === 'string' &&
      media.contentType.length > 0 &&
      typeof media.url === 'string' &&
      media.url.length > 0
    ) {
      return `data:${media.contentType};base64,${media.url}`;
    } else {
      throw new Error(
        'Failed to make data URL. Invalid or missing contentType or url'
      );
    }
  }
  throw new Error(
    'Failed to make data URL. Unexpected media type: ' + typeof media
  );
}

const chunkingConfig: Parameters<typeof chunk>[1] = {
  minLength: 800, // number of minimum characters into chunk
  maxLength: 1000, // number of maximum characters into chunk
  splitter: 'sentence', // paragraph | sentence
  overlap: 100, // number of overlap chracters
  delimiters: '', // regex for base split method
};

// Define a flow to index documents into the "vector store"
// genkit flow:run indexMultimodalPdf '"./docs/BirthdayPets.pdf"'
export const indexMultimodalPdf = ai.defineFlow(
  {
    name: 'indexMultimodalPdf',
    inputSchema: z
      .string()
      .describe('PDF file path')
      .default('./docs/BirthdayPets.pdf'),
  },
  async (filePath: string) => {
    let documents: Document[] = [];
    if (filePath.startsWith('gs://') || filePath.startsWith('http')) {
      // non local file, use url for pdf file
      // e.g gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf
      documents = [Document.fromMedia(filePath, 'application/pdf')];
    } else {
      // local file (e.g. ./docs/BirthdayPets.pdf)
      // use data URLs for images
      filePath = path.resolve(filePath);
      const pdfTxt = await ai.run('extract-text', () => extractText(filePath));

      const chunks = await ai.run('chunk-it', async () =>
        chunk(pdfTxt, chunkingConfig)
      );

      const imageDocs = await ai.run('extract-images', () =>
        extractImages(filePath)
      );

      const textDocs: Document[] = chunks.map((text: string) => {
        return Document.fromText(text, { filePath });
      });
      documents = imageDocs.concat(textDocs);
    }

    await pdfStore.index(documents);
  }
);

async function extractImages(filePath: string): Promise<Document[]> {
  const imgDocs: Document[] = [];
  const pdfDoc = await PDFDocument.load(fs.readFileSync(filePath));
  const indirectObjects = pdfDoc.context.enumerateIndirectObjects();
  for (const [ref, obj] of indirectObjects) {
    if (obj instanceof PDFRawStream) {
      const detectedFileInfo = fileTypeChecker.detectFile(obj.contents);
      if (
        detectedFileInfo?.mimeType &&
        detectedFileInfo?.mimeType.startsWith('image/')
      ) {
        const base64 = Buffer.from(obj.contents).toString('base64');
        imgDocs.push(Document.fromMedia(base64, detectedFileInfo.mimeType));
      }
    }
  }
  return imgDocs;
}

async function extractText(filePath: string) {
  const dataBuffer = fs.readFileSync(filePath);
  const data = await pdf(dataBuffer);
  return data.text;
}
