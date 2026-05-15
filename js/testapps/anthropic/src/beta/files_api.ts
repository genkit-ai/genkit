/**
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

import { anthropic } from '@genkit-ai/anthropic';
import * as fs from 'fs';
import { genkit } from 'genkit';
import * as path from 'path';

// Ensure the API key is set.
const API_KEY = process.env.ANTHROPIC_API_KEY;
// If you have a file ID, you can set it here. Otherwise, the flow will upload a new PDF to Anthropic.
const FILE_ID = process.env.ANTHROPIC_FILE_ID;

/**
 * Uploads a PDF file to Anthropic's Files API.
 * @returns The file object containing `id` field.
 */
export async function uploadPdfToAnthropic() {
  if (!API_KEY) throw new Error('Missing ANTHROPIC_API_KEY env variable');

  // Path to the PDF file to upload
  const pdfPath = path.join(__dirname, '../attention-first-page.pdf');
  const fileBuffer = fs.readFileSync(pdfPath);

  const form = new FormData();
  form.append(
    'file',
    new Blob([new Uint8Array(fileBuffer)], { type: 'application/pdf' }),
    'attention-first-page.pdf'
  );

  const response = await fetch('https://api.anthropic.com/v1/files', {
    method: 'POST',
    headers: {
      'x-api-key': API_KEY,
      'anthropic-version': '2023-06-01',
      'anthropic-beta': 'files-api-2025-04-14',
    },
    body: form,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Anthropic file upload failed: ${response.status} ${text}`);
  }
  const result = await response.json();
  return result as { id: string };
}

const ai = genkit({
  plugins: [
    // Default all flows in this sample to the beta surface
    anthropic({
      apiVersion: 'beta',
      apiKey: API_KEY,
    }),
  ],
});

/**
 * This flow demonstrates PDF document processing via the Anthropic Files API.
 * The PDF is uploaded to Anthropic and referenced by file ID.
 */
ai.defineFlow('beta-pdf-url', async () => {
  let fileId = FILE_ID;

  if (!fileId) {
    const fileResult = await uploadPdfToAnthropic();
    if (!fileResult || !fileResult.id) {
      throw new Error('File ID not found');
    }
    fileId = fileResult.id;
  }

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'What are the key findings or main points in this document?',
          },
          {
            media: {
              url: fileId,
              contentType: 'anthropic/file',
            },
          },
        ],
      },
    ],
  });

  return text;
});
