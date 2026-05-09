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

const ai = genkit({
  plugins: [anthropic()],
});

/**
 * This flow demonstrates PDF document processing with Claude using base64 encoding.
 * The PDF is read from the source directory and sent as a base64 data URL.
 */
ai.defineFlow('stable-pdf-base64', async () => {
  // Read PDF file from the same directory as this source file
  const pdfPath = path.join(__dirname, '../attention-first-page.pdf');
  const pdfBuffer = fs.readFileSync(pdfPath);
  const pdfBase64 = pdfBuffer.toString('base64');

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
              url: `data:application/pdf;base64,${pdfBase64}`,
              contentType: 'application/pdf',
            },
          },
        ],
      },
    ],
  });

  return text;
});

/**
 * This flow demonstrates PDF document processing with a URL reference.
 * Note: This requires the PDF to be hosted at a publicly accessible URL.
 */
ai.defineFlow('stable-pdf-url', async () => {
  // Example: Using a publicly hosted PDF URL
  // In a real application, you would use your own hosted PDF
  const pdfUrl =
    'https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf';

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'Summarize the key points from this document.',
          },
          {
            media: {
              url: pdfUrl,
              contentType: 'application/pdf',
            },
          },
        ],
      },
    ],
  });

  return text;
});

/**
 * This flow demonstrates analyzing specific aspects of a PDF document.
 * Claude can understand both text and visual elements (charts, tables, images) in PDFs.
 */
ai.defineFlow('stable-pdf-analysis', async () => {
  const pdfPath = path.join(__dirname, 'attention-first-page.pdf');
  const pdfBuffer = fs.readFileSync(pdfPath);
  const pdfBase64 = pdfBuffer.toString('base64');

  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'Analyze this document and provide:\n1. The main topic or subject\n2. Any key technical concepts mentioned\n3. Any visual elements (charts, tables, diagrams) if present',
          },
          {
            media: {
              url: `data:application/pdf;base64,${pdfBase64}`,
              contentType: 'application/pdf',
            },
          },
        ],
      },
    ],
  });

  return text;
});
