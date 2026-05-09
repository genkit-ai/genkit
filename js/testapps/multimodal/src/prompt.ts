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
import { z } from 'genkit';
import { ai } from './genkit.js';

export const augmentedVideoPrompt = ai.definePrompt({
  model: vertexAI.model('gemini-2.5-flash'),
  name: 'augmentedVideoPrompt',
  input: {
    schema: z.object({
      question: z.string(),
      media: z.object({
        gcsUrl: z.string(),
        contentType: z.string(),
        startOffsetSec: z.number(),
        endOffsetSec: z.number(),
      }),
    }),
  },
  output: {
    format: 'text',
  },
  messages: `
    Use the following video to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {{media contentType=media.contentType url=media.gcsUrl}}

    Question: {{question}}
  Helpful Answer: `,
});

// Define a prompt that includes the retrieved context documents
export const augmentedMultimodalPrompt = ai.definePrompt({
  model: vertexAI.model('gemini-2.5-flash'),
  name: 'augmentedMultimodalPrompt',
  input: {
    schema: z.object({
      text: z.optional(z.array(z.string())),
      media: z.optional(
        z.array(
          z
            .object({
              dataUrl: z.string(),
              gcsUrl: z.string(),
              contentType: z.string(),
            })
            .partial()
            .refine((data) => data.dataUrl || (data.gcsUrl && data.contentType))
        )
      ),
      question: z.string(),
    }),
  },
  output: {
    format: 'text',
  },
  messages: `
  Use the following context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  {{#each text}}
    - {{this}}
  {{/each}}
  {{#each media}}
    {{#if this.dataUrl}}
      {{media url=this.dataUrl}}
    {{/if}}
    {{#if this.gcsUrl}}
      {{media contentType=this.contentType url=this.gcsUrl}}
    {{/if}}
  {{/each}}

  Question: {{question}}
  Helpful Answer:
    `,
});
