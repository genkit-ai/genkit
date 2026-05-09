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
import { genkit } from 'genkit';
import { langchain } from 'genkitx-langchain';

// Turn off safety checks for evaluation so that the LLM as an evaluator can
// respond appropriately to potentially harmful content without error.
export const PERMISSIVE_SAFETY_SETTINGS: any = {
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
};

export const ai = genkit({
  plugins: [
    googleAI(),
    genkitEval({
      metrics: [
        {
          type: GenkitMetric.MALICIOUSNESS,
          judge: googleAI.model('gemini-2.5-pro'),
          judgeConfig: PERMISSIVE_SAFETY_SETTINGS,
        },
        {
          type: GenkitMetric.ANSWER_ACCURACY,
          judge: googleAI.model('gemini-2.5-pro'),
          judgeConfig: PERMISSIVE_SAFETY_SETTINGS,
        },
      ],
    }),
    devLocalVectorstore([
      {
        indexName: 'pdfQA',
        embedder: googleAI.embedder('gemini-embedding-001'),
      },
    ]),
    langchain({
      evaluators: {
        criteria: ['coherence'],
        labeledCriteria: ['correctness'],
        judge: googleAI.model('gemini-2.5-pro'),
      },
    }),
  ],
});
