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

import { devLocalRetrieverRef } from '@genkit-ai/dev-local-vectorstore';
import { vertexAI } from '@genkit-ai/google-genai';
import { z } from 'genkit';
import { ai } from './genkit.js';

// Define the retriever reference
export const menuRetriever = devLocalRetrieverRef('menuQA');

export const menuQAFlow = ai.defineFlow(
  { name: 'menuQA', inputSchema: z.string(), outputSchema: z.string() },
  async (input: string) => {
    // retrieve relevant documents
    const docs = await ai.retrieve({
      retriever: menuRetriever,
      query: input,
      options: { k: 3 },
    });

    // generate a response
    const llmResponse = await ai.generate({
      model: vertexAI.model('gemini-2.5-flash'),
      prompt: `
    You are acting as a helpful AI assistant that can answer 
    questions about the food available on the menu at Genkit Grub Pub.
    
    Use only the context provided to answer the question.
    If you don't know, do not make up an answer.
    Do not add or change items on the menu.

    Question: ${input}
    `,
      docs,
    });

    const output = llmResponse.text;
    return output;
  }
);
