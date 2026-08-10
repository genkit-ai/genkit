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

import type { MessageData } from '@genkit-ai/ai/model';
import { googleAI } from '@genkit-ai/google-genai';
import { z } from 'genkit';
import { defineAgent, type HistoryStore } from './agent.js';
import { ai } from './genkit.js';

const weatherTool = ai.defineTool(
  {
    name: 'weatherTool',
    description: 'use this tool to display weather',
    inputSchema: z.object({
      date: z
        .string()
        .describe('date (use datePicker tool if user did not specify)'),
      location: z.string().describe('location (ZIP, city, etc.)'),
    }),
    outputSchema: z.string().optional(),
  },
  async () => undefined
);

const datePicker = ai.defineTool(
  {
    name: 'datePicker',
    description:
      'user can use this UI tool to enter a date (prefer this over asking the user to enter the date manually)',
    inputSchema: z.object({
      ignore: z.string().describe('ignore this (set to undefined)').optional(),
    }),
    outputSchema: z.string().optional(),
  },
  async () => undefined
);

export const chatbotFlow = defineAgent(ai, {
  name: 'chatbotFlow',
  model: googleAI.model('gemini-flash-latest'),
  tools: [weatherTool, datePicker],
  returnToolRequests: true,
  systemPrompt:
    'You are a helpful agent. You have the personality of Agent Smith from Matrix. ' +
    'There are tools/functions at your disposal, ' +
    'feel free to call them. If you think a tool/function can help but you do ' +
    'not have sufficient context make sure to ask clarifying questions.',
  historyStore: inMemoryStore(),
});

const chatHistory: Record<string, MessageData[]> = {};

function inMemoryStore(): HistoryStore {
  return {
    async load(id: string): Promise<MessageData[] | undefined> {
      return chatHistory[id];
    },
    async save(id: string, history: MessageData[]) {
      chatHistory[id] = history;
    },
  };
}
