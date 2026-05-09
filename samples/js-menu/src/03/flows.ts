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

import type { MessageData } from '@genkit-ai/ai/model';
import { googleAI } from '@genkit-ai/google-genai';
import { ai } from '../genkit.js';
import type { MenuItem } from '../types';
import {
  ChatHistoryStore,
  ChatSessionInputSchema,
  ChatSessionOutputSchema,
} from './chats';

// Load the menu data from a JSON file.
const menuData = require('../../data/menu.json') as Array<MenuItem>;

// Render the preamble prompt that seeds our chat history.
const preamble: Array<MessageData> = [
  {
    role: 'user',
    content: [
      {
        text: "Hi. What's on the menu today?",
      },
    ],
  },
  {
    role: 'user',
    content: [
      {
        text:
          'I am Walt, a helpful AI assistant here at the restaurant.\n' +
          'I can answer questions about the food on the menu or any other questions\n' +
          "you have about food in general. I probably can't help you with anything else.\n" +
          "Here is today's menu: \n" +
          menuData
            .map((r) => `- ${r.title} ${r.price}\n${r.description}`)
            .join('\n') +
          'Do you have any questions about the menu?\n',
      },
    ],
  },
];

// A simple local storage for chat session history.
// You should probably actually use Firestore for this.
const chatHistoryStore = new ChatHistoryStore(preamble);

// Define a flow which generates a response to each question.

export const s03_multiTurnChatFlow = ai.defineFlow(
  {
    name: 's03_multiTurnChat',
    inputSchema: ChatSessionInputSchema,
    outputSchema: ChatSessionOutputSchema,
  },
  async (input) => {
    // First fetch the chat history. We'll wrap this in a run block.
    // If we were going to a database for the history,
    // we might want to have that db result captured in the trace.
    let history = await ai.run('fetchHistory', async () =>
      chatHistoryStore.read(input.sessionId)
    );

    // Generate the response
    const llmResponse = await ai.generate({
      model: googleAI.model('gemini-flash-latest'),
      messages: history,
      prompt: {
        text: input.question,
      },
    });

    // Add the exchange to the history store and return it
    history = llmResponse.messages;
    chatHistoryStore.write(input.sessionId, history);
    return {
      sessionId: input.sessionId,
      history: history,
    };
  }
);
