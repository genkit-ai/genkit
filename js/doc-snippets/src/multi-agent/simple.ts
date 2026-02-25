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
import { z } from 'genkit';
import { genkit } from 'genkit/beta';

const ai = genkit({});

// [START tools]
const menuLookupTool = ai.defineTool(
  {
    name: 'menuLookupTool',
    description: 'use this tool to look up the menu for a given date',
    inputSchema: z.object({
      date: z.string().describe('the date to look up the menu for'),
    }),
    outputSchema: z.string().describe('the menu for a given date'),
  },
  async (input) => {
    // Retrieve the menu from a database, website, etc.
    // [START_EXCLUDE]
    return '';
    // [END_EXCLUDE]
  }
);

const reservationTool = ai.defineTool(
  {
    name: 'reservationTool',
    description: 'use this tool to try to book a reservation',
    inputSchema: z.object({
      partySize: z.coerce.number().describe('the number of guests'),
      date: z.string().describe('the date to book for'),
    }),
    outputSchema: z
      .string()
      .describe(
        "true if the reservation was successfully booked and false if there's" +
          ' no table available for the requested time'
      ),
  },
  async (input) => {
    // Access your database to try to make the reservation.
    // [START_EXCLUDE]
    return '';
    // [END_EXCLUDE]
  }
);
// [END tools]

// [START chat]
const chat = ai.chat({
  model: googleAI.model('gemini-2.5-pro'),
  system:
    "You are an AI customer service agent for Pavel's Cafe. Use the tools " +
    'available to you to help the customer. If you cannot help the ' +
    'customer with the available tools, politely explain so.',
  tools: [menuLookupTool, reservationTool],
});
// [END chat]
