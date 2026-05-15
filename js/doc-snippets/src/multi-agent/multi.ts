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

import { genkit } from 'genkit/beta';

const ai = genkit({});
const reservationTool = ai.defineTool(
  {
    name: '',
    description: '',
  },
  async () => {}
);
const reservationCancelationTool = reservationTool;
const reservationListTool = reservationTool;

// [START agents]
// Define a prompt that represents a specialist agent
const reservationAgent = ai.definePrompt({
  name: 'reservationAgent',
  model: 'googleai/gemini-2.5-flash',
  description: 'Reservation Agent can help manage guest reservations',
  tools: [reservationTool, reservationCancelationTool, reservationListTool],
  system: 'Help guests make and manage reservations',
});

// Or load agents from .prompt files
const menuInfoAgent = ai.prompt('menuInfoAgent');
const complaintAgent = ai.prompt('complaintAgent');

// The triage agent is the agent that users interact with initially
const triageAgent = ai.definePrompt({
  name: 'triageAgent',
  model: 'googleai/gemini-2.5-flash-lite',
  description: 'Triage Agent',
  tools: [reservationAgent, menuInfoAgent, complaintAgent],
  system: `You are an AI customer service agent for Pavel's Cafe.
  Greet the user and ask them how you can help. If appropriate, transfer to an
  agent that can better handle the request. If you cannot help the customer with
  the available tools, politely explain so.`,
});
// [END agents]

// [START chat]
// Start a chat session, initially with the triage agent
const chat = ai.chat(triageAgent);
// [END chat]
