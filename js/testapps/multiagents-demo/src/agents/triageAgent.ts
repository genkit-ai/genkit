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

import { ai } from '../config/genkit';
import { transferToAgent } from './interrupts';

// The triage agent routes customers to the appropriate specialist
export const agent = ai.definePrompt({
  name: 'triageAgent',
  description: 'Triage Agent',
  model: 'googleai/gemini-2.5-flash',
  tools: [transferToAgent],
  system: `You are an AI customer service agent for TechStore Computer Shop.
    Greet the user and ask them how you can help. Route them to the appropriate specialist agent:
    - Use transferToAgent to route to catalogAgent for browsing products, searching items, or getting product information
    - Use transferToAgent to route to paymentAgent for payment processing
    - Use transferToAgent to route to representativeAgent for providing store information
    If you cannot help the customer with the available tools, politely explain so.`,
});
