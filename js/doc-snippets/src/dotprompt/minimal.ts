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

// [START mini]
import { genkit } from 'genkit';

// Import the model plugins you want to use.
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  // Initialize and configure the model plugins.
  plugins: [
    googleAI({
      apiKey: 'your-api-key', // Or (preferred): export GEMINI_API_KEY=...
    }),
  ],
});
// [END mini]
