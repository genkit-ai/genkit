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

import { checks } from '@genkit-ai/checks';
import { openAICompatible } from '@genkit-ai/compat-oai';
import { deepSeek } from '@genkit-ai/compat-oai/deepseek';
import { openAI } from '@genkit-ai/compat-oai/openai';
import { xAI } from '@genkit-ai/compat-oai/xai';
import { genkitEval } from '@genkit-ai/evaluator';
import { expressHandler } from '@genkit-ai/express';
import { firebaseContext } from '@genkit-ai/firebase/context';
import { modelArmor } from '@genkit-ai/google-cloud';
import {
  googleAI,
  googleAI as newGoogleAI,
  vertexAI as newVertexAI,
} from '@genkit-ai/google-genai';
import {
  createMcpClient,
  createMcpHost,
  createMcpServer,
} from '@genkit-ai/mcp';
import { appRoute } from '@genkit-ai/next';
import { vertexAI } from '@genkit-ai/vertexai';
import { vertexAIEvaluation } from '@genkit-ai/vertexai/evaluation';
import { vertexModelGarden } from '@genkit-ai/vertexai/modelgarden';
import { genkit } from 'genkit';
import { ollama } from 'genkitx-ollama';

firebaseContext();
modelArmor;
checks;
googleAI;
vertexAI;
vertexModelGarden;
vertexAIEvaluation;
ollama;
genkitEval;
createMcpClient;
createMcpServer;
createMcpHost;
openAICompatible;
openAI;
xAI;
deepSeek;
newGoogleAI;
newVertexAI;

process.env.OPENAI_API_KEY = 'fake';
process.env.GEMINI_API_KEY = 'fake';

export const ai = genkit({
  plugins: [
    openAI({ apiKey: 'fake-oai-key' }),
    googleAI({ apiKey: 'fake-goog-key' }),
  ],
});
const hello = ai.defineFlow('hello', () => 'hello');
expressHandler(hello);
appRoute(hello);
