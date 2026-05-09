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
import { createMcpClient } from '@genkit-ai/mcp';
import { genkit } from 'genkit';
import { logger } from 'genkit/logging';

logger.setLogLevel('debug');

const everythingClient = createMcpClient({
  name: 'everything',
  version: '1.0.0',
  serverProcess: {
    command: 'npx',
    args: ['@modelcontextprotocol/server-everything'],
  },
});

const ai = genkit({
  plugins: [googleAI(), everythingClient],
  model: googleAI.model('gemini-2.5-pro'),
});
