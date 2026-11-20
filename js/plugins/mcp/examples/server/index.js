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

import { createMcpServer } from '@genkit-ai/mcp';
import { genkit, z } from 'genkit';

const ai = genkit({});

ai.defineTool(
  {
    name: 'add',
    description: 'add two numbers together',
    inputSchema: z.object({ a: z.number(), b: z.number() }),
    outputSchema: z.number(),
  },
  async ({ a, b }) => {
    return a + b;
  }
);

ai.definePrompt({
  name: 'happy',
  description: 'everybody together now',
  input: {
    schema: z.object({ action: z.string().optional() }),
    default: { action: 'clap your hands' },
  },
  messages: `If you're happy and you know it, {{action}}.`,
});

createMcpServer(ai, { name: 'example_server', version: '0.0.1' }).start();
