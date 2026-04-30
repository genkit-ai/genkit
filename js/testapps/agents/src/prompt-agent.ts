/**
 * Copyright 2026 Google LLC
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

import { z } from 'genkit';
import { ai } from './genkit.js';
import { SessionFlowInit, SessionFlowInput } from '../../../ai/lib/session-flow.js';

// ---------------------------------------------------------------------------
// Writer Agent — demonstrates `defineSessionFlowFromPrompt` with multiple
// prompt input variables.  The system prompt is a Handlebars template that
// references {{ tone }}, {{ format }}, and {{ audience }}.  At runtime the
// client can supply any combination of these variables via `init` to reshape
// the agent's behaviour without changing any server code.
// ---------------------------------------------------------------------------

export const writerPrompt = ai.definePrompt({
  name: 'writerPrompt',
  model: 'googleai/gemini-flash-latest',
  input: {
    schema: z.object({
      tone: z.string(),
      format: z.string(),
      audience: z.string(),
    }),
  },
  system: `You are a versatile writing assistant.

Tone: {{ tone }}
Format: {{ format }}
Target audience: {{ audience }}

Follow these rules strictly:
- Always write in the specified tone.
- Always structure your response using the specified format.
- Always tailor your language and complexity to the target audience.

Help the user with whatever writing task they request.`,
});

export const writerAgent = ai.defineSessionFlowFromPrompt({
  promptName: 'writerPrompt',
  defaultInput: {
    tone: 'Professional',
    format: 'Paragraph',
    audience: 'General',
  },
});

export const testWriterAgent = ai.defineFlow(
  {
    name: 'testWriterAgent',
    inputSchema: z.string().default('Write a short intro about AI safety.'),
    outputSchema: z.any(),
  },
  async (text) => {
    const res = await writerAgent.run(
      <SessionFlowInput>{
        messages: [{ role: 'user', content: [{ text }] }],
      },
      { init: <SessionFlowInit>{}   }
    );
    return res.result;
  }
);
