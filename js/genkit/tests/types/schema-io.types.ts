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

// Type-level assertions: callers pass a schema's *input* type (fields with
// `.default()` are optional), implementations receive its *output* type.
// Never executed: type-checked by `tsconfig.types-test.json`.

import { genkit, z } from '../../src/beta';

const ai = genkit({});

const Input = z.object({ q: z.string(), tone: z.string().default('casual') });

export async function promptCallersPassInput() {
  const p = ai.definePrompt({ name: 'p', input: { schema: Input } });
  await p({ q: 'hi' });
  await p.render({ q: 'hi' });
  p.stream({ q: 'hi' });
  // @ts-expect-error `q` is still required.
  await p({ tone: 'formal' });

  const lazy = ai.prompt<typeof Input>('lazy');
  await lazy({ q: 'hi' });
  await lazy.render({ q: 'hi' });
  lazy.stream({ q: 'hi' });
}

export async function flowCallersPassInput() {
  const f = ai.defineFlow({ name: 'f', inputSchema: Input }, async (input) => {
    // Implementations see the parsed value.
    const tone: string = input.tone;
    return tone;
  });
  await f({ q: 'hi' });
}

export async function retrieverOptionsAreInput() {
  const retriever = ai.defineRetriever(
    {
      name: 'r',
      configSchema: z.object({ k: z.number().default(3) }),
    },
    async (_query, options) => {
      const k: number = options.k;
      return { documents: Array(k).fill({ content: [{ text: 'x' }] }) };
    }
  );
  await ai.retrieve({ retriever, query: 'q', options: {} });
}

export function agentPromptInputIsInput() {
  ai.definePrompt({ name: 'agentPrompt', input: { schema: Input } });
  ai.definePromptAgent<unknown, typeof Input>({
    promptName: 'agentPrompt',
    promptInput: { q: 'hi' },
  });
}
