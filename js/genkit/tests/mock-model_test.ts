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

import { echoModel, mockModel } from '@genkit-ai/ai/testing';
import { z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit, type Genkit } from '../src/genkit';

initNodeFeatures();

describe('mockModel', () => {
  let ai: Genkit;

  beforeEach(() => {
    ai = genkit({});
  });

  it('returns the scripted text and records the request', async () => {
    const model = mockModel(ai, { respond: () => ({ text: 'a summary' }) });

    const summarize = ai.defineFlow('summarize', async (doc: string) =>
      (await ai.generate({ model, prompt: `Summarize: ${doc}` })).text
    );

    assert.strictEqual(await summarize('long text'), 'a summary');
    assert.strictEqual(model.requestCount, 1);
    assert.match(model.lastRequestMessage!.text, /Summarize: long text/);
  });

  it('accepts a bare string as shorthand for text', async () => {
    const model = mockModel(ai, { respond: () => 'hi there' });
    const res = await ai.generate({ model, prompt: 'x' });
    assert.strictEqual(res.text, 'hi there');
  });

  it('streams chunks via sendChunk', async () => {
    const model = mockModel(ai, {
      respond: (_req, { sendChunk }) => {
        sendChunk('hel');
        sendChunk('lo');
        return { text: 'hello' };
      },
    });

    const { response, stream } = ai.generateStream({ model, prompt: 'hi' });
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }

    assert.deepStrictEqual(chunks, ['hel', 'lo']);
    assert.strictEqual((await response).text, 'hello');
  });

  it('emits tool requests that the framework dispatches', async () => {
    const lookup = ai.defineTool(
      {
        name: 'lookup',
        description: 'look something up',
        inputSchema: z.object({ id: z.number() }),
        outputSchema: z.string(),
      },
      async ({ id }) => `item-${id}`
    );

    const model = mockModel(ai, {
      info: { supports: { tools: true } },
      respond: (req) => {
        const toolResponded = req.messages.some((m) =>
          m.content.some((c) => c.toolResponse)
        );
        return toolResponded
          ? { text: 'done' }
          : { toolRequests: [{ name: 'lookup', input: { id: 1 } }] };
      },
    });

    const res = await ai.generate({ model, prompt: 'go', tools: [lookup] });

    assert.strictEqual(res.text, 'done');
    assert.strictEqual(model.requestCount, 2);
  });

  it('records every request, oldest first', async () => {
    const model = mockModel(ai, { respond: () => ({ text: 'ok' }) });
    await ai.generate({ model, prompt: 'first' });
    await ai.generate({ model, prompt: 'second' });

    assert.strictEqual(model.requests.length, 2);
    assert.match(model.requests[0].messages.at(-1)!.content[0].text!, /first/);
    assert.match(model.requests[1].messages.at(-1)!.content[0].text!, /second/);
  });

  it('snapshots the request so later runs do not mutate history', async () => {
    const model = mockModel(ai, { respond: () => ({ text: 'ok' }) });
    await ai.generate({ model, prompt: 'first' });
    const captured = model.lastRequest;
    await ai.generate({ model, prompt: 'second' });

    assert.match(captured!.messages.at(-1)!.content[0].text!, /first/);
  });

  it('flattens the whole assembled request via lastRequestText', async () => {
    // Works even with an output schema, where echoModel can't be used: the mock
    // returns conforming JSON, and assembly is asserted by inspection.
    const model = mockModel(ai, {
      respond: () => ({ text: JSON.stringify({ x: 'ok' }) }),
    });

    await ai.generate({
      model,
      system: 'Be terse',
      prompt: 'hello',
      output: { schema: z.object({ x: z.string() }) },
    });

    assert.match(model.lastRequestText!, /system: Be terse/);
    assert.match(model.lastRequestText!, /hello/);
  });
});

describe('echoModel', () => {
  let ai: Genkit;

  beforeEach(() => {
    ai = genkit({});
  });

  it('echoes the rendered request, for prompt-assembly assertions', async () => {
    const model = echoModel(ai);
    const res = await ai.generate({
      model,
      system: 'Be terse',
      prompt: 'hello',
    });

    assert.match(res.text, /system: Be terse/);
    assert.match(res.text, /hello/);
  });

  it('throws a clear error when the request carries an output schema', async () => {
    const model = echoModel(ai);

    await assert.rejects(
      ai.generate({
        model,
        prompt: 'hi',
        output: { schema: z.object({ x: z.string() }) },
      }),
      /can't satisfy an output schema/
    );
  });
});
