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

import { echoModel, mockModel, type MockResponse } from '@genkit-ai/ai/testing';
import { z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit as genkitBeta, type GenkitBeta } from '../src/beta';
import { genkit, type Genkit } from '../src/genkit';

initNodeFeatures();

describe('mockModel', () => {
  let ai: Genkit;

  beforeEach(() => {
    ai = genkit({});
  });

  it('returns the scripted text and records the request', async () => {
    const model = mockModel(ai, { respond: () => ({ text: 'a summary' }) });

    const summarize = ai.defineFlow(
      'summarize',
      async (doc: string) =>
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

  it('separates adjacent messages so their text does not fuse', async () => {
    const model = mockModel(ai, { respond: () => ({ text: 'ok' }) });
    await ai.generate({ model, system: 'alpha', prompt: 'beta' });

    // The last word of one message must not run into the first of the next,
    // or boundary-spanning assertions silently break.
    assert.doesNotMatch(model.lastRequestText!, /alphabeta/);
    assert.match(model.lastRequestText!, /alpha\nbeta/);
  });

  it('returns a defensive copy from lastRequest and requests', async () => {
    const model = mockModel(ai, { respond: () => ({ text: 'ok' }) });
    await ai.generate({ model, prompt: 'hello' });

    // Mutating the view must not corrupt recorded history.
    model.lastRequest!.messages.length = 0;
    model.requests[0].messages.length = 0;

    assert.ok(model.lastRequest!.messages.length > 0);
    assert.ok(model.requests[0].messages.length > 0);
  });

  it('defaults finishReason to stop when a full response leaves it undefined', async () => {
    const model = mockModel(ai, {
      respond: () => ({
        message: { role: 'model', content: [{ text: 'x' }] },
        finishReason: undefined,
      }),
    });

    const res = await ai.generate({ model, prompt: 'hi' });
    assert.strictEqual(res.finishReason, 'stop');
  });

  it('consumes an array of responses one per call, repeating the last', async () => {
    const model = mockModel(ai, { respond: ['first', 'second'] });

    assert.strictEqual(
      (await ai.generate({ model, prompt: 'a' })).text,
      'first'
    );
    assert.strictEqual(
      (await ai.generate({ model, prompt: 'b' })).text,
      'second'
    );
    // Past the end, the last response repeats rather than throwing.
    assert.strictEqual(
      (await ai.generate({ model, prompt: 'c' })).text,
      'second'
    );
    assert.strictEqual(model.requestCount, 3);
  });

  it('throws a queued Error to inject a failure on a given turn', async () => {
    const model = mockModel(ai, {
      respond: ['ok', new Error('rate limited')],
    });

    assert.strictEqual((await ai.generate({ model, prompt: 'a' })).text, 'ok');
    await assert.rejects(ai.generate({ model, prompt: 'b' }), /rate limited/);
  });

  it('treats an empty-string response as empty text, not a missing message', async () => {
    const model = mockModel(ai, { respond: '' });
    const res = await ai.generate({ model, prompt: 'x' });
    assert.strictEqual(res.text, '');
    assert.deepStrictEqual(res.message?.content, [{ text: '' }]);
  });

  it('accepts a single static response, returned on every call', async () => {
    const model = mockModel(ai, { respond: 'always this' });

    assert.strictEqual(
      (await ai.generate({ model, prompt: 'a' })).text,
      'always this'
    );
    assert.strictEqual(
      (await ai.generate({ model, prompt: 'b' })).text,
      'always this'
    );
  });

  it('swaps behavior for subsequent calls via respondWith', async () => {
    const model = mockModel(ai, { respond: 'initial' });

    assert.strictEqual(
      (await ai.generate({ model, prompt: 'a' })).text,
      'initial'
    );

    model.respondWith(['first', 'second']);
    assert.strictEqual(
      (await ai.generate({ model, prompt: 'b' })).text,
      'first'
    );
    assert.strictEqual(
      (await ai.generate({ model, prompt: 'c' })).text,
      'second'
    );
    // respondWith leaves recorded history untouched.
    assert.strictEqual(model.requestCount, 3);
  });

  it('clears history and re-arms the construction respond via reset', async () => {
    const model = mockModel(ai, { respond: ['first', 'second'] });

    await ai.generate({ model, prompt: 'a' });
    await ai.generate({ model, prompt: 'b' });
    model.respondWith('overridden');
    assert.strictEqual(
      (await ai.generate({ model, prompt: 'c' })).text,
      'overridden'
    );

    model.reset();

    assert.strictEqual(model.requestCount, 0);
    assert.deepStrictEqual(model.requests, []);
    assert.strictEqual(model.lastRequest, undefined);
    // Back to the construction-time queue, re-armed from its first item.
    assert.strictEqual(
      (await ai.generate({ model, prompt: 'd' })).text,
      'first'
    );
  });

  it('exposes tool results fed back to the model via toolResponses', async () => {
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
      // A queued tool loop: request the tool, then answer.
      respond: [
        { toolRequests: [{ name: 'lookup', input: { id: 1 } }] },
        'done',
      ],
    });

    await ai.generate({ model, prompt: 'go', tools: [lookup] });

    assert.deepStrictEqual(
      model.toolResponses.map((t) => t.name),
      ['lookup']
    );
    assert.strictEqual(model.toolResponses[0].output, 'item-1');
  });

  it('treats a falsy respond result as an empty response', async () => {
    // A callback that streams but forgets to return (void) must not crash.
    const model = mockModel(ai, {
      respond: () => undefined as unknown as MockResponse,
    });
    const res = await ai.generate({ model, prompt: 'x' });
    assert.strictEqual(res.text, '');
  });

  it('records a request even when it carries a non-serializable value', async () => {
    const model = mockModel(ai, { respond: () => 'ok' });
    // A function in config can't be structuredClone'd; recording must not throw.
    await ai.generate({
      model,
      prompt: 'hi',
      config: { onEvent: () => 1 } as unknown as Record<string, unknown>,
    });
    assert.strictEqual(model.requestCount, 1);
    assert.match(model.lastRequestText!, /hi/);
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

  it('exposes the output schema to respond by default (native constrained)', async () => {
    let seenSchema: unknown;
    const model = mockModel(ai, {
      respond: (req) => {
        seenSchema = req.output?.schema;
        return { text: JSON.stringify({ x: 'ok' }) };
      },
    });

    await ai.generate({
      model,
      prompt: 'hello',
      output: { schema: z.object({ x: z.string() }) },
    });

    // Native constrained: the schema is handed to the model on the request...
    assert.ok(seenSchema, 'respond should see request.output.schema');
    // ...not injected into the prompt.
    assert.doesNotMatch(
      model.lastRequestText!,
      /conform to the following schema/i
    );
  });

  it('simulates constrained output when supports.constrained is none', async () => {
    let seenSchema: unknown = 'unset';
    const model = mockModel(ai, {
      info: { supports: { constrained: 'none' } },
      respond: (req) => {
        seenSchema = req.output?.schema;
        return { text: JSON.stringify({ x: 'ok' }) };
      },
    });

    await ai.generate({
      model,
      prompt: 'hello',
      output: { schema: z.object({ x: z.string() }) },
    });

    // Simulated path strips the schema from what the model sees...
    assert.strictEqual(seenSchema, undefined);
    // ...and injects it into the prompt instead.
    assert.match(model.lastRequestText!, /conform to the following schema/i);
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

// These cover the two features most likely to need dedicated helpers —
// interrupts (human-in-the-loop) and agent handoff — and show that `mockModel`
// already tests them with no extra machinery: an interrupt is just a tool
// request the framework turns into a pause, and a handoff is just a
// prompt-as-tool request. Both live on the beta surface (`chat`, resume).
describe('mockModel — interrupts and agent handoff', () => {
  let ai: GenkitBeta;

  beforeEach(() => {
    ai = genkitBeta({});
  });

  it('drives an interrupt round-trip, then resumes to completion', async () => {
    // A human-in-the-loop tool: it interrupts on first run, and completes once
    // the generation is resumed with an answer.
    const confirmAction = ai.defineTool(
      { name: 'confirmAction', description: 'needs human confirmation' },
      async (_input, { interrupt, resumed }) => {
        if (resumed) return 'approved';
        return interrupt({ ask: 'Proceed?' });
      }
    );

    let turn = 0;
    const model = mockModel(ai, {
      info: { supports: { tools: true } },
      respond: () =>
        turn++ === 0
          ? { toolRequests: [{ name: 'confirmAction', input: {} }] }
          : { text: 'all done' },
    });

    // Turn 1: the model calls the tool, the tool interrupts, generation pauses.
    const paused = await ai.generate({
      model,
      prompt: 'do the thing',
      tools: [confirmAction],
    });
    assert.strictEqual(paused.interrupts.length, 1);
    assert.strictEqual(paused.interrupts[0].toolRequest.name, 'confirmAction');

    // Turn 2: a human approves; resuming re-runs the tool and the model finishes.
    const finished = await ai.generate({
      model,
      messages: paused.messages,
      tools: [confirmAction],
      resume: { respond: confirmAction.respond(paused.interrupts[0], 'yes') },
    });

    assert.strictEqual(finished.text, 'all done');
    assert.strictEqual(model.requestCount, 2);
  });

  it('drives a defineAgent tool loop with mockModel, exposing the preamble', async () => {
    // Migrated from an `app.chat(promptAgent)` handoff test: the beta Chat API
    // (and its cross-agent preamble swap) was removed in #5248. This keeps the
    // same mockModel surface under test — a multi-turn tool loop, with the
    // agent's rendered system preamble asserted via `lastRequestText` — using
    // the current `defineAgent(...).chat().send(...)` API.
    const app = genkitBeta({ model: 'agentModel' });

    app.defineTool(
      { name: 'lookup', description: "Look up today's special." },
      async () => 'Mushroom risotto'
    );

    const model = mockModel(app, {
      name: 'agentModel',
      info: { supports: { tools: true } },
      respond: (req) => {
        // The model's two-turn view of one `send`: call the tool, then once its
        // result is in the conversation, finish.
        const sawToolResult = req.messages.some((m) =>
          m.content.some((c) => c.toolResponse?.output === 'Mushroom risotto')
        );
        return sawToolResult
          ? { text: 'Booked: Mushroom risotto' }
          : { toolRequests: [{ name: 'lookup', input: {} }] };
      },
    });

    const agent = app.defineAgent({
      name: 'concierge',
      model: 'agentModel',
      system: 'You are the booking concierge.',
      tools: ['lookup'],
    });

    const { text } = await agent.chat().send('book me the special');

    assert.strictEqual(text, 'Booked: Mushroom risotto');
    assert.strictEqual(model.requestCount, 2);
    // The agent's system preamble is what the model saw.
    assert.match(model.lastRequestText!, /You are the booking concierge/);
  });
});
