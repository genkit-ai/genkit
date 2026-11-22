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

import { fallback } from '@genkit-ai/middleware';
import * as assert from 'assert';
import { GenkitError, genkit, z } from 'genkit';
import { after, afterEach, before, describe, it, mock } from 'node:test';
import * as path from 'path';
import {
  choice,
  noul,
  responseInfo,
  score,
  systemOne,
  type Answer,
  type SystemOneOptions,
} from '../src/index.js';
import { FakeServer } from './fake_server.js';

const Triage = z.object({
  department: choice({
    billing: 'Payments, invoicing, refunds',
    technical: 'Bugs, outages, integrations',
    other: 'None of the above',
  }).describe('Which team should handle this?'),
  is_urgent: noul().describe(
    'Does the ticket explicitly communicate time pressure?'
  ),
  frustration: score(['Calm', 'Concerned but civil', 'Very angry']).describe(
    'How frustrated is the customer?'
  ),
});

const ticket = 'I was charged twice and need it fixed today.';
const savedEnv = { ...process.env };

describe('systemOne', () => {
  let server: FakeServer;
  before(async () => {
    server = await new FakeServer().start();
  });
  after(() => server.stop());
  afterEach(() => {
    process.env = { ...savedEnv };
    server.models = undefined;
    server.wrap = undefined;
    server.usage = { input_tokens: 12, output_tokens: 3 };
    server.answers = undefined;
    server.failures = [];
  });

  const typeSafe = () =>
    systemOne.typeSafe({ apiKey: 'ts-key', baseUrl: server.url });

  it('answers a decision schema through generate', async () => {
    server.usage = { input_tokens: 40, output_tokens: 3, cost: 0.0002 };
    const ai = genkit({ plugins: [typeSafe()] });
    const response = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: { schema: Triage },
    });
    const out = response.output!;
    // The answers are typed by the schema: a choice is one of its options.
    const department: 'billing' | 'technical' | 'other' = out.department.choice;
    assert.strictEqual(department, 'billing');
    assert.strictEqual(out.department.confidence, 0.6);
    assert.strictEqual(out.is_urgent.noul, 0.93);
    assert.strictEqual(out.frustration.score, 1.3);
    assert.deepStrictEqual(out.frustration.legend, {
      '0': 'Calm',
      '1': 'Concerned but civil',
      '2': 'Very angry',
    });
    assert.deepStrictEqual(response.usage, {
      inputTokens: 40,
      outputTokens: 3,
      totalTokens: 43,
      custom: { cost: 0.0002 },
    });
    const info = responseInfo(response);
    assert.strictEqual(info.model, 'jev-1.13.0');
    assert.strictEqual(info.answers?.department.type, 'choice');

    const sent = server.bodies.at(-1);
    assert.strictEqual(server.requests.at(-1)!.url, '/v1/systemone');
    assert.strictEqual(
      server.requests.at(-1)!.headers.authorization,
      'Bearer ts-key'
    );
    assert.strictEqual(sent.model, 'jev-1.13.0');
    assert.strictEqual(sent.state, ticket);
    assert.deepStrictEqual(sent.questions.department, {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: {
        billing: 'Payments, invoicing, refunds',
        technical: 'Bugs, outages, integrations',
        other: 'None of the above',
      },
    });
    assert.deepStrictEqual(sent.questions.is_urgent, {
      type: 'noul',
      instructions: 'Does the ticket explicitly communicate time pressure?',
    });
  });

  it('puts a system message in front of every question, never in the state', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    await ai.generate({
      model: 'typesafe/jev-1.13.0',
      system: 'The state is a support ticket.',
      messages: [
        { role: 'user', content: [{ text: 'It broke.' }] },
        { role: 'model', content: [{ text: 'Which part?' }] },
      ],
      prompt: 'The login page.',
      output: { schema: z.object({ urgent: noul().describe('Urgent?') }) },
    });
    const sent = server.bodies.at(-1);
    assert.strictEqual(
      sent.questions.urgent.instructions,
      'The state is a support ticket.\n\nUrgent?'
    );
    assert.deepStrictEqual(sent.state, [
      { role: 'user', content: 'It broke.' },
      { role: 'model', content: 'Which part?' },
      { role: 'user', content: 'The login page.' },
    ]);
  });

  it('answers the enum format as one choice', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    const response = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      system: 'Which team should handle this ticket?',
      prompt: ticket,
      output: {
        format: 'enum',
        schema: z.enum(['technical', 'billing', 'sales']),
      },
    });
    assert.strictEqual(response.output, 'billing');
    const sent = server.bodies.at(-1);
    assert.strictEqual(sent.state, ticket);
    assert.deepStrictEqual(sent.questions, {
      choice: {
        type: 'choice',
        instructions: 'Which team should handle this ticket?',
        criteria: {
          technical: 'technical',
          billing: 'billing',
          sales: 'sales',
        },
      },
    });
  });

  it('answers runtime questions', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    const response = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: 'Book a table for two.',
      output: {
        jsonSchema: systemOne.schema({
          tool: {
            type: 'choice',
            instructions: 'Which tool serves the request?',
            options: [
              { name: 'search', criteria: 'Looks things up' },
              { name: 'booking', criteria: 'Makes reservations' },
            ],
          },
          personal: {
            type: 'noul',
            instructions: "Does the request involve the user's own data?",
            yes: "Names the user's files, mail, or calendar",
            no: 'Asks about the world at large',
          },
        }),
      },
    });
    const answers = response.output as Record<string, Answer>;
    assert.strictEqual(answers.tool.choice, 'booking');
    assert.strictEqual(answers.personal.noul, 0.93);
  });

  it('runs a prompt file that names the decision schema', async () => {
    const ai = genkit({
      plugins: [typeSafe()],
      promptDir: path.join(__dirname, 'prompts'),
    });
    ai.defineSchema('Triage', Triage);
    const triage = ai.prompt('triage');
    const { output } = await triage({ ticket: 'Down again', tier: 'pro' });
    assert.strictEqual((output as any).department.choice, 'billing');
    const sent = server.bodies.at(-1);
    assert.deepStrictEqual(sent.state, { ticket: 'Down again', tier: 'pro' });
    assert.strictEqual(
      sent.questions.department.instructions,
      "The state is a support ticket; tier is the customer's plan.\n\nWhich team should handle this?"
    );
  });

  it('takes a config through modelRef', async () => {
    const ts = typeSafe();
    const ai = genkit({ plugins: [ts] });
    await ai.generate({
      model: ts.modelRef('typesafe/jev-1.13.0', { stateJSON: true }),
      prompt: '{"ticket": "Down", "id": 12345678901234567890}',
      output: { schema: z.object({ u: noul().describe('Urgent?') }) },
    });
    assert.ok(
      server.rawBodies
        .at(-1)!
        .includes('"state":{"ticket": "Down", "id": 12345678901234567890}')
    );
  });

  it('merges extra fields but not over the request', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    const urgent = z.object({ u: noul().describe('Urgent?') });
    await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      config: { extra: { session_id: 's-1' } },
      output: { schema: urgent },
    });
    assert.strictEqual(server.bodies.at(-1).session_id, 's-1');
    await assert.rejects(
      ai.generate({
        model: 'typesafe/jev-1.13.0',
        prompt: ticket,
        config: { extra: { questions: {} } },
        output: { schema: urgent },
      }),
      /extra field "questions" is one the request builds itself/
    );
  });

  it('refuses what a decision model cannot do', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    const tool = ai.defineTool(
      { name: 'lookup', description: 'Looks up', inputSchema: z.object({}) },
      async () => 'x'
    );
    const urgent = z.object({ u: noul().describe('Urgent?') });
    const cases: [string, Promise<unknown>, RegExp][] = [
      [
        'no output schema',
        ai.generate({ model: 'typesafe/jev-1.13.0', prompt: ticket }),
        /answers questions encoded in an output schema/,
      ],
      [
        'plain output schema',
        ai.generate({
          model: 'typesafe/jev-1.13.0',
          prompt: ticket,
          output: { schema: z.object({ team: z.string().describe('Team?') }) },
        }),
        /field "team" is not a question/,
      ],
      [
        'tools',
        ai.generate({
          model: 'typesafe/jev-1.13.0',
          prompt: ticket,
          tools: [tool],
          output: { schema: urgent },
        }),
        /calls no tools/,
      ],
      [
        'media',
        ai.generate({
          model: 'typesafe/jev-1.13.0',
          prompt: [
            { text: ticket },
            { media: { url: 'data:image/png;base64,AA' } },
          ],
          output: { schema: urgent },
        }),
        /reads no image\/png media/,
      ],
    ];
    const before = server.bodies.length;
    for (const [name, call, message] of cases) {
      await assert.rejects(call, message, name);
    }
    assert.strictEqual(server.bodies.length, before, 'nothing was sent');
  });

  it('clamps a score a hair below the rubric to 0', async () => {
    server.answers = {
      frustration: { score: -1e-17, confidence: 0.9, probabilities: {} },
    };
    const ai = genkit({ plugins: [typeSafe()] });
    const { output } = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: { schema: Triage },
    });
    assert.strictEqual(output!.frustration.score, 0);
  });

  it('rejects an answer outside the schema', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    for (const answers of [
      { department: { choice: 'sales', confidence: 0.9 } },
      { is_urgent: { noul: 1.5 } },
    ]) {
      server.answers = answers;
      await assert.rejects(
        ai.generate({
          model: 'typesafe/jev-1.13.0',
          prompt: ticket,
          output: { schema: Triage },
        }),
        /Schema validation failed/,
        JSON.stringify(answers)
      );
    }
  });

  it('lets a caller cancel a request through generate', async () => {
    server.failures = [{ status: 200, body: '{}', delayMs: 2000 }];
    const ai = genkit({ plugins: [typeSafe()] });
    const abort = new AbortController();
    const started = Date.now();
    setTimeout(() => abort.abort(), 100);
    await assert.rejects(
      ai.generate({
        model: 'typesafe/jev-1.13.0',
        prompt: ticket,
        output: { schema: Triage },
        abortSignal: abort.signal,
      }),
      /CANCELLED|abort/i
    );
    assert.ok(Date.now() - started < 1500, 'cancelled before the reply');
  });

  it('asks one choice() behind two fields', async () => {
    const Dept = choice({ billing: 'Payments', other: 'Anything else' });
    const ai = genkit({ plugins: [typeSafe()] });
    const { output } = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: {
        schema: z.object({
          primary: Dept.describe('Which team first?'),
          secondary: Dept.describe('Which team next?'),
        }),
      },
    });
    assert.strictEqual(output!.secondary.choice, 'billing');
    assert.deepStrictEqual(server.bodies.at(-1).questions.secondary, {
      type: 'choice',
      instructions: 'Which team next?',
      criteria: { billing: 'Payments', other: 'Anything else' },
    });
  });

  it('reads the info back from the JSON of a response', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    const response = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: { schema: Triage },
    });
    const json = JSON.parse(JSON.stringify(response.toJSON()));
    assert.strictEqual(responseInfo(json).model, 'jev-1.13.0');
  });

  it('refuses an extra that is not an object, and ignores odd usage', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    const urgent = z.object({ u: noul().describe('Urgent?') });
    await assert.rejects(
      ai.generate({
        model: 'typesafe/jev-1.13.0',
        prompt: ticket,
        config: { extra: ['session'] as any },
        output: { schema: urgent },
      }),
      (e: GenkitError) => e.status === 'INVALID_ARGUMENT'
    );
    server.usage = { input_tokens: '12', output_tokens: 3, cost: 'free' };
    const response = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: { schema: urgent },
    });
    // A count the server leaves out, or sends as something else, stays out.
    assert.deepStrictEqual(response.usage, { outputTokens: 3 });
  });

  it('takes the key from each request only with apiKey: false', async () => {
    process.env.TYPESAFE_API_KEY = 'env-key';
    server.models = '[{"name":"jev-1.13.0"}]';
    const ts = systemOne.typeSafe({ apiKey: false, baseUrl: server.url });
    const ai = genkit({ plugins: [ts] });
    assert.ok((await ts.list!()).length > 0, 'lists without a key');
    assert.strictEqual(server.listingAuth.at(-1), undefined, 'no key listed');
    const urgent = z.object({ u: noul().describe('Urgent?') });
    await assert.rejects(
      ai.generate({
        model: 'typesafe/jev-1.13.0',
        prompt: ticket,
        output: { schema: urgent },
      }),
      (e: GenkitError) =>
        e.status === 'FAILED_PRECONDITION' &&
        /created with apiKey: false, so pass the API key as config.apiKey/.test(
          e.message
        )
    );
    await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      config: { apiKey: 'request-key' },
      output: { schema: urgent },
    });
    assert.strictEqual(
      server.requests.at(-1)!.headers.authorization,
      'Bearer request-key'
    );
  });

  it('needs a key when the server requires one', async () => {
    delete process.env.TYPESAFE_API_KEY;
    assert.throws(
      () => systemOne.typeSafe({ baseUrl: server.url }),
      (e: GenkitError) =>
        e.status === 'FAILED_PRECONDITION' &&
        /set apiKey or the TYPESAFE_API_KEY/.test(e.message)
    );
  });

  it("adds models to a constructor's, an entry replacing a whole spec", async () => {
    const ts = systemOne.typeSafe({
      apiKey: 'k',
      baseUrl: server.url,
      models: { 'jev-2.0.0': {}, 'jev-latest': { label: 'Latest' } },
    });
    const listed = await ts.list!();
    assert.deepStrictEqual(
      listed.map((a) => a.name),
      ['typesafe/jev-1.13.0', 'typesafe/jev-2.0.0', 'typesafe/jev-latest']
    );
    assert.strictEqual((listed[2].metadata?.model as any).label, 'Latest');
    assert.throws(
      () =>
        systemOne.typeSafe({
          apiKey: 'k',
          models: { 'typesafe/jev-latest': {} },
        }),
      /already defined by the preset as jev-latest; use the bare ID/
    );

    const workers = (models: Record<string, object>) =>
      genkit({
        plugins: [
          systemOne.workersAI('acct', {
            apiKey: 't',
            baseUrl: `${server.url}/client/v4/accounts/{account}`,
            route: undefined,
            models,
          }),
        ],
      });
    const hear = (ai: ReturnType<typeof genkit>) =>
      ai.generate({
        model: 'cloudflare-decisions/@cf/cloudflare/clef-omni',
        prompt: [
          { text: 'Hear it.' },
          { media: { url: 'data:audio/wav;base64,AA' } },
        ],
        output: { schema: z.object({ ok: noul().describe('Fine?') }) },
      });
    // The preset's spec stays, and undefined options keep the preset's route.
    await hear(workers({ '@cf/cloudflare/clef-flash': { label: 'Flash' } }));
    assert.strictEqual(
      server.requests.at(-1)!.url,
      '/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-omni'
    );
    assert.deepStrictEqual(server.bodies.at(-1).audio, [
      'data:audio/wav;base64,AA',
    ]);
    // An entry for it replaces the whole spec, audio included.
    await assert.rejects(
      hear(workers({ '@cf/cloudflare/clef-omni': { label: 'Omni' } })),
      /reads no audio\/wav media/
    );
  });

  it('refuses a provider with a slash, and references carry the model info', () => {
    assert.throws(
      () => systemOne({ provider: 'a/b', baseUrl: server.url }),
      /contains a slash/
    );
    const ref = typeSafe().modelRef('jev-1.13.0', { stateJSON: true });
    assert.strictEqual(ref.name, 'typesafe/jev-1.13.0');
    assert.strictEqual(ref.info?.label, 'TypeSafe - jev-1.13.0');
    assert.strictEqual(ref.info?.supports?.constrained, 'all');
    assert.deepStrictEqual(ref.config, { stateJSON: true });
  });

  it('keeps only usable IDs from a listing', async () => {
    server.models = JSON.stringify({
      models: [{ name: 'jev-9' }, { name: 42 }, { id: 'x'.repeat(300) }, {}],
    });
    const names = (await typeSafe().list!()).map((a) => a.name);
    assert.deepStrictEqual(names, [
      'typesafe/jev-1.13.0',
      'typesafe/jev-9',
      'typesafe/jev-latest',
    ]);
  });

  it('takes answers without confidence, typed and at run time', async () => {
    server.answers = {
      department: { choice: 'billing' },
      tool: { choice: 'search' },
    };
    const ai = genkit({ plugins: [typeSafe()] });
    const typed = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: { schema: Triage },
    });
    assert.strictEqual(typed.output!.department.confidence, undefined);
    const runtime = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: {
        jsonSchema: systemOne.schema({
          tool: {
            type: 'choice',
            instructions: 'Which tool?',
            options: [{ name: 'search' }, { name: 'none' }],
          },
        }),
      },
    });
    assert.deepStrictEqual((runtime.output as any).tool, { choice: 'search' });
  });

  it('refuses a runtime answer outside the question set', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    const jsonSchema = systemOne.schema({
      tool: {
        type: 'choice',
        instructions: 'Which tool?',
        options: [{ name: 'search' }, { name: 'none' }],
      },
      personal: { type: 'noul', instructions: 'Personal?' },
    });
    for (const answers of [
      { tool: { choice: 'calendar' } },
      { personal: { noul: 1.5 } },
    ]) {
      server.answers = answers;
      await assert.rejects(
        ai.generate({
          model: 'typesafe/jev-1.13.0',
          prompt: ticket,
          output: { jsonSchema },
        }),
        /Schema validation failed/,
        JSON.stringify(answers)
      );
    }
  });

  it('refuses a probability or a confidence outside 0 to 1', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    for (const department of [
      { choice: 'billing', confidence: 1.5 },
      { choice: 'billing', probabilities: { billing: 2 } },
    ]) {
      server.answers = { department };
      await assert.rejects(
        ai.generate({
          model: 'typesafe/jev-1.13.0',
          prompt: ticket,
          output: { schema: Triage },
        }),
        (e: GenkitError) =>
          e.status === 'INTERNAL' && /not a number from 0 to 1/.test(e.message)
      );
    }
  });

  it('keeps probabilities for undeclared options in the info only', async () => {
    server.answers = {
      department: {
        choice: 'billing',
        probabilities: { billing: 0.9, refunds: 0.1 },
      },
    };
    const ai = genkit({ plugins: [typeSafe()] });
    const response = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: { schema: Triage },
    });
    assert.deepStrictEqual(response.output!.department.probabilities, {
      billing: 0.9,
    });
    assert.deepStrictEqual(
      (responseInfo(response).answers!.department as any).probabilities,
      { billing: 0.9, refunds: 0.1 }
    );
  });

  it('omits usage the server does not report', async () => {
    server.usage = undefined as any;
    const ai = genkit({ plugins: [typeSafe()] });
    const response = await ai.generate({
      model: 'typesafe/jev-1.13.0',
      prompt: ticket,
      output: { schema: Triage },
    });
    assert.strictEqual(response.usage.inputTokens, undefined);
    assert.strictEqual(response.usage.outputTokens, undefined);
  });

  it('bounds each attempt as configured, and passes the caller signal on', async () => {
    const timeouts: number[] = [];
    const realTimeout = AbortSignal.timeout.bind(AbortSignal);
    const timeout = mock.method(AbortSignal, 'timeout', (ms: number) => {
      timeouts.push(ms);
      return realTimeout(ms);
    });
    try {
      const urgent = z.object({ u: noul().describe('Urgent?') });
      const ask = (
        plugin: ReturnType<typeof systemOne>,
        signal?: AbortSignal
      ) =>
        genkit({ plugins: [plugin] }).generate({
          model: 'typesafe/jev-1.13.0',
          prompt: ticket,
          output: { schema: urgent },
          abortSignal: signal,
        });
      // The global fetch, looked up when a request is made, and 30 s.
      const original = globalThis.fetch;
      let globalCalls = 0;
      globalThis.fetch = (input, init) => {
        globalCalls++;
        return original(input, init);
      };
      try {
        await ask(typeSafe());
      } finally {
        globalThis.fetch = original;
      }
      assert.strictEqual(globalCalls, 1);
      assert.deepStrictEqual(timeouts, [30000]);

      // A caller's fetch keeps its own timeouts, and gets the caller's signal.
      timeouts.length = 0;
      let seen: AbortSignal | undefined;
      const fetch: typeof globalThis.fetch = (input, init) => {
        seen = init?.signal ?? undefined;
        return original(input, init);
      };
      const own = systemOne.typeSafe({
        apiKey: 'k',
        baseUrl: server.url,
        fetch,
      });
      await ask(own);
      assert.deepStrictEqual(timeouts, []);
      server.failures = [{ status: 200, body: '{}', delayMs: 1000 }];
      const abort = new AbortController();
      setTimeout(() => abort.abort(), 50);
      await assert.rejects(ask(own, abort.signal), /CANCELLED|abort/i);
      assert.strictEqual(seen?.aborted, true);

      await ask(
        systemOne.typeSafe({
          apiKey: 'k',
          baseUrl: server.url,
          fetch,
          timeoutMs: 1234,
        })
      );
      assert.deepStrictEqual(timeouts, [1234]);
    } finally {
      timeout.mock.restore();
    }
  });

  it('falls back across two plugins without sending one key to the other', async () => {
    const backup = await new FakeServer().start();
    try {
      server.failures = [{ status: 501 }, { status: 501 }];
      const primary = systemOne.typeSafe({
        apiKey: false,
        baseUrl: server.url,
      });
      const secondary = systemOne({
        provider: 'backup',
        baseUrl: backup.url,
        apiKey: false,
      });
      const ai = genkit({ plugins: [primary, secondary] });
      const urgent = z.object({ u: noul().describe('Urgent?') });
      const ask = (use: any) =>
        ai.generate({
          model: 'typesafe/jev-1.13.0',
          prompt: ticket,
          config: { apiKey: 'primary-key' },
          output: { schema: urgent },
          use: [use],
        });
      // A reference with its own config sends its own key.
      await ask(
        fallback({
          models: [secondary.modelRef('d1', { apiKey: 'backup-key' })],
          statuses: ['UNIMPLEMENTED'],
        })
      );
      assert.strictEqual(
        backup.requests.at(-1)!.headers.authorization,
        'Bearer backup-key'
      );
      // isolateConfig keeps the primary key from the backup, which then
      // refuses the request for want of a key of its own.
      const before = backup.requests.length;
      await assert.rejects(
        ask(
          fallback({
            models: [secondary.modelRef('d1')],
            statuses: ['UNIMPLEMENTED'],
            isolateConfig: true,
          })
        ),
        (e: GenkitError) => e.status === 'FAILED_PRECONDITION'
      );
      assert.strictEqual(backup.requests.length, before);
    } finally {
      await backup.stop();
    }
  });

  it('refuses headers or a key HTTP cannot carry', async () => {
    const cases: Partial<SystemOneOptions>[] = [
      { headers: { 'X-Trace': 'a\nb' } },
      { headers: { 'bad header': 'x' } },
      { apiKey: 'key\nwith-newline' },
      { headers: { 'X-Trace': 'a\u000bb' } },
      { headers: { 'X-Trace': 'a\u007fb' } },
      { apiKey: 'key\u000bwith-vt' },
      { apiKey: 'key\u007fwith-del' },
    ];
    for (const options of cases) {
      assert.throws(
        () => systemOne({ provider: 'x', baseUrl: server.url, ...options }),
        (e: GenkitError) =>
          e.status === 'FAILED_PRECONDITION' &&
          /header cannot be sent: its name or value is not valid HTTP/.test(
            e.message
          ) &&
          !/with-|a[\n\u000b\u007f]b/.test(e.message),
        JSON.stringify(options)
      );
    }
    // A horizontal tab is a valid field value.
    assert.ok(
      systemOne({
        provider: 'x',
        baseUrl: server.url,
        headers: { 'X-Trace': 'a\tb' },
      })
    );
    const ai = genkit({
      plugins: [systemOne.typeSafe({ apiKey: false, baseUrl: server.url })],
    });
    for (const apiKey of [
      'key\nwith-newline',
      'key\u000bwith-vt',
      'key\u007fwith-del',
    ]) {
      const before = server.requests.length;
      await assert.rejects(
        ai.generate({
          model: 'typesafe/jev-1.13.0',
          prompt: ticket,
          config: { apiKey },
          output: { schema: z.object({ u: noul().describe('Urgent?') }) },
        }),
        (e: GenkitError) =>
          e.status === 'INVALID_ARGUMENT' &&
          /the Authorization header cannot be sent/.test(e.message) &&
          !e.message.includes('with-') &&
          /header cannot be sent: its name or value is not valid HTTP/.test(
            e.message
          )
      );
      assert.strictEqual(server.requests.length, before, 'nothing was sent');
    }
  });

  it('refuses a baseUrl, a timeoutMs, or models it cannot use', () => {
    for (const baseUrl of ['api.example.com', 'ftp://example.com', 'http://']) {
      assert.throws(
        () => systemOne({ provider: 'x', baseUrl }),
        (e: GenkitError) =>
          e.status === 'FAILED_PRECONDITION' &&
          /baseUrl must be an http or https URL with a host/.test(e.message),
        baseUrl
      );
    }
    for (const timeoutMs of [0, -1, Number.NaN, Infinity]) {
      assert.throws(
        () => systemOne({ provider: 'x', baseUrl: server.url, timeoutMs }),
        /timeoutMs must be a positive number/,
        String(timeoutMs)
      );
    }
  });

  it('escapes the account into the base URL as a path segment', async () => {
    const cf = systemOne.workersAI('team@one 1', {
      apiKey: 't',
      baseUrl: `${server.url}/client/v4/accounts/{account}`,
    });
    await genkit({ plugins: [cf] }).generate({
      model: 'cloudflare-decisions/typesafe/jev',
      prompt: ticket,
      output: { schema: z.object({ ok: noul().describe('Fine?') }) },
    });
    assert.strictEqual(
      server.requests.at(-1)!.url,
      '/client/v4/accounts/team@one%201/ai/run'
    );
  });

  it('lists the configured and the listed models, and keeps the listing', async () => {
    server.models = '{"models":[{"name":"jev-1.14.0"},{"name":"jev-1.13.0"}]}';
    const ts = typeSafe();
    const names = (await ts.list!()).map((a) => a.name);
    assert.deepStrictEqual(names, [
      'typesafe/jev-1.13.0',
      'typesafe/jev-1.14.0',
      'typesafe/jev-latest',
    ]);
    const listings = server.listings.length;
    await ts.list!();
    assert.strictEqual(server.listings.length, listings);
    assert.strictEqual(server.listings.at(-1), '/v1/models');
    const label = (await ts.list!())[0].metadata?.model as any;
    assert.strictEqual(label.label, 'TypeSafe - jev-1.13.0');
    assert.strictEqual(label.supports.constrained, 'all');
  });

  it('advertises the configured models when the listing fails', async () => {
    const names = (await typeSafe().list!()).map((a) => a.name);
    assert.deepStrictEqual(names, [
      'typesafe/jev-1.13.0',
      'typesafe/jev-latest',
    ]);
  });

  it('resolves any model ID', async () => {
    const ai = genkit({ plugins: [typeSafe()] });
    await ai.generate({
      model: 'typesafe/jev-preview',
      prompt: ticket,
      output: { schema: z.object({ u: noul().describe('Urgent?') }) },
    });
    assert.strictEqual(server.bodies.at(-1).model, 'jev-preview');
  });

  it('needs a key, a base URL, and a provider', () => {
    delete process.env.TYPESAFE_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    assert.throws(
      () => systemOne.typeSafe(),
      /set apiKey or the TYPESAFE_API_KEY/
    );
    assert.throws(() => systemOne.openRouter(), /OPENROUTER_API_KEY/);
    assert.throws(() => systemOne({ provider: 'x' }), /set baseUrl/);
    assert.throws(
      () => systemOne({ provider: '', baseUrl: server.url }),
      /set provider/
    );
    assert.throws(
      () =>
        systemOne({
          provider: 'x',
          baseUrl: server.url,
          models: { d1: {}, 'x/d1': {} },
        }),
      /names d1 twice/
    );
    process.env.TYPESAFE_API_KEY = 'env-key';
    process.env.TYPESAFE_BASE_URL = 'http://typesafe.test';
    assert.ok(systemOne.typeSafe());
  });

  it('reaches OpenRouter on its paths, text only', async () => {
    server.models = '{"data":[{"id":"liquid/d1","name":"Liquid: d1"}]}';
    const or = systemOne.openRouter({ apiKey: 'or-key', baseUrl: server.url });
    const ai = genkit({ plugins: [or] });
    await ai.generate({
      model: 'openrouter-decisions/liquid/d1',
      prompt: ticket,
      output: { schema: Triage },
    });
    assert.strictEqual(server.requests.at(-1)!.url, '/api/alpha/decisions');
    assert.strictEqual(server.bodies.at(-1).model, 'liquid/d1');
    await or.list!();
    assert.strictEqual(
      server.listings.at(-1),
      '/api/v1/models?output_modalities=decisions'
    );
    await assert.rejects(
      ai.generate({
        model: 'openrouter-decisions/liquid/d1',
        prompt: [{ text: 'x' }, { media: { url: 'data:image/png;base64,AA' } }],
        output: { schema: Triage },
      }),
      /reads no image/
    );
  });

  it('sends Ollama images as raw base64 and needs text', async () => {
    const ai = genkit({ plugins: [systemOne.ollama({ baseUrl: server.url })] });
    const damage = z.object({ broken: noul().describe('Is it broken?') });
    await ai.generate({
      model: 'ollama-decisions/clef',
      prompt: [
        { text: 'Arrived like this.' },
        { media: { url: 'data:image/jpeg;base64,/9j/AA' } },
      ],
      output: { schema: damage },
    });
    const sent = server.bodies.at(-1);
    assert.strictEqual(server.requests.at(-1)!.url, '/v1/systemone');
    assert.strictEqual(
      server.requests.at(-1)!.headers.authorization,
      undefined
    );
    assert.deepStrictEqual(sent.images, ['/9j/AA']);
    assert.strictEqual(sent.state, 'Arrived like this.');
    await assert.rejects(
      ai.generate({
        model: 'ollama-decisions/clef',
        prompt: [{ media: { url: 'data:image/jpeg;base64,/9j/AA' } }],
        output: { schema: damage },
      }),
      /Ollama requires text in the state, so a request with images needs text too/
    );
  });

  describe('Workers AI', () => {
    const workers = (options = {}) =>
      systemOne.workersAI('acct 1', {
        apiKey: 'cf-token',
        baseUrl: `${server.url}/client/v4/accounts/{account}`,
        ...options,
      });
    const check = z.object({ ok: noul().describe('Is it fine?') });

    it("runs Cloudflare's own model at its path, unwrapping the envelope", async () => {
      server.wrap = (reply) => ({ result: reply, success: true, errors: [] });
      const ai = genkit({ plugins: [workers()] });
      const response = await ai.generate({
        model: 'cloudflare-decisions/@cf/cloudflare/clef',
        prompt: [
          { text: 'Look.' },
          { media: { url: 'data:image/png;base64,AA' } },
        ],
        output: { schema: check },
      });
      assert.strictEqual(
        server.requests.at(-1)!.url,
        '/client/v4/accounts/acct%201/ai/run/@cf/cloudflare/clef'
      );
      assert.strictEqual(server.bodies.at(-1).model, 'clef');
      assert.deepStrictEqual(server.bodies.at(-1).images, [
        'data:image/png;base64,AA',
      ]);
      assert.strictEqual((response.output as any).ok.noul, 0.93);
    });

    it('lets Clef-omni hear and see, and keeps jev text only', async () => {
      const ai = genkit({ plugins: [workers()] });
      await ai.generate({
        model: 'cloudflare-decisions/@cf/cloudflare/clef-omni',
        prompt: [
          { text: 'Clip from the server room.' },
          { media: { url: 'data:audio/wav;base64,AA' } },
          { media: { url: 'data:video/mp4;base64,BB' } },
        ],
        output: { schema: check },
      });
      assert.deepStrictEqual(server.bodies.at(-1).audio, [
        'data:audio/wav;base64,AA',
      ]);
      assert.deepStrictEqual(server.bodies.at(-1).videos, [
        'data:video/mp4;base64,BB',
      ]);
      await assert.rejects(
        ai.generate({
          model: 'cloudflare-decisions/@cf/cloudflare/clef',
          prompt: [
            { text: 'x' },
            { media: { url: 'data:audio/wav;base64,AA' } },
          ],
          output: { schema: check },
        }),
        /reads no audio\/wav media/
      );
      await assert.rejects(
        ai.generate({
          model: 'cloudflare-decisions/typesafe/jev',
          prompt: [
            { text: 'x' },
            { media: { url: 'data:image/png;base64,AA' } },
          ],
          output: { schema: check },
        }),
        /reads no image\/png media/
      );
    });

    it("runs a partner's model with its input wrapped", async () => {
      const ai = genkit({ plugins: [workers()] });
      await ai.generate({
        model: 'cloudflare-decisions/typesafe/jev',
        prompt: ticket,
        output: { schema: check },
      });
      assert.strictEqual(
        server.requests.at(-1)!.url,
        '/client/v4/accounts/acct%201/ai/run'
      );
      const sent = server.bodies.at(-1);
      assert.strictEqual(sent.model, 'typesafe/jev');
      assert.strictEqual(sent.input.state, ticket);
      assert.strictEqual(sent.input.model, undefined);
    });

    it('refuses an ID that would leave the path, and reads a failed envelope', async () => {
      const ai = genkit({ plugins: [workers()] });
      await assert.rejects(
        ai.generate({
          model: 'cloudflare-decisions/@cf/../../tokens',
          prompt: ticket,
          output: { schema: check },
        }),
        /is not a Workers AI model ID/
      );
      server.wrap = () => ({
        result: null,
        success: false,
        errors: [{ code: 5007, message: 'No such model' }],
      });
      await assert.rejects(
        ai.generate({
          model: 'cloudflare-decisions/@cf/cloudflare/clef',
          prompt: ticket,
          output: { schema: check },
        }),
        (e: GenkitError) =>
          e.status === 'UNKNOWN' &&
          e.originalMessage === 'cloudflare-decisions: No such model'
      );
    });

    it('needs an account, unless behind a proxy', () => {
      delete process.env.CLOUDFLARE_ACCOUNT_ID;
      assert.throws(
        () =>
          systemOne.workersAI(undefined, {
            apiKey: 't',
            baseUrl: 'https://x/{account}',
          }),
        /pass an account ID or set the CLOUDFLARE_ACCOUNT_ID/
      );
      assert.throws(
        () => systemOne.workersAI(undefined, { apiKey: 't' }),
        /CLOUDFLARE_ACCOUNT_ID/
      );
      assert.ok(
        systemOne.workersAI(undefined, { apiKey: 't', baseUrl: server.url })
      );
    });
  });

  it('serves any other server through its fields and hooks', async () => {
    server.wrap = (reply) => ({ result: reply });
    const wrapped = systemOne({
      provider: 'wrapped',
      baseUrl: `${server.url}/run`,
      path: '/',
      route: (model, body) => {
        delete body.model;
        return { path: model, body: { input: body } };
      },
      unwrap: (body: any) => body.result,
    });
    const ai = genkit({ plugins: [wrapped] });
    const { output } = await ai.generate({
      model: 'wrapped/d1',
      prompt: ticket,
      output: { schema: Triage },
    });
    assert.strictEqual(output!.department.choice, 'billing');
    assert.strictEqual(server.requests.at(-1)!.url, '/run/d1');
    assert.strictEqual(server.bodies.at(-1).input.state, ticket);
    assert.strictEqual(wrapped.modelRef('wrapped/d1').name, 'wrapped/d1');
    assert.strictEqual(wrapped.modelRef('d1').name, 'wrapped/d1');
  });
});
