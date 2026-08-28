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

import { genkit } from 'genkit';
import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  A2UI_CATALOG_VALUE_TYPE,
  basicCatalog,
  type A2uiCatalog,
} from '../src/catalog.js';
import {
  A2uiOptionsSchema,
  a2ui,
  type A2uiOptions,
} from '../src/middleware.js';
import { a2uiEnvelopesFromParts, isA2uiPart } from '../src/part.js';

const SAMPLE_TEXT = `Here is the weather:
\`\`\`a2ui
[
  { "createSurface": { "surfaceId": "SURFACE_ID", "catalogId": "${basicCatalog.id}" } },
  { "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
    { "id": "root", "component": "Text", "text": "hi" }
  ] } }
]
\`\`\`
`;

/**
 * A minimal fake registry supporting the value APIs the middleware uses,
 * pre-loaded with any catalogs the test needs.
 */
function fakeAi(catalogs: Record<string, A2uiCatalog> = {}) {
  const store: Record<string, Record<string, unknown>> = {
    [A2UI_CATALOG_VALUE_TYPE]: { ...catalogs },
  };
  return {
    registry: {
      registerValue(type: string, name: string, value: unknown) {
        (store[type] ??= {})[name] = value;
      },
      async lookupValue<T = unknown>(
        type: string,
        name: string
      ): Promise<T | undefined> {
        return store[type]?.[name] as T | undefined;
      },
    },
  };
}

/**
 * Instantiates the a2ui generate-middleware and returns its `model` hook —
 * a `(req, ctx, next)` function — for direct unit testing.
 */
function modelHook(config: Partial<A2uiOptions>, ai: any = fakeAi()) {
  const def = (a2ui as any).__def ?? a2ui;
  const impl = def.instantiate({ config, ai });
  return impl.model as (req: any, ctx: any, next: any) => Promise<any>;
}

/** A minimal fake request. */
function req(system?: string) {
  return {
    messages: system
      ? [{ role: 'system', content: [{ text: system }] }]
      : [{ role: 'user', content: [{ text: 'hi' }] }],
  } as any;
}

describe('a2ui() middleware', () => {
  it('defaults to the bundled basic catalog when none is configured', async () => {
    const mw = modelHook({});
    let seen: any;
    await mw(req('You are helpful.'), undefined, async (r: any) => {
      seen = r;
      return { message: { role: 'model', content: [] } };
    });
    const sys = seen.messages.find((m: any) => m.role === 'system');
    const joined = sys.content.map((p: any) => p.text).join('');
    assert.match(joined, /Rendering UI with A2UI/);
  });

  it('resolves a custom catalog registered in the registry by id', async () => {
    const custom: A2uiCatalog = {
      id: 'my-catalog',
      components: [
        { name: 'Widget', description: 'A widget.', props: 'label: string.' },
      ],
    };
    const mw = modelHook(
      { catalog: 'my-catalog' },
      fakeAi({ 'my-catalog': custom })
    );
    let seen: any;
    await mw(req('sys'), undefined, async (r: any) => {
      seen = r;
      return { message: { role: 'model', content: [] } };
    });
    const sys = seen.messages.find((m: any) => m.role === 'system');
    const joined = sys.content.map((p: any) => p.text).join('');
    assert.match(joined, /Widget: A widget\./);
    assert.match(joined, /my-catalog/);
  });

  it('throws when an unknown catalog id is configured', async () => {
    const mw = modelHook({ catalog: 'nope' });
    await assert.rejects(
      () =>
        mw(req('sys'), undefined, async () => ({
          message: { role: 'model', content: [] },
        })),
      /no catalog registered under id "nope"/
    );
  });

  it('injects instructions into an existing system prompt', async () => {
    const mw = modelHook({});
    let seen: any;
    await mw(req('You are helpful.'), undefined, async (r: any) => {
      seen = r;
      return { message: { role: 'model', content: [] } };
    });
    const sys = seen.messages.find((m: any) => m.role === 'system');
    const joined = sys.content.map((p: any) => p.text).join('');
    assert.match(joined, /You are helpful\./);
    assert.match(joined, /Rendering UI with A2UI/);
    assert.match(joined, /Available components/);
  });

  it('creates a system prompt when none exists', async () => {
    const mw = modelHook({});
    let seen: any;
    await mw(req(), undefined, async (r: any) => {
      seen = r;
      return { message: { role: 'model', content: [] } };
    });
    const sys = seen.messages.find((m: any) => m.role === 'system');
    assert.ok(sys, 'expected a system message to be added');
    assert.match(sys.content[0].text, /Rendering UI with A2UI/);
  });

  it('instructions:none injects nothing', async () => {
    const mw = modelHook({ instructions: 'none' });
    let seen: any;
    await mw(req('sys'), undefined, async (r: any) => {
      seen = r;
      return { message: { role: 'model', content: [] } };
    });
    const sys = seen.messages.find((m: any) => m.role === 'system');
    assert.strictEqual(sys.content.length, 1);
    assert.strictEqual(sys.content[0].text, 'sys');
  });

  it('rewrites the final message: prose text + a2ui part', async () => {
    const mw = modelHook({ surfaceId: 'sfc' });

    const res = await mw(req('sys'), undefined, async () => ({
      message: { role: 'model', content: [{ text: SAMPLE_TEXT }] },
    }));
    const content = (res as any).message.content;
    // Should have a prose part and an a2ui part.
    const textPart = content.find((p: any) => typeof p.text === 'string');
    const uiPart = content.find((p: any) => isA2uiPart(p));
    assert.ok(textPart, 'expected a prose text part');
    assert.match(textPart.text, /Here is the weather/);
    assert.ok(uiPart, 'expected an a2ui part');
    const envelopes = a2uiEnvelopesFromParts(content);
    assert.strictEqual(envelopes.length, 2);
    assert.strictEqual((envelopes[0] as any).createSurface.surfaceId, 'sfc');
  });

  it('leaves plain prose responses untouched (no a2ui parts)', async () => {
    const mw = modelHook({});
    const res = await mw(req('sys'), undefined, async () => ({
      message: { role: 'model', content: [{ text: 'just chatting' }] },
    }));
    const content = (res as any).message.content;
    assert.ok(!content.some((p: any) => isA2uiPart(p)));
    assert.strictEqual(a2uiEnvelopesFromParts(content).length, 0);
  });

  it('sanitizes inbound a2ui parts into text for the model', async () => {
    const mw = modelHook({});
    let seen: any;
    await mw(
      {
        messages: [
          {
            role: 'user',
            content: [
              { text: 'clicked:' },
              {
                data: {
                  envelopes: [
                    {
                      action: {
                        name: 'refresh',
                        surfaceId: 's1',
                        sourceComponentId: 'btn',
                        timestamp: 't',
                        context: { city: 'Tokyo' },
                      },
                    },
                  ],
                },
                metadata: { mimeType: 'application/a2ui+json' },
              },
            ],
          },
        ],
      } as any,
      undefined,
      async (r: any) => {
        seen = r;
        return { message: { role: 'model', content: [] } };
      }
    );
    const userMsg = seen.messages.find((m: any) => m.role === 'user');
    // No a2ui data parts should reach the model.
    assert.ok(!userMsg.content.some((p: any) => isA2uiPart(p)));
    const joined = userMsg.content.map((p: any) => p.text).join(' ');
    assert.match(joined, /UI action "refresh"/);
    assert.match(joined, /Tokyo/);
  });

  it('replays a prior assistant surface as a fenced a2ui block, not a sentinel', async () => {
    const mw = modelHook({});
    let seen: any;
    await mw(
      {
        messages: [
          {
            role: 'model',
            content: [
              { text: 'Here you go:' },
              {
                data: {
                  envelopes: [
                    {
                      createSurface: {
                        surfaceId: 's1',
                        catalogId: basicCatalog.id,
                      },
                      version: 'v0.9',
                    },
                    {
                      updateComponents: {
                        surfaceId: 's1',
                        components: [
                          { id: 'root', component: 'Text', text: 'hi' },
                        ],
                      },
                      version: 'v0.9',
                    },
                  ],
                },
                metadata: { mimeType: 'application/a2ui+json' },
              },
            ],
          },
          { role: 'user', content: [{ text: 'thanks' }] },
        ],
      } as any,
      undefined,
      async (r: any) => {
        seen = r;
        return { message: { role: 'model', content: [] } };
      }
    );
    const modelMsg = seen.messages.find((m: any) => m.role === 'model');
    // The a2ui part is gone (the model converter never sees the mime type)...
    assert.ok(!modelMsg.content.some((p: any) => isA2uiPart(p)));
    const joined = modelMsg.content.map((p: any) => p.text).join('\n');
    // ...replaced by the canonical fenced block the model originally emitted,
    // NOT the old `[rendered UI surface]` sentinel that poisoned the model.
    assert.doesNotMatch(joined, /\[rendered UI surface\]/);
    assert.doesNotMatch(joined, /\[UI surface/);
    assert.match(joined, /```a2ui/);
    assert.match(joined, /createSurface/);
    assert.match(joined, /updateComponents/);
    assert.match(joined, /Here you go:/);

    // The reconstructed block round-trips: parsing it yields the envelopes.
    const block = joined.slice(
      joined.indexOf('```a2ui') + '```a2ui'.length,
      joined.lastIndexOf('```')
    );
    const decoded = JSON.parse(block.trim());
    assert.strictEqual(decoded.length, 2);
    assert.ok(decoded[0].createSurface);

    // The real surface id is kept verbatim (NOT scrubbed to a placeholder), so
    // a replayed action `[UI action ... on surface s1]` can still be correlated
    // with this surface. Reuse is prevented at the parser instead: every
    // `createSurface` mints a fresh id (see the distinct-id test).
    assert.strictEqual(decoded[0].createSurface.surfaceId, 's1');
    assert.strictEqual(decoded[1].updateComponents.surfaceId, 's1');
  });

  it('drops a message emptied by sanitizing instead of sending empty content', async () => {
    // A message whose only part is an a2ui part with an empty (or all-
    // unrecognized) envelope array summarizes to nothing. Sending it downstream
    // with `content: []` makes providers like Gemini/Vertex reject the request,
    // so the middleware must drop the whole message instead.
    const mw = modelHook({});
    let seen: any;
    await mw(
      {
        messages: [
          {
            role: 'model',
            content: [
              {
                data: { envelopes: [] },
                metadata: { mimeType: 'application/a2ui+json' },
              },
            ],
          },
          { role: 'user', content: [{ text: 'hi' }] },
        ],
      } as any,
      undefined,
      async (r: any) => {
        seen = r;
        return { message: { role: 'model', content: [] } };
      }
    );

    // The emptied model message is gone entirely; no message has empty content.
    assert.ok(
      !seen.messages.some(
        (m: any) => Array.isArray(m.content) && m.content.length === 0
      )
    );
    // The still-meaningful user message survives.
    const userMsg = seen.messages.find((m: any) => m.role === 'user');
    assert.ok(userMsg);
    assert.strictEqual(userMsg.content[0].text, 'hi');
  });

  it('a new render never reuses a surface id copied from history', async () => {
    // Regression for the "new answer overwrites the prior surface in place"
    // bug: history keeps real ids (for action correlation), so the model can
    // copy an old id into a fresh `createSurface`. The parser must still mint a
    // distinct id for that new render.
    const mw = modelHook({ surfaceId: 'sfc-new' });
    let seen: any;
    const res = await mw(
      {
        messages: [
          {
            role: 'model',
            content: [
              {
                data: {
                  envelopes: [
                    {
                      createSurface: {
                        surfaceId: 's1',
                        catalogId: basicCatalog.id,
                      },
                    },
                    {
                      updateComponents: {
                        surfaceId: 's1',
                        components: [
                          { id: 'root', component: 'Text', text: 'old' },
                        ],
                      },
                    },
                  ],
                },
                metadata: { mimeType: 'application/a2ui+json' },
              },
            ],
          },
          {
            role: 'user',
            content: [
              {
                data: {
                  envelopes: [{ action: { name: 'refresh', surfaceId: 's1' } }],
                },
                metadata: { mimeType: 'application/a2ui+json' },
              },
            ],
          },
        ],
      } as any,
      undefined,
      async (r: any) => {
        seen = r;
        // The model copies the prior surface's real id (`s1`) into a brand-new
        // createSurface - exactly what it does after seeing `s1` in history.
        return {
          message: {
            role: 'model',
            content: [
              {
                text: `Here you go:
\`\`\`a2ui
[
  { "createSurface": { "surfaceId": "s1", "catalogId": "${basicCatalog.id}" } },
  { "updateComponents": { "surfaceId": "s1", "components": [
    { "id": "root", "component": "Text", "text": "new" }
  ] } }
]
\`\`\`
`,
              },
            ],
          },
        };
      }
    );

    // The new render is minted onto the fixed id `sfc-new`, NOT the copied `s1`,
    // so it can't overwrite the prior surface.
    const envelopes = a2uiEnvelopesFromParts((res as any).message.content);
    const create = envelopes.find((e: any) => e.createSurface) as any;
    const update = envelopes.find((e: any) => e.updateComponents) as any;
    assert.strictEqual(create.createSurface.surfaceId, 'sfc-new');
    assert.strictEqual(update.updateComponents.surfaceId, 'sfc-new');

    // Meanwhile, the sanitized history the model saw kept the real id on both
    // the reconstructed surface block and the action line (correlation).
    const modelMsg = seen.messages.find((m: any) => m.role === 'model');
    const modelText = modelMsg.content.map((p: any) => p.text).join('\n');
    assert.match(modelText, /"surfaceId"\s*:\s*"s1"/);

    const userMsg = seen.messages.find((m: any) => m.role === 'user');
    const userText = userMsg.content.map((p: any) => p.text).join('\n');
    assert.match(userText, /on surface s1/);
  });

  it('groups consecutive surface envelopes into one block but splits around an action', async () => {
    const mw = modelHook({});
    let seen: any;
    await mw(
      {
        messages: [
          {
            role: 'user',
            content: [
              {
                data: {
                  envelopes: [
                    {
                      createSurface: {
                        surfaceId: 's1',
                        catalogId: basicCatalog.id,
                      },
                    },
                    { updateComponents: { surfaceId: 's1', components: [] } },
                    { action: { name: 'refresh', surfaceId: 's1' } },
                  ],
                },
                metadata: { mimeType: 'application/a2ui+json' },
              },
            ],
          },
        ],
      } as any,
      undefined,
      async (r: any) => {
        seen = r;
        return { message: { role: 'model', content: [] } };
      }
    );
    const userMsg = seen.messages.find((m: any) => m.role === 'user');
    const joined = userMsg.content.map((p: any) => p.text).join('\n');
    // Exactly one fenced block (the two surface envelopes grouped together)...
    assert.strictEqual((joined.match(/```a2ui/g) ?? []).length, 1);
    // ...plus the action rendered as a text summary after it.
    assert.match(joined, /UI action "refresh"/);
    // The block precedes the action line (source order preserved).
    assert.ok(joined.indexOf('```a2ui') < joined.indexOf('UI action'));
  });

  it('mints the same surface id in the stream and the final message', async () => {
    // Default surface-id policy (random UUID). The stream parse and the final
    // parse must agree on the id for the same surface.
    const mw = modelHook({});
    let streamedId: string | undefined;
    const ctx = {
      onChunk: (c: any) => {
        const envs = a2uiEnvelopesFromParts(c.content);
        const create = envs.find((e: any) => e.createSurface);
        if (create) streamedId = (create as any).createSurface.surfaceId;
      },
    };
    const res = await mw(
      req('sys'),
      ctx as any,
      async (_r: any, wrapped: any) => {
        // Stream the whole block, then return it as the final message too.
        wrapped.onChunk({ content: [{ text: SAMPLE_TEXT }] });
        return { message: { role: 'model', content: [{ text: SAMPLE_TEXT }] } };
      }
    );
    const finalEnvelopes = a2uiEnvelopesFromParts((res as any).message.content);
    const finalCreate = finalEnvelopes.find((e: any) => e.createSurface) as any;
    assert.ok(streamedId, 'expected a streamed surface id');
    assert.ok(finalCreate, 'expected a final createSurface');
    assert.strictEqual(finalCreate.createSurface.surfaceId, streamedId);
  });

  it('transforms streamed chunks via onChunk', async () => {
    const mw = modelHook({ surfaceId: 'sfc' });
    const emitted: any[] = [];

    const ctx = { onChunk: (c: any) => emitted.push(c) };
    // Simulate the runtime calling next() with a wrapped ctx, then streaming.
    await mw(req('sys'), ctx as any, async (_r: any, wrappedCtx: any) => {
      // Stream prose, then the block in pieces.
      wrappedCtx.onChunk({ content: [{ text: 'Here is the weather:\n' }] });
      for (const piece of SAMPLE_TEXT.slice(
        'Here is the weather:\n'.length
      ).match(/.{1,5}/gs) ?? []) {
        wrappedCtx.onChunk({ content: [{ text: piece }] });
      }
      return { message: { role: 'model', content: [] } };
    });
    // Collect all envelopes emitted across chunks.
    const allEnvelopes = emitted.flatMap((c) =>
      a2uiEnvelopesFromParts(c.content)
    );
    assert.strictEqual(allEnvelopes.length, 2);
    // And prose was emitted too.
    const prose = emitted
      .flatMap((c) => c.content ?? [])
      .filter((p: any) => typeof p.text === 'string')
      .map((p: any) => p.text)
      .join('');
    assert.match(prose, /Here is the weather/);
    assert.doesNotMatch(prose, /createSurface/);
  });

  it('defaults to validate:warn — drops a bad block, keeps the turn alive', async () => {
    // A hallucinated component would throw under strict, killing the turn.
    // With the warn default the block is dropped and prose survives.
    const bad = `oops:
\`\`\`a2ui
[{ "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
  { "id": "root", "component": "NotAThing" }
] } }]
\`\`\`
`;
    const mw = modelHook({});
    const original = console.warn;
    console.warn = () => {};
    let res: any;
    try {
      res = await mw(req('sys'), undefined, async () => ({
        message: { role: 'model', content: [{ text: bad }] },
      }));
    } finally {
      console.warn = original;
    }
    const content = res.message.content;
    // No a2ui parts (the bad block was dropped), but prose is preserved.
    assert.ok(!content.some((p: any) => isA2uiPart(p)));
    const text = content
      .filter((p: any) => typeof p.text === 'string')
      .map((p: any) => p.text)
      .join('');
    assert.match(text, /oops/);
  });

  it('flushes the withheld prose tail to the stream at end of turn', async () => {
    // The parser holds back up to a partial-opening-fence tail on every push,
    // so without a final flush the last <=8 chars of trailing prose would never
    // reach the streaming consumer. Stream a short prose turn one char at a time
    // and assert the streamed deltas reconstruct the full text.
    const mw = modelHook({});
    const emitted: any[] = [];
    const ctx = { onChunk: (c: any) => emitted.push(c) };
    const full = 'Hello there, friend!';
    await mw(req('sys'), ctx as any, async (_r: any, wrappedCtx: any) => {
      for (const ch of full) wrappedCtx.onChunk({ content: [{ text: ch }] });
      return { message: { role: 'model', content: [{ text: full }] } };
    });
    const streamedProse = emitted
      .flatMap((c) => c.content ?? [])
      .filter((p: any) => typeof p.text === 'string')
      .map((p: any) => p.text)
      .join('');
    assert.strictEqual(streamedProse, full);
  });

  it('flushes an unterminated trailing block to the stream at end of turn', async () => {
    // A block whose closing fence never arrives on the stream should still be
    // recovered (parsed from what we have) and emitted when the turn ends.
    const mw = modelHook({ surfaceId: 'sfc' });
    const emitted: any[] = [];
    const ctx = { onChunk: (c: any) => emitted.push(c) };
    const unterminated =
      '```a2ui\n[{ "updateComponents": { "surfaceId": "SURFACE_ID", ' +
      '"components": [{ "id": "root", "component": "Text", "text": "hi" }] } }]';
    await mw(req('sys'), ctx as any, async (_r: any, wrappedCtx: any) => {
      wrappedCtx.onChunk({ content: [{ text: unterminated }] });
      return {
        message: { role: 'model', content: [{ text: unterminated }] },
      };
    });
    const streamedEnvelopes = emitted.flatMap((c) =>
      a2uiEnvelopesFromParts(c.content)
    );
    assert.ok(
      streamedEnvelopes.some((e: any) => e.updateComponents),
      'expected the unterminated block to reach the stream after flush'
    );
  });

  it('rejects an unsupported protocol version in the config schema', () => {
    // A typo like 'v0.10' would otherwise be cast to SupportedVersion and emit
    // envelopes the renderer rejects at runtime; the enum catches it at config
    // validation time.
    assert.ok(A2uiOptionsSchema.safeParse({ version: 'v0.9' }).success);
    assert.ok(A2uiOptionsSchema.safeParse({ version: 'v0.9.1' }).success);
    assert.ok(!A2uiOptionsSchema.safeParse({ version: 'v0.10' }).success);
  });

  it('preserves prose ordering around a block in the final message', async () => {
    const mw = modelHook({ surfaceId: 'sfc' });
    const mixed =
      'intro\n' + SAMPLE_TEXT.replace('Here is the weather:\n', '') + 'outro';
    const res = await mw(req('sys'), undefined, async () => ({
      message: { role: 'model', content: [{ text: mixed }] },
    }));
    const content = (res as any).message.content;
    // Expect three ordered parts: prose("intro"), a2ui, prose("outro").
    assert.strictEqual(content.length, 3);
    assert.match(content[0].text, /intro/);
    assert.ok(isA2uiPart(content[1]));
    assert.match(content[2].text, /outro/);
  });
});

describe('a2ui() end-to-end via ai.generate', () => {
  it('rewrites an echo model’s a2ui block into an a2ui part', async () => {
    const ai = genkit({});
    const model = ai.defineModel({ name: 'echo' }, async (request) => {
      // Echo back a fixed mixed prose + a2ui block, ignoring the request.
      return {
        message: { role: 'model', content: [{ text: SAMPLE_TEXT }] },
        finishReason: 'stop',
      };
    });

    const res = await ai.generate({
      model,
      prompt: 'weather please',
      use: [a2ui({ surfaceId: 'sfc' })],
    });

    // The catalog instructions were injected into the request the model saw.
    assert.match(res.text, /Here is the weather/);
    const envelopes = a2uiEnvelopesFromParts(res.message!.content);
    assert.strictEqual(envelopes.length, 2);
    assert.strictEqual((envelopes[0] as any).createSurface.surfaceId, 'sfc');
  });
});
