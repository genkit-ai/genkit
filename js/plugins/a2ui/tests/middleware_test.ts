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

import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  A2UI_CATALOG_VALUE_TYPE,
  basicCatalog,
  type A2uiCatalog,
} from '../src/catalog.js';
import { a2ui, type A2uiOptions } from '../src/middleware.js';
import { a2uiEnvelopes, isA2uiPart } from '../src/part.js';

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
    const mw = modelHook({ surfaceId: () => 'sfc' });
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
    const envelopes = a2uiEnvelopes({ content });
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
    assert.strictEqual(a2uiEnvelopes({ content }).length, 0);
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
                data: [
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

  it('transforms streamed chunks via onChunk', async () => {
    const mw = modelHook({ surfaceId: () => 'sfc' });
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
    const allEnvelopes = emitted.flatMap((c) => a2uiEnvelopes(c));
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
});
