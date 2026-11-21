/**
 * Copyright 2025 Google LLC
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

import { GenkitError } from '@genkit-ai/core';
import { logger } from '@genkit-ai/core/logging';
import * as assert from 'assert';
import { beforeEach, describe, it, mock } from 'node:test';
import { genkit, type GenkitBeta } from '../src/beta';
import { tool } from '../src/index';
import {
  backgroundModel,
  embedderActionMetadata,
  evaluator,
  genkitPlugin,
  model,
  modelActionMetadata,
  type GenkitPlugin,
} from '../src/plugin';
import { defineEchoModel } from './helpers';

const myPlugin = genkitPlugin({
  name: 'myPlugin',
  init() {
    return [
      model({ name: 'model_eager' }, async () => {
        return {};
      }),
    ];
  },
  resolve(actionType, name) {
    switch (actionType) {
      case 'model':
        if (name === 'not-found') {
          return undefined;
        }
        return model({ name }, async () => {
          return {};
        });
      case 'background-model':
        return backgroundModel({
          name,
          async start() {
            return { id: 'abc' };
          },
          async check(operation) {
            return operation;
          },
          async cancel(operation) {
            return operation;
          },
        });
    }
    return undefined;
  },
  list() {
    return [
      modelActionMetadata({
        name: 'potential_model',
      }),
      embedderActionMetadata({
        // Already-namespaced names are kept as is.
        name: 'myPlugin/potential_embedder',
      }),
    ];
  },
});

describe('plugins', () => {
  let ai: GenkitBeta;

  beforeEach(() => {
    ai = genkit({
      model: 'echoModel',
      plugins: [myPlugin],
    });
  });

  it('lists actions', async () => {
    assert.deepStrictEqual(
      new Set(
        Object.keys(await ai.registry.listResolvableActions()).filter(
          (a) => a.startsWith('/model') || a.startsWith('/embedder')
        )
      ),
      new Set([
        '/embedder/myPlugin/potential_embedder',
        '/model/myPlugin/model_eager',
        '/model/myPlugin/potential_model',
      ])
    );
    assert.deepStrictEqual(
      new Set(
        Object.keys(await ai.registry.listActions()).filter(
          (a) => a.startsWith('/model') || a.startsWith('/embedder')
        )
      ),
      new Set(['/model/myPlugin/model_eager'])
    );
  });

  it('resolves actions', async () => {
    const act = await ai.registry.lookupAction(
      '/model/myPlugin/potential_model'
    );
    assert.ok(act);
    assert.strictEqual(act.__action.name, 'myPlugin/potential_model');
    assert.strictEqual(act.__action.actionType, 'model');

    assert.deepStrictEqual(
      new Set(
        Object.keys(await ai.registry.listActions()).filter(
          (a) => a.startsWith('/model') || a.startsWith('/embedder')
        )
      ),
      new Set([
        '/model/myPlugin/model_eager',
        '/model/myPlugin/potential_model',
      ])
    );
  });

  it('returns undefined for actions the plugin does not resolve', async () => {
    assert.strictEqual(
      await ai.registry.lookupAction('/model/myPlugin/not-found'),
      undefined
    );
  });

  it('resolves background models', async () => {
    const act = await ai.registry.lookupAction(
      '/background-model/myPlugin/bg-model'
    );
    assert.ok(act);
    assert.strictEqual(act.__action.name, 'myPlugin/bg-model');
    assert.strictEqual(act.__action.actionType, 'background-model');

    assert.deepStrictEqual(
      new Set(
        Object.keys(await ai.registry.listActions()).filter(
          (a) =>
            a.startsWith('/background-model') ||
            a.startsWith('/check-operation') ||
            a.startsWith('/cancel-operation')
        )
      ),
      new Set([
        '/background-model/myPlugin/bg-model',
        '/check-operation/myPlugin/bg-model/check',
        '/cancel-operation/myPlugin/bg-model/cancel',
      ])
    );
  });
});

describe('plugin context', () => {
  it('passes un-namespaced names to resolve', async () => {
    const names: string[] = [];
    const plugin = genkitPlugin({
      name: 'ns',
      resolve: (actionType, name) => {
        names.push(name);
        return actionType === 'model'
          ? model({ name }, async () => ({}))
          : undefined;
      },
    });
    const ai = genkit({ plugins: [plugin] });
    assert.ok(await ai.registry.lookupAction('/model/ns/foo'));
    assert.deepStrictEqual(names, ['foo']);
  });

  it('gives actions an ai bound to the registry', async () => {
    const judge = genkitPlugin({
      name: 'judge',
      init: ({ ai }) => [
        evaluator(
          {
            name: 'echo',
            displayName: 'echo',
            definition: 'Scores with whatever the model says.',
          },
          async (dataPoint) => {
            const { text } = await ai.generate({
              model: 'echoModel',
              prompt: String(dataPoint.output),
            });
            return {
              testCaseId: dataPoint.testCaseId,
              evaluation: { score: text },
            };
          }
        ),
      ],
    });
    const ai = genkit({ plugins: [judge] });
    defineEchoModel(ai);
    const [response] = await ai.evaluate({
      evaluator: 'judge/echo',
      dataset: [{ testCaseId: 't1', input: 'q', output: 'hello' }],
    });
    const evaluation = response.evaluation;
    assert.ok(!Array.isArray(evaluation));
    assert.match(String(evaluation.score), /^Echo: hello/);
  });

  it('does not expose define* on ctx.ai', async () => {
    let ctxAi: unknown;
    const plugin = genkitPlugin({
      name: 'peek',
      init: ({ ai }) => {
        ctxAi = ai;
        return [];
      },
    });
    const ai = genkit({ plugins: [plugin] });
    await ai.registry.initializeAllPlugins();
    assert.ok(ctxAi);
    assert.notStrictEqual(ctxAi, ai);
    assert.strictEqual(
      (ctxAi as Record<string, unknown>)['defineModel'],
      undefined
    );
  });
});

describe('v1 plugins', () => {
  it('genkitPlugin rejects the v1 call shape', () => {
    // Simulates an un-migrated JS caller.
    const v1GenkitPlugin = genkitPlugin as unknown as (
      name: string,
      initFn: () => void
    ) => GenkitPlugin;
    assert.throws(
      () => v1GenkitPlugin('legacy', () => {}),
      (e: unknown) =>
        e instanceof GenkitError &&
        e.status === 'INVALID_ARGUMENT' &&
        e.message.includes('genkitPlugin({ name, init })')
    );
  });

  it('genkit() rejects function plugins', () => {
    const v1Plugin = (() => ({
      name: 'legacy',
      initializer: () => {},
    })) as unknown as GenkitPlugin;
    assert.throws(
      () => genkit({ plugins: [v1Plugin] }),
      (e: unknown) =>
        e instanceof GenkitError &&
        e.status === 'INVALID_ARGUMENT' &&
        e.message.includes('plugins[0]') &&
        e.message.includes('removed in Genkit 2.0')
    );
  });
});

describe('plugin tools', () => {
  it('are not registered again when passed to generate', async () => {
    const toolPlugin = genkitPlugin({
      name: 'toolPlugin',
      init: () => [
        tool({ name: 'eager', description: 'eager' }, async () => 'ok'),
        tool(
          { name: 'multipart', description: 'multipart', multipart: true },
          async () => ({ output: 'ok' })
        ),
      ],
      resolve: (actionType, name) =>
        actionType === 'tool'
          ? tool({ name, description: name }, async () => 'ok')
          : undefined,
    });
    const ai = genkit({ model: 'echoModel', plugins: [toolPlugin] });
    defineEchoModel(ai);
    const tools = await Promise.all(
      [
        '/tool/toolPlugin/eager',
        '/tool.v2/toolPlugin/multipart',
        '/tool/toolPlugin/lazy',
      ].map((key) => ai.registry.lookupAction(key))
    );
    const errors = mock.method(logger, 'error');

    try {
      await ai.generate({ prompt: 'hi', tools });
      await ai.generate({ prompt: 'hi', tools });
    } finally {
      errors.mock.restore();
    }

    assert.deepStrictEqual(
      errors.mock.calls.map((c) => c.arguments),
      []
    );
    assert.ok(tools.every((t) => t.__registry === ai.registry));
  });
});
