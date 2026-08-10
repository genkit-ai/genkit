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

import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit, type GenkitBeta } from '../src/beta';
import {
  backgroundModel,
  embedderActionMetadata,
  genkitPlugin,
  genkitPluginV2,
  model,
  modelActionMetadata,
} from '../src/plugin';

const v1Plugin = genkitPlugin(
  'myV1Plugin',
  (ai) => {
    ai.defineModel({ name: 'myV1Plugin/model_eager' }, async () => {
      return {};
    });
  },
  async (ai, actionType, name) => {
    switch (actionType) {
      case 'model':
        ai.defineModel({ name: 'myV1Plugin/' + name }, async () => {
          return {};
        });
      case 'background-model':
        ai.defineBackgroundModel({
          name: 'myV1Plugin/' + name,
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
  },
  async () => {
    return [
      modelActionMetadata({
        name: 'myV1Plugin/potential_model',
      }),
      embedderActionMetadata({
        name: 'myV1Plugin/potential_embedder',
      }),
    ];
  }
);

const v2Plugin = genkitPluginV2({
  name: 'myV2Plugin',
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
        name: 'potential_embedder',
      }),
    ];
  },
});

describe('session', () => {
  let ai: GenkitBeta;

  beforeEach(() => {
    ai = genkit({
      model: 'echoModel',
      plugins: [v1Plugin, v2Plugin],
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
        '/embedder/myV1Plugin/potential_embedder',
        '/embedder/myV2Plugin/potential_embedder',
        '/model/myV1Plugin/potential_model',
        '/model/myV1Plugin/model_eager',
        '/model/myV2Plugin/model_eager',
        '/model/myV2Plugin/potential_model',
      ])
    );
    assert.deepStrictEqual(
      new Set(
        Object.keys(await ai.registry.listActions()).filter(
          (a) => a.startsWith('/model') || a.startsWith('/embedder')
        )
      ),
      new Set([
        '/model/myV1Plugin/model_eager',
        '/model/myV2Plugin/model_eager',
      ])
    );
  });

  it('resolves from v1 plugin', async () => {
    const act = await ai.registry.lookupAction(
      '/model/myV1Plugin/potential_model'
    );
    assert.ok(act);
    assert.strictEqual(act.__action.name, 'myV1Plugin/potential_model');
    assert.strictEqual(act.__action.actionType, 'model');

    assert.deepStrictEqual(
      new Set(
        Object.keys(await ai.registry.listActions()).filter(
          (a) => a.startsWith('/model') || a.startsWith('/embedder')
        )
      ),
      new Set([
        '/model/myV1Plugin/model_eager',
        '/model/myV2Plugin/model_eager',
        '/model/myV1Plugin/potential_model',
      ])
    );
  });

  it('resolves background model from v1 plugin', async () => {
    const act = await ai.registry.lookupAction(
      '/background-model/myV1Plugin/bg-model'
    );
    assert.ok(act);
    assert.strictEqual(act.__action.name, 'myV1Plugin/bg-model');
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
        '/background-model/myV1Plugin/bg-model',
        '/check-operation/myV1Plugin/bg-model/check',
        '/cancel-operation/myV1Plugin/bg-model/cancel',
      ])
    );
  });

  it('resolves from v2 plugin', async () => {
    const act = await ai.registry.lookupAction(
      '/model/myV2Plugin/potential_model'
    );
    assert.ok(act);
    assert.strictEqual(act.__action.name, 'myV2Plugin/potential_model');
    assert.strictEqual(act.__action.actionType, 'model');

    assert.deepStrictEqual(
      new Set(
        Object.keys(await ai.registry.listActions()).filter(
          (a) => a.startsWith('/model') || a.startsWith('/embedder')
        )
      ),
      new Set([
        '/model/myV1Plugin/model_eager',
        '/model/myV2Plugin/model_eager',
        '/model/myV2Plugin/potential_model',
      ])
    );
  });

  it('resolves background model from v2 plugin', async () => {
    const act = await ai.registry.lookupAction(
      '/background-model/myV2Plugin/bg-model'
    );
    assert.ok(act);
    assert.strictEqual(act.__action.name, 'myV2Plugin/bg-model');
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
        '/background-model/myV2Plugin/bg-model',
        '/check-operation/myV2Plugin/bg-model/check',
        '/cancel-operation/myV2Plugin/bg-model/cancel',
      ])
    );
  });

  it('resolves model using model resolve helper', async () => {
    const act = await v2Plugin.model('foo-model');
    assert.ok(act);
    assert.strictEqual(act.__action.name, 'foo-model');
    assert.strictEqual(act.__action.actionType, 'model');

    await assert.rejects(v2Plugin.model('not-found'), {
      name: 'GenkitError',
      status: 'NOT_FOUND',
    });
  });
});
