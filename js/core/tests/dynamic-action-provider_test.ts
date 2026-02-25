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
import { setTimeout } from 'timers/promises';
import { z } from 'zod';
import { Action, defineAction } from '../src/action.js';
import {
  defineDynamicActionProvider,
  isDynamicActionProvider,
} from '../src/dynamic-action-provider.js';
import { initNodeFeatures } from '../src/node.js';
import { Registry } from '../src/registry.js';

initNodeFeatures();

describe('dynamic action provider', () => {
  let registry: Registry;
  let tool1: Action<z.ZodTypeAny, z.ZodTypeAny, z.ZodTypeAny>;
  let tool2: Action<z.ZodTypeAny, z.ZodTypeAny, z.ZodTypeAny>;

  beforeEach(() => {
    registry = new Registry();
    tool1 = defineAction(
      registry,
      { name: 'tool1', actionType: 'tool' },
      async () => 'tool1'
    );
    tool2 = defineAction(
      registry,
      { name: 'tool2', actionType: 'tool' },
      async () => 'tool2'
    );
  });

  it('gets a specific action', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      return {
        tool: [tool1, tool2],
      };
    });

    const action = await dap.getAction('tool', 'tool1');
    assert.strictEqual(action, tool1);
    assert.strictEqual(callCount, 1);
  });

  it('lists action metadata', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      return {
        tool: [tool1, tool2],
      };
    });

    const metadata = await dap.listActionMetadata('tool', '*');
    assert.deepStrictEqual(metadata, [tool1.__action, tool2.__action]);
    assert.strictEqual(callCount, 1);
  });

  it('caches the actions', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      return {
        tool: [tool1, tool2],
      };
    });

    let action = await dap.getAction('tool', 'tool1');
    assert.strictEqual(action, tool1);
    assert.strictEqual(callCount, 1);

    // This should be cached
    action = await dap.getAction('tool', 'tool2');
    assert.strictEqual(action, tool2);
    assert.strictEqual(callCount, 1);

    const metadata = await dap.listActionMetadata('tool', '*');
    assert.deepStrictEqual(metadata, [tool1.__action, tool2.__action]);
    assert.strictEqual(callCount, 1);
  });

  it('invalidates the cache', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      return {
        tool: [tool1, tool2],
      };
    });

    await dap.getAction('tool', 'tool1');
    assert.strictEqual(callCount, 1);

    dap.invalidateCache();

    await dap.getAction('tool', 'tool2');
    assert.strictEqual(callCount, 2);
  });

  it('respects cache ttl', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(
      registry,
      { name: 'my-dap', cacheConfig: { ttlMillis: 10 } },
      async () => {
        callCount++;
        return {
          tool: [tool1, tool2],
        };
      }
    );

    await dap.getAction('tool', 'tool1');
    assert.strictEqual(callCount, 1);

    await setTimeout(20);

    await dap.getAction('tool', 'tool2');
    assert.strictEqual(callCount, 2);
  });

  it('lists actions with prefix', async () => {
    let callCount = 0;
    const tool3 = defineAction(
      registry,
      { name: 'other-tool', actionType: 'tool' },
      async () => 'other'
    );
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      return {
        tool: [tool1, tool2, tool3],
      };
    });

    const metadata = await dap.listActionMetadata('tool', 'tool*');
    assert.deepStrictEqual(metadata, [tool1.__action, tool2.__action]);
    assert.strictEqual(callCount, 1);
  });

  it('lists actions with exact match', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      return {
        tool: [tool1, tool2],
      };
    });

    const metadata = await dap.listActionMetadata('tool', 'tool1');
    assert.deepStrictEqual(metadata, [tool1.__action]);
    assert.strictEqual(callCount, 1);
  });

  it('gets action metadata record', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      return {
        tool: [tool1, tool2],
        flow: [tool1],
      };
    });

    const record = await dap.getActionMetadataRecord('dap/my-dap');
    assert.deepStrictEqual(record, {
      'dap/my-dap:tool/tool1': tool1.__action,
      'dap/my-dap:tool/tool2': tool2.__action,
      'dap/my-dap:flow/tool1': tool1.__action,
    });
    assert.strictEqual(callCount, 1);
  });

  it('handles concurrent requests', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      await setTimeout(10);
      return {
        tool: [tool1, tool2],
      };
    });

    const [metadata1, metadata2] = await Promise.all([
      dap.listActionMetadata('tool', '*'),
      dap.listActionMetadata('tool', '*'),
    ]);

    assert.deepStrictEqual(metadata1, [tool1.__action, tool2.__action]);
    assert.deepStrictEqual(metadata2, [tool1.__action, tool2.__action]);
    assert.strictEqual(callCount, 1);
  });

  it('handles fetch errors', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      if (callCount === 1) {
        throw new Error('Fetch failed');
      }
      return {
        tool: [tool1, tool2],
      };
    });

    await assert.rejects(dap.listActionMetadata('tool', '*'), /Fetch failed/);
    assert.strictEqual(callCount, 1);

    const metadata = await dap.listActionMetadata('tool', '*');
    assert.deepStrictEqual(metadata, [tool1.__action, tool2.__action]);
    assert.strictEqual(callCount, 2);
  });

  it('runs the action with transformed metadata when fetching', async () => {
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      return {
        tool: [tool1, tool2],
      };
    });

    let runResult: any;
    const originalRun = dap.run.bind(dap);
    dap.run = async (input, options) => {
      runResult = await originalRun(input, options);
      return runResult;
    };

    await dap.__cache.getOrFetch();

    assert.deepStrictEqual(runResult.result, [tool1.__action, tool2.__action]);
  });

  it('skips trace when requested', async () => {
    let callCount = 0;
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      callCount++;
      return {
        tool: [tool1, tool2],
      };
    });

    const originalRun = dap.run.bind(dap);
    let runCalled = false;
    dap.run = async (input, options) => {
      runCalled = true;
      return originalRun(input, options);
    };

    await dap.__cache.getOrFetch({ skipTrace: true });
    assert.strictEqual(runCalled, false);
    assert.strictEqual(callCount, 1);

    dap.invalidateCache();

    await dap.__cache.getOrFetch();
    assert.strictEqual(runCalled, true);
    assert.strictEqual(callCount, 2);
  });

  it('identifies dynamic action providers', async () => {
    const dap = defineDynamicActionProvider(registry, 'my-dap', async () => {
      return {};
    });
    assert.ok(isDynamicActionProvider(dap));

    const regularAction = defineAction(
      registry,
      { name: 'regular', actionType: 'tool' },
      async () => {}
    );
    assert.ok(!isDynamicActionProvider(regularAction));
  });
});
