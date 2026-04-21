/**
 * Copyright 2024 Google LLC
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
import {
  action,
  defineAction,
  runInActionRuntimeContext,
} from '../src/action.js';
import { defineDynamicActionProvider } from '../src/dynamic-action-provider.js';
import { initNodeFeatures } from '../src/node.js';
import { Registry } from '../src/registry.js';

initNodeFeatures();

describe('registry class', () => {
  var registry: Registry;
  beforeEach(() => {
    registry = new Registry();
  });

  describe('listActions', () => {
    it('returns all registered actions', async () => {
      const fooSomethingAction = action(
        { name: 'foo_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', fooSomethingAction);
      const barSomethingAction = action(
        { name: 'bar_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', barSomethingAction);

      assert.deepEqual(await registry.listActions(), {
        '/model/foo_something': fooSomethingAction,
        '/model/bar_something': barSomethingAction,
      });
    });

    it('returns all registered actions by plugins', async () => {
      registry.registerPluginProvider('foo', {
        name: 'foo',
        async initializer() {
          registry.registerAction('model', fooSomethingAction);
          return {};
        },
      });
      const fooSomethingAction = action(
        {
          name: {
            pluginId: 'foo',
            actionId: 'something',
          },
          actionType: 'model',
        },
        async () => null
      );
      registry.registerPluginProvider('bar', {
        name: 'bar',
        async initializer() {
          registry.registerAction('model', barSomethingAction);
          registry.registerAction('model', barSubSomethingAction);
          return {};
        },
      });
      const barSomethingAction = action(
        {
          name: {
            pluginId: 'bar',
            actionId: 'something',
          },
          actionType: 'model',
        },
        async () => null
      );
      const barSubSomethingAction = action(
        {
          name: {
            pluginId: 'bar',
            actionId: 'sub/something',
          },
          actionType: 'model',
        },
        async () => null
      );

      assert.deepEqual(await registry.listActions(), {
        '/model/foo/something': fooSomethingAction,
        '/model/bar/something': barSomethingAction,
        '/model/bar/sub/something': barSubSomethingAction,
      });
    });

    it('should allow plugin initialization from runtime context', async () => {
      let fooInitialized = false;
      registry.registerPluginProvider('foo', {
        name: 'foo',
        async initializer() {
          defineAction(
            registry,
            {
              actionType: 'model',
              name: 'foo/something',
            },
            async () => null
          );
          fooInitialized = true;
          return {};
        },
      });

      const action = await runInActionRuntimeContext(() =>
        registry.lookupAction('/model/foo/something')
      );

      assert.ok(action);
      assert.ok(fooInitialized);
    });

    it('returns all registered actions, including parent', async () => {
      const child = Registry.withParent(registry);

      const fooSomethingAction = action(
        { name: 'foo_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', fooSomethingAction);
      const barSomethingAction = action(
        { name: 'bar_something', actionType: 'model' },
        async () => null
      );
      child.registerAction('model', barSomethingAction);

      assert.deepEqual(await child.listActions(), {
        '/model/foo_something': fooSomethingAction,
        '/model/bar_something': barSomethingAction,
      });
      assert.deepEqual(await registry.listActions(), {
        '/model/foo_something': fooSomethingAction,
      });
    });
  });

  describe('listResolvableActions', () => {
    it('returns all registered actions', async () => {
      const fooSomethingAction = action(
        { name: 'foo_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', fooSomethingAction);
      const barSomethingAction = action(
        { name: 'bar_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', barSomethingAction);

      assert.deepEqual(await registry.listResolvableActions(), {
        '/model/foo_something': fooSomethingAction.__action,
        '/model/bar_something': barSomethingAction.__action,
      });
    });

    it('returns all registered actions by plugins', async () => {
      registry.registerPluginProvider('foo', {
        name: 'foo',
        async initializer() {
          registry.registerAction('model', fooSomethingAction);
          return {};
        },
      });
      const fooSomethingAction = action(
        {
          name: {
            pluginId: 'foo',
            actionId: 'something',
          },
          actionType: 'model',
        },
        async () => null
      );
      registry.registerPluginProvider('bar', {
        name: 'bar',
        async initializer() {
          registry.registerAction('model', barSomethingAction);
          registry.registerAction('model', barSubSomethingAction);
          return {};
        },
      });
      const barSomethingAction = action(
        {
          name: {
            pluginId: 'bar',
            actionId: 'something',
          },
          actionType: 'model',
        },
        async () => null
      );
      const barSubSomethingAction = action(
        {
          name: {
            pluginId: 'bar',
            actionId: 'sub/something',
          },
          actionType: 'model',
        },
        async () => null
      );

      assert.deepEqual(await registry.listResolvableActions(), {
        '/model/foo/something': fooSomethingAction.__action,
        '/model/bar/something': barSomethingAction.__action,
        '/model/bar/sub/something': barSubSomethingAction.__action,
      });
    });

    it('should allow plugin initialization from runtime context', async () => {
      let fooInitialized = false;
      registry.registerPluginProvider('foo', {
        name: 'foo',
        async initializer() {
          defineAction(
            registry,
            {
              actionType: 'model',
              name: 'foo/something',
            },
            async () => null
          );
          fooInitialized = true;
          return {};
        },
      });

      const action = await runInActionRuntimeContext(() =>
        registry.lookupAction('/model/foo/something')
      );

      assert.ok(action);
      assert.ok(fooInitialized);
    });

    it('returns all registered actions, including parent', async () => {
      const child = Registry.withParent(registry);

      const fooSomethingAction = action(
        { name: 'foo_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', fooSomethingAction);
      const barSomethingAction = action(
        { name: 'bar_something', actionType: 'model' },
        async () => null
      );
      child.registerAction('model', barSomethingAction);

      assert.deepEqual(await child.listResolvableActions(), {
        '/model/foo_something': fooSomethingAction.__action,
        '/model/bar_something': barSomethingAction.__action,
      });
      assert.deepEqual(await registry.listResolvableActions(), {
        '/model/foo_something': fooSomethingAction.__action,
      });
    });

    it('returns all registered actions and ones returned by listActions by plugins', async () => {
      registry.registerPluginProvider('foo', {
        name: 'foo',
        async initializer() {
          registry.registerAction('model', fooSomethingAction);
          return {};
        },
      });
      const fooSomethingAction = action(
        {
          name: {
            pluginId: 'foo',
            actionId: 'something',
          },
          actionType: 'model',
        },
        async () => null
      );
      registry.registerPluginProvider('bar', {
        name: 'bar',
        async initializer() {
          registry.registerAction('model', barSomethingAction);
          registry.registerAction('model', barSubSomethingAction);
          return {};
        },
        async listActions() {
          return [
            {
              name: 'bar/barDynamicallyResolvable',
              actionType: 'model',
              description: 'sings a song',
            },
          ];
        },
      });
      const barSomethingAction = action(
        {
          name: {
            pluginId: 'bar',
            actionId: 'something',
          },
          actionType: 'model',
        },
        async () => null
      );
      const barSubSomethingAction = action(
        {
          name: {
            pluginId: 'bar',
            actionId: 'sub/something',
          },
          actionType: 'model',
        },
        async () => null
      );

      assert.deepEqual(await registry.listResolvableActions(), {
        '/model/foo/something': fooSomethingAction.__action,
        '/model/bar/something': barSomethingAction.__action,
        '/model/bar/sub/something': barSubSomethingAction.__action,
        '/model/bar/barDynamicallyResolvable': {
          name: 'bar/barDynamicallyResolvable',
          actionType: 'model',
          description: 'sings a song',
        },
      });
    });
    it('expands actions from dynamic action providers', async () => {
      const tool1 = action(
        { name: 'fs/tool1', actionType: 'tool' },
        async () => {}
      );
      const tool2 = action(
        { name: 'fs/tool2', actionType: 'tool' },
        async () => {}
      );
      const resource1 = action(
        { name: 'abc/res1', actionType: 'resource' },
        async () => {}
      );
      const resource2 = action(
        { name: 'abc/res2', actionType: 'resource' },
        async () => {}
      );
      const dap = defineDynamicActionProvider(registry, 'my-dap', async () => ({
        tool: [tool1, tool2],
        resource: [resource1, resource2],
      }));

      const resolvableActions = await registry.listResolvableActions();
      assert.deepStrictEqual(resolvableActions, {
        '/dynamic-action-provider/my-dap': dap.__action,
        '/dynamic-action-provider/my-dap:tool/fs/tool1': tool1.__action,
        '/dynamic-action-provider/my-dap:tool/fs/tool2': tool2.__action,
        '/dynamic-action-provider/my-dap:resource/abc/res1': resource1.__action,
        '/dynamic-action-provider/my-dap:resource/abc/res2': resource2.__action,
      });
    });
  });

  describe('lookupAction', () => {
    it('initializes plugin for action first', async () => {
      let fooInitialized = false;
      registry.registerPluginProvider('foo', {
        name: 'foo',
        async initializer() {
          fooInitialized = true;
          return {};
        },
      });
      let barInitialized = false;
      registry.registerPluginProvider('bar', {
        name: 'bar',
        async initializer() {
          barInitialized = true;
          return {};
        },
      });

      await registry.lookupAction('/model/foo/something');

      assert.strictEqual(fooInitialized, true);
      assert.strictEqual(barInitialized, false);

      await registry.lookupAction('/model/bar/something');

      assert.strictEqual(fooInitialized, true);
      assert.strictEqual(barInitialized, true);
    });

    it('returns registered action', async () => {
      const fooSomethingAction = action(
        { name: 'foo_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', fooSomethingAction);
      const barSomethingAction = action(
        { name: 'bar_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', barSomethingAction);
      const barSubSomethingAction = action(
        { name: 'sub/bar_something', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', barSubSomethingAction);

      assert.strictEqual(
        await registry.lookupAction('/model/foo_something'),
        fooSomethingAction
      );
      assert.strictEqual(
        await registry.lookupAction('/model/bar_something'),
        barSomethingAction
      );
      assert.strictEqual(
        await registry.lookupAction('/model/sub/bar_something'),
        barSubSomethingAction
      );
    });

    it('returns registered action with namespace', async () => {
      const fooSomethingAction = action(
        { name: 'foo_something', actionType: 'model' },
        async () => null
      );
      assert.strictEqual(
        fooSomethingAction.__action.key,
        '/model/foo_something'
      );
      registry.registerAction('model', fooSomethingAction, {
        namespace: 'my-plugin',
      });

      const barSomethingAction = action(
        { name: 'my-plugin/bar_something', actionType: 'model' },
        async () => null
      );
      assert.strictEqual(
        barSomethingAction.__action.key,
        '/model/my-plugin/bar_something'
      );
      registry.registerAction('model', barSomethingAction, {
        namespace: 'my-plugin',
      });

      const barSubSomethingAction = action(
        { name: 'sub/bar_something', actionType: 'model' },
        async () => null
      );
      assert.strictEqual(
        barSubSomethingAction.__action.key,
        '/model/sub/bar_something'
      );
      registry.registerAction('model', barSubSomethingAction, {
        namespace: 'my-plugin',
      });

      assert.strictEqual(
        await registry.lookupAction('/model/my-plugin/foo_something'),
        fooSomethingAction
      );
      assert.strictEqual(
        fooSomethingAction.__action.key,
        '/model/my-plugin/foo_something'
      );
      assert.strictEqual(
        await registry.lookupAction('/model/my-plugin/bar_something'),
        barSomethingAction
      );
      assert.strictEqual(
        barSomethingAction.__action.key,
        '/model/my-plugin/bar_something'
      );
      assert.strictEqual(
        await registry.lookupAction('/model/my-plugin/sub/bar_something'),
        barSubSomethingAction
      );
      assert.strictEqual(
        barSubSomethingAction.__action.key,
        '/model/my-plugin/sub/bar_something'
      );
    });

    it('returns action registered by plugin', async () => {
      registry.registerPluginProvider('foo', {
        name: 'foo',
        async initializer() {
          registry.registerAction('model', somethingAction);
          registry.registerAction('model', subSomethingAction);
          return {};
        },
      });
      const somethingAction = action(
        {
          name: {
            pluginId: 'foo',
            actionId: 'something',
          },
          actionType: 'model',
        },
        async () => null
      );
      const subSomethingAction = action(
        {
          name: {
            pluginId: 'foo',
            actionId: 'sub/something',
          },
          actionType: 'model',
        },
        async () => null
      );

      assert.strictEqual(
        await registry.lookupAction('/model/foo/something'),
        somethingAction
      );

      assert.strictEqual(
        await registry.lookupAction('/model/foo/sub/something'),
        subSomethingAction
      );
    });

    it('returns action dynamically resolved by plugin', async () => {
      registry.registerPluginProvider('foo', {
        name: 'foo',
        async initializer() {},
        async resolver(actionType, actionName) {
          if (actionType !== 'model') {
            return;
          }
          switch (actionName) {
            case 'something':
              registry.registerAction('model', somethingAction);
              return;
            case 'sub/something':
              registry.registerAction('model', subSomethingAction);
              return;
          }
        },
      });
      const somethingAction = action(
        {
          name: {
            pluginId: 'foo',
            actionId: 'something',
          },
          actionType: 'model',
        },
        async () => null
      );
      const subSomethingAction = action(
        {
          name: {
            pluginId: 'foo',
            actionId: 'sub/something',
          },
          actionType: 'model',
        },
        async () => null
      );

      assert.strictEqual(
        await registry.lookupAction('/model/foo/something'),
        somethingAction
      );

      assert.strictEqual(
        await registry.lookupAction('/model/foo/sub/something'),
        subSomethingAction
      );
    });

    it('returns undefined for unknown action', async () => {
      assert.strictEqual(
        await registry.lookupAction('/model/foo/something'),
        undefined
      );
    });

    it('should lookup parent registry when child missing action', async () => {
      const childRegistry = new Registry(registry);

      const fooAction = action(
        { name: 'foo', actionType: 'model' },
        async () => null
      );
      registry.registerAction('model', fooAction);

      assert.strictEqual(await registry.lookupAction('/model/foo'), fooAction);
      assert.strictEqual(
        await childRegistry.lookupAction('/model/foo'),
        fooAction
      );
    });

    it('registration on the child registry should not modify parent', async () => {
      const childRegistry = Registry.withParent(registry);

      assert.strictEqual(childRegistry.parent, registry);

      const fooAction = action(
        { name: 'foo', actionType: 'model' },
        async () => null
      );
      childRegistry.registerAction('model', fooAction);

      assert.strictEqual(await registry.lookupAction('/model/foo'), undefined);
      assert.strictEqual(
        await childRegistry.lookupAction('/model/foo'),
        fooAction
      );
    });
  });
});
