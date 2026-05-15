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

import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import {
  defineResource,
  dynamicResource,
  findMatchingResource,
  isDynamicResourceAction,
  resource,
} from '../../src/resource.js';
import { defineEchoModel } from '../helpers.js';

initNodeFeatures();

describe('resource', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
  });

  it('defines and matches static resource uri', async () => {
    const testResource = defineResource(
      registry,
      {
        name: 'testResource',
        uri: 'foo://bar',
        description: 'does foo things',
        metadata: { foo: 'bar' },
      },
      () => {
        return { content: [{ text: 'foo stuff' }] };
      }
    );

    assert.ok(testResource);
    assert.strictEqual(testResource.__action.description, 'does foo things');
    assert.deepStrictEqual(testResource.__action.metadata, {
      foo: 'bar',
      resource: {
        template: undefined,
        uri: 'foo://bar',
      },
      type: 'resource',
      dynamic: false,
    });

    assert.strictEqual(testResource.matches({ uri: 'foo://bar' }), true);
    assert.strictEqual(testResource.matches({ uri: 'foo://baz' }), false);

    assert.deepStrictEqual(await testResource({ uri: 'foo://bar' }), {
      content: [
        {
          text: 'foo stuff',
          metadata: {
            resource: {
              uri: 'foo://bar',
            },
          },
        },
      ],
    });

    assert.ok(await registry.lookupAction('/resource/testResource'));
  });

  it('defines and matches templates resource uri', async () => {
    const testResource = defineResource(
      registry,
      {
        template: 'foo://bar/{baz}',
        description: 'does foo things',
      },
      (input) => {
        return { content: [{ text: `foo stuff ${input.uri}` }] };
      }
    );

    assert.ok(testResource);
    assert.strictEqual(testResource.__action.name, 'foo://bar/{baz}');
    assert.strictEqual(testResource.__action.description, 'does foo things');

    assert.strictEqual(
      testResource.matches({ uri: 'foo://bar/something' }),
      true
    );
    assert.strictEqual(
      testResource.matches({ uri: 'foo://baz/something' }),
      false
    );

    assert.deepStrictEqual(await testResource({ uri: 'foo://bar/something' }), {
      content: [
        {
          text: 'foo stuff foo://bar/something',
          metadata: {
            resource: {
              template: 'foo://bar/{baz}',
              uri: 'foo://bar/something',
            },
          },
        },
      ],
    });

    assert.ok(await registry.lookupAction('/resource/foo://bar/{baz}'));
  });

  it('handle parent resources', async () => {
    const testResource = defineResource(
      registry,
      {
        name: 'testResource',
        template: 'file://{/id*}',
        description: 'does foo things',
      },
      (file) => {
        return {
          content: [
            {
              text: `sub1`,
              metadata: { resource: { uri: `${file.uri}/sub1.txt` } },
            },
            {
              text: `sub2`,
              metadata: { resource: { uri: `${file.uri}/sub2.txt` } },
            },
          ],
        };
      }
    );
    assert.strictEqual(
      testResource.matches({ uri: 'file:///some/directory' }),
      true
    );

    assert.deepStrictEqual(
      await testResource({ uri: 'file:///some/directory' }),
      {
        content: [
          {
            metadata: {
              resource: {
                parent: {
                  template: 'file://{/id*}',
                  uri: 'file:///some/directory',
                },
                uri: 'file:///some/directory/sub1.txt',
              },
            },
            text: 'sub1',
          },
          {
            metadata: {
              resource: {
                parent: {
                  template: 'file://{/id*}',
                  uri: 'file:///some/directory',
                },
                uri: 'file:///some/directory/sub2.txt',
              },
            },
            text: 'sub2',
          },
        ],
      }
    );
  });

  it('finds matching resource', async () => {
    defineResource(
      registry,
      {
        name: 'testTemplateResource',
        template: 'foo://bar/{baz}',
        description: 'does foo things',
      },
      (input) => {
        return { content: [{ text: `foo stuff ${input.uri}` }] };
      }
    );
    defineResource(
      registry,
      {
        name: 'testResource',
        uri: 'bar://baz',
        description: 'does bar things',
      },
      () => {
        return { content: [{ text: `bar` }] };
      }
    );
    const resList = [
      dynamicResource({ uri: 'baz://qux' }, () => ({
        content: [{ text: `baz` }],
      })),
    ];

    const gotBar = await findMatchingResource(registry, resList, {
      uri: 'bar://baz',
    });
    assert.ok(gotBar);
    assert.strictEqual(gotBar.__action.name, 'testResource');

    const gotFoo = await findMatchingResource(registry, resList, {
      uri: 'foo://bar/something',
    });
    assert.ok(gotFoo);
    assert.strictEqual(gotFoo.__action.name, 'testTemplateResource');
    assert.deepStrictEqual(gotFoo.__action.metadata, {
      resource: {
        template: 'foo://bar/{baz}',
        uri: undefined,
      },
      type: 'resource',
      dynamic: false,
    });

    const gotBaz = await findMatchingResource(registry, resList, {
      uri: 'baz://qux',
    });
    assert.ok(gotBaz);
    assert.strictEqual(gotBaz.__action.name, 'baz://qux');
    assert.deepStrictEqual(gotBaz.__action.metadata, {
      resource: {
        template: undefined,
        uri: 'baz://qux',
      },
      type: 'resource',
      dynamic: true,
    });

    const gotUnmatched = await findMatchingResource(registry, resList, {
      uri: 'unknown://bar/something',
    });
    assert.strictEqual(gotUnmatched, undefined);
  });
});

describe('isDynamicResourceAction', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
  });

  it('should recognize dynamic resource actions', () => {
    assert.strictEqual(
      isDynamicResourceAction(
        defineResource(registry, { uri: 'bar://baz' }, () => ({
          content: [{ text: `bar` }],
        }))
      ),
      false
    );

    assert.strictEqual(
      isDynamicResourceAction(defineEchoModel(registry)),
      false
    );

    assert.strictEqual(isDynamicResourceAction('banana'), false);

    assert.strictEqual(
      isDynamicResourceAction(
        dynamicResource({ uri: 'bar://baz' }, () => ({
          content: [{ text: `bar` }],
        }))
      ),
      true
    );
  });

  it('should remain dynamic after registration', () => {
    const dynamic = resource({ uri: 'bar://baz' }, () => ({
      content: [{ text: `bar` }],
    }));
    assert.strictEqual(isDynamicResourceAction(dynamic), true);

    registry.registerAction('resource', dynamic);

    assert.strictEqual(isDynamicResourceAction(dynamic), true);
  });
});
