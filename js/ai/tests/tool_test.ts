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

import { action, z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { afterEach, describe, it } from 'node:test';
import {
  defineInterrupt,
  defineTool,
  isDynamicTool,
  isMultipartTool,
  tool,
} from '../src/tool.js';

initNodeFeatures();

describe('defineInterrupt', () => {
  let registry = new Registry();
  registry.apiStability = 'beta';

  afterEach(() => {
    registry = new Registry();
    registry.apiStability = 'beta';
  });

  function expectInterrupt(fn: () => any, metadata?: Record<string, any>) {
    return assert.rejects(fn, { name: 'ToolInterruptError', metadata });
  }

  it('should throw a simple interrupt with no metadata', async () => {
    const simple = defineInterrupt(registry, {
      name: 'simple',
      description: 'simple interrupt',
    });
    await expectInterrupt(async () => {
      await simple({});
    });
  });

  it('should throw a simple interrupt with fixed metadata', async () => {
    const simple = defineInterrupt(registry, {
      name: 'simple',
      description: 'simple interrupt',
      requestMetadata: { foo: 'bar' },
    });
    await expectInterrupt(
      async () => {
        await simple({});
      },
      { foo: 'bar' }
    );
  });

  it('should throw a simple interrupt with function-returned metadata', async () => {
    const simple = defineInterrupt(registry, {
      name: 'simple',
      description: 'simple interrupt',
      inputSchema: z.string(),
      requestMetadata: (foo) => ({ foo }),
    });
    await expectInterrupt(
      async () => {
        await simple('bar');
      },
      { foo: 'bar' }
    );
  });

  it('should throw a simple interrupt with async function-returned metadata', async () => {
    const simple = defineInterrupt(registry, {
      name: 'simple',
      description: 'simple interrupt',
      inputSchema: z.string(),
      requestMetadata: async (foo) => ({ foo }),
    });
    await expectInterrupt(
      async () => {
        await simple('bar');
      },
      { foo: 'bar' }
    );
  });

  it('should register the reply schema / json schema as the output schema of the tool', () => {
    const ResponseSchema = z.object({ foo: z.string() });
    const simple = defineInterrupt(registry, {
      name: 'simple',
      description: 'simple',
      outputSchema: ResponseSchema,
    });
    assert.equal(simple.__action.outputSchema, ResponseSchema);
    const simple2 = defineInterrupt(registry, {
      name: 'simple2',
      description: 'simple2',
      outputJsonSchema: { type: 'string' },
    });
    assert.deepStrictEqual(simple2.__action.outputJsonSchema, {
      type: 'string',
    });
  });

  describe('multipart tools', () => {
    it('should define a multipart tool', async () => {
      const t = defineTool(
        registry,
        { name: 'test', description: 'test', multipart: true },
        async () => {
          return {
            output: 'main output',
            content: [{ text: 'part 1' }],
          };
        }
      );
      assert.equal(t.__action.metadata.type, 'tool.v2');
      assert.equal(t.__action.actionType, 'tool.v2');
      const result = await t({});
      assert.deepStrictEqual(result, {
        output: 'main output',
        content: [{ text: 'part 1' }],
      });
    });

    it('should handle fallback output', async () => {
      const t = defineTool(
        registry,
        { name: 'test', description: 'test', multipart: true },
        async () => {
          return {
            content: [{ text: 'part 1' }],
          };
        }
      );
      const result = await t({});
      assert.deepStrictEqual(result, {
        content: [{ text: 'part 1' }],
      });
    });
  });
});

describe('isDynamicTool', () => {
  let registry = new Registry();
  registry.apiStability = 'beta';
  afterEach(() => {
    registry = new Registry();
    registry.apiStability = 'beta';
  });

  it('should return true for a dynamic tool', () => {
    const dynamic = tool({ name: 'dynamic', description: 'test' });
    assert.strictEqual(isDynamicTool(dynamic), true);
  });

  it('should return true for a dynamic multipart tool', () => {
    const dynamic = tool({
      name: 'dynamic',
      description: 'test',
      multipart: true,
    });
    assert.strictEqual(isDynamicTool(dynamic), true);
  });

  it('should remain dynamic after registration', () => {
    const dynamic = tool({ name: 'dynamic', description: 'test' });
    assert.strictEqual(isDynamicTool(dynamic), true);

    registry.registerAction('tool', dynamic);

    assert.strictEqual(isDynamicTool(dynamic), true);
  });

  it('should return false for a registered tool', () => {
    const regular = defineTool(
      registry,
      { name: 'regular', description: 'test' },
      async () => {}
    );
    assert.strictEqual(isDynamicTool(regular), false);
  });

  it('should return false for a non-tool action', () => {
    const regularAction = action(
      { actionType: 'util', name: 'regularAction', description: 'test' },
      async () => {}
    );
    assert.strictEqual(isDynamicTool(regularAction), false);
  });

  it('should return false for a non-action', () => {
    assert.strictEqual(isDynamicTool({}), false);
    assert.strictEqual(isDynamicTool('tool'), false);
    assert.strictEqual(isDynamicTool(123), false);
  });
});

describe('isMultipartTool', () => {
  let registry = new Registry();
  registry.apiStability = 'beta';
  afterEach(() => {
    registry = new Registry();
    registry.apiStability = 'beta';
  });

  it('should return true for a multipart tool', () => {
    const multipart = defineTool(
      registry,
      { name: 'multipart', description: 'test', multipart: true },
      async () => {}
    );
    assert.strictEqual(isMultipartTool(multipart), true);
  });

  it('should return false for a non-multipart tool', () => {
    const regular = defineTool(
      registry,
      { name: 'regular', description: 'test' },
      async () => {}
    );
    assert.strictEqual(isMultipartTool(regular), false);
  });

  it('should return false for a non-tool action', () => {
    const regularAction = action(
      { actionType: 'util', name: 'regularAction', description: 'test' },
      async () => {}
    );
    assert.strictEqual(isMultipartTool(regularAction), false);
  });

  it('should return false for a non-action', () => {
    assert.strictEqual(isMultipartTool({}), false);
    assert.strictEqual(isMultipartTool('tool'), false);
    assert.strictEqual(isMultipartTool(123), false);
  });
});

describe('defineTool', () => {
  let registry = new Registry();
  registry.apiStability = 'beta';
  afterEach(() => {
    registry = new Registry();
    registry.apiStability = 'beta';
  });

  describe('.respond()', () => {
    it('constructs a ToolResponsePart', () => {
      const t = defineTool(
        registry,
        { name: 'test', description: 'test' },
        async () => {}
      );
      assert.deepStrictEqual(
        t.respond({ toolRequest: { name: 'test', input: {} } }, 'output'),
        {
          toolResponse: {
            name: 'test',
            output: 'output',
          },
          metadata: {
            interruptResponse: true,
          },
        }
      );
    });

    it('includes metadata', () => {
      const t = defineTool(
        registry,
        { name: 'test', description: 'test' },
        async () => {}
      );
      assert.deepStrictEqual(
        t.respond({ toolRequest: { name: 'test', input: {} } }, 'output', {
          metadata: { extra: 'data' },
        }),
        {
          toolResponse: {
            name: 'test',
            output: 'output',
          },
          metadata: {
            interruptResponse: { extra: 'data' },
          },
        }
      );
    });

    it('validates schema', () => {
      const t = defineTool(
        registry,
        { name: 'test', description: 'test', outputSchema: z.number() },
        async (input, { interrupt }) => interrupt()
      );
      assert.throws(
        () => {
          t.respond(
            { toolRequest: { name: 'test', input: {} } },
            'not_a_number' as any
          );
        },
        { name: 'GenkitError', status: 'INVALID_ARGUMENT' }
      );

      assert.deepStrictEqual(
        t.respond({ toolRequest: { name: 'test', input: {} } }, 55),
        {
          toolResponse: {
            name: 'test',
            output: 55,
          },
          metadata: {
            interruptResponse: true,
          },
        }
      );
    });
  });

  describe('.restart()', () => {
    it('constructs a ToolRequestPart', () => {
      const t = defineTool(
        registry,
        { name: 'test', description: 'test' },
        async () => {}
      );
      assert.deepStrictEqual(
        t.restart({ toolRequest: { name: 'test', input: {} } }),
        {
          toolRequest: {
            name: 'test',
            input: {},
          },
          metadata: {
            resumed: true,
          },
        }
      );
    });

    it('includes metadata', () => {
      const t = defineTool(
        registry,
        { name: 'test', description: 'test' },
        async () => {}
      );
      assert.deepStrictEqual(
        t.restart(
          { toolRequest: { name: 'test', input: {} } },
          { extra: 'data' }
        ),
        {
          toolRequest: {
            name: 'test',
            input: {},
          },
          metadata: {
            resumed: { extra: 'data' },
          },
        }
      );
    });

    it('validates schema', () => {
      const t = defineTool(
        registry,
        { name: 'test', description: 'test', inputSchema: z.number() },
        async (input, { interrupt }) => interrupt()
      );
      assert.throws(
        () => {
          t.restart({ toolRequest: { name: 'test', input: {} } }, undefined, {
            replaceInput: 'not_a_number' as any,
          });
        },
        { name: 'GenkitError', status: 'INVALID_ARGUMENT' }
      );

      assert.deepStrictEqual(
        t.restart({ toolRequest: { name: 'test', input: {} } }, undefined, {
          replaceInput: 55,
        }),
        {
          toolRequest: {
            name: 'test',
            input: 55,
          },
          metadata: {
            resumed: true,
            replacedInput: {},
          },
        }
      );
    });
  });

  it('should register a v1 tool as v2 as well', async () => {
    defineTool(registry, { name: 'test', description: 'test' }, async () => {});
    assert.ok(await registry.lookupAction('/tool/test'));
    assert.ok(await registry.lookupAction('/tool.v2/test'));
  });

  it('should only register a multipart tool as v2', async () => {
    defineTool(
      registry,
      { name: 'test', description: 'test', multipart: true },
      async () => {}
    );
    assert.ok(await registry.lookupAction('/tool.v2/test'));
    assert.equal(await registry.lookupAction('/tool/test'), undefined);
  });

  it('should wrap v1 tool output when called as v2', async () => {
    defineTool(
      registry,
      { name: 'test', description: 'test' },
      async () => 'foo'
    );
    const action = await registry.lookupAction('/tool.v2/test');
    assert.ok(action);
    const result = await action!({});
    assert.deepStrictEqual(result, { output: 'foo' });
  });
});
