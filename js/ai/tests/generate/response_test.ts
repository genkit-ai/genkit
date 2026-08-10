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

import { z } from '@genkit-ai/core';
import { toJsonSchema } from '@genkit-ai/core/schema';
import * as assert from 'assert';
import { describe, it } from 'node:test';
import {
  GenerateResponse,
  GenerationBlockedError,
  GenerationResponseError,
} from '../../src/generate.js';
import { Message } from '../../src/message.js';
import type { GenerateRequest, GenerateResponseData } from '../../src/model.js';

describe('GenerateResponse', () => {
  describe('#toJSON()', () => {
    const testCases = [
      {
        should: 'serialize correctly when custom is undefined',
        responseData: {
          message: {
            role: 'model',
            content: [{ text: '{"name": "Bob"}' }],
          },
          finishReason: 'stop',
          finishMessage: '',
          usage: {},
          // No 'custom' property
        },
        expectedOutput: {
          message: { content: [{ text: '{"name": "Bob"}' }], role: 'model' },
          finishReason: 'stop',
          usage: {},
          custom: {},
        },
      },
    ];

    for (const test of testCases) {
      it(test.should, () => {
        const response = new GenerateResponse(
          test.responseData as GenerateResponseData
        );
        assert.deepStrictEqual(response.toJSON(), test.expectedOutput);
      });
    }
  });

  describe('#output()', () => {
    const testCases = [
      {
        should: 'return structured data from the data part',
        responseData: {
          message: new Message({
            role: 'model',
            content: [{ data: { name: 'Alice', age: 30 } }],
          }),
          finishReason: 'stop',
          finishMessage: '',
          usage: {},
        },
        expectedOutput: { name: 'Alice', age: 30 },
      },
      {
        should: 'parse JSON from text when the data part is absent',
        responseData: {
          message: new Message({
            role: 'model',
            content: [{ text: '{"name": "Bob"}' }],
          }),
          finishReason: 'stop',
          finishMessage: '',
          usage: {},
        },
        expectedOutput: { name: 'Bob' },
      },
    ];

    for (const test of testCases) {
      it(test.should, () => {
        const response = new GenerateResponse(
          test.responseData as GenerateResponseData
        );
        assert.deepStrictEqual(response.output, test.expectedOutput);
      });
    }
  });

  describe('#assertValid()', () => {
    it('throws GenerationBlockedError if finishReason is blocked', () => {
      const response = new GenerateResponse({
        finishReason: 'blocked',
        finishMessage: 'Content was blocked',
      });

      assert.throws(
        () => {
          response.assertValid();
        },
        (err: unknown) => {
          return err instanceof GenerationBlockedError;
        }
      );
    });

    it('throws GenerationResponseError if no message is generated', () => {
      const response = new GenerateResponse({
        finishReason: 'length',
        finishMessage: 'Reached max tokens',
      });

      assert.throws(
        () => {
          response.assertValid();
        },
        (err: unknown) => {
          return err instanceof GenerationResponseError;
        }
      );
    });

    it('throws error if output does not conform to schema', () => {
      const schema = z.object({
        name: z.string(),
        age: z.number(),
      });

      const response = new GenerateResponse({
        message: {
          role: 'model',
          content: [{ text: '{"name": "John", "age": "30"}' }],
        },
        finishReason: 'stop',
      });

      const request: GenerateRequest = {
        messages: [],
        output: {
          schema: toJsonSchema({ schema }),
        },
      };

      assert.throws(
        () => {
          response.assertValidSchema(request);
        },
        (err: unknown) => {
          return err instanceof Error && err.message.includes('must be number');
        }
      );
    });

    it('does not throw if output conforms to schema', () => {
      const schema = z.object({
        name: z.string(),
        age: z.number(),
      });

      const response = new GenerateResponse({
        message: {
          role: 'model',
          content: [{ text: '{"name": "John", "age": 30}' }],
        },
        finishReason: 'stop',
      });

      const request: GenerateRequest = {
        messages: [],
        output: {
          schema: toJsonSchema({ schema }),
        },
      };

      assert.doesNotThrow(() => {
        response.assertValidSchema(request);
      });
    });
  });

  describe('#toolRequests()', () => {
    it('returns empty array if no tools requests found', () => {
      const response = new GenerateResponse({
        message: new Message({
          role: 'model',
          content: [{ text: '{"abc":"123"}' }],
        }),
        finishReason: 'stop',
      });
      assert.deepStrictEqual(response.toolRequests, []);
    });
    it('returns tool call if present', () => {
      const toolCall = {
        toolRequest: {
          name: 'foo',
          ref: 'abc',
          input: 'banana',
        },
      };
      const response = new GenerateResponse({
        message: new Message({
          role: 'model',
          content: [toolCall],
        }),
        finishReason: 'stop',
      });
      assert.deepStrictEqual(response.toolRequests, [toolCall]);
    });
    it('returns all tool calls', () => {
      const toolCall1 = {
        toolRequest: {
          name: 'foo',
          ref: 'abc',
          input: 'banana',
        },
      };
      const toolCall2 = {
        toolRequest: {
          name: 'bar',
          ref: 'bcd',
          input: 'apple',
        },
      };
      const response = new GenerateResponse({
        message: new Message({
          role: 'model',
          content: [toolCall1, toolCall2],
        }),
        finishReason: 'stop',
      });
      assert.deepStrictEqual(response.toolRequests, [toolCall1, toolCall2]);
    });
  });

  it('returns metadata for output conformance', () => {
    const request: GenerateRequest = {
      messages: [],
      output: {
        constrained: true,
        format: 'json',
        contentType: 'application/json',
        schema: toJsonSchema({
          schema: z.object({
            name: z.string(),
            age: z.number(),
          }),
        }),
      },
    };

    const response = new GenerateResponse(
      {
        message: {
          role: 'model',
          content: [{ text: '{"name": "John", "age": "30"}' }],
        },
        finishReason: 'stop',
      },
      {
        request,
      }
    );

    assert.deepEqual(response.message?.metadata, {
      generate: {
        output: { contentType: 'application/json', format: 'json' },
      },
    });
  });
});
