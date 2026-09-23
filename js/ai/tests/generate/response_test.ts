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

import { GenkitError, z } from '@genkit-ai/core';
import { toJsonSchema } from '@genkit-ai/core/schema';
import * as assert from 'assert';
import { describe, it } from 'node:test';
import {
  GenerateResponse,
  GenerationAbortedError,
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

describe('GenerateResponse partials', () => {
  const request: GenerateRequest = {
    messages: [
      { role: 'user', content: [{ text: 'hi' }] },
      { role: 'model', content: [{ toolRequest: { name: 't', input: {} } }] },
      { role: 'tool', content: [{ toolResponse: { name: 't', output: 1 } }] },
    ],
  };

  it('returns the request messages as history when there is no message', () => {
    const response = new GenerateResponse(
      { finishReason: 'failed', finishMessage: 'model melted' },
      { request }
    );
    assert.deepStrictEqual(response.messages, request.messages);
    assert.notStrictEqual(response.messages, request.messages);
  });

  it('still requires a request to build history', () => {
    const response = new GenerateResponse({ finishReason: 'failed' });
    assert.throws(() => response.messages, /without request reference/);
  });

  it('round-trips the classified error through toJSON', () => {
    const error = { status: 'UNAVAILABLE', message: 'model melted' };
    const response = new GenerateResponse({
      finishReason: 'failed',
      finishMessage: 'model melted',
      error,
    });
    assert.deepStrictEqual(response.error, error);
    assert.deepStrictEqual(response.toJSON().error, error);
    assert.strictEqual(
      'error' in new GenerateResponse({ finishReason: 'stop' }).toJSON(),
      false
    );
  });

  it('stamps the error on the response assertValid rejects', () => {
    const blocked = new GenerateResponse({
      finishReason: 'blocked',
      finishMessage: 'unsafe',
    });
    assert.throws(() => blocked.assertValid(), GenerationBlockedError);
    assert.deepStrictEqual(blocked.error, {
      status: 'FAILED_PRECONDITION',
      message: 'Generation blocked: unsafe',
    });

    const empty = new GenerateResponse({ finishReason: 'length' });
    assert.throws(() => empty.assertValid(), GenerationResponseError);
    assert.strictEqual(empty.error?.status, 'FAILED_PRECONDITION');
  });

  it('leaves the response unchanged after isValid', () => {
    const blocked = new GenerateResponse({ finishReason: 'blocked' });
    assert.strictEqual(blocked.isValid(), false);
    assert.strictEqual(blocked.error, undefined);
    assert.strictEqual('error' in blocked.toJSON(), false);
  });
});

describe('GenerationResponseError', () => {
  const partial = new GenerateResponse(
    {
      finishReason: 'failed',
      finishMessage: 'model melted',
      error: { status: 'UNAVAILABLE', message: 'model melted' },
    },
    {
      request: { messages: [{ role: 'user', content: [{ text: 'secret' }] }] },
    }
  );

  it('keeps the response in-process and off the wire', () => {
    const err = new GenerationResponseError(
      partial,
      'model melted',
      'UNAVAILABLE',
      { attempt: 2 }
    );
    assert.strictEqual(err.detail.response, partial);
    assert.strictEqual(err.detail.attempt, 2);
    assert.deepStrictEqual(err.toJSON(), {
      status: 'UNAVAILABLE',
      message: 'model melted',
      details: {
        attempt: 2,
        finishReason: 'failed',
        finishMessage: 'model melted',
      },
    });
    assert.strictEqual(JSON.stringify(err).includes('secret'), false);
  });

  it('wraps a cause losslessly', () => {
    const cause = new GenkitError({
      status: 'UNAVAILABLE',
      message: 'model melted',
      detail: { provider: 'x' },
      source: 'provider',
      responseMetadata: { retryAfterMs: 5 },
    });
    const err = new GenerationResponseError(
      partial,
      'model melted',
      'UNAVAILABLE',
      undefined,
      { cause }
    );
    assert.strictEqual(err.cause, cause);
    assert.strictEqual(err.detail.provider, 'x');
    assert.strictEqual(err.source, 'provider');
    assert.strictEqual(err.responseMetadata?.retryAfterMs, 5);
    assert.strictEqual(err.message, 'provider: UNAVAILABLE: model melted');
  });

  it('sends the public message when the cause text is not for a client', () => {
    const err = new GenerationResponseError(
      partial,
      'db password rejected',
      'INTERNAL',
      undefined,
      {
        cause: new Error('db password rejected'),
        publicMessage: 'generation failed',
      }
    );
    assert.strictEqual(err.toJSON().message, 'generation failed');
    assert.strictEqual(err.originalMessage, 'db password rejected');
  });

  it('is the base of GenerationAbortedError and GenerationBlockedError', () => {
    assert.ok(
      new GenerationAbortedError(partial, 'stopped', 'CANCELLED') instanceof
        GenerationResponseError
    );
    assert.ok(
      new GenerationBlockedError(partial, 'blocked') instanceof
        GenerationResponseError
    );
  });
});
