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

import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  CompletionChunk,
} from '@mistralai/mistralai-gcp/models/components';
import * as assert from 'assert';
import type { GenerateRequest, GenerateResponseData } from 'genkit';
import { describe, it } from 'node:test';
import {
  fromMistralCompletionChunk,
  fromMistralResponse,
  toMistralRequest,
  type MistralConfigSchema,
} from '../../../src/modelgarden/legacy/mistral';

const MODEL_ID = 'mistral-large-2411';

describe('toMistralRequest', () => {
  const testCases: {
    should: string;
    input: GenerateRequest<typeof MistralConfigSchema>;
    expectedOutput: ChatCompletionRequest;
  }[] = [
    {
      should: 'should transform genkit message (text content) correctly',
      input: {
        messages: [
          {
            role: 'user' as const,
            content: [{ text: 'Tell a joke about dogs.' }],
          },
        ],
      },
      expectedOutput: {
        model: MODEL_ID,
        messages: [
          {
            role: 'user',
            content: 'Tell a joke about dogs.',
          },
        ],
        maxTokens: 1024,
        temperature: 0.7,
      },
    },
    {
      should: 'should transform system message',
      input: {
        messages: [
          {
            role: 'system' as const,
            content: [{ text: 'Talk like a pirate.' }],
          },
          {
            role: 'user' as const,
            content: [{ text: 'Tell a joke about dogs.' }],
          },
        ],
      },
      expectedOutput: {
        model: MODEL_ID,
        messages: [
          {
            role: 'system',
            content: 'Talk like a pirate.',
          },
          {
            role: 'user',
            content: 'Tell a joke about dogs.',
          },
        ],
        maxTokens: 1024,
        temperature: 0.7,
      },
    },
    {
      should: 'should transform tool request correctly',
      input: {
        messages: [
          {
            role: 'user' as const,
            content: [{ text: "What's the weather like today?" }],
          },
        ],
        tools: [
          {
            name: 'get_weather',
            description: 'Get the weather for a location.',
            inputSchema: {
              type: 'object',
              properties: {
                location: {
                  type: 'string',
                  description: 'The city and state, e.g. San Francisco, CA',
                },
              },
              required: ['location'],
            },
          },
        ],
      },
      expectedOutput: {
        model: MODEL_ID,
        messages: [
          {
            role: 'user',
            content: "What's the weather like today?",
          },
        ],
        maxTokens: 1024,
        temperature: 0.7,
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              description: 'Get the weather for a location.',
              parameters: {
                type: 'object',
                properties: {
                  location: {
                    type: 'string',
                    description: 'The city and state, e.g. San Francisco, CA',
                  },
                },
                required: ['location'],
              },
            },
          },
        ],
      },
    },
  ];
  for (const test of testCases) {
    it(test.should, () => {
      assert.deepEqual(
        toMistralRequest(MODEL_ID, test.input),
        test.expectedOutput
      );
    });
  }
});

describe('fromMistralResponse', () => {
  const testCases: {
    should: string;
    input: GenerateRequest<typeof MistralConfigSchema>;
    response: ChatCompletionResponse;
    expectedOutput: GenerateResponseData;
  }[] = [
    {
      should: 'should transform mistral message (text content) correctly',
      input: {
        messages: [
          {
            role: 'user' as const,
            content: [{ text: 'Tell a joke about dogs.' }],
          },
        ],
      },
      response: {
        id: 'abcd1234',
        object: 'chat.completion',
        model: MODEL_ID,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content:
                'Why do dogs make terrible comedians? Their jokes are too ruff!',
            },
            finishReason: 'stop',
          },
        ],
        usage: {
          promptTokens: 123,
          completionTokens: 234,
          totalTokens: 357,
        },
      },
      expectedOutput: {
        message: {
          role: 'model' as const,
          content: [
            {
              text: 'Why do dogs make terrible comedians? Their jokes are too ruff!',
            },
          ],
        },
        finishReason: 'stop',
        usage: {
          inputTokens: 123,
          outputTokens: 234,
        },
        custom: {
          id: 'abcd1234',
          model: MODEL_ID,
          created: undefined,
        },
        raw: {
          id: 'abcd1234',
          object: 'chat.completion',
          model: MODEL_ID,
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content:
                  'Why do dogs make terrible comedians? Their jokes are too ruff!',
              },
              finishReason: 'stop',
            },
          ],
          usage: {
            promptTokens: 123,
            completionTokens: 234,
            totalTokens: 357,
          },
        },
      },
    },
    {
      should: 'should transform tool calls correctly',
      input: {
        messages: [
          {
            role: 'user' as const,
            content: [{ text: "What's the weather like today?" }],
          },
        ],
      },
      response: {
        id: 'abcd1234',
        object: 'chat.completion',
        model: MODEL_ID,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              toolCalls: [
                {
                  id: 'call_abc123',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"location":"San Francisco, CA"}',
                  },
                },
              ],
            },
            finishReason: 'tool_calls',
          },
        ],
        usage: {
          promptTokens: 123,
          completionTokens: 234,
          totalTokens: 357,
        },
      },
      expectedOutput: {
        message: {
          role: 'model' as const,
          content: [
            {
              toolRequest: {
                ref: 'call_abc123',
                name: 'get_weather',
                input: { location: 'San Francisco, CA' },
              },
            },
          ],
        },
        finishReason: 'stop',
        usage: {
          inputTokens: 123,
          outputTokens: 234,
        },
        custom: {
          id: 'abcd1234',
          model: MODEL_ID,
          created: undefined,
        },
        raw: {
          id: 'abcd1234',
          object: 'chat.completion',
          model: MODEL_ID,
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: null,
                toolCalls: [
                  {
                    id: 'call_abc123',
                    type: 'function',
                    function: {
                      name: 'get_weather',
                      arguments: '{"location":"San Francisco, CA"}',
                    },
                  },
                ],
              },
              finishReason: 'tool_calls',
            },
          ],
          usage: {
            promptTokens: 123,
            completionTokens: 234,
            totalTokens: 357,
          },
        },
      },
    },
  ];
  for (const test of testCases) {
    it(test.should, () => {
      assert.deepEqual(
        fromMistralResponse(test.input, test.response),
        test.expectedOutput
      );
    });
  }
});

describe('validateToolSequence', () => {
  it('should handle valid tool call and response sequence', () => {
    const input = {
      messages: [
        {
          role: 'user' as const,
          content: [{ text: "What's the weather?" }],
        },
        {
          role: 'model' as const,
          content: [
            {
              toolRequest: {
                ref: 'call_123',
                name: 'get_weather',
                input: { location: 'San Francisco' },
              },
            },
          ],
        },
        {
          role: 'tool' as const,
          content: [
            {
              toolResponse: {
                ref: 'call_123',
                name: 'get_weather',
                output: { temperature: 72, condition: 'sunny' },
              },
            },
          ],
        },
      ],
    };

    // Should not throw an error
    assert.doesNotThrow(() => toMistralRequest(MODEL_ID, input));
  });

  it('should throw error when tool response is missing', () => {
    const input = {
      messages: [
        {
          role: 'user' as const,
          content: [{ text: "What's the weather?" }],
        },
        {
          role: 'model' as const,
          content: [
            {
              toolRequest: {
                ref: 'call_123',
                name: 'get_weather',
                input: { location: 'San Francisco' },
              },
            },
          ],
        },
      ],
    };

    assert.throws(
      () => toMistralRequest(MODEL_ID, input),
      /Mismatch between tool calls/
    );
  });

  it('should throw error when tool response id does not match call', () => {
    const input = {
      messages: [
        {
          role: 'user' as const,
          content: [{ text: "What's the weather?" }],
        },
        {
          role: 'model' as const,
          content: [
            {
              toolRequest: {
                ref: 'call_123',
                name: 'get_weather',
                input: { location: 'San Francisco' },
              },
            },
          ],
        },
        {
          role: 'tool' as const,
          content: [
            {
              toolResponse: {
                ref: 'wrong_id',
                name: 'get_weather',
                output: { temperature: 72, condition: 'sunny' },
              },
            },
          ],
        },
      ],
    };

    assert.throws(
      () => toMistralRequest(MODEL_ID, input),
      /Tool response with ID wrong_id has no matching call/
    );
  });
});

describe('edge cases', () => {
  it('should handle empty message content', () => {
    const input = {
      messages: [
        {
          role: 'user' as const,
          content: [],
        },
      ],
    };

    const result = toMistralRequest(MODEL_ID, input);
    assert.equal(result.messages[0].content, '');
  });

  it('should handle multiple text parts in content', () => {
    const input = {
      messages: [
        {
          role: 'user' as const,
          content: [{ text: 'Hello' }, { text: ' ' }, { text: 'World' }],
        },
      ],
    };

    const result = toMistralRequest(MODEL_ID, input);
    assert.equal(result.messages[0].content, 'Hello World');
  });

  it('should handle custom configuration options', () => {
    const input = {
      messages: [
        {
          role: 'user' as const,
          content: [{ text: 'test' }],
        },
      ],
      config: {
        temperature: 0.5,
        maxOutputTokens: 500,
        topP: 0.9,
        stopSequences: ['END'],
      },
    };

    const result = toMistralRequest(MODEL_ID, input);
    assert.equal(result.temperature, 0.5);
    assert.equal(result.maxTokens, 500);
    assert.equal(result.topP, 0.9);
    assert.deepEqual(result.stop, ['END']);
  });
});

describe('fromMistralResponse error handling', () => {
  it('should handle response with no choices', () => {
    const input = {
      messages: [
        {
          role: 'user' as const,
          content: [{ text: 'test' }],
        },
      ],
    };

    const response = {
      id: 'test',
      object: 'chat.completion',
      model: MODEL_ID,
      choices: [],
      usage: {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      },
    };

    const result = fromMistralResponse(input, response);
    assert.deepEqual(result.message!.content, []);
  });
});

describe('fromMistralCompletionChunk', () => {
  it('should handle text content chunk', () => {
    const chunk = {
      id: 'chunk_1',
      model: MODEL_ID,
      choices: [
        {
          index: 0,
          delta: {
            role: 'assistant',
            content: 'Hello world',
          },
        },
      ],
    };

    const parts = fromMistralCompletionChunk(
      chunk as unknown as CompletionChunk
    );
    assert.deepEqual(parts, [{ text: 'Hello world' }]);
  });

  it('should handle tool call chunk', () => {
    const chunk = {
      id: 'chunk_1',
      model: MODEL_ID,
      choices: [
        {
          index: 0,
          delta: {
            role: 'assistant',
            toolCalls: [
              {
                id: 'call_123',
                type: 'function',
                function: {
                  name: 'get_weather',
                  arguments: '{"location":"San Francisco"}',
                },
              },
            ],
          },
        },
      ],
    };

    const parts = fromMistralCompletionChunk(
      chunk as unknown as CompletionChunk
    );
    assert.deepEqual(parts, [
      {
        toolRequest: {
          ref: 'call_123',
          name: 'get_weather',
          input: { location: 'San Francisco' },
        },
      },
    ]);
  });

  it('should handle empty chunk', () => {
    const chunk = {
      id: 'chunk_1',
      model: MODEL_ID,
      choices: [],
    };

    const parts = fromMistralCompletionChunk(chunk);
    assert.deepEqual(parts, []);
  });

  it('should handle chunk with empty delta', () => {
    const chunk = {
      id: 'chunk_1',
      model: MODEL_ID,
      choices: [
        {
          index: 0,
          delta: {},
        },
      ],
    };

    const parts = fromMistralCompletionChunk(
      chunk as unknown as CompletionChunk
    );
    assert.deepEqual(parts, []);
  });

  it('should handle chunk with invalid tool call', () => {
    const chunk = {
      id: 'chunk_1',
      model: MODEL_ID,
      choices: [
        {
          index: 0,
          delta: {
            toolCalls: [
              {
                id: 'call_123',
                type: 'function',
                // Missing function property
              },
            ],
          },
        },
      ],
    };

    const parts = fromMistralCompletionChunk(
      chunk as unknown as CompletionChunk
    );
    assert.deepEqual(parts, []); // Should skip invalid tool call
  });

  it('should handle mixed content chunk', () => {
    const chunk = {
      id: 'chunk_1',
      model: MODEL_ID,
      choices: [
        {
          index: 0,
          delta: {
            content: 'Some text',
            toolCalls: [
              {
                id: 'call_123',
                type: 'function',
                function: {
                  name: 'get_weather',
                  arguments: '{"location":"San Francisco"}',
                },
              },
            ],
          },
        },
      ],
    };

    const parts = fromMistralCompletionChunk(
      chunk as unknown as CompletionChunk
    );
    assert.deepEqual(parts, [
      { text: 'Some text' },
      {
        toolRequest: {
          ref: 'call_123',
          name: 'get_weather',
          input: { location: 'San Francisco' },
        },
      },
    ]);
  });
});
