/**
Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
import * as assert from 'assert';
import { z } from 'genkit';
import type {
  CandidateData,
  Part as GenkitPart,
  MessageData,
} from 'genkit/model';
import { toJsonSchema } from 'genkit/schema';
import { describe, it } from 'node:test';
import {
  TEST_ONLY,
  applyGeminiPartialArgs,
  fromGeminiCandidate,
  toGeminiFunctionModeEnum,
  toGeminiMessage,
  toGeminiSystemInstruction,
  toGeminiTool,
} from '../../src/common/converters.js';
import type {
  FunctionCall,
  Part as GeminiPart,
  GenerateContentCandidate,
} from '../../src/common/types.js';
import {
  ExecutableCodeLanguage,
  FunctionCallingMode,
  Outcome,
  SchemaType,
} from '../../src/common/types.js';

describe('toGeminiMessage', () => {
  const testCases = [
    {
      should: 'should transform genkit message (text content) correctly',
      inputMessage: {
        role: 'user',
        content: [{ text: 'Tell a joke about dogs.' }],
      },
      expectedOutput: {
        role: 'user',
        parts: [{ text: 'Tell a joke about dogs.' }],
      },
    },
    {
      should:
        'should transform genkit message (tool request content) correctly',
      inputMessage: {
        role: 'model',
        content: [
          { toolRequest: { name: 'tellAFunnyJoke', input: { topic: 'dogs' } } },
        ],
      },
      expectedOutput: {
        role: 'model',
        parts: [
          { functionCall: { name: 'tellAFunnyJoke', args: { topic: 'dogs' } } },
        ],
      },
    },
    {
      should:
        'should transform genkit message (tool response content) correctly and sort by ref',
      inputMessage: {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'tellAFunnyJoke',
              output: 'Why did the dogs cross the road?',
              ref: '1',
            },
          },
          {
            toolResponse: {
              name: 'tellAFunnyJoke',
              output: 'Why did the chicken cross the road?',
              ref: '0',
            },
          },
        ],
      },
      expectedOutput: {
        role: 'function',
        parts: [
          {
            functionResponse: {
              id: '0',
              name: 'tellAFunnyJoke',
              response: {
                name: 'tellAFunnyJoke',
                content: 'Why did the chicken cross the road?',
              },
            },
          },
          {
            functionResponse: {
              id: '1',
              name: 'tellAFunnyJoke',
              response: {
                name: 'tellAFunnyJoke',
                content: 'Why did the dogs cross the road?',
              },
            },
          },
        ],
      },
    },
    {
      should:
        'should transform genkit message (tool response with media content) correctly',
      inputMessage: {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'screenshot',
              output: 'success',
              ref: '0',
              content: [
                {
                  media: {
                    contentType: 'image/png',
                    url: 'data:image/png;base64,SHORTENED_BASE64_DATA',
                  },
                },
              ],
            },
          },
        ],
      },
      expectedOutput: {
        role: 'function',
        parts: [
          {
            functionResponse: {
              id: '0',
              name: 'screenshot',
              response: {
                name: 'screenshot',
                content: 'success',
              },
              parts: [
                {
                  inlineData: {
                    mimeType: 'image/png',
                    data: 'SHORTENED_BASE64_DATA',
                  },
                },
              ],
            },
          },
        ],
      },
    },
    {
      should:
        'should transform genkit message (inline base64 image content) correctly',
      inputMessage: {
        role: 'user',
        content: [
          { text: 'describe the following image:' },
          {
            media: {
              contentType: 'image/jpeg',
              url: 'data:image/jpeg;base64,SHORTENED_BASE64_DATA',
            },
            metadata: {
              mediaResolution: {
                level: 'MEDIA_RESOLUTION_HIGH',
              },
            },
          },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          { text: 'describe the following image:' },
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: 'SHORTENED_BASE64_DATA',
            },
            mediaResolution: {
              level: 'MEDIA_RESOLUTION_HIGH',
            },
          },
        ],
      },
    },
    {
      should:
        'should transform genkit message (fileData image content) correctly',
      inputMessage: {
        role: 'user',
        content: [
          { text: 'describe the following image:' },
          {
            media: {
              contentType: 'image/png',
              url: 'gs://bucket/image.png',
            },
          },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          { text: 'describe the following image:' },
          {
            fileData: {
              mimeType: 'image/png',
              fileUri: 'gs://bucket/image.png',
            },
          },
        ],
      },
    },
    {
      should:
        'should transform genkit message (fileData video content with metadata) correctly',
      inputMessage: {
        role: 'user',
        content: [
          { text: 'describe the following video:' },
          {
            media: {
              contentType: 'video/mp4',
              url: 'gs://bucket/video.mp4',
            },
            metadata: {
              videoMetadata: {
                startOffset: '10.0s',
                endOffset: '20.5s',
                fps: 0.5,
              },
            },
          },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          { text: 'describe the following video:' },
          {
            fileData: {
              mimeType: 'video/mp4',
              fileUri: 'gs://bucket/video.mp4',
            },
            videoMetadata: {
              startOffset: '10.0s',
              endOffset: '20.5s',
              fps: 0.5,
            },
          },
        ],
      },
    },
    {
      should:
        'should transform genkit message (fileData video content with partial metadata) correctly',
      inputMessage: {
        role: 'user',
        content: [
          { text: 'describe the following video:' },
          {
            media: {
              contentType: 'video/mp4',
              url: 'gs://bucket/video.mp4',
            },
            metadata: {
              videoMetadata: {
                startOffset: '5.3s',
                endOffset: '15.7s',
              },
            },
          },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          { text: 'describe the following video:' },
          {
            fileData: {
              mimeType: 'video/mp4',
              fileUri: 'gs://bucket/video.mp4',
            },
            videoMetadata: {
              startOffset: '5.3s',
              endOffset: '15.7s',
            },
          },
        ],
      },
    },
    {
      should: 'should re-populate thoughtSignature from reasoning metadata',
      inputMessage: {
        role: 'model',
        content: [{ reasoning: '', metadata: { thoughtSignature: 'abc123' } }],
      },
      expectedOutput: {
        role: 'model',
        parts: [{ thought: true, thoughtSignature: 'abc123' }],
      },
    },
    {
      should: 'should transform reasoning with text',
      inputMessage: {
        role: 'model',
        content: [
          {
            reasoning: 'I should call a tool',
            metadata: { thoughtSignature: 'def456' },
          },
        ],
      },
      expectedOutput: {
        role: 'model',
        parts: [
          {
            thought: true,
            text: 'I should call a tool',
            thoughtSignature: 'def456',
          },
        ],
      },
    },
    {
      should: 'should transform executableCode custom part',
      inputMessage: {
        role: 'model',
        content: [
          {
            custom: {
              executableCode: {
                language: ExecutableCodeLanguage.PYTHON,
                code: 'print(1+1)',
              },
            },
          },
        ],
      },
      expectedOutput: {
        role: 'model',
        parts: [
          {
            executableCode: {
              language: ExecutableCodeLanguage.PYTHON,
              code: 'print(1+1)',
            },
          },
        ],
      },
    },
    {
      should: 'should transform codeExecutionResult custom part',
      inputMessage: {
        role: 'tool',
        content: [
          {
            custom: {
              codeExecutionResult: {
                outcome: Outcome.OUTCOME_OK,
                output: '2',
              },
            },
          },
        ],
      },
      expectedOutput: {
        role: 'function',
        parts: [
          {
            codeExecutionResult: {
              outcome: Outcome.OUTCOME_OK,
              output: '2',
            },
          },
        ],
      },
    },
  ];
  for (const test of testCases) {
    it(test.should, () => {
      assert.deepStrictEqual(
        toGeminiMessage(test.inputMessage as MessageData),
        test.expectedOutput
      );
    });
  }

  it('should throw on unsupported part type', () => {
    const inputMessage = {
      role: 'user',
      content: [{ unsupported: 'data' } as any],
    };
    assert.throws(
      () => toGeminiMessage(inputMessage as MessageData),
      /Unsupported Part type/
    );
  });

  it('should throw on media part missing contentType for non-data URL', () => {
    const inputMessage = {
      role: 'user',
      content: [
        {
          media: {
            url: 'gs://bucket/file',
          },
        } as any,
      ],
    };
    assert.throws(
      () => toGeminiMessage(inputMessage as MessageData),
      /Must supply a (`)?contentType(`)? when sending File URIs to Gemini/
    );
  });
});

describe('fromGeminiParts', () => {
  it('should transform toolCall, toolResponse, and functionCall with thoughtSignatures correctly', () => {
    const parts = [
      {
        thoughtSignature: 'AAAAA=',
        toolCall: {
          toolType: 'GOOGLE_SEARCH_WEB',
          args: {
            queries: ['northernmost city in Canada'],
          },
          id: 'a111',
        },
      },
      {
        thoughtSignature: 'BBBBB=',
        toolResponse: {
          toolType: 'GOOGLE_SEARCH_WEB',
          response: {
            search_suggestions: '<tags>stuff</tags>\n',
          },
          id: 'a111',
        },
      },
      {
        functionCall: {
          name: 'getWeather',
          args: {
            location: 'Iqaluit, NU',
          },
          id: 'b222',
        },
        thoughtSignature: 'CCCCC=',
      },
    ];

    const result = fromGeminiCandidate({
      index: 0,
      content: { role: 'model', parts: parts as any },
      finishReason: 'STOP' as any,
    });

    assert.deepStrictEqual(result.message.content, [
      {
        custom: {
          geminiToolCall: {
            toolType: 'GOOGLE_SEARCH_WEB',
            id: 'a111',
            args: {
              queries: ['northernmost city in Canada'],
            },
          },
        },
        metadata: { thoughtSignature: 'AAAAA=' },
      },
      {
        custom: {
          geminiToolResponse: {
            toolType: 'GOOGLE_SEARCH_WEB',
            id: 'a111',
            response: {
              search_suggestions: '<tags>stuff</tags>\n',
            },
          },
        },
        metadata: { thoughtSignature: 'BBBBB=' },
      },
      {
        toolRequest: {
          name: 'getWeather',
          ref: 'b222',
          input: {
            location: 'Iqaluit, NU',
          },
        },
        metadata: { thoughtSignature: 'CCCCC=' },
      },
    ]);
  });
});

describe('toGeminiRole fallback and exceptions', () => {
  it('should throw an error for system role if model does not support it', () => {
    const inputMessage: MessageData = {
      role: 'system',
      content: [{ text: 'You are a bot.' }],
    };
    assert.throws(
      () => toGeminiMessage(inputMessage),
      /system role is not supported/
    );
  });

  it('should throw an error for system role if model explicitly flags systemRole as false', () => {
    const inputMessage: MessageData = {
      role: 'system',
      content: [{ text: 'You are a bot.' }],
    };
    const mockModel: any = { info: { supports: { systemRole: false } } };
    assert.throws(
      () => toGeminiMessage(inputMessage, mockModel),
      /system role is not supported/
    );
  });

  it('should throw an error for system role if model supports it but it is passed through toGeminiMessage', () => {
    const inputMessage: MessageData = {
      role: 'system',
      content: [{ text: 'You are a bot.' }],
    };
    const mockModel: any = { info: { supports: { systemRole: true } } };
    assert.throws(
      () => toGeminiMessage(inputMessage, mockModel),
      /system role is only supported for a single message in the first position/
    );
  });

  it('should default unknown roles to user', () => {
    const inputMessage: any = {
      role: 'unknown_role',
      content: [{ text: 'Hello.' }],
    };
    const result = toGeminiMessage(inputMessage);
    assert.strictEqual(result.role, 'user');
  });
});

describe('toGeminiSystemInstruction', () => {
  const testCases = [
    {
      should: 'should transform from system to user',
      inputMessage: {
        role: 'system',
        content: [{ text: 'You are an expert in all things cats.' }],
      },
      expectedOutput: {
        role: 'user',
        parts: [{ text: 'You are an expert in all things cats.' }],
      },
    },
    {
      should: 'should transform from system to user with multiple parts',
      inputMessage: {
        role: 'system',
        content: [
          { text: 'You are an expert in all things animals.' },
          { text: 'You love cats.' },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          { text: 'You are an expert in all things animals.' },
          { text: 'You love cats.' },
        ],
      },
    },
  ];
  for (const test of testCases) {
    it(test.should, () => {
      assert.deepStrictEqual(
        toGeminiSystemInstruction(test.inputMessage as MessageData),
        test.expectedOutput
      );
    });
  }
});

describe('fromGeminiCandidate', () => {
  const testCases = [
    {
      should:
        'should transform gemini candidate to genkit candidate (text parts) correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              text: 'Why did the dog go to the bank?\n\nTo get his bones cashed!',
            },
          ],
        },
        finishReason: 'STOP',
        safetyRatings: [
          { category: 'HARM_CATEGORY_HATE_SPEECH', probability: 'NEGLIGIBLE' },
        ],
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              text: 'Why did the dog go to the bank?\n\nTo get his bones cashed!',
            },
          ],
        },
        finishReason: 'stop',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: [
            {
              category: 'HARM_CATEGORY_HATE_SPEECH',
              probability: 'NEGLIGIBLE',
            },
          ],
        },
      },
    },
    {
      should:
        'should transform gemini candidate with thoughtSignature correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              text: 'I have a thought.',
              thoughtSignature: 'xyz-789',
            },
          ],
        },
        finishReason: 'STOP',
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              text: 'I have a thought.',
              metadata: { thoughtSignature: 'xyz-789' },
            },
          ],
        },
        finishReason: 'stop',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: undefined,
        },
      },
    },
    {
      should:
        'should transform gemini candidate to genkit candidate (function call parts) correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              functionCall: { name: 'tellAFunnyJoke', args: { topic: 'dog' } },
            },
            {
              functionCall: {
                name: 'my__tool__name',
                args: { param: 'value' },
              },
            },
          ],
        },
        finishReason: 'STOP',
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'tellAFunnyJoke',
                input: { topic: 'dog' },
              },
            },
            {
              toolRequest: {
                name: 'my__tool__name', // Expected no conversion for functionCall
                input: { param: 'value' },
              },
            },
          ],
        },
        finishReason: 'stop',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: undefined,
        },
      },
    },
    {
      should:
        'should transform gemini candidate to genkit candidate (thought parts) correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              thought: true,
              thoughtSignature: 'abc123',
            },
            {
              thought: true,
              text: 'thought with text',
              thoughtSignature: 'def456',
            },
          ],
        },
        finishReason: 'STOP',
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              reasoning: '',
              metadata: { thoughtSignature: 'abc123' },
            },
            {
              reasoning: 'thought with text',
              metadata: { thoughtSignature: 'def456' },
            },
          ],
        },
        finishReason: 'stop',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: undefined,
        },
      },
    },
    {
      should: 'should transform gemini candidate (inlineData) correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              inlineData: {
                mimeType: 'image/jpeg',
                data: 'SHORTENED_BASE64_DATA',
              },
            },
          ],
        },
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              media: {
                contentType: 'image/jpeg',
                url: 'data:image/jpeg;base64,SHORTENED_BASE64_DATA',
              },
            },
          ],
        },
        finishReason: 'unknown',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: undefined,
        },
      },
    },
    {
      should: 'should transform gemini candidate (fileData) correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              fileData: {
                mimeType: 'image/png',
                fileUri: 'gs://bucket/image.png',
              },
            },
          ],
        },
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              media: {
                contentType: 'image/png',
                url: 'gs://bucket/image.png',
              },
            },
          ],
        },
        finishReason: 'unknown',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: undefined,
        },
      },
    },
    {
      should: 'should transform gemini candidate (executableCode) correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              executableCode: {
                language: ExecutableCodeLanguage.PYTHON,
                code: 'print(1+1)',
              },
            },
          ],
        },
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              custom: {
                executableCode: {
                  language: ExecutableCodeLanguage.PYTHON,
                  code: 'print(1+1)',
                },
              },
            },
          ],
        },
        finishReason: 'unknown',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: undefined,
        },
      },
    },
    {
      should:
        'should transform gemini candidate (codeExecutionResult) correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              codeExecutionResult: {
                outcome: Outcome.OUTCOME_OK,
                output: '2',
              },
            },
          ],
        },
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              custom: {
                codeExecutionResult: {
                  outcome: Outcome.OUTCOME_OK,
                  output: '2',
                },
              },
            },
          ],
        },
        finishReason: 'unknown',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: undefined,
        },
      },
    },
    {
      should: 'handle various finish reasons',
      geminiCandidate: {
        index: 0,
        content: { role: 'model', parts: [] },
        finishReason: 'MAX_TOKENS',
      },
      expectedOutput: {
        index: 0,
        message: { role: 'model', content: [] },
        finishReason: 'length',
        finishMessage: undefined,
        custom: { citationMetadata: undefined, safetyRatings: undefined },
      },
    },
    {
      should: 'handle SAFETY finish reason',
      geminiCandidate: {
        index: 0,
        content: { role: 'model', parts: [] },
        finishReason: 'SAFETY',
      },
      expectedOutput: {
        index: 0,
        message: { role: 'model', content: [] },
        finishReason: 'blocked',
        finishMessage: undefined,
        custom: { citationMetadata: undefined, safetyRatings: undefined },
      },
    },
    {
      should: 'handle RECITATION finish reason',
      geminiCandidate: {
        index: 0,
        content: { role: 'model', parts: [] },
        finishReason: 'RECITATION',
      },
      expectedOutput: {
        index: 0,
        message: { role: 'model', content: [] },
        finishReason: 'blocked',
        finishMessage: undefined,
        custom: { citationMetadata: undefined, safetyRatings: undefined },
      },
    },
    {
      should: 'handle OTHER finish reason',
      geminiCandidate: {
        index: 0,
        content: { role: 'model', parts: [] },
        finishReason: 'OTHER',
      },
      expectedOutput: {
        index: 0,
        message: { role: 'model', content: [] },
        finishReason: 'other',
        finishMessage: undefined,
        custom: { citationMetadata: undefined, safetyRatings: undefined },
      },
    },
    {
      should: 'should ignore empty parts',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {}, // this one should be skipped
            {
              text: 'Why did the dog go to the bank?\n\nTo get his bones cashed!',
            },
          ],
        },
        finishReason: 'STOP',
        safetyRatings: [
          { category: 'HARM_CATEGORY_HATE_SPEECH', probability: 'NEGLIGIBLE' },
        ],
      },
      expectedOutput: {
        index: 0,
        message: {
          role: 'model',
          content: [
            {
              text: 'Why did the dog go to the bank?\n\nTo get his bones cashed!',
            },
          ],
        },
        finishReason: 'stop',
        finishMessage: undefined,
        custom: {
          citationMetadata: undefined,
          safetyRatings: [
            {
              category: 'HARM_CATEGORY_HATE_SPEECH',
              probability: 'NEGLIGIBLE',
            },
          ],
        },
      },
    },
  ];
  for (const test of testCases) {
    it(test.should, () => {
      const result = fromGeminiCandidate(
        test.geminiCandidate as GenerateContentCandidate
      );
      assert.deepStrictEqual(result, test.expectedOutput);
    });
  }

  describe('fromGeminiFunctionCall partial tool requests', () => {
    it('should handle streaming function calls', () => {
      const chunks: CandidateData[] = [];
      // First chunk, defines the function call
      let result = fromGeminiCandidate(
        {
          index: 0,
          content: {
            role: 'model',
            parts: [
              {
                functionCall: {
                  name: 'getWeather',
                  id: '1234',
                  args: {},
                  willContinue: true,
                },
                thoughtSignature: 'thoughtSignature1234',
              },
            ],
          },
        },
        chunks
      );
      chunks.push(result);

      assert.deepStrictEqual(result.message.content[0].toolRequest, {
        name: 'getWeather',
        ref: '1234',
        input: {},
        partial: true,
      });

      // Second chunk, adds a partial argument
      result = fromGeminiCandidate(
        {
          index: 0,
          content: {
            role: 'model',
            parts: [
              {
                functionCall: {
                  partialArgs: [
                    {
                      jsonPath: '$.location',
                      stringValue: 'Paris, France',
                    },
                  ],
                  willContinue: true,
                },
              },
            ],
          },
        },
        chunks
      );
      chunks.push(result);

      assert.deepStrictEqual(result.message.content[0].toolRequest, {
        name: 'getWeather',
        ref: '1234',
        input: { location: 'Paris, France' },
        partial: true,
      });

      // Third chunk, adds another partial argument
      result = fromGeminiCandidate(
        {
          index: 0,
          content: {
            role: 'model',
            parts: [
              {
                functionCall: {
                  name: '',
                  partialArgs: [
                    {
                      jsonPath: '$.unit',
                      stringValue: 'celsius',
                    },
                  ],
                  willContinue: true,
                },
              },
            ],
          },
        },
        chunks
      );
      chunks.push(result);

      assert.deepStrictEqual(result.message.content[0].toolRequest, {
        name: 'getWeather',
        ref: '1234',
        input: { location: 'Paris, France', unit: 'celsius' },
        partial: true,
      });

      // Final chunk, finishes the call
      result = fromGeminiCandidate(
        {
          index: 0,
          content: {
            role: 'model',
            parts: [
              {
                functionCall: {},
              },
            ],
          },
        },
        chunks
      );
      chunks.push(result);

      assert.deepStrictEqual(result.message.content[0].toolRequest, {
        name: 'getWeather',
        ref: '1234',
        input: { location: 'Paris, France', unit: 'celsius' },
      });
    });
  });
});

describe('Part conversions back and forth', () => {
  it('should be symmetric for Genkit Part -> Gemini Part -> Genkit Part', () => {
    const genkitParts: GenkitPart[] = [
      { text: 'hello' },
      {
        media: { url: 'data:image/png;base64,123', contentType: 'image/png' },
      },
      {
        media: { url: 'gs://bucket/file.png', contentType: 'image/png' },
      },
      {
        toolRequest: { name: 'myTool', input: { arg: 1 }, ref: 'ref1' },
      },
      {
        toolResponse: { name: 'myTool', output: { res: 2 }, ref: 'ref1' },
      },
      {
        reasoning: 'I think therefore I am',
        metadata: { thoughtSignature: 'sig123' },
      },
      {
        custom: {
          executableCode: {
            language: ExecutableCodeLanguage.PYTHON,
            code: 'print("hello")',
          },
        },
      },
      {
        custom: {
          codeExecutionResult: {
            outcome: Outcome.OUTCOME_OK,
            output: 'hello\\n',
          },
        },
      },
      {
        custom: {
          geminiToolCall: {
            toolType: 'GOOGLE_SEARCH_WEB',
            args: { queries: ['Canada'] },
            id: 'a111',
          },
        },
        metadata: { metadata: { foo: 'bar' } },
      },
      {
        custom: {
          geminiToolResponse: {
            toolType: 'GOOGLE_SEARCH_WEB',
            response: { results: [] },
            id: 'a111',
          },
        },
      },
    ];

    for (const part of genkitParts) {
      const geminiPart = TEST_ONLY.toGeminiPart(part);
      const convertedBack = TEST_ONLY.fromGeminiPart(geminiPart);
      assert.deepStrictEqual(convertedBack, part);
    }
  });

  it('should be symmetric for Gemini Part -> Genkit Part -> Gemini Part', () => {
    const geminiParts: GeminiPart[] = [
      { text: 'hello' },
      { inlineData: { mimeType: 'image/png', data: '123' } },
      { fileData: { mimeType: 'image/png', fileUri: 'gs://bucket/file.png' } },
      { functionCall: { name: 'myTool', args: { arg: 1 }, id: 'ref1' } },
      {
        functionResponse: {
          name: 'myTool',
          response: { name: 'myTool', content: { res: 2 } },
          id: 'ref1',
        },
      },
      {
        thought: true,
        text: 'I think therefore I am',
        thoughtSignature: 'sig123',
      },
      {
        executableCode: {
          language: ExecutableCodeLanguage.PYTHON,
          code: 'print("hello")',
        },
      },
      {
        codeExecutionResult: {
          outcome: Outcome.OUTCOME_OK,
          output: 'hello\\n',
        },
      },
      {
        toolCall: {
          toolType: 'GOOGLE_SEARCH_WEB',
          args: { queries: ['Canada'] },
          id: 'a111',
        },
        partMetadata: { foo: 'bar' },
      },
      {
        toolResponse: {
          toolType: 'GOOGLE_SEARCH_WEB',
          response: { results: [] },
          id: 'a111',
        },
      },
    ];

    for (const part of geminiParts) {
      const genkitPart = TEST_ONLY.fromGeminiPart(part);
      const convertedBack = TEST_ONLY.toGeminiPart(genkitPart);
      assert.deepStrictEqual(convertedBack, part);
    }
  });
});

describe('toGeminiTool', () => {
  it('should convert Genkit tool to Gemini FunctionDeclaration', async () => {
    const got = toGeminiTool({
      name: 'foo',
      description: 'tool foo',
      inputSchema: toJsonSchema({
        schema: z.object({
          simpleString: z.string().describe('a string').nullable(),
          simpleNumber: z.number().describe('a number'),
          simpleBoolean: z.boolean().describe('a boolean').optional(),
          simpleArray: z.array(z.string()).describe('an array').optional(),
          simpleEnum: z
            .enum(['choice_a', 'choice_b'])
            .describe('an enum')
            .optional(),
          nestedObject: z
            .object({
              innerString: z.string(),
            })
            .describe('nested object')
            .optional(),
        }),
      }),
    });

    const want = {
      description: 'tool foo',
      name: 'foo',
      parameters: {
        properties: {
          simpleArray: {
            description: 'an array',
            items: {
              type: SchemaType.STRING,
            },
            type: SchemaType.ARRAY,
          },
          simpleBoolean: {
            description: 'a boolean',
            type: SchemaType.BOOLEAN,
          },
          simpleEnum: {
            description: 'an enum',
            enum: ['choice_a', 'choice_b'],
            type: SchemaType.STRING,
          },
          simpleNumber: {
            description: 'a number',
            type: SchemaType.NUMBER,
          },
          simpleString: {
            description: 'a string',
            nullable: true,
            type: SchemaType.STRING,
          },
          nestedObject: {
            description: 'nested object',
            type: SchemaType.OBJECT,
            properties: {
              innerString: {
                type: SchemaType.STRING,
              },
            },
            required: ['innerString'],
          },
        },
        required: ['simpleString', 'simpleNumber'],
        type: SchemaType.OBJECT,
      },
    };
    assert.deepStrictEqual(got, want);
  });

  it('should replace slashes in tool names', async () => {
    const got = toGeminiTool({
      name: 'my/tool/name',
      description: 'tool with slashes',
    });
    const want = {
      description: 'tool with slashes',
      name: 'my__tool__name',
      parameters: undefined,
    };
    assert.deepStrictEqual(got, want);
  });

  it('should throw an error for unsupported schema types', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'badSchemaTool',
          description: 'A tool with an invalid schema type',
          inputSchema: {
            type: 'function', // This maps to an invalid SchemaType enum
          } as any,
        }),
      /Unsupported property type FUNCTION/
    );
  });
});

describe('toGeminiFunctionModeEnum', () => {
  const testCases = [
    { input: undefined, expected: undefined },
    {
      input: 'MODE_UNSPECIFIED',
      expected: FunctionCallingMode.MODE_UNSPECIFIED,
    },
    { input: 'required', expected: FunctionCallingMode.ANY },
    { input: 'ANY', expected: FunctionCallingMode.ANY },
    { input: 'auto', expected: FunctionCallingMode.AUTO },
    { input: 'AUTO', expected: FunctionCallingMode.AUTO },
    { input: 'none', expected: FunctionCallingMode.NONE },
    { input: 'NONE', expected: FunctionCallingMode.NONE },
  ];

  for (const test of testCases) {
    it(`should return ${test.expected} for input '${test.input}'`, () => {
      assert.strictEqual(toGeminiFunctionModeEnum(test.input), test.expected);
    });
  }

  it('should throw for unsupported mode', () => {
    assert.throws(
      () => toGeminiFunctionModeEnum('unsupported'),
      /unsupported function calling mode: unsupported/
    );
  });
});

describe('applyGeminiPartialArgs', () => {
  const testCases = [
    {
      should: 'apply a simple string value',
      initialArgs: {},
      partialArgs: [{ jsonPath: '$.foo', stringValue: 'bar' }],
      expectedArgs: { foo: 'bar' },
    },
    {
      should: 'apply a simple number value',
      initialArgs: {},
      partialArgs: [{ jsonPath: '$.count', numberValue: 42 }],
      expectedArgs: { count: 42 },
    },
    {
      should: 'apply a simple boolean value',
      initialArgs: {},
      partialArgs: [{ jsonPath: '$.enabled', boolValue: true }],
      expectedArgs: { enabled: true },
    },
    {
      should: 'apply a null value',
      initialArgs: { key: 'not-null' },
      partialArgs: [{ jsonPath: '$.key', nullValue: 'NULL_VALUE' as const }],
      expectedArgs: { key: null },
    },
    {
      should: 'apply a value to a nested object',
      initialArgs: {},
      partialArgs: [{ jsonPath: '$.config.setting', stringValue: 'value' }],
      expectedArgs: { config: { setting: 'value' } },
    },
    {
      should: 'apply a value to an array index',
      initialArgs: { items: ['a', 'b'] },
      partialArgs: [{ jsonPath: '$.items[1]', stringValue: 'c' }],
      expectedArgs: { items: ['a', 'bc'] },
    },
    {
      should: 'create and apply a value to an array index',
      initialArgs: { items: [] },
      partialArgs: [{ jsonPath: '$.items[0]', stringValue: 'a' }],
      expectedArgs: { items: ['a'] },
    },
    {
      should: 'apply a value to a nested object within an array',
      initialArgs: { data: [{ id: 1, value: 'old' }] },
      partialArgs: [{ jsonPath: '$.data[0].value', stringValue: 'new' }],
      expectedArgs: { data: [{ id: 1, value: 'oldnew' }] },
    },
    {
      should: 'apply multiple partial args',
      initialArgs: {},
      partialArgs: [
        { jsonPath: '$.name', stringValue: 'test' },
        { jsonPath: '$.config.version', numberValue: 2 },
      ],
      expectedArgs: { name: 'test', config: { version: 2 } },
    },
    {
      should: 'overwrite an existing value',
      initialArgs: { property: 'initial' },
      partialArgs: [{ jsonPath: '$.property', stringValue: 'updated' }],
      expectedArgs: { property: 'initialupdated' },
    },
    {
      should: 'create deeply nested objects and arrays',
      initialArgs: {},
      partialArgs: [{ jsonPath: '$.a.b[0].c', stringValue: 'deep' }],
      expectedArgs: { a: { b: [{ c: 'deep' }] } },
    },
    {
      should: 'create a nested array',
      initialArgs: {},
      partialArgs: [{ jsonPath: '$.data[0][0]', stringValue: 'nested' }],
      expectedArgs: { data: [['nested']] },
    },
    {
      should: 'apply a value to a multi-dimensional array',
      initialArgs: {
        matrix: [
          [1, 2],
          [3, 4],
        ],
      },
      partialArgs: [{ jsonPath: '$.matrix[1][0]', numberValue: 5 }],
      expectedArgs: {
        matrix: [
          [1, 2],
          [5, 4],
        ],
      },
    },
    {
      should: 'add a new property to an existing object',
      initialArgs: { user: { name: 'John' } },
      partialArgs: [{ jsonPath: '$.user.age', numberValue: 30 }],
      expectedArgs: { user: { name: 'John', age: 30 } },
    },
    {
      should: 'add a new element to an existing array',
      initialArgs: { scores: [10, 20] },
      partialArgs: [{ jsonPath: '$.scores[2]', numberValue: 30 }],
      expectedArgs: { scores: [10, 20, 30] },
    },
  ];
  for (const test of testCases) {
    it(`should ${test.should}`, () => {
      const functionCall: FunctionCall = {
        name: 'test',
        args: test.initialArgs,
      };
      const part: GeminiPart = {
        functionCall: {
          name: 'test',
          args: {},
          partialArgs: test.partialArgs,
        },
      } as GeminiPart;
      applyGeminiPartialArgs(
        functionCall.args!,
        part.functionCall?.partialArgs!
      );
      assert.deepStrictEqual(functionCall.args, test.expectedArgs);
    });
  }
});
