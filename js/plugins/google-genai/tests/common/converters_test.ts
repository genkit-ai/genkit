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
        'should send speechMetadata (TTS speaker and style) on text parts',
      inputMessage: {
        role: 'user',
        content: [
          {
            text: "How's it going today, Jane?",
            metadata: {
              speechMetadata: { speaker: 'Joe', style: 'cheerful' },
            },
          },
          {
            text: 'Not too bad.',
            metadata: { speechMetadata: { speaker: 'Jane' } },
          },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          {
            text: "How's it going today, Jane?",
            speechMetadata: { speaker: 'Joe', style: 'cheerful' },
          },
          { text: 'Not too bad.', speechMetadata: { speaker: 'Jane' } },
        ],
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
        role: 'user',
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
        role: 'user',
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
        'should transform genkit message (tool response with missing output and mixed content) correctly',
      inputMessage: {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'screenshot',
              ref: '1',
              content: [
                { text: 'this is a test' },
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
        role: 'user',
        parts: [
          {
            functionResponse: {
              id: '1',
              name: 'screenshot',
              response: {
                name: 'screenshot',
              },
              parts: [
                {
                  text: 'this is a test',
                },
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
        'should transform genkit message (tool response with output object and text content) correctly',
      inputMessage: {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'screenshot',
              ref: '2',
              output: { status: 'ok' },
              content: [{ text: 'this is a test' }],
            },
          },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          {
            functionResponse: {
              id: '2',
              name: 'screenshot',
              response: {
                name: 'screenshot',
                content: { status: 'ok' },
              },
              parts: [
                {
                  text: 'this is a test',
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
        role: 'user',
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
    {
      should: 'should transform resource part',
      inputMessage: {
        role: 'user',
        content: [
          {
            resource: {
              uri: 'file:///some/file.txt',
            },
            metadata: {
              mimeType: 'text/plain',
            },
          },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          {
            fileData: {
              fileUri: 'file:///some/file.txt',
              mimeType: 'text/plain',
            },
          },
        ],
      },
    },
    {
      should: 'should transform resource part with default mimeType',
      inputMessage: {
        role: 'user',
        content: [
          {
            resource: {
              uri: 'file:///some/file.txt',
            },
          },
        ],
      },
      expectedOutput: {
        role: 'user',
        parts: [
          {
            fileData: {
              fileUri: 'file:///some/file.txt',
              mimeType: 'application/octet-stream',
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

  it('should not throw on media part missing contentType for non-data URL', () => {
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
    assert.deepStrictEqual(toGeminiMessage(inputMessage as MessageData), {
      role: 'user',
      parts: [
        {
          fileData: {
            fileUri: 'gs://bucket/file',
          },
        },
      ],
    });
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
      should:
        'should transform gemini candidate (fileData without mimeType) correctly',
      geminiCandidate: {
        index: 0,
        content: {
          role: 'model',
          parts: [
            {
              fileData: {
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

  it('should resolve array item references in MCP tool schemas', () => {
    const got = toGeminiTool({
      name: 'create_draft',
      description: 'Create a draft',
      inputSchema: {
        type: 'object',
        properties: {
          attachments: {
            type: 'array',
            items: { $ref: '#/$defs/Attachment' },
          },
        },
        $defs: {
          Attachment: {
            type: 'object',
            properties: { content: { type: 'string' } },
            required: ['content'],
          },
        },
      },
    });

    assert.deepStrictEqual(got.parameters?.properties?.attachments, {
      type: SchemaType.ARRAY,
      items: {
        type: SchemaType.OBJECT,
        properties: { content: { type: SchemaType.STRING } },
        required: ['content'],
      },
    });
  });

  it('should resolve a root reference with definitions beside it', () => {
    const got = toGeminiTool({
      name: 'root_ref',
      description: 'Tool with a referenced input schema',
      inputSchema: {
        $ref: '#/$defs/Args',
        $defs: {
          Args: {
            type: 'object',
            properties: { query: { type: 'string' } },
            required: ['query'],
          },
        },
      },
    });
    assert.deepStrictEqual(got.parameters, {
      type: SchemaType.OBJECT,
      properties: { query: { type: SchemaType.STRING } },
      required: ['query'],
    });
  });

  it('should allow Genkit annotations beside a tool schema reference', () => {
    const got = toGeminiTool({
      name: 'annotated_ref',
      description: 'Tool with an annotated reference',
      inputSchema: {
        type: 'object',
        properties: {
          name: {
            $ref: '#/$defs/Name',
            'x-genkit-ui-label': 'Name',
          },
        },
        $defs: { Name: { type: 'string' } },
      },
    });
    assert.deepStrictEqual(got.parameters?.properties?.name, {
      type: SchemaType.STRING,
    });
  });

  it('should preserve untyped descendants inside referenced schemas', () => {
    const item = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        note: { anyOf: [{ type: 'string' }, { type: 'null' }], default: null },
        values: {
          type: 'array',
          items: { anyOf: [{ type: 'string' }, { type: 'null' }] },
        },
      },
    };
    const inline = toGeminiTool({
      name: 'nullable_note',
      description: 'Tool with an optional note',
      inputSchema: { type: 'object', properties: { item } },
    });
    const referenced = toGeminiTool({
      name: 'nullable_note',
      description: 'Tool with an optional note',
      inputSchema: {
        type: 'object',
        properties: { item: { $ref: '#/$defs/Item' } },
        $defs: { Item: item },
      },
    });
    assert.deepStrictEqual(referenced, inline);
    assert.deepStrictEqual(referenced.parameters?.properties?.item, {
      type: SchemaType.OBJECT,
      properties: {
        name: { type: SchemaType.STRING },
        values: { type: SchemaType.ARRAY, items: undefined },
      },
      required: undefined,
    });
  });

  it('should reject directly referenced schemas without a supported type', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'untyped_ref',
          description: 'Tool with an untyped reference target',
          inputSchema: {
            $ref: '#/$defs/Value',
            $defs: { Value: { anyOf: [{ type: 'string' }, { type: 'null' }] } },
          },
        }),
      /Referenced tool schema has no type supported by Gemini/
    );
  });

  it('should reject recursive references through typed descendants', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'recursive_ref',
          description: 'Tool with a recursive reference',
          inputSchema: {
            $ref: '#/$defs/Node',
            $defs: {
              Node: {
                type: 'object',
                properties: { next: { $ref: '#/$defs/Node' } },
              },
            },
          },
        }),
      /Recursive tool schema reference/
    );
  });

  it('should reject unresolved tool schema references before sending them to Gemini', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'bad_ref',
          description: 'Tool with a missing definition',
          inputSchema: {
            type: 'object',
            properties: {
              items: { type: 'array', items: { $ref: '#/$defs/Missing' } },
            },
          },
        }),
      /Unresolved tool schema reference #\/\$defs\/Missing/
    );
  });

  it('should reject constraints beside a tool schema reference', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'bad_combination',
          description: 'Tool with intersecting constraints',
          inputSchema: {
            type: 'object',
            properties: {
              item: {
                $ref: '#/$defs/Item',
                properties: { label: { type: 'string' } },
              },
            },
            $defs: {
              Item: {
                type: 'object',
                properties: { id: { type: 'string' } },
                required: ['id'],
              },
            },
          },
        }),
      /Unsupported sibling keyword properties alongside tool schema reference/
    );
  });

  it('should reject nullable beside a tool schema reference', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'nullable_ref',
          description: 'Tool with a nullable reference',
          inputSchema: {
            type: 'object',
            properties: {
              value: { $ref: '#/$defs/Value', nullable: true },
            },
            $defs: { Value: { type: 'string', nullable: false } },
          },
        }),
      /Unsupported sibling keyword nullable alongside tool schema reference/
    );
  });

  it('should reject nested schema resources with their own reference scope', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'nested_resource',
          description: 'Tool with a nested schema resource',
          inputSchema: {
            type: 'object',
            properties: { item: { $ref: '#/$defs/Item' } },
            $defs: {
              Item: {
                $id: 'https://example.test/item',
                type: 'object',
                properties: { name: { $ref: '#/$defs/Name' } },
                $defs: { Name: { type: 'string' } },
              },
              Name: { type: 'integer' },
            },
          },
        }),
      /nested schema resource/i
    );
    assert.throws(
      () =>
        toGeminiTool({
          name: 'inline_nested_resource',
          description: 'Tool with an inline nested schema resource',
          inputSchema: {
            type: 'object',
            properties: {
              item: {
                $id: 'https://example.test/item',
                type: 'object',
                properties: { name: { $ref: '#/$defs/Name' } },
                $defs: { Name: { type: 'string' } },
              },
            },
            $defs: { Name: { type: 'integer' } },
          },
        }),
      /Nested tool schema resources cannot be converted/
    );
    assert.throws(
      () =>
        toGeminiTool({
          name: 'draft4_nested_resource',
          description: 'Draft 4 tool with a nested id schema resource',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-04/schema#',
            type: 'object',
            properties: {
              child: {
                id: 'child.json',
                type: 'object',
                properties: { val: { $ref: '#/definitions/Value' } },
                definitions: { Value: { type: 'string' } },
              },
            },
            definitions: { Value: { type: 'integer' } },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message: /Nested tool schema resources cannot be converted/,
      }
    );

    const draft4DollarId = toGeminiTool({
      name: 'draft4_child_dollar_id_annotation',
      description: 'Draft 4 tool with non-resource $id keyword',
      inputSchema: {
        $schema: 'http://json-schema.org/draft-04/schema#',
        type: 'object',
        properties: {
          meta: {
            $id: 'metadata',
            type: 'object',
            properties: { name: { $ref: '#/definitions/Name' } },
          },
        },
        definitions: { Name: { type: 'string' } },
      },
    });
    assert.deepStrictEqual(draft4DollarId.parameters?.properties?.meta, {
      type: SchemaType.OBJECT,
      properties: { name: { type: SchemaType.STRING } },
      required: undefined,
    });

    const draft2020 = toGeminiTool({
      name: 'draft2020_child_id_annotation',
      description: 'Draft 2020-12 tool with non-resource id keyword',
      inputSchema: {
        $schema: 'https://json-schema.org/draft/2020-12/schema',
        type: 'object',
        properties: {
          address: {
            id: 'address',
            type: 'object',
            properties: { street: { $ref: '#/$defs/Street' } },
          },
        },
        $defs: { Street: { type: 'string' } },
      },
    });
    assert.deepStrictEqual(draft2020.parameters?.properties?.address, {
      type: SchemaType.OBJECT,
      properties: { street: { type: SchemaType.STRING } },
      required: undefined,
    });

    const customDialect = toGeminiTool({
      name: 'custom_dialect_child_id_annotation',
      description: 'Custom dialect URI containing draft-04 substring',
      inputSchema: {
        $schema: 'https://schemas.example/draft-04-compat/schema',
        type: 'object',
        properties: {
          address: {
            id: 'address',
            type: 'object',
            properties: { street: { $ref: '#/$defs/Street' } },
          },
        },
        $defs: { Street: { type: 'string' } },
      },
    });
    assert.deepStrictEqual(customDialect.parameters?.properties?.address, {
      type: SchemaType.OBJECT,
      properties: { street: { type: SchemaType.STRING } },
      required: undefined,
    });

    const draft7Anchor = toGeminiTool({
      name: 'draft7_plain_name_fragment_id',
      description: 'Draft 7 child with plain-name fragment $id',
      inputSchema: {
        $schema: 'http://json-schema.org/draft-07/schema#',
        type: 'object',
        properties: {
          contact: {
            $id: '#contact',
            type: 'object',
            properties: { name: { $ref: '#/definitions/Name' } },
          },
        },
        definitions: { Name: { type: 'string' } },
      },
    });
    assert.deepStrictEqual(draft7Anchor.parameters?.properties?.contact, {
      type: SchemaType.OBJECT,
      properties: { name: { type: SchemaType.STRING } },
      required: undefined,
    });
  });

  it('should ignore $ref sibling keywords for Draft 4-7 schemas', () => {
    const draft7 = toGeminiTool({
      name: 'draft7_ref_siblings',
      description: 'Draft 7 tool with sibling keywords alongside $ref',
      inputSchema: {
        $schema: 'http://json-schema.org/draft-07/schema#',
        type: 'object',
        properties: {
          names: {
            type: 'array',
            items: { $ref: '#/definitions/Name', type: 'string', $id: 'alias' },
          },
          alias: {
            $ref: '#/definitions/Name',
            type: 'integer',
            description: 'ignored sibling description',
          },
          chained: {
            $ref: '#/definitions/Alias',
          },
        },
        definitions: {
          Name: { type: 'string', description: 'canonical name' },
          Alias: { $ref: '#/definitions/Name', $id: 'chained-alias' },
        },
      },
    });
    assert.deepStrictEqual(draft7.parameters?.properties, {
      names: {
        type: SchemaType.ARRAY,
        items: { type: SchemaType.STRING, description: 'canonical name' },
      },
      alias: { type: SchemaType.STRING, description: 'canonical name' },
      chained: { type: SchemaType.STRING, description: 'canonical name' },
    });

    assert.throws(
      () =>
        toGeminiTool({
          name: 'draft7_enclosing_nested_resource',
          description: 'Draft 7 tool with enclosing nested $id over a $ref',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
            type: 'object',
            properties: {
              child: {
                $id: 'https://example.test/child',
                type: 'object',
                properties: {
                  name: { $ref: '#/definitions/Name', $id: 'ignored' },
                },
                definitions: { Name: { type: 'string' } },
              },
            },
            definitions: { Name: { type: 'integer' } },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message: /Nested tool schema resources cannot be converted/,
      }
    );

    assert.throws(
      () =>
        toGeminiTool({
          name: 'draft2020_ref_sibling_type',
          description: 'Draft 2020-12 tool with sibling type alongside $ref',
          inputSchema: {
            $schema: 'https://json-schema.org/draft/2020-12/schema',
            type: 'object',
            properties: {
              names: {
                type: 'array',
                items: { $ref: '#/$defs/Name', type: 'string' },
              },
            },
            $defs: { Name: { type: 'string' } },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message:
          /Unsupported sibling keyword type alongside tool schema reference/,
      }
    );

    const draft2020Anchor = toGeminiTool({
      name: 'draft2020_ref_sibling_annotations',
      description:
        'Draft 2020-12 tool with sibling $anchor, description, and content annotations alongside $ref',
      inputSchema: {
        $schema: 'https://json-schema.org/draft/2020-12/schema',
        type: 'object',
        properties: {
          names: {
            type: 'array',
            items: {
              $ref: '#/$defs/Name',
              $anchor: 'nameItem',
              description: 'item name',
            },
          },
          attachment: {
            $ref: '#/$defs/Text',
            contentEncoding: 'base64',
            contentMediaType: 'application/json',
            contentSchema: { type: 'object' },
          },
        },
        $defs: {
          Name: { type: 'string', description: 'canonical name' },
          Text: { type: 'string', description: 'raw text' },
        },
      },
    });
    assert.deepStrictEqual(draft2020Anchor.parameters?.properties, {
      names: {
        type: SchemaType.ARRAY,
        items: { type: SchemaType.STRING, description: 'item name' },
      },
      attachment: { type: SchemaType.STRING, description: 'raw text' },
    });

    assert.throws(
      () =>
        toGeminiTool({
          name: 'draft2020_ref_sibling_id',
          description: 'Draft 2020-12 tool with sibling $id alongside $ref',
          inputSchema: {
            $schema: 'https://json-schema.org/draft/2020-12/schema',
            type: 'object',
            properties: {
              names: {
                type: 'array',
                items: { $ref: '#/$defs/Name', $id: 'alias' },
              },
            },
            $defs: { Name: { type: 'string' } },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message: /Nested tool schema resources cannot be converted/,
      }
    );
  });

  it('should reject pointers that cross nested schema resource boundaries', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'crossed_resource',
          description: 'Tool with a pointer into a nested resource',
          inputSchema: {
            type: 'object',
            properties: {
              item: { $ref: '#/$defs/Item/properties/name' },
            },
            $defs: {
              Item: {
                $id: 'https://example.test/item',
                type: 'object',
                properties: { name: { $ref: '#/$defs/Name' } },
                $defs: { Name: { type: 'string' } },
              },
              Name: { type: 'integer' },
            },
          },
        }),
      /Tool schema reference crosses a nested schema resource/
    );
  });

  it('should allow nested schema IDs when no references need resolving', () => {
    const got = toGeminiTool({
      name: 'nested_id',
      description: 'Tool with a nested schema ID',
      inputSchema: {
        type: 'object',
        properties: {
          item: {
            $id: 'https://example.test/item',
            type: 'object',
            properties: { name: { type: 'string' } },
          },
        },
      },
    });
    assert.deepStrictEqual(got.parameters?.properties?.item, {
      type: SchemaType.OBJECT,
      properties: { name: { type: SchemaType.STRING } },
      required: undefined,
    });
  });

  it('should not treat a $defs entry named $id as a nested schema resource', () => {
    const got = toGeminiTool({
      name: 'defs_named_id',
      description: 'Tool with a $defs entry named $id',
      inputSchema: {
        type: 'object',
        properties: {
          alias: { $ref: '#/$defs/Alias' },
          idValue: { $ref: '#/$defs/$id' },
        },
        $defs: {
          $id: { type: 'string' },
          Alias: { $ref: '#/$defs/Name' },
          Name: { type: 'string' },
        },
      },
    });
    assert.deepStrictEqual(got.parameters?.properties, {
      alias: { type: SchemaType.STRING },
      idValue: { type: SchemaType.STRING },
    });
  });

  it('should decode percent-encoded pointer separators and escaped keys', () => {
    const got = toGeminiTool({
      name: 'encoded_pointers',
      description: 'Tool with percent-encoded and escaped JSON Pointers',
      inputSchema: {
        type: 'object',
        properties: {
          encoded: { $ref: '#%2F$defs%2FName' },
          escaped: { $ref: '#/$defs/a~1b~0c' },
        },
        $defs: {
          Name: { type: 'string' },
          'a/b~c': { type: 'integer' },
        },
      },
    });
    assert.deepStrictEqual(got.parameters?.properties, {
      encoded: { type: SchemaType.STRING },
      escaped: { type: SchemaType.INTEGER },
    });
  });

  it('should reject malformed JSON Pointer tilde escapes even if literal key exists', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'bad_escape',
          description: 'Tool with malformed ~2 escape',
          inputSchema: {
            type: 'object',
            properties: { value: { $ref: '#/$defs/a~2b' } },
            $defs: { 'a~2b': { type: 'string' } },
          },
        }),
      /Invalid tool schema reference #\/\$defs\/a~2b/
    );
  });

  it('should reject recursive references across encoded and unencoded pointer spellings', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'encoded_recursive_ref',
          description: 'Tool with an encoded recursive reference',
          inputSchema: {
            type: 'object',
            properties: { node: { $ref: '#/$defs/Node' } },
            $defs: {
              Node: {
                type: 'object',
                properties: { next: { $ref: '#%2F$defs%2FNode' } },
              },
            },
          },
        }),
      /Recursive tool schema reference/
    );
  });

  it('should resolve referenced nullable primitives and reject null-only type arrays', () => {
    const got = toGeminiTool({
      name: 'nullable_primitive_ref',
      description: 'Tool with a referenced nullable primitive',
      inputSchema: {
        type: 'object',
        properties: { label: { $ref: '#/$defs/NullableString' } },
        $defs: { NullableString: { type: ['string', 'null'] } },
      },
    });
    assert.deepStrictEqual(got.parameters?.properties?.label, {
      type: SchemaType.STRING,
      nullable: true,
    });

    assert.throws(
      () =>
        toGeminiTool({
          name: 'null_only_ref',
          description: 'Tool with a referenced null-only type array',
          inputSchema: {
            type: 'object',
            properties: { nothing: { $ref: '#/$defs/NullOnly' } },
            $defs: { NullOnly: { type: ['null'] } },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message: /Referenced tool schema has no type supported by Gemini/,
      }
    );
  });

  it('should preserve the fallback for boolean item schemas', () => {
    const got = toGeminiTool({
      name: 'boolean_item',
      description: 'Tool with a boolean item schema',
      inputSchema: {
        type: 'object',
        properties: { items: { type: 'array', items: true } },
      },
    });
    assert.deepStrictEqual(got.parameters?.properties?.items, {
      type: SchemaType.ARRAY,
      items: undefined,
    });
  });

  it('should reject a reference to a boolean schema', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'boolean_true_ref',
          description: 'Tool with an unconstrained referenced item',
          inputSchema: {
            type: 'object',
            properties: {
              items: { type: 'array', items: { $ref: '#/$defs/Any' } },
            },
            $defs: { Any: true },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message: /resolves to an unsupported true schema/,
      }
    );

    assert.throws(
      () =>
        toGeminiTool({
          name: 'boolean_false_ref',
          description: 'Tool with a false referenced item',
          inputSchema: {
            type: 'object',
            properties: {
              items: { type: 'array', items: { $ref: '#/$defs/Never' } },
            },
            $defs: { Never: false },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message: /resolves to an unsupported false schema/,
      }
    );
  });

  it('should reject empty-string sibling keywords alongside a tool schema reference', () => {
    assert.throws(
      () =>
        toGeminiTool({
          name: 'empty_sibling_ref',
          description: 'Tool with an empty-string keyword alongside a $ref',
          inputSchema: {
            type: 'object',
            properties: {
              value: { $ref: '#/$defs/S', '': 0, type: 'integer' },
            },
            $defs: { S: { type: 'string' } },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message:
          /Unsupported sibling keyword "" alongside tool schema reference/,
      }
    );
  });

  it('should allow nested schema IDs with unused items and without double-charging node budget', () => {
    const unusedItemsGot = toGeminiTool({
      name: 'nested_id_unused_items',
      description: 'Object with nested $id and ignored items keyword',
      inputSchema: {
        type: 'object',
        properties: {
          item: {
            $id: 'https://example.test/item',
            type: 'object',
            properties: { name: { type: 'string' } },
            items: { $ref: '#/$defs/T' },
          },
        },
        $defs: { T: { type: 'string' } },
      },
    });
    assert.deepStrictEqual(unusedItemsGot.parameters?.properties?.item, {
      type: SchemaType.OBJECT,
      properties: { name: { type: SchemaType.STRING } },
      required: undefined,
    });

    const leafProperties: Record<string, any> = {};
    for (let i = 0; i < 1400; i++) {
      leafProperties[`p${i}`] = { type: 'string' };
    }
    const nestedGot = toGeminiTool({
      name: 'nested_ids_budget',
      description: 'Three nested $id objects over 1400 reference-free leaves',
      inputSchema: {
        type: 'object',
        properties: {
          outer: {
            $id: 'https://example.test/outer',
            type: 'object',
            properties: {
              middle: {
                $id: 'https://example.test/middle',
                type: 'object',
                properties: {
                  inner: {
                    $id: 'https://example.test/inner',
                    type: 'object',
                    properties: leafProperties,
                  },
                },
              },
            },
          },
        },
      },
    });
    const innerProps =
      nestedGot.parameters?.properties?.outer?.properties?.middle?.properties
        ?.inner?.properties;
    assert.strictEqual(Object.keys(innerProps ?? {}).length, 1400);
    assert.deepStrictEqual(innerProps?.p1399, { type: SchemaType.STRING });
  });

  it('should convert wide inline schemas without hitting the reference expansion budget', () => {
    const flatProperties: Record<string, any> = {};
    for (let i = 0; i < 4096; i++) {
      flatProperties[`p${i}`] = { type: 'string' };
    }
    const flatGot = toGeminiTool({
      name: 'wide_inline_schema',
      description: 'Flat tool with 4096 inline properties and no references',
      inputSchema: {
        type: 'object',
        properties: flatProperties,
      },
    });
    assert.strictEqual(
      Object.keys(flatGot.parameters?.properties ?? {}).length,
      4096
    );
    assert.deepStrictEqual(flatGot.parameters?.properties?.p4095, {
      type: SchemaType.STRING,
    });

    const nestedGot = toGeminiTool({
      name: 'wide_nested_id',
      description: 'Tool with a wide inline schema under a nested $id',
      inputSchema: {
        type: 'object',
        properties: {
          wide: {
            $id: 'https://example.test/wide',
            type: 'object',
            properties: flatProperties,
          },
        },
      },
    });
    assert.strictEqual(
      Object.keys(nestedGot.parameters?.properties?.wide?.properties ?? {})
        .length,
      4096
    );
    assert.deepStrictEqual(
      nestedGot.parameters?.properties?.wide?.properties?.p4095,
      { type: SchemaType.STRING }
    );

    let deepInline: Record<string, any> = { type: 'string' };
    for (let i = 0; i < 130; i++) {
      deepInline = { type: 'object', properties: { child: deepInline } };
    }
    const deepGot = toGeminiTool({
      name: 'deep_inline_schema',
      description: 'Tool with 130 nested inline objects and no references',
      inputSchema: deepInline,
    });
    let cursor = deepGot.parameters as any;
    for (let i = 0; i < 130; i++) {
      assert.strictEqual(cursor?.type, SchemaType.OBJECT);
      cursor = cursor.properties?.child;
    }
    assert.deepStrictEqual(cursor, { type: SchemaType.STRING });
  });

  it('should reject $ref-expanded schemas exceeding the node expansion budget', () => {
    const refProperties: Record<string, any> = {};
    for (let i = 0; i < 4097; i++) {
      refProperties[`p${i}`] = { $ref: '#/$defs/Leaf' };
    }
    assert.throws(
      () =>
        toGeminiTool({
          name: 'wide_ref_expansion',
          description: 'Tool expanding more than 4096 referenced nodes',
          inputSchema: {
            type: 'object',
            properties: refProperties,
            $defs: { Leaf: { type: 'string' } },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message: /Tool schema is too large to convert/,
      }
    );
  });

  it('should account for repeated enum values in the schema expansion budget', () => {
    const inlineLargeEnum = Array.from({ length: 4095 }, (_, i) => `e${i}`);
    const inlineGot = toGeminiTool({
      name: 'inline_large_enum',
      description: 'Tool with a large inline enum and no references',
      inputSchema: {
        type: 'object',
        properties: {
          code: { type: 'string', enum: inlineLargeEnum },
        },
      },
    });
    assert.strictEqual(
      inlineGot.parameters?.properties?.code?.enum?.length,
      4095
    );

    const smallEnum = toGeminiTool({
      name: 'small_enum_ref',
      description: 'Tool referencing a small enum twice',
      inputSchema: {
        type: 'object',
        properties: {
          first: { $ref: '#/$defs/Color' },
          second: { $ref: '#/$defs/Color' },
        },
        $defs: { Color: { type: 'string', enum: ['red', 'blue'] } },
      },
    });
    assert.deepStrictEqual(smallEnum.parameters?.properties, {
      first: { type: SchemaType.STRING, enum: ['red', 'blue'] },
      second: { type: SchemaType.STRING, enum: ['red', 'blue'] },
    });

    const bigEnum = Array.from({ length: 500 }, (_, i) => `v${i}`);
    const properties: Record<string, any> = {};
    for (let i = 0; i < 10; i++) {
      properties[`field${i}`] = { $ref: '#/$defs/BigEnum' };
    }
    assert.throws(
      () =>
        toGeminiTool({
          name: 'large_enum_ref',
          description: 'Tool repeating a large enum via $ref',
          inputSchema: {
            type: 'object',
            properties,
            $defs: { BigEnum: { type: 'string', enum: bigEnum } },
          },
        }),
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
        message: /Tool schema is too large to convert/,
      }
    );
  });

  it('should bound expansion of referenced tool schemas', () => {
    const defs: Record<string, any> = { D14: { type: 'string' } };
    for (let i = 13; i >= 0; i--) {
      defs[`D${i}`] = {
        type: 'object',
        properties: {
          left: { $ref: `#/$defs/D${i + 1}` },
          right: { $ref: `#/$defs/D${i + 1}` },
        },
      };
    }
    assert.throws(
      () =>
        toGeminiTool({
          name: 'large_schema',
          description: 'Tool with repeated references',
          inputSchema: {
            type: 'object',
            properties: { root: { $ref: '#/$defs/D0' } },
            $defs: defs,
          },
        }),
      /Tool schema is too large to convert/
    );
  });

  it('should reject reference chains before they overflow the call stack', () => {
    const defs: Record<string, any> = { D200: { type: 'string' } };
    for (let i = 199; i >= 0; i--) {
      defs[`D${i}`] = { $ref: `#/$defs/D${i + 1}` };
    }
    assert.throws(
      () =>
        toGeminiTool({
          name: 'deep_schema',
          description: 'Tool with a deep reference chain',
          inputSchema: {
            type: 'object',
            properties: { value: { $ref: '#/$defs/D0' } },
            $defs: defs,
          },
        }),
      /Tool schema is too deep to convert/
    );
  });

  it('should preserve a __proto__ property on referenced objects during serialization', () => {
    const inputSchema = JSON.parse(`{
      "type": "object",
      "properties": {
        "payload": { "$ref": "#/$defs/Payload" }
      },
      "$defs": {
        "Payload": {
          "type": "object",
          "properties": {
            "__proto__": { "type": "string" }
          },
          "required": ["__proto__"]
        }
      }
    }`);
    const got = toGeminiTool({
      name: 'proto_property_ref',
      description: 'Tool referencing an object with a __proto__ property',
      inputSchema,
    });
    const payloadProps = got.parameters?.properties?.payload?.properties;
    assert.ok(payloadProps);
    assert.strictEqual(
      Object.prototype.hasOwnProperty.call(payloadProps, '__proto__'),
      true
    );
    assert.deepStrictEqual(
      Object.getOwnPropertyDescriptor(payloadProps, '__proto__')?.value,
      { type: SchemaType.STRING }
    );
    assert.deepStrictEqual(got.parameters?.properties?.payload?.required, [
      '__proto__',
    ]);
    assert.deepStrictEqual(JSON.parse(JSON.stringify(got.parameters)), {
      type: 'OBJECT',
      properties: {
        payload: {
          type: 'OBJECT',
          properties: JSON.parse('{"__proto__":{"type":"STRING"}}'),
          required: ['__proto__'],
        },
      },
    });
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
