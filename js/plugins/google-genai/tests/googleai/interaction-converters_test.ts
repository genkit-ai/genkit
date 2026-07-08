/**
 * Copyright 2026 Google LLC
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
import { MessageData, Part } from 'genkit';
import { ToolDefinition } from 'genkit/model';
import { describe, it } from 'node:test';
import {
  ensureToolIds,
  fromInteraction,
  fromInteractionContent,
  fromInteractionDelta,
  fromInteractionStep,
  toInteractionConfigTool,
  toInteractionContent,
  toInteractionGenerationConfig,
  toInteractionGoogleSearch,
  toInteractionRole,
  toInteractionSteps,
  toInteractionTool,
} from '../../src/googleai/interaction-converters.js';
import {
  Content,
  GeminiInteraction,
  Step,
  StepDeltaData,
} from '../../src/googleai/interaction-types.js';

describe('Interaction Converters', () => {
  describe('ensureToolIds', () => {
    it('should assign IDs to tool requests without refs', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'tool1', input: {} } },
            { toolRequest: { name: 'tool2', input: {} } },
          ],
        },
      ];
      const result = ensureToolIds(messages);
      const req1 = result[0].content[0].toolRequest!;
      const req2 = result[0].content[1].toolRequest!;
      assert.ok(req1.ref && req1.ref.startsWith('genkit-auto-id-'));
      assert.ok(req2.ref && req2.ref.startsWith('genkit-auto-id-'));
      assert.notStrictEqual(req1.ref, req2.ref);
    });

    it('should assign matching IDs to tool responses without refs based on order', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'tool1', input: {} } },
            { toolRequest: { name: 'tool2', input: {} } },
          ],
        },
        {
          role: 'tool',
          content: [
            { toolResponse: { name: 'tool1', output: {} } },
            { toolResponse: { name: 'tool2', output: {} } },
          ],
        },
      ];
      const result = ensureToolIds(messages);
      const req1 = result[0].content[0].toolRequest!;
      const req2 = result[0].content[1].toolRequest!;
      const res1 = result[1].content[0].toolResponse!;
      const res2 = result[1].content[1].toolResponse!;

      assert.ok(req1.ref);
      assert.strictEqual(req1.ref, res1.ref);
      assert.ok(req2.ref);
      assert.strictEqual(req2.ref, res2.ref);
    });

    it('should assign orphan ID to tool response if no matching request', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [{ toolResponse: { name: 'tool1', output: {} } }],
        },
      ];
      const result = ensureToolIds(messages);
      const res1 = result[0].content[0].toolResponse!;
      assert.ok(res1.ref && res1.ref.startsWith('genkit-orphan-id-'));
    });

    it('should preserve existing refs', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'tool1', input: {}, ref: 'existing-id' } },
          ],
        },
      ];
      const result = ensureToolIds(messages);
      const req1 = result[0].content[0].toolRequest!;
      assert.strictEqual(req1.ref, 'existing-id');
    });
  });

  describe('toInteractionRole', () => {
    it('should convert user role', () => {
      assert.strictEqual(toInteractionRole('user'), 'user');
    });
    it('should convert model role', () => {
      assert.strictEqual(toInteractionRole('model'), 'model');
    });
    it('should convert tool role to user', () => {
      assert.strictEqual(toInteractionRole('tool'), 'user');
    });
    it('should throw for system role', () => {
      assert.throws(
        () => toInteractionRole('system'),
        /System role should be handled as system_instruction/
      );
    });
  });

  describe('toInteractionTool', () => {
    it('should convert ToolDefinition to InteractionTool', () => {
      const tool: ToolDefinition = {
        name: 'myFunc',
        description: 'desc',
        inputSchema: {
          type: 'object',
          properties: { arg: { type: 'string' } },
        },
      };
      const result = toInteractionTool(tool);
      assert.deepStrictEqual(result, {
        type: 'function',
        name: 'myFunc',
        description: 'desc',
        parameters: {
          type: 'object',
          properties: { arg: { type: 'string' } },
        },
      });
    });
  });

  describe('toInteractionGenerationConfig', () => {
    it('should flatten thinkingConfig and map casing', () => {
      const config = {
        thinkingConfig: {
          thinkingLevel: 'HIGH',
          includeThoughts: true,
        },
      };
      const result = toInteractionGenerationConfig(config);
      assert.deepStrictEqual(result, {
        thinking_level: 'high',
        thinking_summaries: 'auto',
      });
    });

    it('should preserve other properties in thinkingConfig', () => {
      const config = {
        thinkingConfig: {
          thinkingLevel: 'LOW',
          includeThoughts: false,
          thinkingBudget: 1024,
          unknownProp: 'test',
        },
      };
      const result = toInteractionGenerationConfig(config);
      assert.deepStrictEqual(result, {
        thinking_level: 'low',
        thinking_summaries: 'none',
        thinking_config: {
          thinking_budget: 1024,
          unknown_prop: 'test',
        },
      });
    });

    it('should handle already snake_cased thinking_config', () => {
      const config = {
        thinking_config: {
          thinking_level: 'MEDIUM',
          include_thoughts: true,
          thinking_budget: 2048,
        },
      };
      const result = toInteractionGenerationConfig(config);
      assert.deepStrictEqual(result, {
        thinking_level: 'medium',
        thinking_summaries: 'auto',
        thinking_config: {
          thinking_budget: 2048,
        },
      });
    });
  });

  describe('toInteractionGoogleSearch', () => {
    it('should handle boolean true config', () => {
      const result = toInteractionGoogleSearch(true);
      assert.deepStrictEqual(result, { type: 'google_search' });
    });

    it('should handle array format in snake_case', () => {
      const result = toInteractionGoogleSearch({
        search_types: ['web_search', 'image_search'],
      });
      assert.deepStrictEqual(result, {
        type: 'google_search',
        search_types: ['web_search', 'image_search'],
      });
    });

    it('should handle object format in camelCase', () => {
      const result = toInteractionGoogleSearch({
        searchTypes: { webSearch: {}, enterpriseWebSearch: {} },
      });
      assert.deepStrictEqual(result, {
        type: 'google_search',
        search_types: ['web_search', 'enterprise_web_search'],
      });
    });

    it('should handle empty config object', () => {
      const result = toInteractionGoogleSearch({});
      assert.deepStrictEqual(result, { type: 'google_search' });
    });

    it('should pass through unrecognized fields from search types array', () => {
      const result = toInteractionGoogleSearch({
        searchTypes: ['web_search', 'my_custom_search'],
      });
      assert.deepStrictEqual(result, {
        type: 'google_search',
        search_types: ['web_search', 'my_custom_search'],
      });
    });

    it('should pass through unrecognized fields from search types object', () => {
      const result = toInteractionGoogleSearch({
        searchTypes: { webSearch: {}, customSearchPlugin: {} },
      });
      assert.deepStrictEqual(result, {
        type: 'google_search',
        search_types: ['web_search', 'custom_search_plugin'],
      });
    });

    it('should throw an error for invalid top-level config', () => {
      assert.throws(
        () => toInteractionGoogleSearch('invalid'),
        /Invalid configuration for googleSearch tool/
      );
    });

    it('should throw an error for invalid searchTypes format', () => {
      assert.throws(
        () => toInteractionGoogleSearch({ searchTypes: 'invalid' }),
        /Invalid searchTypes configuration/
      );
    });

    it('should throw an error for invalid search type in array', () => {
      assert.throws(
        () => toInteractionGoogleSearch({ searchTypes: [123] }),
        /Invalid search type/
      );
    });
  });

  describe('toInteractionConfigTool', () => {
    it('should throw an error if toolRaw is not an object', () => {
      assert.throws(
        () => toInteractionConfigTool('not-an-object'),
        /Invalid tool configuration/
      );
      assert.throws(
        () => toInteractionConfigTool(['not-an-object']),
        /Invalid tool configuration/
      );
      assert.throws(
        () => toInteractionConfigTool(null),
        /Invalid tool configuration/
      );
    });

    it('should handle built-in tool format with valid inputs', () => {
      const tool = { codeExecution: { someProp: 123 } };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'code_execution',
        some_prop: 123,
      });
    });

    it('should handle built-in tool format with boolean true', () => {
      const tool = { codeExecution: true };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'code_execution',
      });
    });

    it('should throw an error if built-in tool config is invalid type', () => {
      const tool = { codeExecution: 'invalid' };
      assert.throws(
        () => toInteractionConfigTool(tool),
        /Invalid configuration for codeExecution tool/
      );
    });

    it('should handle fileSearch configurations and validate array', () => {
      const tool = { fileSearch: { fileSearchStoreNames: ['store1'] } };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'file_search',
        file_search_store_names: ['store1'],
      });

      const invalidTool = { fileSearch: { fileSearchStoreNames: 'store1' } };
      assert.throws(
        () => toInteractionConfigTool(invalidTool),
        /fileSearchStoreNames must be an array of strings/
      );
    });

    it('should pass through arbitrary custom tools un-nested', () => {
      const tool = {
        type: 'function',
        name: 'myFunction',
        parameters: { type: 'object' },
      };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'function',
        name: 'myFunction',
        parameters: { type: 'object' },
      });
    });

    it('should pass through unrecognized fields from tool configurations', () => {
      const tool = {
        urlContext: {
          myUrlParam: 1,
        },
      };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'url_context',
        my_url_param: 1,
      });
    });
  });

  describe('toInteractionContent', () => {
    it('should convert TextPart', () => {
      const part: Part = { text: 'Hello' };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, { type: 'text', text: 'Hello' });
    });

    it('should convert MediaPart (image data)', () => {
      const part: Part = {
        media: {
          url: 'data:image/png;base64,DATA',
          contentType: 'image/png',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'image',
        data: 'DATA',
        mime_type: 'image/png',
      });
    });

    it('should convert MediaPart (image uri)', () => {
      const part: Part = {
        media: {
          url: 'gs://bucket/image.png',
          contentType: 'image/png',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'image',
        uri: 'gs://bucket/image.png',
        mime_type: 'image/png',
      });
    });

    it('should convert MediaPart (audio)', () => {
      const part: Part = {
        media: {
          url: 'data:audio/mp3;base64,DATA',
          contentType: 'audio/mp3',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'audio',
        data: 'DATA',
        mime_type: 'audio/mp3',
      });
    });

    it('should convert MediaPart (document)', () => {
      const part: Part = {
        media: {
          url: 'gs://bucket/doc.pdf',
          contentType: 'application/pdf',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'document',
        uri: 'gs://bucket/doc.pdf',
        mime_type: 'application/pdf',
      });
    });
  });

  describe('toInteractionSteps', () => {
    it('should convert ToolRequestPart to step', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'func',
                input: { a: 1 },
                ref: 'ref1',
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'function_call',
          name: 'func',
          arguments: { a: 1 },
          id: 'ref1',
        },
      ]);
    });

    it('should convert ToolResponsePart to step', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'func',
                output: { result: 'ok' },
                ref: 'ref1',
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'function_result',
          name: 'func',
          result: { result: 'ok' },
          call_id: 'ref1',
        },
      ]);
    });

    it('should group text contents into model_output step', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [{ text: 'Thinking' }, { text: 'Done' }],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'model_output',
          content: [
            { type: 'text', text: 'Thinking' },
            { type: 'text', text: 'Done' },
          ],
        },
      ]);
    });

    it('should convert custom googleSearchCall to google_search_call step', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              custom: {
                googleSearchCall: {
                  id: 'gs1',
                  arguments: { queries: ['genkit'] },
                },
              },
              metadata: { thoughtSignature: 'sig' },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'google_search_call',
          id: 'gs1',
          arguments: { queries: ['genkit'] },
          signature: 'sig',
        },
      ]);
    });

    it('should convert custom googleSearchResult to google_search_result step', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              custom: {
                googleSearchResult: {
                  callId: 'gs1',
                  result: { answer: 'framework' },
                },
              },
              metadata: { thoughtSignature: 'sig' },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'google_search_result',
          call_id: 'gs1',
          result: { answer: 'framework' },
          signature: 'sig',
        },
      ]);
    });

    it('should convert custom executableCode to code_execution_call step', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              custom: {
                executableCode: { code: 'print("hello")', language: 'PYTHON' },
              },
              metadata: { thoughtSignature: 'sig', callId: 'ce1' },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'code_execution_call',
          id: 'ce1',
          arguments: { code: 'print("hello")', language: 'PYTHON' },
          signature: 'sig',
        },
      ]);
    });

    it('should convert custom codeExecutionResult to code_execution_result step', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              custom: {
                codeExecutionResult: { output: 'hello\n' },
              },
              metadata: { thoughtSignature: 'sig', callId: 'ce1' },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'code_execution_result',
          call_id: 'ce1',
          result: 'hello\n',
          signature: 'sig',
        },
      ]);
    });
  });

  describe('fromInteractionContent', () => {
    it('should convert TextContent', () => {
      const content: Content = {
        type: 'text',
        text: 'Hello world',
        annotations: [{ start_index: 0, end_index: 5, source: 'source' }],
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        text: 'Hello world',
        metadata: {
          annotations: [{ start_index: 0, end_index: 5, source: 'source' }],
        },
      });
    });

    it('should convert ImageContent with data', () => {
      const content: Content = {
        type: 'image',
        data: 'BASE64DATA',
        mime_type: 'image/png',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        media: {
          url: 'data:image/png;base64,BASE64DATA',
          contentType: 'image/png',
        },
      });
    });

    it('should convert ImageContent with uri', () => {
      const content: Content = {
        type: 'image',
        uri: 'gs://bucket/image.png',
        mime_type: 'image/png',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        media: {
          url: 'gs://bucket/image.png',
          contentType: 'image/png',
        },
      });
    });

    it('should convert ImageContent with resolution', () => {
      const content: Content = {
        type: 'image',
        uri: 'gs://bucket/image.png',
        mime_type: 'image/png',
        resolution: 'high',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        media: {
          url: 'gs://bucket/image.png',
          contentType: 'image/png',
        },
        metadata: { resolution: 'high' },
      });
    });

    it('should convert ThoughtContent', () => {
      const content: Content = {
        type: 'thought',
        signature: 'SIG',
        summary: [{ type: 'text', text: 'Thinking...' }],
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        reasoning: 'Thinking...',
        metadata: {
          thoughtSignature: 'SIG',
        },
        custom: {
          thought: content,
        },
      });
    });

    it('should convert ThoughtContent with mixed summary', () => {
      const content: Content = {
        type: 'thought',
        signature: 'SIG',
        summary: [
          { type: 'text', text: 'Thinking about...' },
          { type: 'image', uri: 'gs://image.png' },
          { type: 'text', text: '...this image.' },
        ],
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        reasoning: 'Thinking about...\n[Image]\n...this image.',
        metadata: {
          thoughtSignature: 'SIG',
        },
        custom: {
          thought: content,
        },
      });
    });

    it('should convert FunctionCallContent', () => {
      const content: Content = {
        type: 'function_call',
        name: 'get_weather',
        arguments: { location: 'Paris' },
        id: 'call_123',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        toolRequest: {
          name: 'get_weather',
          input: { location: 'Paris' },
          ref: 'call_123',
        },
      });
    });

    it('should convert FunctionResultContent', () => {
      const content: Content = {
        type: 'function_result',
        name: 'get_weather',
        result: { temperature: 20 },
        call_id: 'call_123',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        toolResponse: {
          name: 'get_weather',
          output: { temperature: 20 },
          ref: 'call_123',
        },
      });
    });
  });

  describe('fromInteractionStep', () => {
    it('should convert model_output step', () => {
      const step: any = {
        type: 'model_output',
        content: [{ type: 'text', text: 'Hello' }],
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        { text: 'Hello', metadata: { annotations: undefined } },
      ]);
    });

    it('should skip user_input step', () => {
      const step: any = {
        type: 'user_input',
        content: [{ type: 'text', text: 'Hello' }],
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, []);
    });

    it('should convert google_search_call step', () => {
      const step: any = {
        type: 'google_search_call',
        id: '123',
        arguments: { queries: ['foo'] },
        signature: 'sig',
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          custom: {
            googleSearchCall: {
              id: '123',
              arguments: { queries: ['foo'] },
            },
          },
          metadata: { thoughtSignature: 'sig' },
        },
      ]);
    });

    it('should convert google_search_result step', () => {
      const step: any = {
        type: 'google_search_result',
        call_id: '123',
        result: { content: 'bar' },
        signature: 'sig',
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          custom: {
            googleSearchResult: {
              callId: '123',
              result: { content: 'bar' },
            },
          },
          metadata: { thoughtSignature: 'sig' },
        },
      ]);
    });

    it('should convert code_execution_call step', () => {
      const step: any = {
        type: 'code_execution_call',
        id: '123',
        arguments: { code: 'print(1)', language: 'PYTHON' },
        signature: 'sig',
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          custom: {
            executableCode: { code: 'print(1)', language: 'PYTHON' },
          },
          metadata: { callId: '123', thoughtSignature: 'sig' },
        },
      ]);
    });

    it('should convert code_execution_result step', () => {
      const step: any = {
        type: 'code_execution_result',
        call_id: '123',
        result: '1\n',
        signature: 'sig',
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          custom: {
            codeExecutionResult: { output: '1\n', outcome: 'OUTCOME_OK' },
          },
          metadata: { callId: '123', thoughtSignature: 'sig' },
        },
      ]);
    });

    it('should convert thought step', () => {
      const step: Step = {
        type: 'thought',
        signature: '',
        summary: [
          { type: 'text', text: '**Protocol...**' },
          { type: 'text', text: ' **Evalua...**' },
        ],
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          reasoning: '**Protocol...**\n **Evalua...**',
          metadata: { thoughtSignature: '' },
          custom: { thought: step },
        },
      ]);
    });
  });

  describe('fromInteractionDelta', () => {
    it('should convert text delta', () => {
      const delta: StepDeltaData = { type: 'text', text: 'hello' };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, [{ text: 'hello' }]);
    });

    it('should convert image delta', () => {
      const delta: StepDeltaData = {
        type: 'image',
        data: 'XYZ',
        mime_type: 'image/png',
      };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, [
        {
          media: { url: 'data:image/png;base64,XYZ', contentType: 'image/png' },
        },
      ]);
    });

    it('should convert function_call delta', () => {
      const delta: StepDeltaData = {
        type: 'function_call',
        name: 'myTool',
        id: 'ref1',
        arguments: { a: 1 },
      };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, [
        {
          toolRequest: {
            name: 'myTool',
            ref: 'ref1',
            input: { a: 1 },
            partial: true,
          },
        },
      ]);
    });

    it('should ignore arguments_delta', () => {
      const delta: StepDeltaData = {
        type: 'arguments_delta',
        arguments: '{"a":',
      };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, []);
    });
  });

  describe('fromInteraction', () => {
    it('should convert cancelled interaction', () => {
      const interaction: GeminiInteraction = {
        id: '123',
        status: 'cancelled',
      };
      const result = fromInteraction(interaction);
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.output?.finishReason, 'aborted');
      assert.strictEqual(result.output?.finishMessage, 'Operation cancelled');
      assert.deepStrictEqual(result.output?.message?.content, [
        { text: 'Operation cancelled.' },
      ]);
    });
  });
});
