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
import { GenkitError, MessageData, Part } from 'genkit';
import { ToolDefinition } from 'genkit/model';
import { describe, it } from 'node:test';
import {
  ensureToolIds,
  fromInteraction,
  fromInteractionContent,
  fromInteractionDelta,
  fromInteractionStep,
  fromInteractionSync,
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

    it('should handle mcpServer tool configuration and normalize string array allowed_tools to objects', () => {
      const tool = {
        mcpServer: {
          name: 'my-server',
          url: 'https://mcp.example.com',
          allowedTools: ['search', 'read_doc'],
        },
      };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'mcp_server',
        name: 'my-server',
        url: 'https://mcp.example.com',
        allowed_tools: [{ tools: ['search', 'read_doc'] }],
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

    it('throws GenkitError when tool output contains non-text/image content', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'pdfTool',
                ref: 'call-1',
                output: undefined,
                content: [
                  {
                    media: {
                      url: 'data:application/pdf;base64,ABC',
                      contentType: 'application/pdf',
                    },
                  },
                ],
              },
            },
          ],
        },
      ];
      assert.throws(
        () => toInteractionSteps(messages),
        (err: any) => {
          assert.strictEqual(err.status, 'INVALID_ARGUMENT');
          assert.strictEqual(
            err.originalMessage,
            'Tool output for pdfTool may only contain text or image content.'
          );
          return true;
        }
      );
    });

    it('wraps plain array tool output in result object', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'listTool',
                ref: 'call-1',
                output: [1, 2, 3],
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'function_result',
          name: 'listTool',
          call_id: 'call-1',
          result: { result: [1, 2, 3] },
        },
      ]);
    });

    it('should convert empty reasoning part with thoughtSignature to thought step without summary', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              reasoning: '',
              metadata: {
                thoughtSignature: 'sig-123',
              },
              custom: {
                thought: {
                  type: 'thought',
                  summary: [],
                  signature: 'sig-123',
                },
              },
            },
            {
              toolRequest: {
                name: 'getWeather',
                ref: 'call-1',
                input: { location: 'London' },
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'thought',
          signature: 'sig-123',
        },
        {
          type: 'function_call',
          name: 'getWeather',
          id: 'call-1',
          arguments: { location: 'London' },
        },
      ]);
    });

    it('should convert non-empty reasoning part to thought step with summary and signature', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              reasoning: 'Looking up weather data',
              metadata: {
                thoughtSignature: 'sig-456',
              },
            },
            {
              text: 'The weather is sunny.',
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'thought',
          summary: [{ type: 'text', text: 'Looking up weather data' }],
          signature: 'sig-456',
        },
        {
          type: 'model_output',
          content: [{ type: 'text', text: 'The weather is sunny.' }],
        },
      ]);
    });

    it('should preserve rich summary from custom.thought when available', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              reasoning: 'Plan',
              custom: {
                thought: {
                  type: 'thought',
                  summary: [
                    {
                      type: 'text',
                      text: 'Plan',
                      annotations: [
                        { title: 'Doc', url: 'https://example.com' },
                      ],
                    },
                  ],
                  signature: 'custom-sig-789',
                },
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'thought',
          summary: [
            {
              type: 'text',
              text: 'Plan',
              annotations: [{ title: 'Doc', url: 'https://example.com' }],
            },
          ],
          signature: 'custom-sig-789',
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
        reasoning: 'Thinking about...[Image]...this image.',
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

    it('should convert FunctionResultContent with plain array result into tool output', () => {
      const content: Content = {
        type: 'function_result',
        name: 'list_items',
        result: [1, 2, 3] as any,
        call_id: 'call_123',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        toolResponse: {
          name: 'list_items',
          output: [1, 2, 3],
          ref: 'call_123',
        },
      });
    });

    it('should convert FunctionResultContent with multimodal Content array into tool content', () => {
      const content: Content = {
        type: 'function_result',
        name: 'multimodal_tool',
        result: [
          { type: 'text', text: 'description' },
          { type: 'image', uri: 'https://example.com/img.png' },
        ] as any,
        call_id: 'call_123',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        toolResponse: {
          name: 'multimodal_tool',
          content: [
            { text: 'description', metadata: { annotations: undefined } },
            {
              media: {
                url: 'https://example.com/img.png',
                contentType: undefined,
              },
            },
          ],
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
          reasoning: '**Protocol...** **Evalua...**',
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

    it('should convert thought_summary delta to reasoning', () => {
      const delta: StepDeltaData = {
        type: 'thought_summary',
        content: { type: 'text', text: 'thinking process' },
      };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, [{ reasoning: 'thinking process' }]);
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

    it('should finish a failed operation with an error (no infinite polling)', () => {
      const result = fromInteraction({
        id: '123',
        status: 'failed',
        errors: [{ code: 'api_error', message: 'boom' }],
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.output, undefined);
      assert.ok(result.error?.message.includes('[api_error] boom'));
      assert.strictEqual(result.error?.code, 'api_error');
    });

    it('should finish a failed operation without errors[] with a generic error', () => {
      const result = fromInteraction({ id: '123', status: 'failed' });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.error?.message, 'Interaction failed');
    });

    it('should convert a content-blocked failed operation to finishReason blocked', () => {
      const result = fromInteraction({
        id: '123',
        status: 'failed',
        errors: [{ code: 'recitation', message: 'Recitation blocked.' }],
        usage: { total_input_tokens: 7, total_tokens: 7 },
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.error, undefined);
      assert.strictEqual(result.output?.finishReason, 'blocked');
      assert.strictEqual(result.output?.finishMessage, 'Recitation blocked.');
      assert.strictEqual(result.output?.usage?.inputTokens, 7);
    });

    it('should carry usage on a cancelled operation', () => {
      const result = fromInteraction({
        id: '123',
        status: 'cancelled',
        usage: { total_input_tokens: 3, total_tokens: 3 },
      });
      assert.strictEqual(result.output?.usage?.inputTokens, 3);
    });

    it('should finish a requires_action operation (collaborative planning) and surface the plan', () => {
      const result = fromInteraction({
        id: '123',
        status: 'requires_action',
        steps: [
          {
            type: 'model_output',
            content: [
              {
                type: 'text',
                text: "Here is the research plan I've prepared: 1. ...",
              },
            ],
          },
        ],
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.error, undefined);
      assert.strictEqual(result.output?.finishReason, 'stop');
      assert.deepStrictEqual(
        result.output?.message?.content.map((p) => p.text),
        ["Here is the research plan I've prepared: 1. ..."]
      );
      assert.strictEqual(
        result.output?.message?.metadata?.interactionStatus,
        'requires_action'
      );
      assert.strictEqual(
        result.output?.message?.metadata?.interactionId,
        '123'
      );
    });

    it('should surface pending client function calls as toolRequests on requires_action', () => {
      const result = fromInteraction({
        id: '123',
        status: 'requires_action',
        steps: [
          {
            type: 'function_call',
            id: 'call-1',
            name: 'lookup',
            arguments: { q: 'x' },
          },
        ],
      });
      assert.strictEqual(result.done, true);
      assert.deepStrictEqual(result.output?.message?.content, [
        { toolRequest: { name: 'lookup', ref: 'call-1', input: { q: 'x' } } },
      ]);
    });

    it('should keep polling for in_progress', () => {
      const result = fromInteraction({ id: '123', status: 'in_progress' });
      assert.strictEqual(result.done, false);
    });

    it('should keep polling for queued (waiting for capacity)', () => {
      const result = fromInteraction({ id: '123', status: 'queued' });
      assert.strictEqual(result.done, false);
      assert.strictEqual(result.error, undefined);
    });

    it('should finish with an error for an unknown status (no infinite polling)', () => {
      const result = fromInteraction({
        id: '123',
        status: 'expired' as any,
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.output, undefined);
      assert.strictEqual(
        result.error?.message,
        'Unknown interaction status: expired'
      );
    });

    it('should convert an incomplete operation to finishReason length', () => {
      const result = fromInteraction({
        id: '123',
        status: 'incomplete',
        steps: [
          { type: 'model_output', content: [{ type: 'text', text: 'partial' }] },
        ],
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.output?.finishReason, 'length');
      assert.deepStrictEqual(
        result.output?.message?.content.map((p) => p.text),
        ['partial']
      );
    });
  });

  describe('fromInteractionSync', () => {
    it('returns finishReason blocked for a post-execution safety block', () => {
      const result = fromInteractionSync({
        id: 'int-1',
        status: 'failed',
        errors: [
          {
            code: 'safety',
            message: 'Response blocked due to safety violations.',
          },
        ],
      });
      assert.strictEqual(result.finishReason, 'blocked');
      assert.strictEqual(
        result.finishMessage,
        'Response blocked due to safety violations.'
      );
      assert.deepStrictEqual(result.message?.content, []);
      assert.strictEqual(result.message?.metadata?.interactionId, 'int-1');
    });

    it('returns finishReason blocked for every content-block code', () => {
      for (const code of [
        'safety',
        'recitation',
        'language',
        'prohibited_content',
        'spii',
        'blocklist',
        'image_safety',
        'image_prohibited_content',
        'image_recitation',
        'image_other',
        'content_blocked',
        'jailbreak',
        'model_armor',
      ]) {
        const result = fromInteractionSync({
          status: 'failed',
          errors: [{ code, message: 'blocked' }],
        });
        assert.strictEqual(result.finishReason, 'blocked', `code "${code}"`);
      }
    });

    it('throws a retryable ABORTED GenkitError for malformed_function_call', () => {
      const advice =
        'Model generated invalid JSON syntax and the output could not be parsed. Please retry the request.';
      assert.throws(
        () =>
          fromInteractionSync({
            status: 'failed',
            errors: [{ code: 'malformed_function_call', message: advice }],
          }),
        (err: any) => {
          assert.ok(err instanceof GenkitError);
          assert.strictEqual(err.status, 'ABORTED');
          assert.ok(err.message.includes('[malformed_function_call]'));
          assert.ok(err.message.includes(advice));
          return true;
        }
      );
    });

    it('throws ABORTED for unexpected_tool_call and no_image', () => {
      for (const code of ['unexpected_tool_call', 'no_image']) {
        assert.throws(
          () =>
            fromInteractionSync({
              status: 'failed',
              errors: [{ code, message: 'retry' }],
            }),
          (err: any) => err instanceof GenkitError && err.status === 'ABORTED',
          `code "${code}"`
        );
      }
    });

    it('throws a non-retryable UNKNOWN GenkitError for a failure without errors[]', () => {
      assert.throws(
        () => fromInteractionSync({ status: 'failed' }),
        (err: any) => {
          assert.ok(err instanceof GenkitError);
          assert.strictEqual(err.status, 'UNKNOWN');
          assert.ok(err.message.includes('Interaction failed'));
          return true;
        }
      );
    });

    it('throws a non-retryable UNKNOWN GenkitError for an unrecognized code', () => {
      assert.throws(
        () =>
          fromInteractionSync({
            status: 'failed',
            errors: [{ code: 'something_new', message: 'huh' }],
          }),
        (err: any) =>
          err instanceof GenkitError &&
          err.status === 'UNKNOWN' &&
          err.message.includes('[something_new] huh')
      );
    });

    it('returns finishReason length with partial content for an incomplete interaction', () => {
      const result = fromInteractionSync({
        id: 'int-1',
        status: 'incomplete',
        steps: [
          { type: 'model_output', content: [{ type: 'text', text: 'partial' }] },
        ],
      });
      assert.strictEqual(result.finishReason, 'length');
      assert.strictEqual(
        result.finishMessage,
        'Interaction incomplete (truncated output)'
      );
      assert.deepStrictEqual(
        result.message?.content.map((p) => p.text),
        ['partial']
      );
    });

    it('carries usage (including per-modality counts) on a blocked response', () => {
      const result = fromInteractionSync({
        status: 'failed',
        errors: [{ code: 'safety', message: 'blocked' }],
        usage: {
          total_input_tokens: 10,
          total_output_tokens: 0,
          total_tokens: 10,
          input_tokens_by_modality: [
            { modality: 'text', tokens: 8 },
            { modality: 'image', tokens: 2 },
          ],
        },
      });
      assert.strictEqual(result.finishReason, 'blocked');
      assert.deepStrictEqual(result.usage, {
        inputTokens: 10,
        outputTokens: 0,
        totalTokens: 10,
        cachedContentTokens: undefined,
        thoughtsTokens: undefined,
        inputCharacters: 8,
        inputImages: 2,
      });
    });

    it('carries usage on a cancelled response', () => {
      const result = fromInteractionSync({
        status: 'cancelled',
        usage: { total_input_tokens: 4, total_tokens: 4 },
      });
      assert.strictEqual(result.finishReason, 'aborted');
      assert.strictEqual(result.usage?.inputTokens, 4);
    });

    it('carries usage even when the interaction has no steps', () => {
      const result = fromInteractionSync({
        status: 'completed',
        usage: { total_input_tokens: 5, total_tokens: 5 },
      });
      assert.strictEqual(result.usage?.inputTokens, 5);
    });

    it('still returns finishReason stop for a completed interaction', () => {
      const result = fromInteractionSync({
        status: 'completed',
        steps: [
          { type: 'model_output', content: [{ type: 'text', text: 'done' }] },
        ],
      });
      assert.strictEqual(result.finishReason, 'stop');
    });
  });
});
