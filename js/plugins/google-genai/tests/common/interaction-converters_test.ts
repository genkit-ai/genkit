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
  toInteractionContent,
  toInteractionRole,
  toInteractionTool,
} from '../../src/common/interaction-converters.js';
import {
  Content,
  GeminiInteraction,
} from '../../src/common/interaction-types.js';

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

    it('should convert ToolRequestPart', () => {
      const part: Part = {
        toolRequest: {
          name: 'func',
          input: { a: 1 },
          ref: 'ref1',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'function_call',
        name: 'func',
        arguments: { a: 1 },
        id: 'ref1',
      });
    });

    it('should convert ToolResponsePart', () => {
      const part: Part = {
        toolResponse: {
          name: 'func',
          output: { result: 'ok' },
          ref: 'ref1',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'function_result',
        name: 'func',
        result: { result: 'ok' },
        call_id: 'ref1',
      });
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
        metadata: { resolution: undefined },
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
        metadata: { resolution: undefined },
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
