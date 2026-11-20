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

import { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import {
  CallToolResult,
  GetPromptResult,
  JSONRPCMessage,
  JSONRPCRequest,
  Prompt,
  ReadResourceResult,
  Resource,
  ResourceTemplate,
  Root,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import { Genkit } from 'genkit';
import { ModelAction } from 'genkit/model';

export function defineEchoModel(ai: Genkit): ModelAction {
  const model = ai.defineModel(
    {
      name: 'echoModel',
    },
    async (request, { streamingRequested, sendChunk }) => {
      (model as any).__test__lastRequest = request;
      (model as any).__test__lastStreamingRequested = streamingRequested;
      if (streamingRequested) {
        sendChunk({
          content: [
            {
              text: '3',
            },
          ],
        });
        sendChunk({
          content: [
            {
              text: '2',
            },
          ],
        });
        sendChunk({
          content: [
            {
              text: '1',
            },
          ],
        });
      }
      return {
        message: {
          role: 'model',
          content: [
            {
              text:
                'Echo: ' +
                request.messages
                  .map(
                    (m) =>
                      (m.role === 'user' || m.role === 'model'
                        ? ''
                        : `${m.role}: `) + m.content.map((c) => c.text).join()
                  )
                  .join(),
            },
            {
              text: '; config: ' + JSON.stringify(request.config),
            },
          ],
        },
        finishReason: 'stop',
      };
    }
  );
  return model;
}

export class FakeTransport implements Transport {
  tools: Tool[] = [];
  prompts: Prompt[] = [];
  resources: Resource[] = [];
  resourceTemplates: ResourceTemplate[] = [];
  callToolResult?: CallToolResult;
  getPromptResult?: GetPromptResult;
  readResourceResult?: ReadResourceResult;
  roots?: Root[];

  async start(): Promise<void> {}

  async send(message: JSONRPCMessage): Promise<void> {
    const request = message as JSONRPCRequest;
    console.log(' - - - - -send', JSON.stringify(request, undefined, 2));
    if (request.method === 'initialize') {
      this.onmessage?.({
        result: {
          protocolVersion: '2024-11-05',
          capabilities: {
            prompts: {},
            tools: {},
            resources: {},
          },
          serverInfo: {
            name: 'mock-server',
            version: '0.0.1',
          },
        },
        jsonrpc: '2.0',
        id: request.id,
      });
    } else if (request.method === 'notifications/initialized') {
      // do nothing...
    } else if (request.method === 'tools/list') {
      this.onmessage?.({
        result: {
          tools: this.tools,
        },
        jsonrpc: '2.0',
        id: request.id,
      });
    } else if (request.method === 'tools/call') {
      const result = {
        ...this.callToolResult,
      };
      if (request.params?._meta)
        result.content = [
          ...(result.content || []),
          { type: 'text', text: JSON.stringify(request.params._meta) },
        ];
      this.onmessage?.({
        result,
        jsonrpc: '2.0',
        id: request.id,
      });
    } else if (request.method === 'prompts/list') {
      this.onmessage?.({
        result: {
          prompts: this.prompts,
        },
        jsonrpc: '2.0',
        id: request.id,
      });
    } else if (request.method === 'prompts/get') {
      const result = {
        ...this.getPromptResult,
      };
      if (request.params?._meta)
        result.messages = [
          ...(result.messages || []),
          {
            role: 'assistant',
            content: {
              type: 'text',
              text: JSON.stringify(request.params._meta),
            },
          },
        ];
      this.onmessage?.({
        result,
        jsonrpc: '2.0',
        id: request.id,
      });
    } else if (request.method === 'resources/list') {
      this.onmessage?.({
        result: {
          resources: this.resources,
        },
        jsonrpc: '2.0',
        id: request.id,
      });
    } else if (request.method === 'resources/templates/list') {
      this.onmessage?.({
        result: {
          resourceTemplates: this.resourceTemplates,
        },
        jsonrpc: '2.0',
        id: request.id,
      });
    } else if (request.method === 'resources/read') {
      this.onmessage?.({
        result: {
          ...this.readResourceResult,
        },
        jsonrpc: '2.0',
        id: request.id,
      });
    } else if (request.method === 'notifications/roots/list_changed') {
      this.onmessage?.({
        jsonrpc: '2.0',
        id: 1,
        method: 'roots/list',
      });
    } else if ((request as any).result?.roots) {
      this.roots = (request as any).result?.roots;
      console.log('updated roots', this.roots);
    } else {
      throw new Error(`Unknown request method: ${request.method}`);
    }
  }

  async close(): Promise<void> {
    this.onclose?.();
  }

  onclose?: () => void;
  onerror?: (error: Error) => void;
  onmessage?: (message: JSONRPCMessage) => void;
}
