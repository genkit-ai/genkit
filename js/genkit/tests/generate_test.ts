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

import type { GenerateResponseChunkData, MessageData } from '@genkit-ai/ai';
import { ModelAction } from '@genkit-ai/ai/model';
import { Operation, z, type JSONSchema7 } from '@genkit-ai/core';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { modelRef } from '../../ai/src/model';
import { interrupt } from '../../ai/src/tool';
import { dynamicResource, genkit, tool, type GenkitBeta } from '../src/beta';
import {
  defineEchoModel,
  defineProgrammableModel,
  runAsync,
  type ProgrammableModel,
} from './helpers';

describe('generate', () => {
  describe('default model', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({
        model: 'echoModel',
      });
      defineEchoModel(ai);
    });

    it('calls the default model', async () => {
      const response = await ai.generate({
        prompt: 'hi',
      });
      assert.strictEqual(response.text, 'Echo: hi; config: {}');
    });

    it('calls the default model with just a string prompt', async () => {
      const response = await ai.generate('hi');
      assert.strictEqual(response.text, 'Echo: hi; config: {}');
    });

    it('calls the default model with just parts prompt', async () => {
      const response = await ai.generate([{ text: 'hi' }]);
      assert.strictEqual(response.text, 'Echo: hi; config: {}');
    });

    it('calls the default model system', async () => {
      const response = await ai.generate({
        prompt: 'hi',
        system: 'talk like a pirate',
      });
      assert.strictEqual(
        response.text,
        'Echo: system: talk like a pirate,hi; config: {}'
      );
      assert.deepStrictEqual(response.request, {
        config: {
          version: undefined,
        },
        docs: undefined,
        messages: [
          {
            role: 'system',
            content: [{ text: 'talk like a pirate' }],
          },
          {
            role: 'user',
            content: [{ text: 'hi' }],
          },
        ],
        output: {},
        tools: [],
      });
    });

    it('calls the default model with tool choice', async () => {
      const response = await ai.generate({
        prompt: 'hi',
        toolChoice: 'required',
      });
      assert.strictEqual(response.text, 'Echo: hi; config: {}');
      assert.deepStrictEqual(response.request, {
        config: {
          version: undefined,
        },
        docs: undefined,
        messages: [
          {
            role: 'user',
            content: [{ text: 'hi' }],
          },
        ],
        output: {},
        tools: [],
        toolChoice: 'required',
      });
    });

    it('streams the default model', async () => {
      const { response, stream } = await ai.generateStream('hi');

      const chunks: string[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk.text);
      }
      assert.strictEqual((await response).text, 'Echo: hi; config: {}');
      assert.deepStrictEqual(chunks, ['3', '2', '1']);
    });
  });

  describe('explicit model', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({});
      defineEchoModel(ai);
    });

    it('calls the explicitly passed in model', async () => {
      const response = await ai.generate({
        model: 'echoModel',
        prompt: 'hi',
      });
      assert.strictEqual(response.text, 'Echo: hi; config: {}');
    });

    it('rejects on invalid model', async () => {
      const response = ai.generate({
        model: 'modelThatDoesNotExist',
        prompt: 'hi',
      });
      await assert.rejects(response, 'Model modelThatDoesNotExist not found');
    });
  });

  describe('streaming', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({});
    });

    it('rethrows response errors', async () => {
      ai.defineModel(
        {
          apiVersion: 'v2',
          name: 'blockingModel',
        },
        async (request, { sendChunk, streamingRequested }) => {
          if (streamingRequested) {
            await runAsync(() => {
              sendChunk({
                content: [
                  {
                    text: '3',
                  },
                ],
              });
            });
            await runAsync(() => {
              sendChunk({
                content: [
                  {
                    text: '2',
                  },
                ],
              });
            });
            await runAsync(() => {
              sendChunk({
                content: [
                  {
                    text: '1',
                  },
                ],
              });
            });
          }
          return await runAsync(() => ({
            message: {
              role: 'model',
              content: [],
            },
            finishReason: 'blocked',
          }));
        }
      );

      await assert.rejects(async () => {
        const { response, stream } = ai.generateStream({
          prompt: 'hi',
          model: 'blockingModel',
        });
        for await (const chunk of stream) {
          // nothing
        }
        await response;
      });
    });

    it('rethrows initialization errors', async () => {
      await assert.rejects(
        async () => {
          const { stream } = ai.generateStream({
            prompt: 'hi',
            model: 'modelNotFound',
          });
          for await (const chunk of stream) {
            // nothing
          }
        },
        { status: 'NOT_FOUND' }
      );
    });

    it('passes the streaming callback to the model', async () => {
      const model = defineEchoModel(ai);
      const flow = ai.defineFlow('wrapper', async (_, streamingCallback) => {
        const response = await ai.generate({
          model: model,
          prompt: 'hi',
          onChunk: console.log,
        });
        return response.text;
      });
      const text = await flow();
      assert.ok((model as any).__test__lastStreamingCallback);
    });

    it('strips out the noop streaming callback', async () => {
      const model = defineEchoModel(ai);
      const flow = ai.defineFlow('wrapper', async (_, streamingCallback) => {
        const response = await ai.generate({
          model: model,
          prompt: 'hi',
          onChunk: streamingCallback,
        });
        return response.text;
      });
      const text = await flow();
      assert.ok(!(model as any).__test__lastStreamingCallback);
    });
  });

  describe('config', () => {
    let ai: GenkitBeta;

    beforeEach(() => {
      ai = genkit({});
      defineEchoModel(ai);
    });

    it('takes config passed to generate', async () => {
      const response = await ai.generate({
        prompt: 'hi',
        model: 'echoModel',
        config: {
          temperature: 11,
        },
      });
      assert.strictEqual(response.text, 'Echo: hi; config: {"temperature":11}');
    });

    it('merges config from the ref', async () => {
      const response = await ai.generate({
        prompt: 'hi',
        model: modelRef({ name: 'echoModel' }).withConfig({
          version: 'abc',
        }),
        config: {
          temperature: 11,
        },
      });
      assert.strictEqual(
        response.text,
        'Echo: hi; config: {"version":"abc","temperature":11}'
      );
    });

    it('picks up the top-level version from the ref', async () => {
      const response = await ai.generate({
        prompt: 'hi',
        model: modelRef({ name: 'echoModel' }).withVersion('bcd'),
        config: {
          temperature: 11,
        },
      });
      assert.strictEqual(
        response.text,
        'Echo: hi; config: {"version":"bcd","temperature":11}'
      );
    });
  });

  describe('tools', () => {
    let ai: GenkitBeta;
    let pm: ProgrammableModel;
    let echo: ModelAction;

    beforeEach(() => {
      class Extra {
        toJSON() {
          return 'extra';
        }
      }
      ai = genkit({
        model: 'programmableModel',
        // testing with a non-serializable data in the context
        context: { something: new Extra() },
      });
      pm = defineProgrammableModel(ai);
      echo = defineEchoModel(ai);
    });

    it('call the tool', async () => {
      ai.defineTool(
        { name: 'testTool', description: 'description' },
        async () => 'tool called'
      );

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        return {
          message: {
            role: 'model',
            content: [
              reqCounter++ === 0
                ? {
                    toolRequest: {
                      name: 'testTool',
                      input: {},
                      ref: 'ref123',
                    },
                  }
                : { text: 'done' },
            ],
          },
        };
      };

      const { text } = await ai.generate({
        prompt: 'call the tool',
        tools: ['testTool'],
      });

      assert.strictEqual(text, 'done');
      assert.deepStrictEqual(
        pm.lastRequest,

        {
          config: {},
          messages: [
            {
              role: 'user',
              content: [{ text: 'call the tool' }],
            },
            {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    input: {},
                    name: 'testTool',
                    ref: 'ref123',
                  },
                },
              ],
            },
            {
              role: 'tool',
              content: [
                {
                  toolResponse: {
                    name: 'testTool',
                    output: 'tool called',
                    ref: 'ref123',
                  },
                },
              ],
            },
          ],
          output: {},
          tools: [
            {
              description: 'description',
              inputSchema: {
                $schema: 'http://json-schema.org/draft-07/schema#',
              },
              name: 'testTool',
              outputSchema: {
                $schema: 'http://json-schema.org/draft-07/schema#',
              },
            },
          ],
        }
      );
    });

    it('call the tool with context', async () => {
      ai.defineTool(
        { name: 'testTool', description: 'description' },
        async (_, { context }) => JSON.stringify(context)
      );

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        return {
          message: {
            role: 'model',
            content: [
              reqCounter++ === 0
                ? {
                    toolRequest: {
                      name: 'testTool',
                      input: {},
                      ref: 'ref123',
                    },
                  }
                : { text: 'done' },
            ],
          },
        };
      };

      const { messages } = await ai.generate({
        prompt: 'call the tool',
        tools: ['testTool'],
      });

      assert.deepStrictEqual(messages[2], {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'testTool',
              output: '{"something":"extra"}',
              ref: 'ref123',
            },
          },
        ],
      });
    });

    it('calls the dynamic tool', async () => {
      const schema = {
        properties: {
          foo: { type: 'string' },
        },
      } as JSONSchema7;
      const dynamicTestTool1 = tool(
        {
          name: 'dynamicTestTool1',
          inputJsonSchema: schema,
          description: 'description',
        },
        async () => 'tool called 1'
      );
      const dynamicTestTool2 = ai.dynamicTool(
        {
          name: 'dynamicTestTool2',
          inputJsonSchema: schema,
          description: 'description 2',
        },
        async () => 'tool called 2'
      );
      const dynamicMultipartTool = tool(
        {
          multipart: true,
          name: 'dynamicMultipartTool',
          inputJsonSchema: schema,
          description: 'description',
        },
        async () => {
          return {
            output: 'main output',
            content: [{ text: 'part 1' }],
          };
        }
      );

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        return {
          message: {
            role: 'model',
            content:
              reqCounter++ === 0
                ? [
                    {
                      toolRequest: {
                        name: 'dynamicTestTool1',
                        input: { foo: 'bar' },
                        ref: 'ref123',
                      },
                    },
                    {
                      toolRequest: {
                        name: 'dynamicTestTool2',
                        input: { foo: 'baz' },
                        ref: 'ref234',
                      },
                    },
                    {
                      toolRequest: {
                        name: 'dynamicMultipartTool',
                        input: { foo: 'baz' },
                        ref: 'ref234',
                      },
                    },
                  ]
                : [{ text: 'done' }],
          },
        };
      };

      const { text } = await ai.generate({
        prompt: 'call the tool',
        tools: [dynamicTestTool1, dynamicTestTool2, dynamicMultipartTool],
      });

      assert.strictEqual(text, 'done');
      // remove properties from outputSchema -- too verbose, not important for the test
      delete pm.lastRequest?.tools?.[2]?.outputSchema?.properties;
      assert.deepStrictEqual(pm.lastRequest, {
        config: {},
        messages: [
          {
            role: 'user',
            content: [{ text: 'call the tool' }],
          },
          {
            role: 'model',
            content: [
              {
                toolRequest: {
                  input: { foo: 'bar' },
                  name: 'dynamicTestTool1',
                  ref: 'ref123',
                },
              },
              {
                toolRequest: {
                  input: { foo: 'baz' },
                  name: 'dynamicTestTool2',
                  ref: 'ref234',
                },
              },
              {
                toolRequest: {
                  input: {
                    foo: 'baz',
                  },
                  name: 'dynamicMultipartTool',
                  ref: 'ref234',
                },
              },
            ],
          },
          {
            role: 'tool',
            content: [
              {
                toolResponse: {
                  name: 'dynamicTestTool1',
                  output: 'tool called 1',
                  ref: 'ref123',
                },
              },
              {
                toolResponse: {
                  name: 'dynamicTestTool2',
                  output: 'tool called 2',
                  ref: 'ref234',
                },
              },
              {
                toolResponse: {
                  content: [
                    {
                      text: 'part 1',
                    },
                  ],
                  name: 'dynamicMultipartTool',
                  output: 'main output',
                  ref: 'ref234',
                },
              },
            ],
          },
        ],
        output: {},
        tools: [
          {
            description: 'description',
            inputSchema: schema,
            name: 'dynamicTestTool1',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
          {
            description: 'description 2',
            inputSchema: schema,
            name: 'dynamicTestTool2',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
          {
            description: 'description',
            inputSchema: {
              properties: {
                foo: {
                  type: 'string',
                },
              },
            },
            name: 'dynamicMultipartTool',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              additionalProperties: true,
              type: 'object',
            },
          },
        ],
      });
    });

    it('calls the dynamic resource', async () => {
      const dynamicTestResource = dynamicResource(
        {
          name: 'dynamicTestTool',
          uri: 'foo://foo',
          description: 'description',
        },
        async () => ({ content: [{ text: 'dynamic text' }] })
      );
      ai.defineResource(
        {
          name: 'regularResource',
          template: 'bar://{value}',
          description: 'description 2',
        },
        async () => ({ content: [{ text: 'regular text' }] })
      );

      const { text } = await ai.generate({
        model: 'echoModel',
        prompt: [
          { text: 'some text' },
          { resource: { uri: 'foo://foo' } },
          { resource: { uri: 'bar://bar' } },
        ],
        resources: [dynamicTestResource],
      });
      assert.strictEqual(
        text,
        'Echo: some text,dynamic text,regular text; config: {}'
      );
      assert.deepStrictEqual((echo as any).__test__lastRequest.messages, [
        {
          role: 'user',
          content: [
            { text: 'some text' },
            {
              metadata: {
                resource: {
                  uri: 'foo://foo',
                },
              },
              text: 'dynamic text',
            },
            {
              metadata: {
                resource: {
                  template: 'bar://{value}',
                  uri: 'bar://bar',
                },
              },
              text: 'regular text',
            },
          ],
        },
      ]);
    });

    it('interrupts the dynamic tool with no impl', async () => {
      const schema = {
        properties: {
          foo: { type: 'string' },
        },
      } as JSONSchema7;
      const dynamicTestTool = ai.dynamicTool({
        name: 'dynamicTestTool',
        inputJsonSchema: schema,
        description: 'description',
      });

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        return {
          message: {
            role: 'model',
            content: [
              reqCounter++ === 0
                ? {
                    toolRequest: {
                      name: 'dynamicTestTool',
                      input: { foo: 'bar' },
                      ref: 'ref123',
                    },
                  }
                : { text: 'done' },
            ],
          },
        };
      };

      const response = await ai.generate({
        prompt: 'call the tool',
        tools: [dynamicTestTool],
      });

      assert.deepStrictEqual(response.interrupts, [
        {
          metadata: {
            interrupt: true,
          },
          toolRequest: {
            input: {
              foo: 'bar',
            },
            name: 'dynamicTestTool',
            ref: 'ref123',
          },
        },
      ]);
    });

    it('call the tool with output schema', async () => {
      const schema = z.object({
        foo: z.string(),
      });

      ai.defineTool(
        {
          name: 'testTool',
          description: 'description',
          inputSchema: schema,
          outputSchema: schema,
        },
        async () => {
          return {
            foo: 'bar',
          };
        }
      );

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        return {
          message: {
            role: 'model',
            content: [
              reqCounter++ === 0
                ? {
                    toolRequest: {
                      name: 'testTool',
                      input: { foo: 'fromTool' },
                      ref: 'ref123',
                    },
                  }
                : {
                    text: "```\n{foo: 'fromModel'}\n```",
                  },
            ],
          },
        };
      };
      const { text, output } = await ai.generate({
        output: { schema },
        prompt: 'call the tool',
        tools: ['testTool'],
      });
      assert.strictEqual(text, "```\n{foo: 'fromModel'}\n```");
      assert.deepStrictEqual(output, {
        foo: 'fromModel',
      });
    });

    it('should propagate context to the tool', async () => {
      const schema = z.object({
        foo: z.string(),
      });

      ai.defineTool(
        {
          name: 'testTool',
          description: 'description',
          inputSchema: schema,
          outputSchema: schema,
        },
        async (_, { context }) => {
          return {
            foo: `bar ${context.auth?.email}`,
          };
        }
      );

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        return {
          message: {
            role: 'model',
            content: [
              reqCounter++ === 0
                ? {
                    toolRequest: {
                      name: 'testTool',
                      input: { foo: 'fromTool' },
                      ref: 'ref123',
                    },
                  }
                : {
                    text: req.messages
                      .splice(-1)
                      .map((m) =>
                        m.content
                          .map(
                            (c) =>
                              c.text || JSON.stringify(c.toolResponse?.output)
                          )
                          .join()
                      )
                      .join(),
                  },
            ],
          },
        };
      };
      const { text } = await ai.generate({
        prompt: 'call the tool',
        tools: ['testTool'],
        context: { auth: { email: 'a@b.c' } },
      });
      assert.strictEqual(text, '{"foo":"bar a@b.c"}');
    });

    it('calls the multipart tool', async () => {
      const t = ai.defineTool(
        { name: 'testTool', description: 'description', multipart: true },
        async () => ({
          output: 'tool called',
          content: [{ text: 'part 1' }],
        })
      );

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        return {
          message: {
            role: 'model',
            content: [
              reqCounter++ === 0
                ? {
                    toolRequest: {
                      name: 'testTool',
                      input: {},
                      ref: 'ref123',
                    },
                  }
                : { text: 'done' },
            ],
          },
        };
      };

      const { text, messages } = await ai.generate({
        prompt: 'call the tool',
        tools: [t],
      });

      assert.strictEqual(text, 'done');
      assert.strictEqual(messages.length, 4);
      const toolMessage = messages[2];
      assert.strictEqual(toolMessage.role, 'tool');
      assert.deepStrictEqual(toolMessage.content, [
        {
          toolResponse: {
            name: 'testTool',
            ref: 'ref123',
            output: 'tool called',
            content: [{ text: 'part 1' }],
          },
        },
      ]);
    });

    it('streams the tool responses', async () => {
      ai.defineTool(
        { name: 'testTool', description: 'description' },
        async () => 'tool called'
      );

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        if (sc) {
          sc({
            content: [
              reqCounter === 0
                ? {
                    toolRequest: {
                      name: 'testTool',
                      input: {},
                      ref: 'ref123',
                    },
                  }
                : { text: 'done' },
            ],
          });
        }
        return {
          message: {
            role: 'model',
            content: [
              reqCounter++ === 0
                ? {
                    toolRequest: {
                      name: 'testTool',
                      input: {},
                      ref: 'ref123',
                    },
                  }
                : { text: 'done' },
            ],
          },
        };
      };

      const { stream, response } = await ai.generateStream({
        prompt: 'call the tool',
        tools: ['testTool'],
      });

      const chunks: any[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk.toJSON());
      }

      assert.strictEqual((await response).text, 'done');
      assert.deepStrictEqual(chunks, [
        {
          content: [
            {
              toolRequest: {
                input: {},
                name: 'testTool',
                ref: 'ref123',
              },
            },
          ],
          index: 0,
          role: 'model',
        },
        {
          content: [
            {
              toolResponse: {
                name: 'testTool',
                output: 'tool called',
                ref: 'ref123',
              },
            },
          ],
          index: 1,
          role: 'tool',
        },
        {
          content: [{ text: 'done' }],
          index: 2,
          role: 'model',
        },
      ]);
    });

    it('throws when exceeding max tool call iterations', async () => {
      ai.defineTool(
        { name: 'testTool', description: 'description' },
        async () => 'tool called'
      );

      // this will result in the tool getting called infinitely in a loop.
      pm.handleResponse = async () => {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'testTool',
                  input: {},
                  ref: 'ref123',
                },
              },
            ],
          },
        };
      };

      await assert.rejects(
        ai.generate({
          prompt: 'call the tool',
          tools: ['testTool'],
          maxTurns: 17,
        }),
        (err: Error) => {
          return err.message.includes(
            'Exceeded maximum tool call iterations (17)'
          );
        }
      );
    });

    it('interrupts tool execution', async () => {
      ai.defineTool(
        { name: 'simpleTool', description: 'description' },
        async (input) => `response: ${input.name}`
      );
      ai.defineTool(
        { name: 'interruptingTool', description: 'description' },
        async (input, { interrupt }) =>
          interrupt({ confirm: 'is it a banana?' })
      );
      ai.defineTool(
        { name: 'resumableTool', description: 'description' },
        async (input, { interrupt, resumed }) => {
          if ((resumed as any)?.status === 'ok') return true;
          return interrupt();
        }
      );
      const dynamicInterrupt = interrupt({
        name: 'dynamicInterrupt',
        description: 'description',
      });

      // first response is a tool call, the subsequent responses are just text response from agent b.
      let reqCounter = 0;
      pm.handleResponse = async (req, sc) => {
        return {
          message: {
            role: 'model',
            content:
              reqCounter++ === 0
                ? [
                    {
                      text: 'reasoning',
                    },
                    {
                      toolRequest: {
                        name: 'interruptingTool',
                        input: {},
                        ref: 'ref123',
                      },
                    },
                    {
                      toolRequest: {
                        name: 'simpleTool',
                        input: { name: 'foo' },
                        ref: 'ref456',
                      },
                    },
                    {
                      toolRequest: {
                        name: 'resumableTool',
                        input: { doIt: true },
                        ref: 'ref789',
                      },
                    },
                    {
                      toolRequest: {
                        name: 'dynamicInterrupt',
                        input: { doIt: true },
                        ref: 'ref890',
                      },
                    },
                  ]
                : [{ text: 'done' }],
          },
        };
      };

      const response = await ai.generate({
        prompt: 'call the tool',
        tools: [
          'interruptingTool',
          'simpleTool',
          'resumableTool',
          dynamicInterrupt,
        ],
      });

      assert.strictEqual(reqCounter, 1);
      assert.deepStrictEqual(response.toolRequests, [
        {
          toolRequest: {
            input: {},
            name: 'interruptingTool',
            ref: 'ref123',
          },
          metadata: {
            interrupt: {
              confirm: 'is it a banana?',
            },
          },
        },
        {
          toolRequest: {
            input: {
              name: 'foo',
            },
            name: 'simpleTool',
            ref: 'ref456',
          },
          metadata: {
            pendingOutput: 'response: foo',
          },
        },
        {
          metadata: {
            interrupt: true,
          },
          toolRequest: {
            name: 'resumableTool',
            ref: 'ref789',
            input: {
              doIt: true,
            },
          },
        },
        {
          metadata: { interrupt: true },
          toolRequest: {
            input: {
              doIt: true,
            },
            name: 'dynamicInterrupt',
            ref: 'ref890',
          },
        },
      ]);
      assert.deepStrictEqual(response.message?.toJSON(), {
        role: 'model',
        content: [
          {
            text: 'reasoning',
          },
          {
            metadata: {
              interrupt: {
                confirm: 'is it a banana?',
              },
            },
            toolRequest: {
              input: {},
              name: 'interruptingTool',
              ref: 'ref123',
            },
          },
          {
            toolRequest: {
              input: {
                name: 'foo',
              },
              name: 'simpleTool',
              ref: 'ref456',
            },
            metadata: {
              pendingOutput: 'response: foo',
            },
          },
          {
            metadata: {
              interrupt: true,
            },
            toolRequest: {
              name: 'resumableTool',
              ref: 'ref789',
              input: {
                doIt: true,
              },
            },
          },
          {
            metadata: { interrupt: true },
            toolRequest: {
              input: {
                doIt: true,
              },
              name: 'dynamicInterrupt',
              ref: 'ref890',
            },
          },
        ],
      });
      assert.deepStrictEqual(pm.lastRequest, {
        config: {},
        messages: [
          {
            role: 'user',
            content: [{ text: 'call the tool' }],
          },
        ],
        output: {},
        tools: [
          {
            description: 'description',
            inputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            name: 'interruptingTool',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
          {
            description: 'description',
            inputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            name: 'simpleTool',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
          {
            description: 'description',
            inputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            name: 'resumableTool',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
          {
            description: 'description',
            inputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            name: 'dynamicInterrupt',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
        ],
      });
    });

    it('can resume generation', async () => {
      const interrupter = ai.defineInterrupt({
        name: 'interrupter',
        description: 'always interrupts',
      });
      const truth = ai.defineTool(
        { name: 'truth', description: 'always returns true' },
        async () => true
      );
      const resumable = ai.defineTool(
        {
          name: 'resumable',
          description: 'interrupts unless resumed with {status: "ok"}',
        },
        async (_input, { interrupt, resumed }) => {
          console.log('RESUMABLE TOOL CALLED WITH:', resumed);
          if ((resumed as any)?.status === 'ok') return true;
          return interrupt();
        }
      );

      const messages: MessageData[] = [
        { role: 'user', content: [{ text: 'hello' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: { name: 'interrupter', input: {} },
              metadata: { interrupt: true },
            },
            {
              toolRequest: { name: 'truth', input: {} },
              metadata: { pendingOutput: true },
            },
            {
              toolRequest: { name: 'resumable', input: {} },
              metadata: { interrupt: true },
            },
          ],
        },
      ];

      const response = await ai.generate({
        model: 'echoModel',
        messages,
        tools: [interrupter, resumable, truth],
        resume: {
          respond: interrupter.respond(
            {
              toolRequest: { name: 'interrupter', input: {} },
              metadata: { interrupt: true },
            },
            23
          ),
          restart: resumable.restart(
            {
              toolRequest: { name: 'resumable', input: {} },
              metadata: { interrupt: true },
            },
            { status: 'ok' }
          ),
        },
      });

      const revisedModelMessage = response.messages.at(-3);
      const toolMessage = response.messages.at(-2);

      assert.deepStrictEqual(
        revisedModelMessage?.content,
        [
          {
            metadata: {
              resolvedInterrupt: true,
            },
            toolRequest: {
              input: {},
              name: 'interrupter',
            },
          },
          {
            metadata: {},
            toolRequest: {
              input: {},
              name: 'truth',
            },
          },
          {
            metadata: {
              resolvedInterrupt: true,
            },
            toolRequest: {
              input: {},
              name: 'resumable',
            },
          },
        ],
        'resuming amends the model message to resolve interrupts'
      );
      assert.deepStrictEqual(
        toolMessage?.content,
        [
          {
            metadata: {
              interruptResponse: true,
            },
            toolResponse: {
              name: 'interrupter',
              output: 23,
            },
          },
          {
            metadata: {
              source: 'pending',
            },
            toolResponse: {
              name: 'truth',
              output: true,
            },
          },
          {
            toolResponse: {
              name: 'resumable',
              output: true,
            },
          },
        ],
        'resuming generates a tool message containing all expected responses'
      );
    });

    it('streams a generated tool message when resumed', async () => {
      pm.handleResponse = async (request, sendChunk) => {
        sendChunk?.({
          role: 'model',
          index: 0,
          content: [{ text: 'final response' }],
        });
        return {
          message: { role: 'model', content: [{ text: 'final response' }] },
        };
      };

      const chunks: GenerateResponseChunkData[] = [];
      await ai.generate({
        onChunk: (chunk) => chunks.push(chunk.toJSON()),
        messages: [
          { role: 'user', content: [{ text: 'use the doThing tool' }] },
          {
            role: 'model',
            content: [
              {
                toolRequest: { name: 'doThing', input: {} },
                metadata: { interrupt: true },
              },
            ],
          },
        ],
        resume: {
          respond: { toolResponse: { name: 'doThing', output: 'did thing' } },
        },
      });

      assert.deepStrictEqual(chunks, [
        {
          content: [
            {
              toolResponse: {
                name: 'doThing',
                output: 'did thing',
              },
            },
          ],
          index: 0,
          role: 'tool',
        },
        {
          content: [
            {
              text: 'final response',
            },
          ],
          index: 1,
          role: 'model',
        },
      ]);
    });
  });

  describe('long running', () => {
    let ai: GenkitBeta;
    let pm: ProgrammableModel;

    beforeEach(() => {
      ai = genkit({
        model: 'programmableModel',
      });
      pm = defineProgrammableModel(ai);
    });

    it('starts the operation', async () => {
      ai.defineTool(
        { name: 'testTool', description: 'description' },
        async () => 'tool called'
      );

      ai.defineBackgroundModel({
        name: 'bkg-model',
        async start(_) {
          return {
            id: '123',
          };
        },
        async check(operation) {
          return {
            id: '123',
          };
        },
      });

      const { operation } = await ai.generate({
        model: 'bkg-model',
        prompt: 'call the tool',
        tools: ['testTool'],
      });

      delete (operation as any).latencyMs;
      assert.deepStrictEqual(operation, {
        action: '/background-model/bkg-model',
        id: '123',
      });
    });

    it('checks operation status', async () => {
      const newOp = {
        id: '123',
        done: true,
        output: {
          finishReason: 'stop',
          message: {
            role: 'model',
            content: [{ text: 'done' }],
          },
        },
      } as Operation;

      ai.defineBackgroundModel({
        name: 'bkg-model',
        async start(_) {
          return {
            id: '123',
          };
        },
        async check(operation) {
          return { ...newOp };
        },
      });

      const operation = await ai.checkOperation({
        action: '/background-model/bkg-model',
        id: '123',
      });

      assert.deepStrictEqual(operation, {
        ...newOp,
        action: '/background-model/bkg-model',
      });
    });

    it('cancels operation', async () => {
      const newOp = {
        id: '123',
        done: true,
        output: {
          finishReason: 'stop',
          message: {
            role: 'model',
            content: [{ text: 'cancelled' }],
          },
        },
      } as Operation;

      ai.defineBackgroundModel({
        name: 'bkg-model',
        async start(_) {
          return {
            id: '123',
          };
        },
        async check(operation) {
          return operation;
        },
        async cancel(operation) {
          return { ...newOp };
        },
      });

      const operation = await ai.cancelOperation({
        action: '/background-model/bkg-model',
        id: '123',
      });

      assert.deepStrictEqual(operation, {
        ...newOp,
        action: '/background-model/bkg-model',
      });
    });
  });
});
