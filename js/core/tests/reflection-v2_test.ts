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

import * as assert from 'assert';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { WebSocket, WebSocketServer } from 'ws';
import { z } from 'zod';
import { action, bidiAction } from '../src/action.js';
import { initNodeFeatures } from '../src/node.js';
import { ReflectionServerV2 } from '../src/reflection-v2.js';
import { Registry } from '../src/registry.js';

initNodeFeatures();

describe('ReflectionServerV2', () => {
  let wss: WebSocketServer;
  let server: ReflectionServerV2;
  let registry: Registry;
  let port: number;
  let connections: WebSocket[] = [];

  beforeEach(() => {
    return new Promise<void>((resolve) => {
      wss = new WebSocketServer({ port: 0 });
      wss.on('listening', () => {
        port = (wss.address() as any).port;
        resolve();
      });
      wss.on('connection', (ws) => {
        connections.push(ws); // Track all connections
      });
      registry = new Registry();
    });
  });

  afterEach(async () => {
    if (server) {
      await server.stop();
    }
    // Terminate all connections to let wss.close() proceed
    for (const ws of connections) {
      ws.terminate();
    }
    connections = [];
    await new Promise<void>((resolve) => {
      wss.close(() => resolve());
    });
  });

  it('should connect to the server and register', async () => {
    const connected = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('connect timeout')),
        2000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          const msg = JSON.parse(data.toString());
          if (msg.method === 'register') {
            assert.strictEqual(msg.params.name, 'test-app');
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                result: {},
                id: msg.id,
              })
            );
            clearTimeout(timer);
            resolve();
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
      name: 'test-app',
    });
    await server.start();
    await connected;
  });

  it('should handle listActions', async () => {
    // Register a dummy action
    const testAction = action(
      {
        name: 'testAction',
        description: 'A test action',
        inputSchema: z.object({ foo: z.string() }),
        outputSchema: z.object({ bar: z.string() }),
        actionType: 'custom',
      },
      async (input) => ({ bar: input.foo })
    );
    registry.registerAction('custom', testAction);

    const gotListActions = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('listActions timeout')),
        2000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          const msg = JSON.parse(data.toString());
          if (msg.method === 'register') {
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                result: {},
                id: msg.id,
              })
            );
            // After registration, request listActions
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                method: 'listActions',
                id: '123',
              })
            );
          } else if (msg.id === '123') {
            const actions = msg.result.actions;
            assert.ok(actions['/custom/testAction']);
            assert.strictEqual(
              actions['/custom/testAction'].name,
              'testAction'
            );
            clearTimeout(timer);
            resolve();
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await gotListActions;
  });

  it('should handle listValues', async () => {
    registry.registerValue('middleware', 'my-mw', { template: 'foo' });

    const gotListValues = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('listValues timeout')),
        2000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          const msg = JSON.parse(data.toString());
          if (msg.method === 'register') {
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                result: {},
                id: msg.id,
              })
            );
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                method: 'listValues',
                params: { type: 'middleware' },
                id: '124',
              })
            );
          } else if (msg.id === '124') {
            assert.ok(msg.result.values['my-mw']);
            assert.strictEqual(msg.result.values['my-mw'].template, 'foo');
            clearTimeout(timer);
            resolve();
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await gotListValues;
  });

  it('should handle listValues with toJson mapping', async () => {
    registry.registerValue('middleware', 'mw1', {
      toJson: () => ({ name: 'mw1' }),
    });
    registry.registerValue('middleware', 'mw2', {
      name: 'mw2',
    });

    const gotListValues = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('listValues toJson timeout')),
        2000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          const msg = JSON.parse(data.toString());
          if (msg.method === 'register') {
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                result: {},
                id: msg.id,
              })
            );
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                method: 'listValues',
                params: { type: 'middleware' },
                id: '125',
              })
            );
          } else if (msg.id === '125') {
            assert.ok(msg.result.values['mw1']);
            assert.strictEqual(msg.result.values['mw1'].name, 'mw1');
            assert.ok(msg.result.values['mw2']);
            assert.strictEqual(msg.result.values['mw2'].name, 'mw2');
            clearTimeout(timer);
            resolve();
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await gotListValues;
  });

  it('should reject unsupported type parameter for listValues in V2', async () => {
    const gotError = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('listValues error timeout')),
        2000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          const msg = JSON.parse(data.toString());
          if (msg.method === 'register') {
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                result: {},
                id: msg.id,
              })
            );
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                method: 'listValues',
                params: { type: 'unsupported_type' },
                id: '126',
              })
            );
          } else if (msg.id === '126') {
            assert.ok(msg.error);
            assert.strictEqual(msg.error.code, -32602);
            assert.match(msg.error.message, /is not supported/);
            clearTimeout(timer);
            resolve();
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await gotError;
  });

  it('should handle runAction', async () => {
    const testAction = action(
      {
        name: 'testAction',
        inputSchema: z.object({ foo: z.string() }),
        outputSchema: z.object({ bar: z.string() }),
        actionType: 'custom',
      },
      async (input) => ({ bar: input.foo })
    );
    registry.registerAction('custom', testAction);

    const actionRun = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('runAction timeout')),
        2000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/testAction',
                    input: { foo: 'baz' },
                  },
                  id: '456',
                })
              );
            } else if (msg.id === '456') {
              if (msg.error) {
                reject(
                  new Error(`runAction error: ${JSON.stringify(msg.error)}`)
                );
                return;
              }
              assert.strictEqual(msg.result.result.bar, 'baz');
              clearTimeout(timeout);
              resolve();
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionRun;
  });

  it('should handle streaming runAction', async () => {
    const streamAction = action(
      {
        name: 'streamAction',
        inputSchema: z.object({ foo: z.string() }),
        outputSchema: z.string(),
        actionType: 'custom',
      },
      async (input, { sendChunk }) => {
        sendChunk('chunk1');
        sendChunk('chunk2');
        return 'done';
      }
    );
    registry.registerAction('custom', streamAction);

    const chunks: any[] = [];
    const actionRun = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('streamAction timeout')),
        2000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/streamAction',
                    input: { foo: 'baz' },
                    stream: true,
                  },
                  id: '789',
                })
              );
            } else if (msg.method === 'streamChunk') {
              chunks.push(msg.params.chunk);
            } else if (msg.id === '789') {
              if (msg.error) {
                reject(
                  new Error(`streamAction error: ${JSON.stringify(msg.error)}`)
                );
                return;
              }
              assert.strictEqual(msg.result.result, 'done');
              assert.deepStrictEqual(chunks, ['chunk1', 'chunk2']);
              clearTimeout(timeout);
              resolve();
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionRun;
  });

  it('should handle cancelAction', async () => {
    let cancelSignal: AbortSignal | undefined;
    const longAction = action(
      {
        name: 'longAction',
        inputSchema: z.any(),
        outputSchema: z.any(),
        actionType: 'custom',
      },
      async (_, { abortSignal }) => {
        cancelSignal = abortSignal;
        await new Promise((resolve, reject) => {
          const timer = setTimeout(resolve, 5000);
          if (abortSignal.aborted) {
            clearTimeout(timer);
            reject(new Error('Action cancelled'));
            return;
          }
          abortSignal.addEventListener('abort', () => {
            clearTimeout(timer);
            reject(new Error('Action cancelled'));
          });
        });
      }
    );
    registry.registerAction('custom', longAction);

    const actionCancelled = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('cancelAction timeout')),
        2000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              // Start action
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/longAction',
                    input: {},
                  },
                  id: '999',
                })
              );
            } else if (msg.method === 'runActionState') {
              // Got traceId, send cancel
              const traceId = msg.params.state.traceId;
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'cancelAction',
                  params: { traceId },
                  id: '1000',
                })
              );
            } else if (msg.id === '1000') {
              // Cancel response
              assert.strictEqual(msg.result.message, 'Action cancelled');
            } else if (msg.id === '999') {
              // Run action response (should be error)
              if (msg.error) {
                // Ensure code indicates cancellation if possible, or just error
                // In implementation we send code -32000 and message 'Action was cancelled'
                assert.match(msg.error.message, /cancelled/);
                assert.ok(cancelSignal?.aborted);
                clearTimeout(timeout);
                resolve();
              } else {
                reject(new Error('Action should have failed'));
              }
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionCancelled;
  });

  it('should reconnect when lost connection and register again', async () => {
    let connectionCount = 0;
    const reconnected = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('reconnect timeout')),
        5000
      );
      wss.on('connection', (ws) => {
        connectionCount++;
        ws.on('message', (data) => {
          const msg = JSON.parse(data.toString());
          if (msg.method === 'register') {
            if (connectionCount === 1) {
              ws.terminate(); // Simulate server drop
            } else if (connectionCount === 2) {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              clearTimeout(timeout);
              resolve();
            }
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    (server as any).baseDelayMs = 10; // Fast for testing
    await server.start();
    await reconnected;
  });

  it('should handle bidi streaming action with input and output streams', async () => {
    // Create a bidi action that echoes input chunks back as output chunks,
    // and returns the count of received chunks as the final result.
    const echoBidiAction = bidiAction(
      {
        name: 'echoBidi',
        inputSchema: z.string(),
        outputSchema: z.object({ count: z.number() }),
        streamSchema: z.string(),
        actionType: 'custom',
      },
      async function* ({ inputStream }) {
        let count = 0;
        for await (const chunk of inputStream) {
          count++;
          yield `echo:${chunk}`;
        }
        return { count };
      }
    );
    registry.registerAction('custom', echoBidiAction);

    const outputChunks: any[] = [];
    const actionRun = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('bidi action timeout')),
        3000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              // Start the bidi action with stream + streamInput
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/echoBidi',
                    input: null,
                    stream: true,
                    streamInput: true,
                  },
                  id: 'bidi-1',
                })
              );
              // Send input stream chunks
              setTimeout(() => {
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-1', chunk: 'hello' },
                  })
                );
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-1', chunk: 'world' },
                  })
                );
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'endInputStream',
                    params: { requestId: 'bidi-1' },
                  })
                );
              }, 50);
            } else if (msg.method === 'streamChunk') {
              outputChunks.push(msg.params.chunk);
            } else if (msg.id === 'bidi-1') {
              if (msg.error) {
                reject(
                  new Error(`bidi action error: ${JSON.stringify(msg.error)}`)
                );
                return;
              }
              assert.deepStrictEqual(msg.result.result, { count: 2 });
              assert.deepStrictEqual(outputChunks, [
                'echo:hello',
                'echo:world',
              ]);
              clearTimeout(timeout);
              resolve();
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionRun;
  });

  it('should handle bidi streaming with multiple interleaved chunks', async () => {
    // Bidi action that transforms each input chunk and yields it
    const transformBidiAction = bidiAction(
      {
        name: 'transformBidi',
        inputSchema: z.number(),
        outputSchema: z.object({ sum: z.number() }),
        streamSchema: z.number(),
        actionType: 'custom',
      },
      async function* ({ inputStream }) {
        let sum = 0;
        for await (const n of inputStream) {
          sum += n;
          yield n * 2; // output is double the input
        }
        return { sum };
      }
    );
    registry.registerAction('custom', transformBidiAction);

    const outputChunks: any[] = [];
    const actionRun = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('transform bidi timeout')),
        3000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/transformBidi',
                    input: null,
                    stream: true,
                    streamInput: true,
                  },
                  id: 'bidi-2',
                })
              );
              // Send chunks with slight delays to simulate real-time input
              setTimeout(() => {
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-2', chunk: 1 },
                  })
                );
              }, 20);
              setTimeout(() => {
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-2', chunk: 2 },
                  })
                );
              }, 40);
              setTimeout(() => {
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-2', chunk: 3 },
                  })
                );
              }, 60);
              setTimeout(() => {
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'endInputStream',
                    params: { requestId: 'bidi-2' },
                  })
                );
              }, 80);
            } else if (msg.method === 'streamChunk') {
              if (msg.params.requestId === 'bidi-2') {
                outputChunks.push(msg.params.chunk);
              }
            } else if (msg.id === 'bidi-2') {
              if (msg.error) {
                reject(
                  new Error(
                    `transform bidi error: ${JSON.stringify(msg.error)}`
                  )
                );
                return;
              }
              assert.deepStrictEqual(msg.result.result, { sum: 6 });
              assert.deepStrictEqual(outputChunks, [2, 4, 6]);
              clearTimeout(timeout);
              resolve();
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionRun;
  });

  it('should handle bidi action with no input chunks (immediate close)', async () => {
    const emptyBidiAction = bidiAction(
      {
        name: 'emptyBidi',
        inputSchema: z.string(),
        outputSchema: z.object({ received: z.number() }),
        streamSchema: z.string(),
        actionType: 'custom',
      },
      async function* ({ inputStream }) {
        let received = 0;
        for await (const _ of inputStream) {
          received++;
        }
        return { received };
      }
    );
    registry.registerAction('custom', emptyBidiAction);

    const actionRun = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('empty bidi timeout')),
        3000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/emptyBidi',
                    input: null,
                    stream: true,
                    streamInput: true,
                  },
                  id: 'bidi-3',
                })
              );
              // Immediately close the input stream without sending any chunks
              setTimeout(() => {
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'endInputStream',
                    params: { requestId: 'bidi-3' },
                  })
                );
              }, 50);
            } else if (msg.id === 'bidi-3') {
              if (msg.error) {
                reject(
                  new Error(`empty bidi error: ${JSON.stringify(msg.error)}`)
                );
                return;
              }
              assert.deepStrictEqual(msg.result.result, { received: 0 });
              clearTimeout(timeout);
              resolve();
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionRun;
  });

  it('should handle bidi action with init data passed as input', async () => {
    const initBidiAction = bidiAction(
      {
        name: 'initBidi',
        inputSchema: z.string(),
        outputSchema: z.object({
          prefix: z.string(),
          messages: z.array(z.string()),
        }),
        streamSchema: z.string(),
        initSchema: z.object({ prefix: z.string() }),
        actionType: 'custom',
      },
      async function* ({ inputStream, init }) {
        const messages: string[] = [];
        for await (const msg of inputStream) {
          const prefixed = `${init.prefix}:${msg}`;
          messages.push(prefixed);
          yield prefixed;
        }
        return { prefix: init.prefix, messages };
      }
    );
    registry.registerAction('custom', initBidiAction);

    const outputChunks: any[] = [];
    const actionRun = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('init bidi timeout')),
        3000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              // Input carries init data for bidi actions
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/initBidi',
                    input: { prefix: 'test' },
                    stream: true,
                    streamInput: true,
                  },
                  id: 'bidi-4',
                })
              );
              setTimeout(() => {
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-4', chunk: 'a' },
                  })
                );
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-4', chunk: 'b' },
                  })
                );
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'endInputStream',
                    params: { requestId: 'bidi-4' },
                  })
                );
              }, 50);
            } else if (msg.method === 'streamChunk') {
              if (msg.params.requestId === 'bidi-4') {
                outputChunks.push(msg.params.chunk);
              }
            } else if (msg.id === 'bidi-4') {
              if (msg.error) {
                reject(
                  new Error(`init bidi error: ${JSON.stringify(msg.error)}`)
                );
                return;
              }
              assert.deepStrictEqual(msg.result.result, {
                prefix: 'test',
                messages: ['test:a', 'test:b'],
              });
              assert.deepStrictEqual(outputChunks, ['test:a', 'test:b']);
              clearTimeout(timeout);
              resolve();
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionRun;
  });

  it('should handle bidi streaming without output streaming (streamInput only)', async () => {
    // Test bidi with streamInput but no output streaming (stream: false)
    const collectBidiAction = bidiAction(
      {
        name: 'collectBidi',
        inputSchema: z.string(),
        outputSchema: z.object({ items: z.array(z.string()) }),
        streamSchema: z.string(),
        actionType: 'custom',
      },
      async function* ({ inputStream }) {
        const items: string[] = [];
        for await (const item of inputStream) {
          items.push(item);
        }
        return { items };
      }
    );
    registry.registerAction('custom', collectBidiAction);

    const actionRun = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('collect bidi timeout')),
        3000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              // stream: false but streamInput: true
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/collectBidi',
                    input: null,
                    stream: false,
                    streamInput: true,
                  },
                  id: 'bidi-5',
                })
              );
              setTimeout(() => {
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-5', chunk: 'x' },
                  })
                );
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'sendInputStreamChunk',
                    params: { requestId: 'bidi-5', chunk: 'y' },
                  })
                );
                ws.send(
                  JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'endInputStream',
                    params: { requestId: 'bidi-5' },
                  })
                );
              }, 50);
            } else if (msg.id === 'bidi-5') {
              if (msg.error) {
                reject(
                  new Error(`collect bidi error: ${JSON.stringify(msg.error)}`)
                );
                return;
              }
              assert.deepStrictEqual(msg.result.result, {
                items: ['x', 'y'],
              });
              clearTimeout(timeout);
              resolve();
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionRun;
  });

  it('should pass init param from runAction to the action handler', async () => {
    const initAction = action(
      {
        name: 'initAction',
        inputSchema: z.string(),
        outputSchema: z.object({
          receivedInput: z.string(),
          receivedInit: z.object({ sessionId: z.string() }),
        }),
        initSchema: z.object({ sessionId: z.string() }),
        actionType: 'custom',
      },
      async (input, { init }) => {
        return { receivedInput: input, receivedInit: init };
      }
    );
    registry.registerAction('custom', initAction);

    const actionRun = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('init action timeout')),
        3000
      );
      wss.on('connection', (ws) => {
        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg.method === 'register') {
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  result: {},
                  id: msg.id,
                })
              );
              ws.send(
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'runAction',
                  params: {
                    key: '/custom/initAction',
                    input: 'hello',
                    init: { sessionId: 'sess-42' },
                    stream: false,
                  },
                  id: 'init-1',
                })
              );
            } else if (msg.id === 'init-1') {
              if (msg.error) {
                reject(
                  new Error(`init action error: ${JSON.stringify(msg.error)}`)
                );
                return;
              }
              assert.deepStrictEqual(msg.result.result, {
                receivedInput: 'hello',
                receivedInit: { sessionId: 'sess-42' },
              });
              clearTimeout(timeout);
              resolve();
            }
          } catch (e) {
            clearTimeout(timeout);
            reject(e);
          }
        });
      });
    });

    server = new ReflectionServerV2(registry, {
      url: `ws://localhost:${port}`,
    });
    await server.start();
    await actionRun;
  });
});
