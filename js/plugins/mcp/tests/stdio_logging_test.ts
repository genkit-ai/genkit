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

import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import { genkit, z } from 'genkit/beta';
import { logger } from 'genkit/logging';
import * as assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { PassThrough } from 'node:stream';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { createMcpServer } from '../src/index.js';

describe('MCP stdio logging', () => {
  let originalReflectionEnabled: string | undefined;

  beforeEach(() => {
    originalReflectionEnabled = process.env.GENKIT_REFLECTION_ENABLED;
    process.env.GENKIT_REFLECTION_ENABLED = 'false';
  });

  afterEach(() => {
    if (originalReflectionEnabled === undefined) {
      delete process.env.GENKIT_REFLECTION_ENABLED;
    } else {
      process.env.GENKIT_REFLECTION_ENABLED = originalReflectionEnabled;
    }
  });
  it('keeps startup registration and runtime logs off stdout when configured before genkit()', async () => {
    const originalDebug = console.debug;
    const originalInfo = console.info;
    const originalError = console.error;
    const originalLevel = logger.defaultLogger.level;
    const originalOutput = logger.defaultLogger.output;
    const stdoutLogs: string[] = [];
    const stderrLogs: string[] = [];

    console.debug = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.info = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.error = (...args: unknown[]) => stderrLogs.push(args.join(' '));
    logger.setLogLevel('debug');
    logger.setDefaultLogOutput('stderr');

    let ai: ReturnType<typeof genkit> | undefined;
    let server: ReturnType<typeof createMcpServer> | undefined;

    try {
      ai = genkit({});
      ai.defineTool(
        {
          name: 'add',
          description: 'add two numbers together',
          inputSchema: z.object({ a: z.number(), b: z.number() }),
          outputSchema: z.number(),
        },
        async ({ a, b }) => a + b
      );
      ai.definePrompt(
        {
          name: 'happy',
          description: 'everybody together now',
        },
        'hello'
      );
      ai.defineResource(
        {
          name: 'my resource',
          uri: 'my://resource',
        },
        async () => ({ content: [{ text: 'my resource' }] })
      );

      server = createMcpServer(ai, {
        name: 'example_server',
        version: '0.0.1',
      });
      await server.start(
        new StdioServerTransport(new PassThrough(), new PassThrough())
      );
      logger.info('stdio runtime info');
      logger.debug('stdio runtime debug');

      assert.deepEqual(stdoutLogs, []);
      assert.ok(stderrLogs.includes('registering /tool/add'));
      assert.ok(
        stderrLogs.some((line) =>
          line.includes(
            "[MCP Server] MCP server 'example_server' started successfully."
          )
        )
      );
      assert.ok(stderrLogs.includes('stdio runtime info'));
      assert.ok(stderrLogs.includes('stdio runtime debug'));
    } finally {
      await server?.server?.close();
      await ai?.stopServers();
      logger.setDefaultLogOutput(originalOutput);
      logger.setLogLevel(originalLevel as 'error' | 'warn' | 'info' | 'debug');
      console.debug = originalDebug;
      console.info = originalInfo;
      console.error = originalError;
    }
  });

  it('keeps Genkit logs off the MCP stdio protocol stream', async () => {
    const originalDebug = console.debug;
    const originalInfo = console.info;
    const originalError = console.error;
    const originalLevel = logger.defaultLogger.level;
    const originalOutput = logger.defaultLogger.output;
    const stdoutLogs: string[] = [];
    const stderrLogs: string[] = [];
    const ai = genkit({});
    const server = createMcpServer(ai, { name: 'stdio-logging-test' });

    console.debug = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.info = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.error = (...args: unknown[]) => stderrLogs.push(args.join(' '));
    logger.setDefaultLogOutput('stdout');
    logger.setLogLevel('debug');

    try {
      await server.start(
        new StdioServerTransport(new PassThrough(), new PassThrough())
      );
      logger.info('stdio info sentinel');
      logger.debug('stdio debug sentinel');

      assert.deepEqual(stdoutLogs, []);
      assert.ok(
        stderrLogs.some((line) =>
          line.includes(
            "[MCP Server] MCP server 'stdio-logging-test' started successfully."
          )
        )
      );
      assert.ok(stderrLogs.includes('stdio info sentinel'));
      assert.ok(stderrLogs.includes('stdio debug sentinel'));
    } finally {
      await server.server?.close();
      await ai.stopServers();
      logger.setDefaultLogOutput(originalOutput);
      logger.setLogLevel(originalLevel as 'error' | 'warn' | 'info' | 'debug');
      console.debug = originalDebug;
      console.info = originalInfo;
      console.error = originalError;
    }
  });

  it('redirects logs to stderr for duplicate SDK StdioServerTransport copies', async () => {
    const originalDebug = console.debug;
    const originalInfo = console.info;
    const originalError = console.error;
    const originalLevel = logger.defaultLogger.level;
    const originalOutput = logger.defaultLogger.output;
    const stdoutLogs: string[] = [];
    const stderrLogs: string[] = [];
    const ai = genkit({});
    const server = createMcpServer(ai, { name: 'dup-stdio-test' });

    // Simulate a StdioServerTransport constructed from a duplicate SDK package.
    class DuplicateSdkStdioTransport implements Transport {
      async start() {}
      async send() {}
      async close() {}
    }
    Object.defineProperty(DuplicateSdkStdioTransport, 'name', {
      value: 'StdioServerTransport',
    });

    console.debug = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.info = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.error = (...args: unknown[]) => stderrLogs.push(args.join(' '));
    logger.setDefaultLogOutput('stdout');
    logger.setLogLevel('debug');

    try {
      await server.start(new DuplicateSdkStdioTransport());
      logger.info('duplicate sdk info sentinel');

      assert.deepEqual(stdoutLogs, []);
      assert.ok(stderrLogs.includes('duplicate sdk info sentinel'));
    } finally {
      await server.server?.close();
      await ai.stopServers();
      logger.setDefaultLogOutput(originalOutput);
      logger.setLogLevel(originalLevel as 'error' | 'warn' | 'info' | 'debug');
      console.debug = originalDebug;
      console.info = originalInfo;
      console.error = originalError;
    }
  });

  it('selects stderr synchronously when start() omits transport', async () => {
    const originalDebug = console.debug;
    const originalInfo = console.info;
    const originalError = console.error;
    const originalLevel = logger.defaultLogger.level;
    const originalOutput = logger.defaultLogger.output;
    const stdoutLogs: string[] = [];
    const stderrLogs: string[] = [];
    const ai = genkit({});
    const server = createMcpServer(ai, { name: 'default-stdio-sync-test' });

    console.debug = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.info = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.error = (...args: unknown[]) => stderrLogs.push(args.join(' '));
    logger.setDefaultLogOutput('stdout');
    logger.setLogLevel('debug');

    try {
      const ready = server.start();
      logger.info('booting before await');
      await ready;
      logger.info('running after await');

      assert.deepEqual(stdoutLogs, []);
      assert.ok(stderrLogs.includes('booting before await'));
      assert.ok(stderrLogs.includes('running after await'));
    } finally {
      await server.server?.close();
      await ai.stopServers();
      logger.setDefaultLogOutput(originalOutput);
      logger.setLogLevel(originalLevel as 'error' | 'warn' | 'info' | 'debug');
      console.debug = originalDebug;
      console.info = originalInfo;
      console.error = originalError;
    }
  });

  it('redirects logs to stderr for CommonJS StdioServerTransport subclasses', async () => {
    const originalDebug = console.debug;
    const originalInfo = console.info;
    const originalError = console.error;
    const originalLevel = logger.defaultLogger.level;
    const originalOutput = logger.defaultLogger.output;
    const stdoutLogs: string[] = [];
    const stderrLogs: string[] = [];
    const ai = genkit({});
    const server = createMcpServer(ai, { name: 'cjs-stdio-subclass-test' });

    const require = createRequire(import.meta.url);
    const { StdioServerTransport: CjsStdioServerTransport } =
      require('@modelcontextprotocol/sdk/server/stdio.js') as {
        StdioServerTransport: typeof StdioServerTransport;
      };
    class CustomCjsStdioTransport extends CjsStdioServerTransport {}
    const transport = new CustomCjsStdioTransport(
      new PassThrough(),
      new PassThrough()
    );

    assert.equal(transport instanceof StdioServerTransport, false);
    assert.equal(transport.constructor.name, 'CustomCjsStdioTransport');

    console.debug = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.info = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.error = (...args: unknown[]) => stderrLogs.push(args.join(' '));
    logger.setDefaultLogOutput('stdout');
    logger.setLogLevel('debug');

    try {
      const ready = server.start(transport);
      logger.info('cjs subclass before await');
      await ready;
      logger.info('cjs subclass after await');

      assert.deepEqual(stdoutLogs, []);
      assert.ok(stderrLogs.includes('cjs subclass before await'));
      assert.ok(stderrLogs.includes('cjs subclass after await'));
    } finally {
      await server.server?.close();
      await ai.stopServers();
      logger.setDefaultLogOutput(originalOutput);
      logger.setLogLevel(originalLevel as 'error' | 'warn' | 'info' | 'debug');
      console.debug = originalDebug;
      console.info = originalInfo;
      console.error = originalError;
    }
  });

  it('preserves stdout logging for non-stdio transports', async () => {
    const originalDebug = console.debug;
    const originalInfo = console.info;
    const originalError = console.error;
    const originalLevel = logger.defaultLogger.level;
    const originalOutput = logger.defaultLogger.output;
    const stdoutLogs: string[] = [];
    const stderrLogs: string[] = [];
    const ai = genkit({});
    const server = createMcpServer(ai, { name: 'non-stdio-test' });

    class FakeHttpTransport implements Transport {
      async start() {}
      async send() {}
      async close() {}
    }

    console.debug = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.info = (...args: unknown[]) => stdoutLogs.push(args.join(' '));
    console.error = (...args: unknown[]) => stderrLogs.push(args.join(' '));
    logger.setDefaultLogOutput('stdout');
    logger.setLogLevel('debug');

    try {
      await server.start(new FakeHttpTransport());
      logger.info('http info sentinel');
      logger.debug('http debug sentinel');

      assert.ok(stdoutLogs.includes('http info sentinel'));
      assert.ok(stdoutLogs.includes('http debug sentinel'));
      assert.deepEqual(stderrLogs, []);
    } finally {
      await server.server?.close();
      await ai.stopServers();
      logger.setDefaultLogOutput(originalOutput);
      logger.setLogLevel(originalLevel as 'error' | 'warn' | 'info' | 'debug');
      console.debug = originalDebug;
      console.info = originalInfo;
      console.error = originalError;
    }
  });
});
