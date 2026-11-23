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
import { genkit } from 'genkit';
import { logger } from 'genkit/logging';
import * as assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import { describe, it } from 'node:test';
import { createMcpServer } from '../src/index.js';

describe('MCP stdio logging', () => {
  it('keeps Genkit logs off the MCP stdio protocol stream', async () => {
    const originalDebug = console.debug;
    const originalInfo = console.info;
    const originalError = console.error;
    const originalLevel = logger.defaultLogger.level;
    const originalOutput = logger.defaultLogger.output;
    const stdoutLogs: string[] = [];
    const stderrLogs: string[] = [];
    const server = createMcpServer(genkit({}), { name: 'stdio-logging-test' });

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
    const server = createMcpServer(genkit({}), { name: 'dup-stdio-test' });

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
    const server = createMcpServer(genkit({}), { name: 'non-stdio-test' });

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
      logger.setDefaultLogOutput(originalOutput);
      logger.setLogLevel(originalLevel as 'error' | 'warn' | 'info' | 'debug');
      console.debug = originalDebug;
      console.info = originalInfo;
      console.error = originalError;
    }
  });
});
