#!/usr/bin/env node
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

/**
 * Zero-code runner: `agent-dirs serve [dir]` serves every agent directory
 * without a host `index.ts`. The agent folder is the only user input.
 *
 * Everything variable here - model provider, session store, host/port - is
 * deliberately NOT read from a project config file. In the product this
 * configuration belongs to the platform (a `firebase.json` block / the
 * hosting service), which hands it to the runner already resolved; a config
 * file read here would become a second, competing source of truth. The
 * defaults below are the local-dev story only.
 *
 * @module @genkit-ai/agent-dirs/cli
 */

import { vertexAI } from '@genkit-ai/google-genai';
import { genkit } from 'genkit/beta';
import { logger } from 'genkit/logging';
import { spawn, type ChildProcess } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { agentDirs } from './index.js';
import { serveAgents } from './server.js';

const USAGE = `Usage: agent-dirs serve [dir] [--port <n>] [--host <h>] [--watch]

  dir      agents directory (default ./agents)
  --port   listen port (default PORT env or 8080)
  --host   interface to bind (default 127.0.0.1; the endpoints carry no
           auth, so pass --host 0.0.0.0 only behind a platform ingress,
           e.g. in a container)
  --watch  restart the server when anything under dir changes (dev only)`;

interface ServeArgs {
  dir?: string;
  port?: number;
  host?: string;
  watch?: boolean;
}

function parseArgs(argv: string[]): ServeArgs | undefined {
  const [command, ...rest] = argv;
  if (command !== 'serve') return undefined;
  const args: ServeArgs = {};
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i];
    if (arg === '--port') {
      args.port = Number(rest[++i]);
    } else if (arg.startsWith('--port=')) {
      args.port = Number(arg.slice('--port='.length));
    } else if (arg === '--host') {
      args.host = rest[++i];
    } else if (arg.startsWith('--host=')) {
      args.host = arg.slice('--host='.length);
    } else if (arg === '--watch') {
      args.watch = true;
    } else if (!arg.startsWith('-')) {
      if (args.dir !== undefined) return undefined; // one dir only
      args.dir = arg;
    } else {
      return undefined;
    }
  }
  if (
    args.port !== undefined &&
    (!Number.isInteger(args.port) || args.port < 0 || args.port > 65535)
  ) {
    return undefined;
  }
  if (args.host !== undefined && !args.host) return undefined;
  return args;
}

/**
 * `--watch`: supervise a child server process and restart it when anything
 * under the agents directory changes. A process restart (rather than
 * in-process re-registration) is deliberate: tool modules are ESM imports
 * and the module cache would serve stale code, and the registry has no
 * un-register. Skill and knowledge bodies are already re-read per turn
 * without this.
 */
function runSupervisor(dir: string, args: ServeArgs): void {
  const resolved = path.resolve(dir);
  if (!fs.existsSync(resolved)) {
    console.error(`agent-dirs: --watch: no such directory: ${resolved}`);
    process.exit(1);
  }

  const childArgs = [
    ...process.execArgv,
    process.argv[1],
    'serve',
    dir,
    ...(args.port !== undefined ? [`--port=${args.port}`] : []),
    ...(args.host !== undefined ? [`--host=${args.host}`] : []),
  ];

  let child: ChildProcess | undefined;
  let restartTimer: NodeJS.Timeout | undefined;
  let shuttingDown = false;

  const start = () => {
    child = spawn(process.execPath, childArgs, { stdio: 'inherit' });
    child.on('exit', (code) => {
      child = undefined;
      // A startup crash (e.g. broken frontmatter under strict mode) must not
      // kill the watcher: stay up so fixing the file restarts the server.
      if (!shuttingDown && code !== 0) {
        logger.warn('[agent-dirs] server exited; waiting for changes');
      }
    });
  };

  const restart = (filename: string | null) => {
    clearTimeout(restartTimer);
    restartTimer = setTimeout(() => {
      logger.info(
        `[agent-dirs] change detected (${filename ?? dir}), restarting`
      );
      if (child) {
        child.once('exit', start);
        child.kill('SIGTERM');
      } else {
        start();
      }
    }, 300);
  };

  fs.watch(resolved, { recursive: true }, (_event, filename) =>
    restart(filename)
  );

  const shutdown = (signal: NodeJS.Signals) => {
    shuttingDown = true;
    clearTimeout(restartTimer);
    child?.kill(signal);
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  logger.info(`[agent-dirs] watching ${resolved}`);
  start();
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (!args) {
    console.error(USAGE);
    process.exit(1);
  }
  const dir = args.dir ?? './agents';

  if (args.watch) {
    runSupervisor(dir, args);
    return;
  }

  // Default provider and store. Platform-resolved config (firebase.json /
  // the hosting service) should select these per project: provider from the
  // project's enabled AI backend, store from an environment-aware default
  // (file locally, Firestore on GCP).
  const ai = genkit({
    plugins: [vertexAI(), agentDirs({ dir })],
  });

  const { agents, server } = await serveAgents(ai, {
    ...(args.port !== undefined && { port: args.port }),
    // Loopback unless told otherwise: the endpoints are unauthenticated.
    host: args.host ?? process.env.HOST ?? '127.0.0.1',
  });
  if (agents.length === 0) {
    logger.warn(`[agent-dirs] no agents found under ${path.resolve(dir)}`);
  }

  let closing = false;
  const shutdown = (signal: NodeJS.Signals) => {
    // Graceful close waits on open connections (a streaming turn is the
    // normal case), so a second signal - or a 5s timeout, matching typical
    // container grace periods - must force-exit or Ctrl-C becomes inert.
    if (closing) process.exit(signal === 'SIGINT' ? 130 : 143);
    closing = true;
    setTimeout(() => process.exit(0), 5000).unref();
    if (!server) process.exit(0);
    server.close(() => process.exit(0));
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch((e) => {
  const err = e as NodeJS.ErrnoException;
  if (err?.code === 'EADDRINUSE') {
    console.error(`agent-dirs: port already in use (${err.message})`);
  } else if (err?.code === 'ERR_SOCKET_BAD_PORT' || err?.code === 'EACCES') {
    console.error(`agent-dirs: cannot bind requested port (${err.message})`);
  } else {
    console.error(`agent-dirs: ${err?.message ?? err}`);
  }
  process.exit(1);
});
