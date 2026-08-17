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
 * Everything variable here - model provider, session store, port - is
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
import * as path from 'node:path';
import { agentDirs } from './index.js';
import { serveAgents } from './server.js';

const USAGE = `Usage: agent-dirs serve [dir] [--port <n>]

  dir     agents directory (default ./agents)
  --port  listen port (default PORT env or 8080)`;

interface ServeArgs {
  dir: string;
  port?: number;
}

function parseArgs(argv: string[]): ServeArgs | undefined {
  const [command, ...rest] = argv;
  if (command !== 'serve') return undefined;
  const args: ServeArgs = { dir: './agents' };
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i];
    if (arg === '--port') {
      args.port = Number(rest[++i]);
    } else if (arg.startsWith('--port=')) {
      args.port = Number(arg.slice('--port='.length));
    } else if (!arg.startsWith('-')) {
      args.dir = arg;
    } else {
      return undefined;
    }
    if (args.port !== undefined && !Number.isInteger(args.port)) {
      return undefined;
    }
  }
  return args;
}

/**
 * Tool files are TypeScript; plain `node` cannot import them. Locally, tsx
 * (an optional peer) is registered as a module loader. A production image
 * should instead precompile `tools/*.ts` to `.mjs` at build time - the
 * loader accepts both - so tsx never ships to prod.
 */
async function registerTsLoader(): Promise<void> {
  try {
    const tsx = (await import('tsx/esm/api')) as {
      register: () => unknown;
    };
    tsx.register();
  } catch {
    logger.warn(
      '[agent-dirs] tsx is not installed - TypeScript tool files will fail ' +
        'to load. Install tsx, or precompile tools to .mjs.'
    );
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (!args) {
    console.error(USAGE);
    process.exit(1);
  }
  await registerTsLoader();

  // Default provider and store. Platform-resolved config (firebase.json /
  // the hosting service) should select these per project: provider from the
  // project's enabled AI backend, store from an environment-aware default
  // (file locally, Firestore on GCP).
  const ai = genkit({
    plugins: [vertexAI(), agentDirs({ dir: args.dir })],
  });

  const { agents, server } = await serveAgents(ai, {
    ...(args.port !== undefined && { port: args.port }),
  });
  if (agents.length === 0) {
    logger.warn(
      `[agent-dirs] no agents found under ${path.resolve(args.dir)}`
    );
  }
  const shutdown = () => {
    server?.close(() => process.exit(0));
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

void main();
