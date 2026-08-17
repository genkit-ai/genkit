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
 * One-call HTTP serving for registered agents.
 *
 * Nothing here is agent-dirs specific - any `defineAgent` user needs the same
 * three routes per agent, and both the agents testapp and this package's demo
 * previously hand-rolled them. Natural upstream home: `@genkit-ai/express`,
 * as a sibling of `startFlowServer` (which serves flows only and wires none
 * of the agent companion actions). Lives here while the plugin is
 * experimental.
 *
 * @module @genkit-ai/agent-dirs/server
 */

import { expressHandler } from '@genkit-ai/express';
import cors, { type CorsOptions } from 'cors';
import express from 'express';
import type { Genkit } from 'genkit';
import type { StreamManager } from 'genkit/beta';
import { logger } from 'genkit/logging';
import type { Server } from 'node:http';
import { listAgents } from './lookup.js';

/** Options for {@link serveAgents}. */
export interface ServeAgentsOptions {
  /** Port to listen on. Defaults to `PORT` env or 8080. */
  port?: number;
  /**
   * Host/interface to bind. Defaults to `HOST` env, else all interfaces.
   * The endpoints carry no auth, so anything that can reach them can run
   * turns on the server's credentials - bind `127.0.0.1` unless deploying
   * behind a platform ingress.
   */
  host?: string;
  /** Route prefix for agent endpoints. Default `/api`. */
  pathPrefix?: string;
  /** CORS options (`cors` package). Default allows all origins. */
  cors?: CorsOptions;
  /**
   * Mount routes on an existing express app instead of creating one. When
   * provided, no body parser, CORS middleware or listener is installed - the
   * caller owns the app lifecycle.
   */
  app?: express.Express;
  /**
   * Stream manager enabling durable stream reconnects (the client re-attaches
   * to an in-flight turn via the `X-Genkit-Stream-Id` header). Without one,
   * streams are plain per-request SSE.
   */
  streamManager?: StreamManager;
}

/** The result of {@link serveAgents}. */
export interface AgentServer {
  app: express.Express;
  /** The HTTP server. Absent when mounting on a caller-owned app. */
  server?: Server;
  /** Names of the agents exposed. */
  agents: string[];
}

/**
 * Exposes every registered agent over HTTP:
 *
 * - `POST <prefix>/<name>` - chat turns (streaming supported), consumable
 *   with `remoteAgent({ url })` from `genkit/beta/client`
 * - `POST <prefix>/<name>/getSnapshot` - snapshot inspection
 * - `POST <prefix>/<name>/abort` - abort an in-flight detached turn
 *
 * ```ts
 * const ai = genkit({ plugins: [vertexAI(), agentDirs()] });
 * await serveAgents(ai);
 * ```
 */
export async function serveAgents(
  ai: Genkit,
  options: ServeAgentsOptions = {}
): Promise<AgentServer> {
  const agents = await listAgents(ai);
  const names = Object.keys(agents);
  if (names.length === 0) {
    logger.warn('[agent-dirs] serveAgents: no agents are registered');
  }

  const app = options.app ?? express();
  if (!options.app) {
    app.use(express.json());
    // Durable stream reconnects (streamManager option) require the client to
    // read the X-Genkit-Stream-Id response header, so it is CORS-exposed by
    // default.
    app.use(
      cors(
        options.cors ?? {
          allowedHeaders: ['Content-Type', 'Accept', 'X-Genkit-Stream-Id'],
          exposedHeaders: ['X-Genkit-Stream-Id'],
        }
      )
    );
  }

  const handlerOpts = options.streamManager
    ? { streamManager: options.streamManager }
    : undefined;
  const prefix = (options.pathPrefix ?? '/api').replace(/\/+$/, '');
  for (const [name, agent] of Object.entries(agents)) {
    app.post(`${prefix}/${name}`, expressHandler(agent, handlerOpts));
    app.post(
      `${prefix}/${name}/getSnapshot`,
      expressHandler(agent.getSnapshotDataAction)
    );
    app.post(`${prefix}/${name}/abort`, expressHandler(agent.abortAgentAction));
  }

  let server: Server | undefined;
  if (!options.app) {
    const port = options.port ?? Number(process.env.PORT ?? 8080);
    const host = options.host ?? process.env.HOST;
    server = await new Promise<Server>((resolve, reject) => {
      const s = host ? app.listen(port, host) : app.listen(port);
      // Bind failures (EADDRINUSE, bad port) arrive as 'error' events after
      // listen() returns; unhandled, they crash the process with a raw stack.
      s.once('error', reject);
      s.once('listening', () => {
        s.removeListener('error', reject);
        const address = s.address();
        const bound =
          typeof address === 'object' && address !== null
            ? address.port
            : port;
        logger.info(
          `[agent-dirs] serving ${names.length} agent(s) on ` +
            `${host ?? ''}:${bound} - ` +
            names.map((n) => `${prefix}/${n}`).join(', ')
        );
        resolve(s);
      });
    });
  }
  return { app, server, agents: names };
}
