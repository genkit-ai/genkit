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

import {
  UserBuilder,
  agentCardHandler,
  jsonRpcHandler,
} from '@a2a-js/sdk/server/express';
import express from 'express';
import { InMemorySessionStore, genkit, type GenkitBeta } from 'genkit/beta';
import type { ModelAction } from 'genkit/model';

import assert from 'node:assert';
import type * as http from 'node:http';
import type { AddressInfo } from 'node:net';
import { after, before, describe, it } from 'node:test';

import { defineA2aAgent } from '../src/a2a-agent.js';
import { GenkitA2ARequestHandler } from '../src/request-handler.js';

/**
 * Defines a simple model that streams a couple of chunks and echoes the last
 * user message back in its final response.
 */
function defineEchoModel(ai: GenkitBeta): ModelAction {
  return ai.defineModel({ name: 'echoModel' }, async (request, sendChunk) => {
    sendChunk?.({ content: [{ text: 'Hello' }] });
    sendChunk?.({ content: [{ text: ' world' }] });
    const lastUser = [...request.messages]
      .reverse()
      .find((m) => m.role === 'user');
    const echoed = (lastUser?.content ?? []).map((c) => c.text ?? '').join('');
    return {
      message: { role: 'model', content: [{ text: `Echo: ${echoed}` }] },
      finishReason: 'stop',
    };
  });
}

/**
 * Starts an HTTP server exposing the given Genkit agent over A2A. Returns the
 * base URL and a stop() function.
 */
async function startA2aServer(
  ai: GenkitBeta,
  agent: any
): Promise<{ url: string; stop: () => Promise<void> }> {
  const app = express();
  app.use(express.json());

  // The handler card url is filled in after we know the port; use a lazy
  // agentCardProvider so we can bind the real url.
  let baseUrl = '';
  const handler = new GenkitA2ARequestHandler({
    agent,
    // url is set to a placeholder; overwritten by the lazy provider below.
    url: 'http://localhost',
  });

  app.use(
    '/',
    jsonRpcHandler({
      requestHandler: handler,
      userBuilder: UserBuilder.noAuthentication,
    }) as any
  );
  app.use(
    '/.well-known/agent-card.json',
    agentCardHandler({
      agentCardProvider: async () => {
        const card = await handler.getAgentCard();
        return { ...card, url: baseUrl };
      },
    }) as any
  );

  const server: http.Server = await new Promise((resolve) => {
    const s = app.listen(0, () => resolve(s));
  });
  const port = (server.address() as AddressInfo).port;
  baseUrl = `http://localhost:${port}`;

  return {
    url: baseUrl,
    stop: () =>
      new Promise<void>((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve()))
      ),
  };
}

describe('defineA2aAgent (round-trip)', () => {
  let server: { url: string; stop: () => Promise<void> };
  let clientAi: GenkitBeta;

  before(async () => {
    // Server side: a Genkit agent exposed over A2A.
    const serverAi = genkit({});
    defineEchoModel(serverAi);
    const serverAgent = serverAi.defineAgent({
      name: 'echoAgent',
      description: 'Echoes what you say.',
      model: 'echoModel',
      // A store is required for the A2A handler, which resumes a session per
      // A2A contextId (server-managed state).
      store: new InMemorySessionStore(),
    });

    server = await startA2aServer(serverAi, serverAgent);

    // Client side: a separate Genkit instance consuming the remote A2A agent.
    clientAi = genkit({});
  });

  after(async () => {
    await server.stop();
  });

  it('derives name/description from the remote agent card', async () => {
    const remote = await defineA2aAgent(clientAi, { url: server.url });
    assert.strictEqual(remote.__action.name, 'echoAgent');
    assert.strictEqual(remote.__action.description, 'Echoes what you say.');
  });

  it('registers the agent in the registry', async () => {
    await defineA2aAgent(clientAi, {
      url: server.url,
      name: 'registeredAgent',
    });
    const action = await clientAi.registry.lookupAction(
      '/agent/registeredAgent'
    );
    assert.ok(action, 'expected the A2A agent to be registered');
  });

  it('runs a turn and returns the remote response', async () => {
    const remote = await defineA2aAgent(clientAi, {
      url: server.url,
      name: 'turnAgent',
    });
    const chat = remote.chat();
    const res = await chat.send('hi there');
    assert.match(res.text, /Echo: hi there/);
  });

  it('streams chunks from the remote agent', async () => {
    const remote = await defineA2aAgent(clientAi, {
      url: server.url,
      name: 'streamAgent',
    });
    const chat = remote.chat();
    const turn = chat.sendStream('stream please');
    let streamed = '';
    for await (const chunk of turn.stream) {
      streamed += chunk.text;
    }
    const res = await turn.response;
    assert.ok(streamed.length > 0, 'expected streamed text');
    assert.match(res.text, /Echo: stream please/);
  });

  it('keeps conversation continuity across turns in one chat', async () => {
    const remote = await defineA2aAgent(clientAi, {
      url: server.url,
      name: 'multiTurnAgent',
    });
    const chat = remote.chat();
    await chat.send('first');
    const res2 = await chat.send('second');
    // The client-side session retains both turns.
    assert.ok(chat.messages.length >= 4);
    assert.match(res2.text, /Echo: second/);
  });
});
