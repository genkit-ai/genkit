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

import { Message } from '@genkit-ai/ai';
import type { SessionStore } from '@genkit-ai/ai/session';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit, type GenkitBeta } from '../src/beta';
import { TestMemorySessionStore, defineEchoModel } from './helpers';

describe('session', () => {
  let ai: GenkitBeta;

  beforeEach(() => {
    ai = genkit({
      model: 'echoModel',
    });
    defineEchoModel(ai);
  });

  it('maintains history in the session', async () => {
    const session = ai.createSession();
    const chat = session.chat();
    let response = await chat.send('hi');

    assert.strictEqual(response.text, 'Echo: hi; config: {}');

    response = await chat.send('bye');

    assert.strictEqual(
      response.text,
      'Echo: hi,Echo: hi,; config: {},bye; config: {}'
    );
    assert.deepStrictEqual(response.messages, [
      { content: [{ text: 'hi' }], role: 'user' },
      {
        content: [{ text: 'Echo: hi' }, { text: '; config: {}' }],
        role: 'model',
      },
      { content: [{ text: 'bye' }], role: 'user' },
      {
        content: [
          { text: 'Echo: hi,Echo: hi,; config: {},bye' },
          { text: '; config: {}' },
        ],
        role: 'model',
      },
    ]);
  });

  it('sends ready-to-serialize data to the session store', async () => {
    let savedData: any;
    const store: SessionStore = {
      save: async (id, data) => {
        savedData = data;
      },
      get: async (id) => savedData,
    };
    const session = ai.createSession({
      store,
      initialState: { foo: 'bar' },
    });

    session.updateMessages('main', [
      new Message({ role: 'user', content: [{ text: 'hello there' }] }),
    ]);

    assert.deepStrictEqual(savedData, {
      id: session.id,
      state: { foo: 'bar' },
      threads: {
        main: [{ content: [{ text: 'hello there' }], role: 'user' }],
      },
    });
  });

  it('maintains multithreaded history in the session', async () => {
    const store = new TestMemorySessionStore();
    const session = ai.createSession({
      store,
      initialState: {
        name: 'Genkit',
      },
    });

    const mainChat = session.chat();
    let response = await mainChat.send('hi main');
    assert.strictEqual(response.text, 'Echo: hi main; config: {}');

    const lawyerChat = session.chat('lawyerChat', {
      system: 'talk like a lawyer',
    });
    response = await lawyerChat.send('hi lawyerChat');
    assert.strictEqual(
      response.text,
      'Echo: system: talk like a lawyer,hi lawyerChat; config: {}'
    );

    const pirateChat = session.chat('pirateChat', {
      system: 'talk like a pirate',
    });
    response = await pirateChat.send('hi pirateChat');
    assert.strictEqual(
      response.text,
      'Echo: system: talk like a pirate,hi pirateChat; config: {}'
    );

    const gotState = await store.get(session.id);
    delete (gotState as any).id; // ignore
    assert.deepStrictEqual(gotState, {
      state: {
        name: 'Genkit',
      },
      threads: {
        main: [
          { content: [{ text: 'hi main' }], role: 'user' },
          {
            content: [{ text: 'Echo: hi main' }, { text: '; config: {}' }],
            role: 'model',
          },
        ],
        lawyerChat: [
          {
            content: [{ text: 'talk like a lawyer' }],
            role: 'system',
            metadata: { preamble: true },
          },
          { content: [{ text: 'hi lawyerChat' }], role: 'user' },
          {
            content: [
              { text: 'Echo: system: talk like a lawyer,hi lawyerChat' },
              { text: '; config: {}' },
            ],
            role: 'model',
          },
        ],
        pirateChat: [
          {
            content: [{ text: 'talk like a pirate' }],
            role: 'system',
            metadata: { preamble: true },
          },
          { content: [{ text: 'hi pirateChat' }], role: 'user' },
          {
            content: [
              { text: 'Echo: system: talk like a pirate,hi pirateChat' },
              { text: '; config: {}' },
            ],
            role: 'model',
          },
        ],
      },
    });
  });

  it('maintains history in the session with streaming', async () => {
    const session = ai.createSession();
    const chat = session.chat();

    let { response, stream } = chat.sendStream('hi');

    let chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }
    assert.strictEqual((await response).text, 'Echo: hi; config: {}');
    assert.deepStrictEqual(chunks, ['3', '2', '1']);

    ({ response, stream } = chat.sendStream('bye'));

    chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }

    assert.deepStrictEqual(chunks, ['3', '2', '1']);
    assert.strictEqual(
      (await response).text,
      'Echo: hi,Echo: hi,; config: {},bye; config: {}'
    );
    assert.deepStrictEqual((await response).messages, [
      { content: [{ text: 'hi' }], role: 'user' },
      {
        role: 'model',
        content: [{ text: 'Echo: hi' }, { text: '; config: {}' }],
      },
      { content: [{ text: 'bye' }], role: 'user' },
      {
        role: 'model',
        content: [
          { text: 'Echo: hi,Echo: hi,; config: {},bye' },
          { text: '; config: {}' },
        ],
      },
    ]);
  });

  it('stores state and messages in the store', async () => {
    const store = new TestMemorySessionStore();
    const session = ai.createSession({
      store,
      initialState: {
        foo: 'bar',
      },
    });
    const chat = session.chat();

    await chat.send('hi');
    await chat.send('bye');

    const state = await store.get(session.id);
    delete (state as any).id;
    assert.deepStrictEqual(state, {
      state: {
        foo: 'bar',
      },
      threads: {
        main: [
          { content: [{ text: 'hi' }], role: 'user' },
          {
            content: [{ text: 'Echo: hi' }, { text: '; config: {}' }],
            role: 'model',
          },
          { content: [{ text: 'bye' }], role: 'user' },
          {
            content: [
              { text: 'Echo: hi,Echo: hi,; config: {},bye' },
              { text: '; config: {}' },
            ],
            role: 'model',
          },
        ],
      },
    });
  });

  describe('loadChat', () => {
    it('loads session from store', async () => {
      const store = new TestMemorySessionStore();
      // init the store
      const originalSession = ai.createSession({ store });
      const originalMainChat = originalSession.chat({
        config: {
          temperature: 1,
        },
      });
      await originalMainChat.send('hi');
      await originalMainChat.send('bye');

      const sessionId = originalSession.id;

      // load
      const session = await ai.loadSession(sessionId, { store });
      const mainChat = session.chat();
      assert.deepStrictEqual(mainChat.messages, [
        { content: [{ text: 'hi' }], role: 'user' },
        {
          role: 'model',
          content: [
            { text: 'Echo: hi' },
            { text: '; config: {"temperature":1}' },
          ],
        },
        {
          content: [{ text: 'bye' }],
          role: 'user',
        },
        {
          content: [
            { text: 'Echo: hi,Echo: hi,; config: {"temperature":1},bye' },
            { text: '; config: {"temperature":1}' },
          ],
          role: 'model',
        },
      ]);
      const response = await mainChat.send('hi again');
      assert.strictEqual(
        response.text,
        'Echo: hi,Echo: hi,; config: {"temperature":1},bye,Echo: hi,Echo: hi,; config: {"temperature":1},bye,; config: {"temperature":1},hi again; config: {}'
      );
      assert.deepStrictEqual(mainChat.messages, [
        { content: [{ text: 'hi' }], role: 'user' },
        {
          role: 'model',
          content: [
            { text: 'Echo: hi' },
            { text: '; config: {"temperature":1}' },
          ],
        },
        {
          content: [{ text: 'bye' }],
          role: 'user',
        },
        {
          content: [
            { text: 'Echo: hi,Echo: hi,; config: {"temperature":1},bye' },
            { text: '; config: {"temperature":1}' },
          ],
          role: 'model',
        },
        { content: [{ text: 'hi again' }], role: 'user' },
        {
          role: 'model',
          content: [
            {
              text: 'Echo: hi,Echo: hi,; config: {"temperature":1},bye,Echo: hi,Echo: hi,; config: {"temperature":1},bye,; config: {"temperature":1},hi again',
            },
            { text: '; config: {}' },
          ],
        },
      ]);

      const state = await store.get(sessionId);
      assert.deepStrictEqual(state?.threads, {
        main: [
          { content: [{ text: 'hi' }], role: 'user' },
          {
            role: 'model',
            content: [
              { text: 'Echo: hi' },
              { text: '; config: {"temperature":1}' },
            ],
          },
          {
            content: [{ text: 'bye' }],
            role: 'user',
          },
          {
            content: [
              { text: 'Echo: hi,Echo: hi,; config: {"temperature":1},bye' },
              { text: '; config: {"temperature":1}' },
            ],
            role: 'model',
          },
          { content: [{ text: 'hi again' }], role: 'user' },
          {
            role: 'model',
            content: [
              {
                text: 'Echo: hi,Echo: hi,; config: {"temperature":1},bye,Echo: hi,Echo: hi,; config: {"temperature":1},bye,; config: {"temperature":1},hi again',
              },
              { text: '; config: {}' },
            ],
          },
        ],
      });
    });
  });

  it('can start chat from a prompt', async () => {
    const agent = ai.definePrompt({
      name: 'agent',
      config: { temperature: 1 },
      description: 'Agent description',
      system: 'hello from template',
    });

    const session = ai.createSession();
    const chat = session.chat(agent);
    const response = await chat.send('hi');
    assert.deepStrictEqual(response.messages, [
      {
        role: 'system',
        content: [{ text: 'hello from template' }],
        metadata: { preamble: true },
      },
      {
        content: [{ text: 'hi' }],
        role: 'user',
      },
      {
        content: [
          { text: 'Echo: system: hello from template,hi' },
          { text: '; config: {"temperature":1}' },
        ],
        role: 'model',
      },
    ]);
  });

  it('can start chat from a prompt with input', async () => {
    const agent = ai.definePrompt({
      name: 'agent',
      config: { temperature: 1 },
      description: 'Agent description',
      system: 'hello {{ name }} from template',
    });

    const session = ai.createSession();
    const chat = session.chat(agent, {
      input: {
        name: 'Genkit',
      },
    });
    const response = await chat.send('hi');
    assert.deepStrictEqual(response.messages, [
      {
        role: 'system',
        content: [{ text: 'hello Genkit from template' }],
        metadata: { preamble: true },
      },
      {
        content: [{ text: 'hi' }],
        role: 'user',
      },
      {
        content: [
          { text: 'Echo: system: hello Genkit from template,hi' },
          { text: '; config: {"temperature":1}' },
        ],
        role: 'model',
      },
    ]);
  });

  it('can start chat thread from a prompt with input', async () => {
    const agent = ai.definePrompt({
      name: 'agent',
      config: { temperature: 1 },
      description: 'Agent description',
      system: 'hello {{ name }} from template',
    });
    const store = new TestMemorySessionStore();
    const session = ai.createSession({ store });
    const chat = session.chat('mythread', agent, {
      input: {
        name: 'Genkit',
      },
    });

    await chat.send('hi');

    const gotState = await store.get(session.id);
    delete (gotState as any).id; // ignore
    assert.deepStrictEqual(gotState?.threads, {
      mythread: [
        {
          role: 'system',
          content: [{ text: 'hello Genkit from template' }],
          metadata: { preamble: true },
        },
        {
          content: [{ text: 'hi' }],
          role: 'user',
        },
        {
          content: [
            { text: 'Echo: system: hello Genkit from template,hi' },
            { text: '; config: {"temperature":1}' },
          ],
          role: 'model',
        },
      ],
    });
  });

  it('can read current session state from a prompt', async () => {
    const agent = ai.definePrompt({
      name: 'agent',
      config: { temperature: 1 },
      description: 'Agent description',
      system: 'foo={{@state.foo}}',
    });

    const session = ai.createSession({
      initialState: {
        foo: 'bar',
      },
    });
    const chat = session.chat(agent);
    const response = await chat.send('hi');
    assert.deepStrictEqual(response.messages, [
      {
        role: 'system',
        content: [{ text: 'foo=bar' }],
        metadata: { preamble: true },
      },
      {
        content: [{ text: 'hi' }],
        role: 'user',
      },
      {
        content: [
          { text: 'Echo: system: foo=bar,hi' },
          { text: '; config: {"temperature":1}' },
        ],
        role: 'model',
      },
    ]);
  });

  it('can run arbitrary code within the session context', async () => {
    const testFlow = ai.defineFlow('text', () => ai.currentSession().state);

    const sess = ai.createSession({
      initialState: {
        foo: 'bar',
      },
    });

    // running the flow directly throws because it's trying to access currentSession
    assert.rejects(() => testFlow(), {
      message: 'not running within a session',
    });

    const response = await sess.run(testFlow);
    assert.deepStrictEqual(response, {
      foo: 'bar',
    });
  });
});
