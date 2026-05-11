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

import type { MessageData } from '@genkit-ai/ai';
import { z } from '@genkit-ai/core';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit, type GenkitBeta } from '../src/beta';
import {
  defineEchoModel,
  defineProgrammableModel,
  type ProgrammableModel,
} from './helpers';

describe('chat', () => {
  let ai: GenkitBeta;

  beforeEach(() => {
    ai = genkit({
      model: 'echoModel',
      promptDir: './tests/prompts',
    });
    defineEchoModel(ai);
  });

  it('maintains history in the session', async () => {
    const session = ai.chat();
    let response = await session.send('hi');

    assert.strictEqual(response.text, 'Echo: hi; config: {}');

    response = await session.send('bye');

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

  it('maintains history in the session with streaming', async () => {
    const chat = ai.chat();
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

  it('can init a session with a prompt', async () => {
    const prompt = ai.definePrompt({ name: 'hi', prompt: 'hi {{ name }}' });

    const session = await ai.chat(
      await prompt.render(
        { name: 'Genkit' },
        {
          config: { temperature: 11 },
        }
      )
    );
    const response = await session.send('hi');

    assert.strictEqual(
      response.text,
      'Echo: hi Genkit,hi; config: {"temperature":11}'
    );
  });

  it('can start chat from a prompt', async () => {
    const preamble = ai.definePrompt({
      name: 'hi',
      config: { version: 'abc' },
      messages: 'hi from template',
    });
    const session = await ai.chat(preamble);
    const response = await session.send('send it');

    assert.strictEqual(
      response.text,
      'Echo: hi from template,send it; config: {"version":"abc"}'
    );
  });

  it('can start chat from a prompt with input', async () => {
    const preamble = ai.definePrompt({
      name: 'hi',
      config: { version: 'abc' },
      messages: 'hi {{ name }} from template',
    });
    const session = await ai.chat(preamble, {
      input: { name: 'Genkit' },
    });
    const response = await session.send('send it');

    assert.strictEqual(
      response.text,
      'Echo: hi Genkit from template,send it; config: {"version":"abc"}'
    );
  });

  it('can start chat from a prompt file with input', async () => {
    const preamble = ai.prompt('chat_preamble');
    const session = await ai.chat(preamble, {
      input: { name: 'Genkit' },
    });
    const response = await session.send('send it');

    assert.strictEqual(
      response.text,
      'Echo: hi Genkit from template,send it; config: {"version":"abc"}'
    );
  });

  it('can send a rendered prompt to chat', async () => {
    const prompt = ai.definePrompt({
      name: 'hi',
      config: { version: 'abc' },
      prompt: 'hi {{ name }}',
    });
    const session = ai.chat();
    const response = await session.send(
      await prompt.render(
        { name: 'Genkit' },
        {
          config: { temperature: 11 },
        }
      )
    );

    assert.strictEqual(
      response.text,
      'Echo: hi Genkit; config: {"version":"abc","temperature":11}'
    );
  });
});

describe('preamble', () => {
  let ai: GenkitBeta;
  let pm: ProgrammableModel;

  beforeEach(() => {
    ai = genkit({
      model: 'programmableModel',
    });
    pm = defineProgrammableModel(ai);
    defineEchoModel(ai);
  });

  it('swaps out preamble on prompt tool invocation', async () => {
    const agentB = ai.definePrompt({
      name: 'agentB',
      config: { temperature: 1 },
      description: 'Agent B description',
      tools: ['agentA'],
      toolChoice: 'required',
      system: 'agent b',
    });

    const agentA = ai.definePrompt({
      name: 'agentA',
      config: { temperature: 2 },
      description: 'Agent A description',
      tools: [agentB],
      toolChoice: 'required',
      messages: async () => {
        return [
          {
            role: 'system',
            content: [{ text: ' agent a' }],
          },
        ];
      },
    });

    // simple hi, nothing interesting...
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [
            { text: `hi from agent a (toolChoice: ${req.toolChoice})` },
          ],
        },
      };
    };

    const session = ai.chat(agentA);
    let { text } = await session.send('hi');
    assert.strictEqual(text, 'hi from agent a (toolChoice: required)');
    assert.deepStrictEqual(pm.lastRequest, {
      config: {
        temperature: 2,
      },
      messages: [
        {
          content: [{ text: ' agent a' }],
          metadata: { preamble: true },
          role: 'system',
        },
        {
          content: [{ text: 'hi' }],
          role: 'user',
        },
      ],
      output: {},
      tools: [
        {
          name: 'agentB',
          description: 'Agent B description',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
          outputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
        },
      ],
      toolChoice: 'required',
    });

    // transfer to agent B...

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
                    name: 'agentB',
                    input: {},
                    ref: 'ref123',
                  },
                }
              : { text: `hi from agent b (toolChoice: ${req.toolChoice})` },
          ],
        },
      };
    };

    ({ text } = await session.send('pls transfer to b'));

    assert.deepStrictEqual(text, 'hi from agent b (toolChoice: required)');
    assert.deepStrictEqual(pm.lastRequest, {
      config: {
        temperature: 1,
      },
      messages: [
        {
          role: 'system',
          content: [{ text: 'agent b' }], // <--- NOTE: swapped out the preamble
          metadata: { preamble: true },
        },
        {
          role: 'user',
          content: [{ text: 'hi' }],
        },
        {
          role: 'model',
          content: [{ text: 'hi from agent a (toolChoice: required)' }],
        },
        {
          role: 'user',
          content: [{ text: 'pls transfer to b' }],
        },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                input: {},
                name: 'agentB',
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
                name: 'agentB',
                output: 'transferred to agentB',
                ref: 'ref123',
              },
            },
          ],
        },
      ],
      output: {},
      tools: [
        {
          description: 'Agent A description',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
          name: 'agentA',
          outputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
        },
      ],
      toolChoice: 'required',
    });

    // transfer back to to agent A...

    // first response is a tool call, the subsequent responses are just text response from agent a.
    reqCounter = 0;
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [
            reqCounter++ === 0
              ? {
                  toolRequest: {
                    name: 'agentA',
                    input: {},
                    ref: 'ref123',
                  },
                }
              : { text: 'hi from agent a' },
          ],
        },
      };
    };

    ({ text } = await session.send('pls transfer to a'));

    assert.deepStrictEqual(text, 'hi from agent a');
    assert.deepStrictEqual(pm.lastRequest, {
      config: {
        temperature: 2,
      },
      messages: [
        {
          role: 'system',
          content: [{ text: ' agent a' }], // <--- NOTE: swapped out the preamble
          metadata: { preamble: true },
        },
        {
          role: 'user',
          content: [{ text: 'hi' }],
        },
        {
          role: 'model',
          content: [{ text: 'hi from agent a (toolChoice: required)' }],
        },
        {
          role: 'user',
          content: [{ text: 'pls transfer to b' }],
        },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                input: {},
                name: 'agentB',
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
                name: 'agentB',
                output: 'transferred to agentB',
                ref: 'ref123',
              },
            },
          ],
        },
        {
          role: 'model',
          content: [{ text: 'hi from agent b (toolChoice: required)' }],
        },
        {
          role: 'user',
          content: [{ text: 'pls transfer to a' }],
        },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                input: {},
                name: 'agentA',
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
                name: 'agentA',
                output: 'transferred to agentA',
                ref: 'ref123',
              },
            },
          ],
        },
      ],
      output: {},
      tools: [
        {
          description: 'Agent B description',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
          name: 'agentB',
          outputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
        },
      ],
      toolChoice: 'required',
    });
  });

  it('updates the preamble on fresh chat instance', async () => {
    const agent = ai.definePrompt({
      name: 'agent',
      config: { temperature: 2 },
      description: 'Agent A description',
      messages: '{{ role "system"}} greet {{ @state.name }}',
    });

    const session = ai.createSession({ initialState: { name: 'Pavel' } });

    const chat = session.chat(agent, { model: 'echoModel' });
    let response = await chat.send('hi');

    assert.deepStrictEqual(response.messages, [
      {
        role: 'system',
        content: [{ text: ' greet Pavel' }],
        metadata: { preamble: true },
      },
      {
        role: 'user',
        content: [{ text: 'hi' }],
      },
      {
        role: 'model',
        content: [
          { text: 'Echo: system:  greet Pavel,hi' },
          { text: '; config: {"temperature":2}' },
        ],
      },
    ]);

    await session.updateState({ name: 'Michael' });

    const freshChat = session.chat(agent, { model: 'echoModel' });
    response = await freshChat.send('hi');

    assert.deepStrictEqual(response.messages, [
      {
        role: 'system',
        content: [{ text: ' greet Michael' }],
        metadata: { preamble: true },
      },
      {
        role: 'user',
        content: [{ text: 'hi' }],
      },
      {
        role: 'model',
        content: [
          { text: 'Echo: system:  greet Pavel,hi' },
          { text: '; config: {"temperature":2}' },
        ],
      },
      {
        role: 'user',
        content: [{ text: 'hi' }],
      },
      {
        role: 'model',
        content: [
          {
            text: 'Echo: system:  greet Michael,hi,Echo: system:  greet Pavel,hi,; config: {"temperature":2},hi',
          },
          { text: '; config: {"temperature":2}' },
        ],
      },
    ]);
  });

  it('initializes chat with history', async () => {
    const history: MessageData[] = [
      {
        role: 'user',
        content: [{ text: 'hi' }],
      },
      {
        role: 'model',
        content: [{ text: 'bye' }],
      },
    ];

    const chat = ai.chat({
      model: 'echoModel',
      system: 'system instructions',
      messages: history,
    });

    const response = await chat.send('hi again');
    assert.deepStrictEqual(response.messages, [
      {
        role: 'system',
        content: [{ text: 'system instructions' }],
        metadata: { preamble: true },
      },
      {
        role: 'user',
        content: [{ text: 'hi' }],
        metadata: { preamble: true },
      },
      {
        role: 'model',
        content: [{ text: 'bye' }],
        metadata: { preamble: true },
      },
      {
        role: 'user',
        content: [{ text: 'hi again' }],
      },
      {
        role: 'model',
        content: [
          { text: 'Echo: system: system instructions,hi,bye,hi again' },
          { text: '; config: {}' },
        ],
      },
    ]);
  });

  it('initializes chat with history in preamble', async () => {
    const hi = ai.definePrompt({
      name: 'hi',
      model: 'echoModel',
      input: {
        schema: z.object({
          name: z.string(),
        }),
      },
      system: 'system instructions',
      prompt: 'hi {{ name }}',
    });

    const history: MessageData[] = [
      {
        role: 'user',
        content: [{ text: 'hi' }],
      },
      {
        role: 'model',
        content: [{ text: 'bye' }],
      },
    ];

    const chat = ai.chat(hi, { input: { name: 'Genkit' }, messages: history });

    const response = await chat.send('hi again');
    assert.deepStrictEqual(response.messages, [
      {
        role: 'system',
        content: [{ text: 'system instructions' }],
        metadata: { preamble: true },
      },
      {
        role: 'user',
        content: [{ text: 'hi' }],
        metadata: {
          preamble: true,
        },
      },
      {
        role: 'model',
        content: [{ text: 'bye' }],
        metadata: {
          preamble: true,
        },
      },
      {
        role: 'user',
        content: [{ text: 'hi Genkit' }],
        metadata: {
          preamble: true,
        },
      },
      {
        role: 'user',
        content: [{ text: 'hi again' }],
      },
      {
        role: 'model',
        content: [
          {
            text: 'Echo: system: system instructions,hi,bye,hi Genkit,hi again',
          },
          { text: '; config: {}' },
        ],
      },
    ]);
  });
});

describe('session state', () => {
  let ai: Genkit;

  beforeEach(() => {
    ai = genkit({
      model: 'echoModel',
    });
    defineEchoModel(ai);
  });

  it('properly patches partial state updates', async () => {
    const session = ai.createSession({
      initialState: { userName: 'John', color: 'Blue', age: 25 },
    });

    // Update single property
    await session.updateState({ color: 'Green' });
    assert.deepStrictEqual(
      session.state,
      { userName: 'John', color: 'Green', age: 25 },
      'should preserve existing properties when updating a single field'
    );

    // Update multiple properties
    await session.updateState({ userName: 'Jane', color: 'red', age: 26 });
    assert.deepStrictEqual(
      session.state,
      { userName: 'Jane', color: 'red', age: 26 },
      'should preserve non-updated properties when updating multiple fields'
    );

    // Update with undefined state
    const emptySession = ai.createSession();
    await emptySession.updateState({ newProp: 'value' });
    assert.deepStrictEqual(
      emptySession.state,
      { newProp: 'value' },
      'should handle updates when initial state is undefined'
    );
  });

  it('maintains state across chat messages', async () => {
    const session = ai.createSession({
      initialState: { userName: 'Pavel' },
    });

    const chat = session.chat();
    await chat.send('hi');

    assert.deepStrictEqual(
      session.state,
      { userName: 'Pavel' },
      'should preserve state between chat messages'
    );
    await session.updateState({ userName: 'Jacob' });

    await chat.send('bye');
    assert.deepStrictEqual(
      session.state,
      { userName: 'Jacob' },
      'state should remain stable after chat messages'
    );
  });
});
