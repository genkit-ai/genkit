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

import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  A2A_PROTOCOL_VERSION,
  deriveAgentCard,
  type AgentLike,
} from '../src/agent-card.js';

const fakeAgent: AgentLike = {
  __action: { name: 'weatherAgent', description: 'Tells the weather.' },
};

describe('deriveAgentCard', () => {
  it('derives a complete card from an agent and url', () => {
    const card = deriveAgentCard(fakeAgent, { url: 'http://localhost:3000' });
    assert.strictEqual(card.name, 'weatherAgent');
    assert.strictEqual(card.description, 'Tells the weather.');
    assert.strictEqual(card.url, 'http://localhost:3000');
    assert.strictEqual(card.protocolVersion, A2A_PROTOCOL_VERSION);
    assert.strictEqual(card.preferredTransport, 'JSONRPC');
    assert.strictEqual(card.capabilities.streaming, true);
    assert.deepStrictEqual(card.defaultInputModes, ['text/plain']);
    assert.deepStrictEqual(card.defaultOutputModes, ['text/plain']);
    assert.strictEqual(card.skills.length, 1);
    assert.strictEqual(card.skills[0].name, 'weatherAgent');
  });

  it('falls back to a generic description when the agent has none', () => {
    const card = deriveAgentCard(
      { __action: { name: 'bareAgent' } },
      { url: 'http://localhost:3000' }
    );
    assert.strictEqual(card.description, 'The bareAgent agent.');
  });

  it('lets an explicit card override derived values', () => {
    const card = deriveAgentCard(fakeAgent, {
      url: 'http://localhost:3000',
      card: {
        description: 'Custom description',
        version: '1.2.3',
        skills: [{ id: 's1', name: 'skill', description: 'd', tags: ['x'] }],
      },
    });
    assert.strictEqual(card.description, 'Custom description');
    assert.strictEqual(card.version, '1.2.3');
    assert.strictEqual(card.skills[0].id, 's1');
  });

  it('throws when no url is available', () => {
    assert.throws(() => deriveAgentCard(fakeAgent), /no `url` provided/);
  });

  it('throws when the agent has no name and no card name', () => {
    assert.throws(
      () => deriveAgentCard({}, { url: 'http://localhost:3000' }),
      /has no name/
    );
  });
});
