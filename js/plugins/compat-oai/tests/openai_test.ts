/**
 * Copyright 2024 The Fire Company
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

import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { genkit, modelActionMetadata, type ActionMetadata } from 'genkit';
import { modelRef, type GenerateRequest } from 'genkit/model';
import type OpenAI from 'openai';
import type { Fetch } from 'openai/core';
import 'openai/shims/web';
import {
  ChatCompletionCommonConfigSchema,
  defineCompatOpenAIModel,
  toOpenAIRequestBody,
} from '../src/model';
import { openAI } from '../src/openai';
import { SUPPORTED_GPT_MODELS, gptModelRef } from '../src/openai/gpt';
import * as LISTING_FIXTURE from './openai_listing_fixture.json';

describe('gptModel', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should correctly define supported GPT models', () => {
    const model = defineCompatOpenAIModel({
      name: 'openai/gpt-4o',
      client: {} as OpenAI,
      modelRef: testModelRef('openai/gpt-4o'),
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-4o',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    });
  });

  it('should correctly define gpt-4.1, gpt-4.1-mini, and gpt-4.1-nano', () => {
    const gpt41 = defineCompatOpenAIModel({
      name: 'openai/gpt-4.1',
      client: {} as OpenAI,
      modelRef: testModelRef('openai/gpt-4.1'),
    });
    expect({
      name: gpt41.__action.name,
      supports: gpt41.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-4.1',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    });

    const gpt41mini = defineCompatOpenAIModel({
      name: 'openai/gpt-4.1-mini',
      client: {} as OpenAI,
      modelRef: testModelRef('openai/gpt-4.1-mini'),
    });
    expect({
      name: gpt41mini.__action.name,
      supports: gpt41mini.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-4.1-mini',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    });

    const gpt41nano = defineCompatOpenAIModel({
      name: 'openai/gpt-4.1-nano',
      client: {} as OpenAI,
      modelRef: testModelRef('openai/gpt-4.1-nano'),
    });
    expect({
      name: gpt41nano.__action.name,
      supports: gpt41nano.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-4.1-nano',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    });
  });

  it('should correctly define gpt-5', () => {
    const gpt5 = defineCompatOpenAIModel({
      name: 'openai/gpt-5',
      client: {} as OpenAI,
      modelRef: SUPPORTED_GPT_MODELS['gpt-5'],
    });
    expect({
      name: gpt5.__action.name,
      supports: gpt5.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-5',
      supports: {
        multiturn: true,
        tools: true,
        toolChoice: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
        constrained: 'all',
      },
    });
  });
});

describe('gpt-6-astra', () => {
  const expectedSupports = {
    multiturn: true,
    tools: false,
    toolChoice: false,
    media: true,
    systemRole: true,
    output: ['text', 'json'],
    constrained: 'all',
  };

  it('is in the supported GPT model catalog without tool support', () => {
    const modelRef = SUPPORTED_GPT_MODELS['gpt-6-astra'];
    expect(modelRef.name).toBe('openai/gpt-6-astra');
    expect(modelRef.info?.supports).toStrictEqual(expectedSupports);
  });

  it('defines a model action with the catalog supports', () => {
    const model = defineCompatOpenAIModel({
      name: 'openai/gpt-6-astra',
      client: {} as OpenAI,
      modelRef: SUPPORTED_GPT_MODELS['gpt-6-astra'],
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-6-astra',
      supports: expectedSupports,
    });
  });
});

describe('GPT model catalog', () => {
  // Chat ids OpenAI shut down on or before 2026-11-12.
  const SHUT_DOWN_IDS = [
    'o1',
    'o3-mini',
    'o4-mini',
    'gpt-4o-2024-05-13',
    'gpt-4.1-nano',
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
    'gpt-4-turbo-preview',
    'gpt-4-0125-preview',
    'gpt-4-1106-preview',
    'gpt-4-vision',
    'gpt-4-vision-preview',
    'gpt-4-1106-vision-preview',
    'gpt-4',
    'gpt-4-0613',
    'gpt-4-32k',
    'gpt-4-32k-0613',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-1106',
    'gpt-4.5',
    'gpt-4.5-preview',
    'gpt-5-chat-latest',
  ];

  it('registers no shut-down model', () => {
    const registered = Object.keys(SUPPORTED_GPT_MODELS);
    for (const id of SHUT_DOWN_IDS) {
      expect(registered).not.toContain(id);
    }
  });

  it('marks the 2026-12-11 shutdown batch as deprecated', () => {
    for (const id of ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'o3'] as const) {
      expect(SUPPORTED_GPT_MODELS[id].info?.stage).toBe('deprecated');
    }
  });

  it('marks the 2027-01-14 shutdown batch as deprecated', () => {
    for (const id of [
      'gpt-4o',
      'gpt-4o-mini',
      'gpt-4o-mini-2024-07-18',
      'gpt-4.1',
      'gpt-4.1-mini',
    ] as const) {
      expect(SUPPORTED_GPT_MODELS[id].info?.stage).toBe('deprecated');
      expect(SUPPORTED_GPT_MODELS[id].info?.supports?.tools).toBe(true);
    }
  });

  it('gpt-5-search-api advertises no tools', () => {
    expect(SUPPORTED_GPT_MODELS['gpt-5-search-api'].info?.supports?.tools).toBe(
      false
    );
  });

  it('gives o3 constrained output', () => {
    expect(SUPPORTED_GPT_MODELS.o3.info?.supports?.constrained).toBe('all');
  });

  it('leaves current models without a stage', () => {
    for (const id of ['gpt-5.5', 'gpt-5.6-sol', 'gpt-6-sol'] as const) {
      expect(SUPPORTED_GPT_MODELS[id].info?.stage).toBeUndefined();
    }
  });

  describe('GPT-6 models', () => {
    it('registers gpt-6-sol, gpt-6-luna, gpt-6.1-sol and gpt-6.1-terra', () => {
      for (const id of [
        'gpt-6-sol',
        'gpt-6-luna',
        'gpt-6.1-sol',
        'gpt-6.1-terra',
      ] as const) {
        expect(SUPPORTED_GPT_MODELS[id].name).toBe(`openai/${id}`);
      }
    });

    it('gpt-6-astra, gpt-6.1-sol and gpt-6.1-terra advertise no tools', () => {
      for (const id of [
        'gpt-6-astra',
        'gpt-6.1-sol',
        'gpt-6.1-terra',
      ] as const) {
        expect(SUPPORTED_GPT_MODELS[id].info?.supports?.tools).toBe(false);
      }
    });
  });

  it('gpt-5.4 and later effort-none tool models advertise tools', () => {
    expect(LISTING_FIXTURE.tools.only_at_reasoning_effort_none).toEqual([
      'gpt-5.4',
      'gpt-5.4-mini',
      'gpt-5.4-nano',
      'gpt-5.5',
      'gpt-5.6-sol',
      'gpt-5.6-terra',
      'gpt-5.6-luna',
      'gpt-6-sol',
      'gpt-6-luna',
    ]);
    for (const id of LISTING_FIXTURE.tools.only_at_reasoning_effort_none) {
      expect(gptModelRef(id).info?.supports?.tools).toBe(true);
    }
  });

  it('advertises toolChoice exactly when it advertises tools', () => {
    for (const [id, ref] of Object.entries(SUPPORTED_GPT_MODELS)) {
      const supports = ref.info?.supports;
      expect({ id, toolChoice: supports?.toolChoice }).toEqual({
        id,
        toolChoice: supports?.tools,
      });
    }
    expect(SUPPORTED_GPT_MODELS.o3.info?.supports?.toolChoice).toBe(true);
    expect(gptModelRef('gpt-4o-2024-08-06').info?.supports?.toolChoice).toBe(
      true
    );
  });

  describe('gptModelRef', () => {
    it('looks up ids with or without the openai/ prefix', () => {
      for (const name of ['gpt-6-sol', 'openai/gpt-6-sol']) {
        const ref = gptModelRef(name);
        expect({ name: ref.name, info: ref.info }).toEqual({
          name: 'openai/gpt-6-sol',
          info: SUPPORTED_GPT_MODELS['gpt-6-sol'].info,
        });
      }
    });

    it('gives dated snapshots the info of their alias', () => {
      for (const [snapshot, alias] of [
        ['gpt-4o-2024-08-06', 'gpt-4o'],
        ['gpt-5-2025-08-07', 'gpt-5'],
        ['gpt-5.5-2026-04-23', 'gpt-5.5'],
        ['gpt-4.1-2025-04-14', 'gpt-4.1'],
      ] as const) {
        const ref = gptModelRef(snapshot);
        expect({ name: ref.name, info: ref.info }).toEqual({
          name: `openai/${snapshot}`,
          info: SUPPORTED_GPT_MODELS[alias].info,
        });
      }
      expect(gptModelRef('gpt-5-2025-08-07').info?.stage).toBe('deprecated');
      expect(gptModelRef('gpt-5.5-2026-04-23').info?.stage).toBeUndefined();
      expect(gptModelRef('gpt-4.1-2025-04-14').info?.stage).toBe('deprecated');
    });

    it('gives shut-down snapshots generic info, not their alias info', () => {
      const ref = gptModelRef('gpt-4o-2024-05-13');
      expect(ref.name).toBe('openai/gpt-4o-2024-05-13');
      expect(ref.info).not.toEqual(SUPPORTED_GPT_MODELS['gpt-4o'].info);
      expect(ref.info?.stage).toBeUndefined();
    });

    it('gives uncatalogued ids no stage', () => {
      expect(
        gptModelRef('gpt-4.1-nano-2025-04-14').info?.stage
      ).toBeUndefined();
      expect(
        gptModelRef('ft:gpt-4o-mini:acme::abc123').info?.stage
      ).toBeUndefined();
    });
  });
});

describe('openAI listActions', () => {
  // Shut-down families the filter still guards against.
  const RETIRED = ['computer-use-preview', 'o3-deep-research', 'sora-2'];
  const ALL_IDS = [
    ...LISTING_FIXTURE.chat,
    ...LISTING_FIXTURE.hidden,
    ...LISTING_FIXTURE.other,
    ...RETIRED,
  ];

  let requestedUrls: string[] = [];
  const fakeFetch: Fetch = async (url) => {
    requestedUrls.push(String(url));
    return new Response(
      JSON.stringify({
        object: 'list',
        data: ALL_IDS.map((id) => ({
          id,
          object: 'model',
          created: 0,
          owned_by: 'openai',
        })),
      }),
      { status: 200, headers: { 'content-type': 'application/json' } }
    );
  };

  async function listActions() {
    requestedUrls = [];
    const plugin = openAI({ apiKey: 'test-key', fetch: fakeFetch });
    const actions = (await plugin.list!()) ?? [];
    expect(requestedUrls).toEqual([expect.stringMatching(/\/models$/)]);
    return actions;
  }

  function findAction(actions: ActionMetadata[], id: string) {
    return actions.find((a) => a.name === id || a.name === `openai/${id}`);
  }

  function chatMetadata(id: string) {
    const ref = gptModelRef(id);
    return modelActionMetadata({
      name: ref.name,
      info: ref.info,
      configSchema: ref.configSchema,
    });
  }

  it.each(LISTING_FIXTURE.chat)('lists %s as a chat model', async (id) => {
    const action = findAction(await listActions(), id);
    expect(action).toEqual(chatMetadata(id));
  });

  it.each([...LISTING_FIXTURE.hidden, ...RETIRED])(
    'does not list %s',
    async (id) => {
      expect(findAction(await listActions(), id)).toBeUndefined();
    }
  );

  it.each(LISTING_FIXTURE.other)(
    'does not list %s as a chat model',
    async (id) => {
      const action = findAction(await listActions(), id);
      if (action) {
        expect(action.metadata).not.toEqual(chatMetadata(id).metadata);
      }
    }
  );

  it('advertises tools for every chat id except the never-tools models', async () => {
    const actions = await listActions();
    const noTools: string[] = LISTING_FIXTURE.tools.never_on_chat_completions;
    for (const id of LISTING_FIXTURE.chat) {
      const supports = findAction(actions, id)?.metadata?.model?.supports;
      expect({ id, tools: supports?.tools }).toEqual({
        id,
        tools: !noTools.includes(id),
      });
    }
  });

  it('uses the alias info for dated snapshots', async () => {
    const actions = await listActions();
    for (const [snapshot, alias] of [
      ['gpt-5-2025-08-07', 'gpt-5'],
      ['gpt-4o-2024-08-06', 'gpt-4o'],
      ['gpt-5.5-2026-04-23', 'gpt-5.5'],
    ] as const) {
      const model = findAction(actions, snapshot)?.metadata?.model;
      expect({
        snapshot,
        stage: model?.stage,
        supports: model?.supports,
      }).toEqual({
        snapshot,
        stage: SUPPORTED_GPT_MODELS[alias].info?.stage,
        supports: SUPPORTED_GPT_MODELS[alias].info?.supports,
      });
    }
    expect(
      findAction(actions, 'gpt-5-2025-08-07')?.metadata?.model?.stage
    ).toBe('deprecated');
  });
});

describe('openAI resolve', () => {
  it('resolves a dated snapshot with the supports of its alias', async () => {
    const ai = genkit({ plugins: [openAI({ apiKey: 'test-key' })] });
    const action = await ai.registry.lookupAction(
      '/model/openai/gpt-4o-2024-08-06'
    );
    expect(action?.__action.metadata?.model?.supports).toEqual(
      SUPPORTED_GPT_MODELS['gpt-4o'].info?.supports
    );
  });
});

// Additional test to ensure toOpenAiRequestBody works for new models

describe('toOpenAiRequestBody for new GPT-4.1 variants', () => {
  const baseRequest = { messages: [] } as GenerateRequest;
  it('should not throw for gpt-4.1', () => {
    expect(() => toOpenAIRequestBody('gpt-4.1', baseRequest)).not.toThrow();
  });
  it('should not throw for gpt-4.1-mini', () => {
    expect(() =>
      toOpenAIRequestBody('gpt-4.1-mini', baseRequest)
    ).not.toThrow();
  });
  it('should not throw for gpt-4.1-nano', () => {
    expect(() =>
      toOpenAIRequestBody('gpt-4.1-nano', baseRequest)
    ).not.toThrow();
  });
});

function testModelRef(name: string) {
  return modelRef({
    name,
    info: {
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    },
    configSchema: ChatCompletionCommonConfigSchema,
  });
}
