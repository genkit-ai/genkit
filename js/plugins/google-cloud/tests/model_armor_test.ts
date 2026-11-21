/**
 * Copyright 2025 Google LLC
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

import { beforeEach, describe, expect, it } from '@jest/globals';
import { genkit, type Genkit } from 'genkit';
import {
  modelArmor,
  type ApplyDeidentificationResultsFn,
} from '../src/model-armor.js';

// Mock ModelArmorClient
class MockModelArmorClient {
  sanitizeUserPrompt = async () => [{}];
  sanitizeModelResponse = async () => [{}];
}

function createEmptyResult() {
  return { sanitizationResult: {} };
}

function createSdpResult(replacementText: string) {
  return {
    sanitizationResult: {
      filterResults: {
        sdp: {
          sdpFilterResult: {
            deidentifyResult: {
              data: { text: replacementText },
            },
          },
        },
      },
    },
  };
}

function createRaiBlockResult() {
  return {
    sanitizationResult: {
      filterMatchState: 'MATCH_FOUND',
      filterResults: {
        rai: { raiFilterResult: { matchState: 'MATCH_FOUND' } },
      },
    },
  };
}

function createSdpBlockResult(replacementText: string) {
  const res = createSdpResult(replacementText);
  (res.sanitizationResult as any).filterMatchState = 'MATCH_FOUND';
  return res;
}

describe('modelArmor', () => {
  let ai: Genkit;
  let mockClient: any;

  beforeEach(() => {
    mockClient = new MockModelArmorClient();
    // The client is instance-level, so it is set via the plugin.
    ai = genkit({ plugins: [modelArmor.plugin({ client: mockClient })] });
    ai.defineModel({ name: 'echoModel' }, async (req: any) => {
      return {
        message: {
          role: 'model',
          content: [{ text: `Echo: ${req.messages[0].content[0].text}` }],
        },
      };
    });
  });

  it('passes through when no sanitization triggers', async () => {
    mockClient.sanitizeUserPrompt = async () => [createEmptyResult()];
    mockClient.sanitizeModelResponse = async () => [createEmptyResult()];

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'hello',
      use: [modelArmor({ templateName: 'test' })],
    });

    expect(response.text).toMatch(/Echo: hello/);
  });

  it('replaces user prompt on SDP match', async () => {
    mockClient.sanitizeUserPrompt = async () => [
      createSdpResult('sanitized_hello'),
    ];
    mockClient.sanitizeModelResponse = async () => [createEmptyResult()];

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'hello',
      use: [
        modelArmor({
          templateName: 'test',
          applyDeidentificationResults: true,
        }),
      ],
    });

    // The echo model should receive the SANITIZED prompt
    expect(response.text).toMatch(/Echo: sanitized_hello/);
  });

  it('blocks user prompt on RAI match', async () => {
    mockClient.sanitizeUserPrompt = async () => [createRaiBlockResult()];

    await expect(
      ai.generate({
        model: 'echoModel',
        prompt: 'bad stuff',
        use: [modelArmor({ templateName: 'test' })],
      })
    ).rejects.toThrow(
      expect.objectContaining({
        status: 'PERMISSION_DENIED',
        message: expect.stringContaining('Model Armor blocked user prompt.'),
      })
    );
  });

  it('replaces model response on SDP match', async () => {
    mockClient.sanitizeUserPrompt = async () => [createEmptyResult()];
    mockClient.sanitizeModelResponse = async () => [
      createSdpResult('sanitized_response'),
    ];

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'hello',
      use: [
        modelArmor({
          templateName: 'test',
          applyDeidentificationResults: true,
        }),
      ],
    });

    expect(response.text).toBe('sanitized_response');
  });

  it('blocks model response on RAI match', async () => {
    mockClient.sanitizeUserPrompt = async () => [createEmptyResult()];
    mockClient.sanitizeModelResponse = async () => [createRaiBlockResult()];

    await expect(
      ai.generate({
        model: 'echoModel',
        prompt: 'hello',
        use: [modelArmor({ templateName: 'test' })],
      })
    ).rejects.toThrow(
      expect.objectContaining({
        status: 'PERMISSION_DENIED',
        message: expect.stringContaining('Model Armor blocked model response.'),
      })
    );
  });

  it('respects protectionTarget=userPrompt', async () => {
    // Should sanitize prompt but NOT response
    mockClient.sanitizeUserPrompt = async () => [createSdpResult('sanitized')];
    // This one would block if called
    mockClient.sanitizeModelResponse = async () => [createRaiBlockResult()];

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'hello',
      use: [
        modelArmor({
          templateName: 'test',
          protectionTarget: 'userPrompt',
          applyDeidentificationResults: true,
        }),
      ],
    });

    expect(response.text).toMatch(/Echo: sanitized/);
  });

  it('strictSdpEnforcement blocks even if remediated', async () => {
    mockClient.sanitizeUserPrompt = async () => [
      createSdpBlockResult('sanitized_hello'),
    ];

    await expect(
      ai.generate({
        model: 'echoModel',
        prompt: 'sensitive',
        use: [
          modelArmor({
            templateName: 'test',
            strictSdpEnforcement: true,
            applyDeidentificationResults: true,
          }),
        ],
      })
    ).rejects.toThrow(
      expect.objectContaining({
        status: 'PERMISSION_DENIED',
        message: expect.stringContaining('Model Armor blocked user prompt.'),
      })
    );
  });

  it('respects filters option', async () => {
    // RAI match found, but we only filter 'csam'
    mockClient.sanitizeUserPrompt = async () => [
      {
        sanitizationResult: {
          filterMatchState: 'MATCH_FOUND',
          filterResults: {
            rai: { raiFilterResult: { matchState: 'MATCH_FOUND' } },
          },
        },
      },
    ];

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'bad stuff',
      use: [
        modelArmor({
          templateName: 'test',
          filters: ['csam'],
        }),
      ],
    });

    expect(response.text).toMatch(/Echo: bad stuff/);
  });

  it('preserves non-text parts when SDP replaces text', async () => {
    mockClient.sanitizeUserPrompt = async () => [
      createSdpResult('sanitized_text'),
    ];

    ai.defineModel({ name: 'inspectionModel' }, async (req: any) => {
      return {
        message: {
          role: 'model',
          content: [{ text: JSON.stringify(req.messages) }],
        },
      };
    });

    const response = await ai.generate({
      model: 'inspectionModel',
      messages: [
        {
          role: 'user',
          content: [{ text: 'old stuff' }],
        },
        {
          role: 'model',
          content: [{ text: 'response' }],
        },
        {
          role: 'user',
          content: [
            { text: 'sensitive info' },
            { media: { url: 'http://example.com/image.png' } },
          ],
        },
      ],
      use: [
        modelArmor({
          templateName: 'test',
          applyDeidentificationResults: true,
        }),
      ],
    });

    const content = JSON.parse(response.text);
    // content should have preserved media and replaced text
    expect(content).toEqual([
      {
        role: 'user',
        content: [{ text: 'old stuff' }],
      },
      {
        role: 'model',
        content: [{ text: 'response' }],
      },
      {
        role: 'user',
        content: [
          { media: { url: 'http://example.com/image.png' } },
          { text: 'sanitized_text' },
        ],
      },
    ]);
  });

  it('supports custom function for applying SDP', async () => {
    mockClient.sanitizeUserPrompt = async () => [
      createSdpResult('sanitized_text'),
    ];

    const applyFn: ApplyDeidentificationResultsFn = ({ messages }) => {
      // Custom logic: replace with "CUSTOM APPLIED" instead of sdpResult data
      const newContent = [{ text: 'CUSTOM APPLIED' }];
      return [{ ...messages[0], content: newContent }];
    };
    const customAi = genkit({
      plugins: [
        modelArmor.plugin({
          client: mockClient,
          applyDeidentificationResults: applyFn,
        }),
      ],
    });
    customAi.defineModel({ name: 'echoModel' }, async (req) => ({
      message: {
        role: 'model',
        content: [{ text: `Echo: ${req.messages[0].content[0].text}` }],
      },
    }));

    const response = await customAi.generate({
      model: 'echoModel',
      prompt: 'hello',
      use: [modelArmor({ templateName: 'test' })],
    });
    expect(response.text).toMatch(/Echo: CUSTOM APPLIED/);

    // An explicit `false` turns the plugin's function off for a call.
    const unchanged = await customAi.generate({
      model: 'echoModel',
      prompt: 'hello',
      use: [
        modelArmor({
          templateName: 'test',
          applyDeidentificationResults: false,
        }),
      ],
    });
    expect(unchanged.text).toMatch(/Echo: hello/);
  });

  it('keeps the client out of the traced middleware config', () => {
    const ref = modelArmor({ templateName: 'test', filters: ['rai'] });
    expect(JSON.parse(JSON.stringify(ref))).toEqual({
      name: 'modelArmor',
      config: { templateName: 'test', filters: ['rai'] },
    });
  });

  it('requires a templateName', async () => {
    await expect(
      ai.generate({ model: 'echoModel', prompt: 'hi', use: [modelArmor()] })
    ).rejects.toThrow(expect.objectContaining({ status: 'INVALID_ARGUMENT' }));
  });
});
