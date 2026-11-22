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

import type { v2 } from '@google-cloud/dlp';
import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { Genkit, genkit } from 'genkit';
import * as auth from '../src/auth.js';
import { sensitiveDataProtection } from '../src/sdp/index.js';

const mockDeidentifyContent =
  jest.fn<typeof v2.DlpServiceClient.prototype.deidentifyContent>();

jest.mock('@google-cloud/dlp', () => ({
  v2: {
    DlpServiceClient: jest.fn().mockImplementation((opts: any) => ({
      ...opts,
      deidentifyContent: mockDeidentifyContent,
    })),
  },
}));

jest.mock('../src/auth.js', () => ({
  credentialsFromEnvironment: jest.fn(),
}));

describe('sensitiveDataProtection middleware', () => {
  let ai: Genkit;

  beforeEach(() => {
    jest.clearAllMocks();

    ai = genkit({});
    ai.defineModel({ name: 'echoModel' }, async (req: any) => {
      return {
        message: {
          role: 'model',
          content: [{ text: `Echo: ${req.messages[0].content[0].text}` }],
        },
      };
    });

    ai.defineModel({ name: 'leakyModel' }, async (req: any) => {
      return {
        message: {
          role: 'model',
          content: [{ text: `Here is my credit card` }],
        },
      };
    });

    // Setup mock for middleware auth
    const mockCreds = { client_email: 'test@example.com' };
    jest.mocked(auth.credentialsFromEnvironment).mockResolvedValue({
      projectId: 'env-project',
      credentials: mockCreds as any,
    });
  });

  it('Applies default InfoTypes configuration to DLP API request', async () => {
    mockDeidentifyContent.mockImplementation(async (req: any) => {
      const text = req.item.value;
      const redacted = text.replace(
        /my credit card|my ssn/g,
        '[REDACTED_BY_MOCK]'
      );
      return [{ item: { value: redacted } }] as any;
    });

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'Here is my credit card',
      use: [sensitiveDataProtection({})],
    });

    // Validates that request was intercepted, passed to model, and returned
    expect(response.text).toBe('Echo: Here is [REDACTED_BY_MOCK]');

    // Verify DLP was called with default config
    expect(mockDeidentifyContent).toHaveBeenCalledWith(
      expect.objectContaining({
        inspectConfig: {
          infoTypes: [
            { name: 'CREDIT_CARD_NUMBER' },
            { name: 'EMAIL_ADDRESS' },
            { name: 'PHONE_NUMBER' },
          ],
        },
        deidentifyConfig: expect.objectContaining({
          infoTypeTransformations: {
            transformations: [
              {
                primitiveTransformation: { replaceWithInfoTypeConfig: {} },
              },
            ],
          },
        }),
      })
    );
  });

  it('Applies custom string transformation configuration to DLP API request', async () => {
    mockDeidentifyContent.mockImplementation(async (req: any) => {
      const text = req.item.value;
      const redacted = text.replace(/my credit card|my ssn/g, '[HIDDEN]');
      return [{ item: { value: redacted } }] as any;
    });

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'Here is my credit card',
      use: [
        sensitiveDataProtection({
          inline: {
            transformation: 'CUSTOM_STRING',
            customConfig: '[HIDDEN]',
          },
        }),
      ],
    });

    expect(response.text).toBe('Echo: Here is [HIDDEN]');

    // Verify DLP was called with custom string config
    expect(mockDeidentifyContent).toHaveBeenCalledWith(
      expect.objectContaining({
        deidentifyConfig: expect.objectContaining({
          infoTypeTransformations: {
            transformations: [
              {
                primitiveTransformation: {
                  replaceConfig: {
                    newValue: { stringValue: '[HIDDEN]' },
                  },
                },
              },
            ],
          },
        }),
      })
    );
  });

  it('Applies character masking transformation configuration to DLP API request', async () => {
    mockDeidentifyContent.mockImplementation(async (req: any) => {
      const text = req.item.value;
      const redacted = text.replace(/my credit card|my ssn/g, '#####');
      return [{ item: { value: redacted } }] as any;
    });

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'Here is my credit card',
      use: [
        sensitiveDataProtection({
          inline: {
            transformation: 'MASK',
            maskConfig: { maskingCharacter: '#' },
          },
        }),
      ],
    });

    expect(response.text).toBe('Echo: Here is #####');

    // Verify DLP was called with mask config
    expect(mockDeidentifyContent).toHaveBeenCalledWith(
      expect.objectContaining({
        deidentifyConfig: expect.objectContaining({
          infoTypeTransformations: {
            transformations: [
              {
                primitiveTransformation: {
                  characterMaskConfig: { maskingCharacter: '#' },
                },
              },
            ],
          },
        }),
      })
    );
  });

  it('Applies custom InfoTypes configuration to DLP API request', async () => {
    mockDeidentifyContent.mockImplementation(async (req: any) => {
      const text = req.item.value;
      const redacted = text.replace(
        /my credit card|my ssn/g,
        '[REDACTED_BY_MOCK]'
      );
      return [{ item: { value: redacted } }] as any;
    });

    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'Here is my ssn',
      use: [
        sensitiveDataProtection({
          inline: {
            infoTypes: [
              'US_SOCIAL_SECURITY_NUMBER',
              { name: 'EMAIL_ADDRESS', version: 'latest' },
            ],
          },
        }),
      ],
    });

    expect(response.text).toBe('Echo: Here is [REDACTED_BY_MOCK]');

    // Verify DLP was called with custom infoTypes
    expect(mockDeidentifyContent).toHaveBeenCalledWith(
      expect.objectContaining({
        inspectConfig: {
          infoTypes: [
            { name: 'US_SOCIAL_SECURITY_NUMBER' },
            { name: 'EMAIL_ADDRESS', version: 'latest' },
          ],
        },
      })
    );
  });

  it("Redacts sensitive data from the model's response before returning to user", async () => {
    mockDeidentifyContent.mockImplementation(async (req: any) => {
      const text = req.item.value;
      const redacted = text.replace(/my credit card/g, '[REDACTED_BY_MOCK]');
      return [{ item: { value: redacted } }] as any;
    });

    const response = await ai.generate({
      model: 'leakyModel',
      prompt: 'Hello',
      use: [sensitiveDataProtection({})],
    });

    expect(response.text).toBe('Here is [REDACTED_BY_MOCK]');
  });
});
