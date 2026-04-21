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

import * as assert from 'assert';
import { GenerateRequest, Operation, z } from 'genkit';
import { describe, it } from 'node:test';
import { HarmBlockThreshold, HarmCategory } from '../../src/common/types.js';
import {
  fromImagenResponse,
  fromLyriaResponse,
  fromVeoOperation,
  toGeminiLabels,
  toGeminiSafetySettings,
  toImagenPredictRequest,
  toLyriaPredictRequest,
  toVeoClientOptions,
  toVeoMedia,
  toVeoModel,
  toVeoOperationRequest,
  toVeoPredictRequest,
} from '../../src/vertexai/converters.js';
import { SafetySettingsSchema } from '../../src/vertexai/gemini.js';
import {
  ImagenConfigSchema,
  ImagenTryOnConfigSchema,
} from '../../src/vertexai/imagen.js';
import { LyriaConfigSchema } from '../../src/vertexai/lyria.js';
import {
  ClientOptions,
  ImagenPredictResponse,
  LyriaPredictResponse,
  VeoOperation,
} from '../../src/vertexai/types.js';
import { VeoConfigSchema } from '../../src/vertexai/veo.js';

describe('Vertex AI Converters', () => {
  describe('toGeminiSafetySettings', () => {
    it('returns undefined for undefined input', () => {
      const result = toGeminiSafetySettings(undefined);
      assert.strictEqual(result, undefined);
    });

    it('returns an empty array for an empty array input', () => {
      const result = toGeminiSafetySettings([]);
      assert.deepStrictEqual(result, []);
    });

    it('converts genkit safety settings to Gemini safety settings', () => {
      const genkitSettings: z.infer<typeof SafetySettingsSchema>[] = [
        {
          category: 'HARM_CATEGORY_HATE_SPEECH',
          threshold: 'BLOCK_LOW_AND_ABOVE',
        },
        {
          category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
          threshold: 'BLOCK_NONE',
        },
      ];

      const expected = [
        {
          category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
          threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        },
        {
          category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
          threshold: HarmBlockThreshold.BLOCK_NONE,
        },
      ];

      const result = toGeminiSafetySettings(genkitSettings);
      assert.deepStrictEqual(result, expected);
    });
  });

  describe('toGeminiLabels', () => {
    it('returns undefined for undefined input', () => {
      const result = toGeminiLabels(undefined);
      assert.strictEqual(result, undefined);
    });

    it('returns undefined for an empty object input', () => {
      const result = toGeminiLabels({});
      assert.strictEqual(result, undefined);
    });

    it('converts an object with valid labels', () => {
      const labels = {
        env: 'production',
        'my-label': 'my-value',
      };
      const result = toGeminiLabels(labels);
      assert.deepStrictEqual(result, labels);
    });

    it('filters out empty string keys', () => {
      const labels = {
        env: 'dev',
        '': 'should-be-ignored',
        'valid-key': 'valid-value',
      };
      const expected = {
        env: 'dev',
        'valid-key': 'valid-value',
      };
      const result = toGeminiLabels(labels);
      assert.deepStrictEqual(result, expected);
    });

    it('returns undefined if all keys are empty strings', () => {
      const labels = {
        '': 'value1',
      };
      const result = toGeminiLabels(labels);
      assert.strictEqual(result, undefined);
    });

    it('handles labels with empty values', () => {
      const labels = {
        key1: '',
        key2: 'value2',
      };
      const expected = {
        key1: '',
        key2: 'value2',
      };
      const result = toGeminiLabels(labels);
      assert.deepStrictEqual(result, expected);
    });
  });

  describe('toImagenPredictRequest', () => {
    const baseRequest: GenerateRequest<typeof ImagenConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'A cat on a mat' }] }],
    };

    it('should create a basic ImagenPredictRequest with default sampleCount', () => {
      const result = toImagenPredictRequest(baseRequest);
      assert.deepStrictEqual(result, {
        instances: [{ prompt: 'A cat on a mat' }],
        parameters: { sampleCount: 1 },
      });
    });

    it('should handle candidates and config parameters', () => {
      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        ...baseRequest,
        candidates: 2,
        config: {
          seed: 42,
          negativePrompt: 'ugly',
          aspectRatio: '16:9',
        },
      };
      const result = toImagenPredictRequest(request);
      assert.deepStrictEqual(result, {
        instances: [{ prompt: 'A cat on a mat' }],
        parameters: {
          sampleCount: 2,
          seed: 42,
          negativePrompt: 'ugly',
          aspectRatio: '16:9',
        },
      });
    });

    it('should omit undefined or null config parameters', () => {
      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        ...baseRequest,
        config: {
          addWatermark: false,
          foo: 0,
          negativePrompt: undefined,
          seed: null as any,
          aspectRatio: '1:1',
        },
      };
      const result = toImagenPredictRequest(request);
      assert.deepStrictEqual(result, {
        instances: [{ prompt: 'A cat on a mat' }],
        parameters: {
          addWatermark: false,
          sampleCount: 1,
          foo: 0,
          aspectRatio: '1:1',
        },
      });
    });

    it('should handle image and mask media', () => {
      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        messages: [
          {
            role: 'user',
            content: [
              { text: 'A dog on a rug' },
              {
                media: {
                  url: 'data:image/png;base64,IMAGEDATA',
                  contentType: 'image/png',
                },
                metadata: { type: 'image' },
              },
              {
                media: {
                  url: 'data:image/png;base64,MASKDATA',
                  contentType: 'image/png',
                },
                metadata: { type: 'mask' },
              },
            ],
          },
        ],
      };
      const result = toImagenPredictRequest(request);
      assert.deepStrictEqual(result, {
        instances: [
          {
            prompt: 'A dog on a rug',
            image: { bytesBase64Encoded: 'IMAGEDATA' },
            mask: { image: { bytesBase64Encoded: 'MASKDATA' } },
          },
        ],
        parameters: { sampleCount: 1 },
      });
    });

    it('should handle TryOn config', () => {
      const request: GenerateRequest<typeof ImagenTryOnConfigSchema> = {
        messages: [
          {
            role: 'user',
            content: [
              {
                media: {
                  url: 'data:image/jpeg;base64,PERSON_IMAGE',
                  contentType: 'image/jpeg',
                },
                metadata: { type: 'personImage' },
              },
              {
                media: {
                  url: 'gs://bucket/product.png',
                  contentType: 'image/png',
                },
                metadata: { type: 'productImage' },
              },
            ],
          },
        ],
        config: {
          sampleCount: 2,
          storageUri: 'gs://bucket',
          seed: 123,
          baseSteps: 50,
          personGeneration: 'allow_adult',
          safetySetting: 'block_few',
          outputOptions: {
            mimeType: 'image/jpeg',
            compressionQuality: 80,
          },
        },
      };
      const result = toImagenPredictRequest(request);
      assert.deepStrictEqual(result, {
        instances: [
          {
            personImage: {
              image: { bytesBase64Encoded: 'PERSON_IMAGE' },
            },
            productImages: [
              {
                image: { gcsUri: 'gs://bucket/product.png' },
              },
            ],
          },
        ],
        parameters: {
          sampleCount: 2,
          storageUri: 'gs://bucket',
          seed: 123,
          baseSteps: 50,
          personGeneration: 'allow_adult',
          safetySetting: 'block_few',
          outputOptions: {
            mimeType: 'image/jpeg',
            compressionQuality: 80,
          },
        },
      });
    });
  });

  describe('fromImagenResponse', () => {
    it('should convert ImagenPredictResponse to GenerateResponseData', () => {
      const response: ImagenPredictResponse = {
        predictions: [
          { bytesBase64Encoded: 'IMAGE1', mimeType: 'image/jpeg' },
          { bytesBase64Encoded: 'IMAGE2', mimeType: 'image/png' },
        ],
      };
      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        messages: [{ role: 'user', content: [{ text: 'test' }] }],
      };
      const result = fromImagenResponse(response, request);

      assert.strictEqual(result.candidates?.length, 2);
      // Test structure from fromImagenPrediction logic
      assert.deepStrictEqual(result.candidates?.[0], {
        index: 0,
        finishReason: 'stop',
        message: {
          role: 'model',
          content: [
            {
              media: {
                url: 'data:image/jpeg;base64,IMAGE1',
                contentType: 'image/jpeg',
              },
            },
          ],
        },
      });
      assert.deepStrictEqual(result.candidates?.[1], {
        index: 1,
        finishReason: 'stop',
        message: {
          role: 'model',
          content: [
            {
              media: {
                url: 'data:image/png;base64,IMAGE2',
                contentType: 'image/png',
              },
            },
          ],
        },
      });
      assert.strictEqual(result.custom, response);
      assert.ok(result.usage);
    });
  });

  describe('toLyriaPredictRequest', () => {
    const baseRequest: GenerateRequest<typeof LyriaConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'A happy song' }] }],
    };

    it('should create a basic LyriaPredictRequest', () => {
      const result = toLyriaPredictRequest(baseRequest);
      assert.deepStrictEqual(result, {
        instances: [{ prompt: 'A happy song' }],
        parameters: { sampleCount: 1 },
      });
    });

    it('should handle config parameters', () => {
      const request: GenerateRequest<typeof LyriaConfigSchema> = {
        ...baseRequest,
        config: {
          negativePrompt: 'sad',
          seed: 123,
          sampleCount: 3,
        },
      };
      const result = toLyriaPredictRequest(request);
      assert.deepStrictEqual(result, {
        instances: [
          { prompt: 'A happy song', negativePrompt: 'sad', seed: 123 },
        ],
        parameters: { sampleCount: 3 },
      });
    });
  });

  describe('fromLyriaResponse', () => {
    it('should convert LyriaPredictResponse to GenerateResponseData', () => {
      const response: LyriaPredictResponse = {
        predictions: [{ bytesBase64Encoded: 'AUDIO1', mimeType: 'audio/wav' }],
      };
      const request: GenerateRequest<typeof LyriaConfigSchema> = {
        messages: [{ role: 'user', content: [{ text: 'test' }] }],
      };
      const result = fromLyriaResponse(response, request);

      assert.strictEqual(result.candidates?.length, 1);

      assert.deepStrictEqual(result.candidates?.[0], {
        index: 0,
        finishReason: 'stop',
        message: {
          role: 'model',
          content: [
            {
              media: {
                url: 'data:audio/wav;base64,AUDIO1',
                contentType: 'audio/wav',
              },
            },
          ],
        },
      });
      assert.strictEqual(result.custom, response);
      assert.ok(result.usage);
    });
  });

  describe('toVeoMedia', () => {
    it('should convert data URL', () => {
      const mediaPart = {
        url: 'data:image/png;base64,VEODATA',
        contentType: 'image/png',
      };
      const result = toVeoMedia(mediaPart);
      assert.deepStrictEqual(result, {
        bytesBase64Encoded: 'VEODATA',
        mimeType: 'image/png',
      });
    });

    it('should convert gs URL', () => {
      const mediaPart = {
        url: 'gs://bucket/object',
        contentType: 'video/mp4',
      };
      const result = toVeoMedia(mediaPart);
      assert.deepStrictEqual(result, {
        gcsUri: 'gs://bucket/object',
        mimeType: 'video/mp4',
      });
    });

    it('should throw on http URL', () => {
      const mediaPart = {
        url: 'http://example.com/image.jpg',
        contentType: 'image/jpeg',
      };
      assert.throws(() => toVeoMedia(mediaPart), /Veo does not support http/);
    });

    it('should infer mimeType if missing', () => {
      const mediaPart = { url: 'data:image/jpeg;base64,VEODATA' };
      const result = toVeoMedia(mediaPart as any);
      assert.deepStrictEqual(result, {
        bytesBase64Encoded: 'VEODATA',
        mimeType: 'image/jpeg',
      });
    });
  });

  describe('toVeoPredictRequest', () => {
    const baseRequest: GenerateRequest<typeof VeoConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'A video of a sunset' }] }],
    };

    it('should create a basic VeoPredictRequest', () => {
      const result = toVeoPredictRequest(baseRequest);
      assert.deepStrictEqual(result, {
        instances: [{ prompt: 'A video of a sunset' }],
        parameters: {},
      });
    });

    it('should handle config parameters', () => {
      const request: GenerateRequest<typeof VeoConfigSchema> = {
        ...baseRequest,
        config: {
          durationSeconds: 5,
          fps: 24,
          aspectRatio: '16:9',
        },
      };
      const result = toVeoPredictRequest(request);
      assert.deepStrictEqual(result, {
        instances: [{ prompt: 'A video of a sunset' }],
        parameters: {
          durationSeconds: 5,
          fps: 24,
          aspectRatio: '16:9',
        },
      });
    });

    it('should handle media parts', () => {
      const request: GenerateRequest<typeof VeoConfigSchema> = {
        messages: [
          {
            role: 'user',
            content: [
              { text: 'A video of a sunrise' },
              {
                media: {
                  url: 'data:image/jpeg;base64,IMAGEDATA',
                  contentType: 'image/jpeg',
                },
                metadata: { type: 'image' },
              },
              {
                media: {
                  url: 'gs://bucket/video.mp4',
                  contentType: 'video/mp4',
                },
                metadata: { type: 'video' },
              },
            ],
          },
        ],
      };
      const result = toVeoPredictRequest(request);
      assert.deepStrictEqual(result, {
        instances: [
          {
            prompt: 'A video of a sunrise',
            image: {
              bytesBase64Encoded: 'IMAGEDATA',
              mimeType: 'image/jpeg',
            },
            video: { gcsUri: 'gs://bucket/video.mp4', mimeType: 'video/mp4' },
          },
        ],
        parameters: {},
      });
    });

    it('should handle referenceImages media parts', () => {
      const request: GenerateRequest<typeof VeoConfigSchema> = {
        messages: [
          {
            role: 'user',
            content: [
              { text: 'A video of a sunrise' },
              {
                media: {
                  url: 'data:image/jpeg;base64,REFIMAGEDATA1',
                  contentType: 'image/jpeg',
                },
                metadata: { type: 'referenceImages', referenceType: 'asset' },
              },
              {
                media: {
                  url: 'gs://bucket/refimage2.png',
                  contentType: 'image/png',
                },
                metadata: { type: 'referenceImages', referenceType: 'style' },
              },
            ],
          },
        ],
      };
      const result = toVeoPredictRequest(request);
      assert.deepStrictEqual(result, {
        instances: [
          {
            prompt: 'A video of a sunrise',
            referenceImages: [
              {
                image: {
                  bytesBase64Encoded: 'REFIMAGEDATA1',
                  mimeType: 'image/jpeg',
                },
                referenceType: 'asset',
              },
              {
                image: {
                  gcsUri: 'gs://bucket/refimage2.png',
                  mimeType: 'image/png',
                },
                referenceType: 'style',
              },
            ],
          },
        ],
        parameters: {},
      });
    });
  });

  describe('fromVeoOperation', () => {
    it('should convert basic pending operation', () => {
      const veoOp: VeoOperation = {
        name: 'operations/123',
        done: false,
      };
      const result = fromVeoOperation(veoOp);
      assert.deepStrictEqual(result, {
        id: 'operations/123',
        done: false,
      });
    });

    it('should convert done operation with videos', () => {
      const veoOp: VeoOperation = {
        name: 'operations/456',
        done: true,
        response: {
          videos: [
            {
              gcsUri: 'gs://bucket/vid1.mp4',
              mimeType: 'video/mp4',
            },
            {
              bytesBase64Encoded: 'VID2DATA',
              mimeType: 'video/webm',
            },
          ],
        },
      };
      const result = fromVeoOperation(veoOp);
      assert.deepStrictEqual(result, {
        id: 'operations/456',
        done: true,
        output: {
          finishReason: 'stop',
          raw: veoOp.response,
          message: {
            role: 'model',
            content: [
              {
                media: {
                  url: 'gs://bucket/vid1.mp4',
                  contentType: 'video/mp4',
                },
              },
              {
                media: {
                  url: 'data:video/webm:base64,VID2DATA',
                  contentType: 'video/webm',
                },
              },
            ],
          },
        },
      });
    });

    it('should convert operation with error', () => {
      const veoOp: VeoOperation = {
        name: 'operations/789',
        done: true,
        error: { code: 3, message: 'Invalid argument' },
      };
      const result = fromVeoOperation(veoOp);
      assert.deepStrictEqual(result, {
        id: 'operations/789',
        done: true,
        error: { message: 'Invalid argument' },
      });
    });

    it('should convert operation with clientOptions', () => {
      const clientOptions: ClientOptions = {
        kind: 'regional',
        location: 'us-west1',
        projectId: 'foo',
        authClient: {} as any,
      };
      const veoOp: VeoOperation = {
        name: 'operations/789',
        done: false,
        clientOptions: clientOptions,
      };
      const result = fromVeoOperation(veoOp);
      assert.deepStrictEqual(result, {
        id: 'operations/789',
        done: false,
        metadata: {
          clientOptions: clientOptions,
        },
      });
    });
  });

  describe('toVeoModel', () => {
    it('should extract model name from operation id', () => {
      const op = {
        id: 'projects/test-project/locations/us-central1/models/veo-1.0/operations/12345',
      };
      const result = toVeoModel(op);
      assert.strictEqual(result, 'veo-1.0');
    });
  });

  describe('toVeoOperationRequest', () => {
    it('should create VeoOperationRequest from Operation', () => {
      const op = {
        id: 'operations/abcdef',
      };
      const result = toVeoOperationRequest(op);
      assert.deepStrictEqual(result, {
        operationName: 'operations/abcdef',
      });
    });
  });

  describe('toVeoClientOptions', () => {
    const defaultClientOptions: ClientOptions = {
      kind: 'regional',
      location: 'us-central1',
      projectId: 'default-project',
      authClient: {} as any,
    };
    const opClientOptions: ClientOptions = {
      kind: 'global',
      location: 'global',
      projectId: 'op-project',
      authClient: {} as any,
    };

    it('should return client options from operation metadata if present', () => {
      const op: Operation = {
        id: 'op1',
        done: false,
        metadata: {
          clientOptions: opClientOptions,
        },
      };
      const result = toVeoClientOptions(op, defaultClientOptions);
      assert.deepStrictEqual(result, opClientOptions);
    });

    it('should return default client options if not in operation metadata', () => {
      const op: Operation = {
        id: 'op2',
        done: false,
      };
      const result = toVeoClientOptions(op, defaultClientOptions);
      assert.deepStrictEqual(result, defaultClientOptions);
    });

    it('should return default client options if metadata is empty', () => {
      const op: Operation = {
        id: 'op3',
        done: false,
        metadata: {},
      };
      const result = toVeoClientOptions(op, defaultClientOptions);
      assert.deepStrictEqual(result, defaultClientOptions);
    });
  });
});
