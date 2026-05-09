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

import {
  GenerateRequest,
  GenerateResponseData,
  GenkitError,
  MediaPart,
  Operation,
  z,
} from 'genkit';
import { CandidateData, getBasicUsageStats } from 'genkit/model';
import {
  HarmBlockThreshold,
  HarmCategory,
  ImagenInstance,
  ImagenParameters,
  ImagenPredictRequest,
  ImagenPredictResponse,
  ImagenPrediction,
  SafetySetting,
} from '../common/types.js';
import { extractMediaArray } from '../common/utils.js';
import { SafetySettingsSchema } from './gemini.js';
import {
  ImagenConfigSchemaType,
  ImagenTryOnConfigSchemaType,
} from './imagen.js';
import { LyriaConfigSchemaType } from './lyria.js';
import {
  ClientOptions,
  LyriaInstance,
  LyriaParameters,
  LyriaPredictRequest,
  LyriaPredictResponse,
  LyriaPrediction,
  VeoInstance,
  VeoMedia,
  VeoOperation,
  VeoOperationRequest,
  VeoPredictRequest,
} from './types.js';
import {
  checkSupportedMimeType,
  extractMedia,
  extractMimeType,
  extractText,
} from './utils.js';
import { VeoConfigSchemaType } from './veo.js';

export function toGeminiSafetySettings(
  genkitSettings?: z.infer<typeof SafetySettingsSchema>[]
): SafetySetting[] | undefined {
  if (!genkitSettings) return undefined;
  return genkitSettings.map((s) => {
    return {
      category: s.category as HarmCategory,
      threshold: s.threshold as HarmBlockThreshold,
    };
  });
}

export function toGeminiLabels(
  labels?: Record<string, string>
): Record<string, string> | undefined {
  if (!labels) {
    return undefined;
  }
  const keys = Object.keys(labels);
  const newLabels: Record<string, string> = {};
  for (const key of keys) {
    const value = labels[key];
    if (!key) {
      continue;
    }
    newLabels[key] = value;
  }

  if (Object.keys(newLabels).length == 0) {
    return undefined;
  }
  return newLabels;
}

export function toImagenMedia(media: MediaPart['media']): {
  bytesBase64Encoded?: string;
  gcsUri?: string;
} {
  if (media.url.startsWith('data:')) {
    return {
      bytesBase64Encoded: media.url?.split(',')[1],
    };
  } else if (media.url.startsWith('gs://')) {
    return {
      gcsUri: media.url,
    };
  } else if (media.url.startsWith('http')) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'Imagen does not support http(s) URIs. Please specify a Cloud Storage URI.',
    });
  } else {
    // Assume it's a non-prefixed data url
    return {
      bytesBase64Encoded: media.url,
    };
  }
}

export function toImagenTryOnRequest(
  request: GenerateRequest<ImagenTryOnConfigSchemaType>
): ImagenPredictRequest {
  const { ...parameters } = request.config || {};

  const instance: ImagenInstance = {};
  const personImageMedia = extractMedia(request, {
    metadataType: 'personImage',
  });
  if (personImageMedia) {
    instance.personImage = {
      image: toImagenMedia(personImageMedia),
    };
  }

  const productImagesMedia = extractMediaArray(request, {
    metadataType: 'productImage',
  });
  if (productImagesMedia) {
    instance.productImages = productImagesMedia.map((media) => ({
      image: toImagenMedia(media.media),
    }));
  }

  return {
    instances: [instance],
    parameters,
  };
}

export function toImagenPredictRequest(
  request: GenerateRequest<ImagenConfigSchemaType | ImagenTryOnConfigSchemaType>
): ImagenPredictRequest {
  const hasTryOnMedia = request.messages.some((m) =>
    m.content.some(
      (p) =>
        p.metadata &&
        (p.metadata.type === 'personImage' ||
          p.metadata.type === 'productImage')
    )
  );
  if (hasTryOnMedia) {
    return toImagenTryOnRequest(
      request as GenerateRequest<ImagenTryOnConfigSchemaType>
    );
  }
  return {
    instances: toImagenInstances(
      request as GenerateRequest<ImagenConfigSchemaType>
    ),
    parameters: toImagenParameters(
      request as GenerateRequest<ImagenConfigSchemaType>
    ),
  };
}

function toImagenInstances(
  request: GenerateRequest<ImagenConfigSchemaType>
): ImagenInstance[] {
  let instance: ImagenInstance = {
    prompt: extractText(request),
  };

  const imageMedia = extractMedia(request, {
    metadataType: 'image',
    isDefault: true,
  });
  if (imageMedia) {
    const image = imageMedia.url.split(',')[1];
    instance.image = {
      bytesBase64Encoded: image,
    };
  }

  const maskMedia = extractMedia(request, { metadataType: 'mask' });
  if (maskMedia) {
    const mask = maskMedia.url.split(',')[1];
    instance.mask = {
      image: {
        bytesBase64Encoded: mask,
      },
    };
  }

  return [instance];
}

function toImagenParameters(
  request: GenerateRequest<ImagenConfigSchemaType>
): ImagenParameters {
  const params = {
    sampleCount: request.candidates ?? 1,
    ...request?.config,
  };

  for (const k in params) {
    if (params[k] === undefined || params[k] === null) {
      delete params[k];
    }
  }

  return params;
}

function fromImagenPrediction(p: ImagenPrediction, i: number): CandidateData {
  const b64data = p.bytesBase64Encoded;
  const mimeType = p.mimeType;
  return {
    index: i,
    finishReason: 'stop',
    message: {
      role: 'model',
      content: [
        {
          media: {
            url: `data:${mimeType};base64,${b64data}`,
            contentType: mimeType,
          },
        },
      ],
    },
  };
}

/**
 *
 * @param response The response to convert
 * @param request The request (for usage stats)
 * @returns The converted response
 */
export function fromImagenResponse(
  response: ImagenPredictResponse,
  request: GenerateRequest
): GenerateResponseData {
  const candidates = response.predictions.map(fromImagenPrediction);
  return {
    candidates,
    usage: {
      ...getBasicUsageStats(request.messages, candidates),
      custom: { generations: candidates.length },
    },
    custom: response,
  };
}

export function toLyriaPredictRequest(
  request: GenerateRequest<LyriaConfigSchemaType>
): LyriaPredictRequest {
  return {
    instances: toLyriaInstances(request),
    parameters: toLyriaParameters(request),
  };
}

function toLyriaInstances(
  request: GenerateRequest<LyriaConfigSchemaType>
): LyriaInstance[] {
  let config = { ...request.config };
  delete config.sampleCount; // Sample count goes in parameters, the rest go in instances
  return [
    {
      prompt: extractText(request),
      ...config,
    },
  ];
}

function toLyriaParameters(
  request: GenerateRequest<LyriaConfigSchemaType>
): LyriaParameters {
  return {
    sampleCount: request.config?.sampleCount || 1,
  };
}

function fromLyriaPrediction(p: LyriaPrediction, i: number): CandidateData {
  const b64data = p.bytesBase64Encoded;
  const mimeType = p.mimeType;
  return {
    index: i,
    finishReason: 'stop',
    message: {
      role: 'model',
      content: [
        {
          media: {
            url: `data:${mimeType};base64,${b64data}`,
            contentType: mimeType,
          },
        },
      ],
    },
  };
}

export function fromLyriaResponse(
  response: LyriaPredictResponse,
  request: GenerateRequest
): GenerateResponseData {
  const candidates: CandidateData[] =
    response.predictions.map(fromLyriaPrediction);
  return {
    candidates,
    usage: {
      ...getBasicUsageStats(request.messages, candidates),
      custom: { generations: candidates.length },
    },
    custom: response,
  };
}

export function toVeoPredictRequest(
  request: GenerateRequest<VeoConfigSchemaType>
): VeoPredictRequest {
  return {
    instances: toVeoInstances(request),
    parameters: { ...request.config },
  };
}

function toVeoInstances(
  request: GenerateRequest<VeoConfigSchemaType>
): VeoInstance[] {
  let instance: VeoInstance = {
    prompt: extractText(request),
  };

  const supportedImageTypes = ['image/jpeg', 'image/png', 'image/webp'];
  const supportedVideoTypes = [
    'video/mov',
    'video/mpeg',
    'video/mp4',
    'video/mpg',
    'video/avi',
    'video/wmv',
    'video/mpegps',
    'video/flv',
  ];

  const imageMedia = extractMedia(request, {
    metadataType: 'image',
    isDefault: true,
  });
  if (imageMedia) {
    checkSupportedMimeType(imageMedia, supportedImageTypes);
    instance.image = toVeoMedia(imageMedia);
  }

  const lastFrameMedia = extractMedia(request, { metadataType: 'lastFrame' });
  if (lastFrameMedia) {
    checkSupportedMimeType(lastFrameMedia, supportedImageTypes);
    instance.lastFrame = toVeoMedia(lastFrameMedia);
  }

  const videoMedia = extractMedia(request, { metadataType: 'video' });
  if (videoMedia) {
    checkSupportedMimeType(videoMedia, supportedVideoTypes);
    instance.video = toVeoMedia(videoMedia);
  }

  const referenceImages = extractMediaArray(request, {
    metadataType: 'referenceImages',
  });
  if (referenceImages) {
    instance.referenceImages = referenceImages.map((refImage) => ({
      image: toVeoMedia(refImage.media),
      referenceType: refImage.metadata?.referenceType as string,
    }));
  }

  return [instance];
}

export function toVeoMedia(media: MediaPart['media']): VeoMedia {
  let mimeType = media.contentType;
  if (!mimeType) {
    mimeType = extractMimeType(media.url);
    if (!mimeType) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: 'Content type is required.',
      });
    }
  }
  if (media.url.startsWith('data:')) {
    return {
      bytesBase64Encoded: media.url?.split(',')[1],
      mimeType,
    };
  } else if (media.url.startsWith('gs://')) {
    return {
      gcsUri: media.url,
      mimeType,
    };
  } else if (media.url.startsWith('http')) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'Veo does not support http(s) URIs. Please specify a Cloud Storage URI.',
    });
  } else {
    // Assume it's a non-prefixed data url
    return {
      bytesBase64Encoded: media.url,
      mimeType,
    };
  }
}

export function fromVeoOperation(
  fromOp: VeoOperation
): Operation<GenerateResponseData> {
  const toOp: Operation<GenerateResponseData> = { id: fromOp.name };
  if (fromOp.done !== undefined) {
    toOp.done = fromOp.done;
  }
  if (fromOp.error) {
    toOp.error = { message: fromOp.error.message };
  }
  if (fromOp.clientOptions) {
    toOp.metadata = {
      clientOptions: fromOp.clientOptions,
    };
  }

  if (fromOp.response) {
    toOp.output = {
      finishReason: 'stop',
      raw: fromOp.response,
      message: {
        role: 'model',
        content: fromOp.response.videos.map((veoMedia) => {
          if (veoMedia.bytesBase64Encoded) {
            return {
              media: {
                url: `data:${veoMedia.mimeType}:base64,${veoMedia.bytesBase64Encoded}`,
                contentType: veoMedia.mimeType,
              },
            };
          }

          return {
            media: {
              url: veoMedia.gcsUri ?? '',
              contentType: veoMedia.mimeType,
            },
          };
        }),
      },
    };
  }

  return toOp;
}

export function toVeoModel(op: Operation<GenerateResponseData>): string {
  return op.id.substring(
    op.id.indexOf('models/') + 7,
    op.id.indexOf('/operations/')
  );
}

export function toVeoOperationRequest(
  op: Operation<GenerateResponseData>
): VeoOperationRequest {
  return {
    operationName: op.id,
  };
}

export function toVeoClientOptions(
  op: Operation<GenerateResponseData>,
  clientOpt: ClientOptions
): ClientOptions {
  return op.metadata?.clientOptions ?? clientOpt;
}
