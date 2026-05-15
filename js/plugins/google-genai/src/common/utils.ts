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
  EmbedderReference,
  GenkitError,
  Part as GenkitPart,
  JSONSchema,
  MediaPart,
  ModelReference,
  getClientHeader as defaultGetClientHeader,
  z,
} from 'genkit';
import { GenerateRequest } from 'genkit/model';
import { applyGeminiPartialArgs } from './converters.js';
import {
  GenerateContentCandidate,
  GenerateContentResponse,
  GenerateContentStreamResult,
  PART_KEYS,
  Part,
  isObject,
} from './types.js';

/**
 * Safely extracts the error message from the error.
 * @param e The error
 * @returns The error message
 */
export function extractErrMsg(e: unknown): string {
  let errorMessage = 'An unknown error occurred';
  if (e instanceof Error) {
    errorMessage = e.message;
  } else if (typeof e === 'string') {
    errorMessage = e;
  } else {
    // Fallback for other types
    try {
      errorMessage = JSON.stringify(e);
    } catch (stringifyError) {
      errorMessage = 'Failed to stringify error object';
    }
  }
  return errorMessage;
}

/**
 * Custom replacer function for JSON.stringify to truncate long string fields.
 * Truncates strings to the first 100 and last 10 characters
 * if the original string is longer than 110 characters.
 *
 * @param key The key of the property being stringified.
 * @param value The value of the property being stringified.
 * @return The transformed value, or the original value if no transformation is needed.
 */
export function stringTruncator(key: string, value: unknown): unknown {
  const beginLength = 100;
  const endLength = 10;
  const totalLength = beginLength + endLength;
  if (typeof value === 'string' && value.length > totalLength) {
    const start = value.substring(0, 100);
    const end = value.substring(value.length - 10);
    return `${start}...[TRUNCATED]...${end}`;
  }
  return value; // Return the original value for other keys or non-string values
}

/**
 * Gets the un-prefixed model name from a modelReference
 */
export function extractVersion(
  model: ModelReference<z.ZodTypeAny> | EmbedderReference<z.ZodTypeAny>
): string {
  return model.version ? model.version : checkModelName(model.name);
}

/**
 * Gets the model name without certain prefixes..
 * e.g. for "models/googleai/gemini-2.5-pro" it returns just 'gemini-2.5-pro'
 * @param name A string containing the model string with possible prefixes
 * @returns the model string stripped of certain prefixes
 */
export function modelName(name?: string): string | undefined {
  if (!name) return name;

  // Remove any of these prefixes:
  const prefixesToRemove =
    /background-model\/|model\/|models\/|embedders\/|googleai\/|vertexai\//g;
  return name.replace(prefixesToRemove, '');
}

/**
 * Gets the suffix of a model string.
 * Throws if the string is empty.
 * @param name A string containing the model string
 * @returns the model string stripped of prefixes and guaranteed not empty.
 */
export function checkModelName(name?: string): string {
  const version = modelName(name);
  if (!version) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'Model name is required.',
    });
  }
  return version;
}

export function extractText(request: GenerateRequest) {
  return (
    request.messages
      .at(-1)
      ?.content.map((c) => c.text || '')
      .join('') ?? ''
  );
}

const KNOWN_MIME_TYPES = {
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
  mp4: 'video/mp4',
  pdf: 'application/pdf',
};

export function extractMimeType(url?: string): string {
  if (!url) {
    return '';
  }

  const dataPrefix = 'data:';
  if (!url.startsWith(dataPrefix)) {
    // Not a data url, try suffix
    url.lastIndexOf('.');
    const key = url.substring(url.lastIndexOf('.') + 1);
    if (Object.keys(KNOWN_MIME_TYPES).includes(key)) {
      return KNOWN_MIME_TYPES[key];
    }
    return '';
  }

  const commaIndex = url.indexOf(',');
  if (commaIndex == -1) {
    // Invalid - missing separator
    return '';
  }

  // The part between 'data:' and the comma
  let mimeType = url.substring(dataPrefix.length, commaIndex);
  const base64Marker = ';base64';
  if (mimeType.endsWith(base64Marker)) {
    mimeType = mimeType.substring(0, mimeType.length - base64Marker.length);
  }

  return mimeType.trim();
}

export function checkSupportedMimeType(
  media: MediaPart['media'],
  supportedTypes: string[]
) {
  if (!supportedTypes.includes(media.contentType ?? '')) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Invalid mimeType for ${displayUrl(media.url)}: "${media.contentType}". Supported mimeTypes: ${supportedTypes.join(', ')}`,
    });
  }
}

/**
 *
 * @param url The url to show (e.g. in an error message)
 * @returns The appropriately  sized url
 */
export function displayUrl(url: string): string {
  if (url.length <= 50) {
    return url;
  }

  return url.substring(0, 25) + '...' + url.substring(url.length - 25);
}

function isMediaPart(part: GenkitPart): part is MediaPart {
  return (part as MediaPart).media !== undefined;
}

/**
 *
 * @param request A generate request to extract from
 * @param metadataType The media must have metadata matching this type if isDefault is false
 * @param isDefault 'true' allows missing metadata type to match as well.
 * @returns
 */
export function extractMedia(
  request: GenerateRequest,
  params: {
    metadataType?: string;
    /* Is there is no metadata type, it will match if isDefault is true */
    isDefault?: boolean;
  }
): MediaPart['media'] | undefined {
  const mediaArray = extractMediaArray(request, params);
  if (mediaArray?.length) {
    return mediaArray[0].media;
  }

  return undefined;
}

/**
 *
 * @param request A generate request to extract from
 * @param metadataType The media must have metadata matching this type if isDefault is false
 * @param isDefault 'true' allows missing metadata type to match as well.
 * @returns
 */
export function extractMediaArray(
  request: GenerateRequest,
  params: {
    metadataType?: string;
    /* If there is no metadata type, it will match if isDefault is true */
    isDefault?: boolean;
  }
): MediaPart[] | undefined {
  // MediaPart filter:  Keeps parts matching `params.metadataType`,
  // or parts with no metadata type if `params.isDefault` is true.
  // Keeps everything if no params are specified.
  const matchesMediaParams = (part: MediaPart) => {
    if (params.metadataType || params.isDefault) {
      // We need to check the metadata type
      const metadata = part.metadata;
      if (!metadata?.type) {
        return !!params.isDefault;
      } else {
        return metadata.type == params.metadataType;
      }
    }
    return true;
  };

  const mediaArray = request.messages
    .at(-1)
    ?.content.filter(isMediaPart)
    .filter(matchesMediaParams)
    ?.map((mediaPart) => {
      let media = mediaPart.media;
      if (media && !media?.contentType) {
        // Add the mimeType
        media = {
          url: media.url,
          contentType: extractMimeType(media.url),
        };
      }

      return {
        media,
        metadata: {
          referenceType: mediaPart.metadata?.referenceType ?? 'asset',
        },
      };
    });

  if (mediaArray?.length) {
    return mediaArray;
  }

  return undefined;
}

/**
 * Cleans a JSON schema by removing specific keys and standardizing types.
 *
 * @param {JSONSchema} schema The JSON schema to clean.
 * @returns {JSONSchema} The cleaned JSON schema.
 */
export function cleanSchema(schema: JSONSchema): JSONSchema {
  const out = structuredClone(schema);
  for (const key in out) {
    if (key === '$schema' || key === 'additionalProperties') {
      delete out[key];
      continue;
    }
    if (typeof out[key] === 'object') {
      out[key] = cleanSchema(out[key]);
    }
    // Zod nullish() and picoschema optional fields will produce type `["string", "null"]`
    // which is not supported by the model API. Convert them to just `"string"`.
    if (key === 'type' && Array.isArray(out[key])) {
      // find the first that's not `null`.
      out[key] = out[key].find((t) => t !== 'null');
    }
  }
  return out;
}

/**
 * Processes the streaming body of a Response object. It decodes the stream as
 * UTF-8 text, parses JSON objects from specially formatted lines (e.g., "data: {}"),
 * and returns both an async generator for individual responses and a promise
 * that resolves to the aggregated final response.
 *
 * @param response The Response object with a streaming body.
 * @returns An object containing:
 *  - stream: An AsyncGenerator yielding each GenerateContentResponse.
 *  - response: A Promise resolving to the aggregated GenerateContentResponse.
 */
export function processStream(response: Response): GenerateContentStreamResult {
  if (!response.body) {
    throw new Error('Error processing stream because response.body not found');
  }
  const inputStream = response.body.pipeThrough(
    new TextDecoderStream('utf8', { fatal: true })
  );
  const responseStream = getResponseStream(inputStream);
  const [stream1, stream2] = responseStream.tee();
  return {
    stream: generateResponseSequence(stream1),
    response: getResponsePromise(stream2),
  };
}

function getResponseStream(
  inputStream: ReadableStream<string>
): ReadableStream<GenerateContentResponse> {
  const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
  const reader = inputStream.getReader();
  const stream = new ReadableStream<GenerateContentResponse>({
    start(controller) {
      let currentText = '';
      return pump();
      function pump(): Promise<(() => Promise<void>) | undefined> {
        return reader
          .read()
          .then(({ value, done }) => {
            if (done) {
              if (currentText.trim()) {
                controller.error(new Error('Failed to parse stream'));
                return;
              }
              controller.close();
              return;
            }

            currentText += value;
            let match = currentText.match(responseLineRE);
            let parsedResponse: GenerateContentResponse;
            while (match) {
              try {
                parsedResponse = JSON.parse(match[1]);
              } catch (e) {
                controller.error(
                  new Error(`Error parsing JSON response: "${match[1]}"`)
                );
                return;
              }
              controller.enqueue(parsedResponse);
              currentText = currentText.substring(match[0].length);
              match = currentText.match(responseLineRE);
            }
            return pump();
          })
          .catch((e: Error) => {
            let err = e;
            err.stack = e.stack;
            if (err.name === 'AbortError') {
              err = new GenkitError({
                status: 'ABORTED',
                message: 'Request aborted when reading from the stream',
              });
            } else {
              err = new Error('Error reading from the stream');
            }
            throw err;
          });
      }
    },
  });
  return stream;
}

async function* generateResponseSequence(
  stream: ReadableStream<GenerateContentResponse>
): AsyncGenerator<GenerateContentResponse> {
  const reader = stream.getReader();
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    yield value;
  }
}

async function getResponsePromise(
  stream: ReadableStream<GenerateContentResponse>
): Promise<GenerateContentResponse> {
  const allResponses: GenerateContentResponse[] = [];
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      return aggregateResponses(allResponses);
    }
    allResponses.push(value);
  }
}

function handleFunctionCall(
  part: Part,
  newPart: Partial<Part>,
  activePartialToolRequest: Part | null
): {
  shouldContinue: boolean;
  newActivePartialToolRequest: Part | null;
} {
  // If there's an active partial tool request, we're in the middle of a stream.
  if (activePartialToolRequest) {
    if (part.functionCall?.partialArgs) {
      applyGeminiPartialArgs(
        activePartialToolRequest.functionCall!.args!,
        part.functionCall.partialArgs
      );
    }
    // If `willContinue` is false, this is the end of the stream.
    if (!part.functionCall!.willContinue) {
      newPart.thoughtSignature = activePartialToolRequest.thoughtSignature;
      part.functionCall = activePartialToolRequest.functionCall;
      delete part.functionCall!.willContinue;
      activePartialToolRequest = null;
    } else {
      // If `willContinue` is true, we're still in the middle of a stream.
      // This is a partial result, so we skip adding it to the parts list.
      return {
        shouldContinue: true,
        newActivePartialToolRequest: activePartialToolRequest,
      };
    }
    // If `willContinue` is true on a part and there's no active partial request,
    // this is the start of a new streaming tool call.
  } else if (part.functionCall!.willContinue) {
    activePartialToolRequest = {
      ...part,
      functionCall: {
        ...part.functionCall,
        args: part.functionCall!.args || {},
      },
    };
    if (part.functionCall?.partialArgs) {
      applyGeminiPartialArgs(
        activePartialToolRequest.functionCall!.args!,
        part.functionCall.partialArgs
      );
    }
    // This is the start of a partial, so we skip adding it to the parts list.
    return {
      shouldContinue: true,
      newActivePartialToolRequest: activePartialToolRequest,
    };
  }

  // If we're here, it's a regular, non-streaming tool call.
  newPart.functionCall = part.functionCall;
  return {
    shouldContinue: false,
    newActivePartialToolRequest: activePartialToolRequest,
  };
}

function aggregateResponses(
  responses: GenerateContentResponse[]
): GenerateContentResponse {
  const lastResponse = responses.at(-1);
  if (lastResponse === undefined) {
    throw new Error(
      'Error aggregating stream chunks because the final response in stream chunk is undefined'
    );
  }
  const aggregatedResponse: GenerateContentResponse = {};
  if (lastResponse.promptFeedback) {
    aggregatedResponse.promptFeedback = lastResponse.promptFeedback;
  }
  let activePartialToolRequest: Part | null = null;
  for (const response of responses) {
    for (const candidate of response.candidates ?? []) {
      const index = candidate.index ?? 0;
      if (!aggregatedResponse.candidates) {
        aggregatedResponse.candidates = [];
      }
      if (!aggregatedResponse.candidates[index]) {
        aggregatedResponse.candidates[index] = {
          index,
        } as GenerateContentCandidate;
      }
      const aggregatedCandidate = aggregatedResponse.candidates[index];
      aggregateMetadata(aggregatedCandidate, candidate, 'citationMetadata');
      aggregateMetadata(aggregatedCandidate, candidate, 'groundingMetadata');
      if (candidate.safetyRatings?.length) {
        aggregatedCandidate.safetyRatings = (
          aggregatedCandidate.safetyRatings ?? []
        ).concat(candidate.safetyRatings);
      }
      if (candidate.finishReason !== undefined) {
        aggregatedCandidate.finishReason = candidate.finishReason;
      }
      if (candidate.finishMessage !== undefined) {
        aggregatedCandidate.finishMessage = candidate.finishMessage;
      }

      if (candidate.avgLogprobs !== undefined) {
        aggregatedCandidate.avgLogprobs = candidate.avgLogprobs;
      }
      if (candidate.logprobsResult !== undefined) {
        aggregatedCandidate.logprobsResult = candidate.logprobsResult;
      }

      /**
       * Candidates should always have content and parts, but this handles
       * possible malformed responses.
       */
      if (candidate.content && candidate.content.parts) {
        if (!aggregatedCandidate.content) {
          aggregatedCandidate.content = {
            role: candidate.content.role || 'user',
            parts: [],
          };
        }

        for (const part of candidate.content.parts) {
          const newPart: Partial<Part> = {};

          // Instead of iterating over the keys in the Object, which might
          // Contain new, unsupported things, we iterate over the PART_KEYS
          // which is the list of Part keys we know about.
          // This keeps us in sync
          for (const key of PART_KEYS) {
            if (key === 'functionCall') {
              // shouldContinue in the functionCall logic below applies to the
              // top level for loop, not this nested one.
              continue;
            } else if (key === 'text') {
              if (typeof part.text === 'string') {
                newPart.text = part.text;
              }
            } else if (part[key]) {
              // Essentially newPart[key] = part[key] with type safety.
              Object.assign(newPart, { [key]: part[key] });
            }
          }

          if (part.functionCall) {
            // function calls are special, there can be partials, so we need aggregate
            // the partials into final functionCall.
            const { shouldContinue, newActivePartialToolRequest } =
              handleFunctionCall(part, newPart, activePartialToolRequest);
            if (shouldContinue) {
              activePartialToolRequest = newActivePartialToolRequest;
              continue;
            }
            activePartialToolRequest = newActivePartialToolRequest;
          }

          if (Object.keys(newPart).length === 0) {
            newPart.text = '';
          }
          aggregatedCandidate.content.parts.push(newPart as Part);
        }
      }
    }
    if (response.usageMetadata) {
      aggregatedResponse.usageMetadata = response.usageMetadata;
    }
  }
  return aggregatedResponse;
}

function aggregateMetadata<K extends keyof GenerateContentCandidate>(
  aggCandidate: GenerateContentCandidate,
  chunkCandidate: GenerateContentCandidate,
  fieldName: K
) {
  const chunkObj = chunkCandidate[fieldName];
  const aggObj = aggCandidate[fieldName];
  if (chunkObj === undefined) return; // Nothing to do

  if (aggObj === undefined) {
    aggCandidate[fieldName] = chunkObj;
    return;
  }

  if (isObject(chunkObj)) {
    for (const k of Object.keys(chunkObj)) {
      if (Array.isArray(aggObj[k]) && Array.isArray(chunkObj[k])) {
        aggObj[k] = aggObj[k].concat(chunkObj[k]);
      } else {
        // last one wins, also handles only one being an array.
        aggObj[k] = chunkObj[k] ?? aggObj[k];
      }
    }
  }
}

export function getGenkitClientHeader() {
  if (process.env.MONOSPACE_ENV == 'true') {
    return defaultGetClientHeader() + ' firebase-studio-vm';
  }
  return defaultGetClientHeader();
}

export function isKnownKey<T extends object>(
  key: string | number | symbol,
  obj: T
): key is keyof T {
  return key in obj;
}

export const TEST_ONLY = { aggregateResponses };
