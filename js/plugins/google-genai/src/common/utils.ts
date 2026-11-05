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
  StatusName,
  StatusNameSchema,
  getClientHeader as defaultGetClientHeader,
  z,
} from 'genkit';
import { logger } from 'genkit/logging';
import { GenerateRequest } from 'genkit/model';
import { applyGeminiPartialArgs } from './converters.js';
import {
  GeminiInteraction,
  InteractionSseEvent,
  InteractionStreamResult,
  Step,
  TextContent,
} from './interaction-types.js';
import {
  GenerateContentCandidate,
  GenerateContentResponse,
  GenerateContentStreamResult,
  PART_KEYS,
  Part,
  isObject,
} from './types.js';

/**
 * `Api-Revision` header value for Interactions API requests. It selects the
 * steps-based schema (`steps` in responses, step list input) that the
 * Interactions converters read and write; without it the server uses the
 * older `outputs` / turn list schema.
 */
export const INTERACTIONS_API_REVISION = '2026-05-20';

export function buildTraceMetadataInput(
  url: string,
  fetchOptions: RequestInit,
  traceOptions: {
    request?: unknown;
    model?: string;
    clientOptions?: { timeout?: number };
  }
): Record<string, unknown> {
  const safeHeaders = { ...(fetchOptions.headers as Record<string, string>) };

  const redactString = (str: string | undefined): string | undefined => {
    if (!str) return str;
    return `<REDACTED> (${str.length} characters)`;
  };

  if (safeHeaders['x-goog-api-key']) {
    safeHeaders['x-goog-api-key'] = redactString(
      safeHeaders['x-goog-api-key']
    )!;
  }
  if (safeHeaders['Authorization']) {
    safeHeaders['Authorization'] = redactString(safeHeaders['Authorization'])!;
  }

  const safeOptions: any = {};
  if (traceOptions.clientOptions?.timeout) {
    safeOptions.timeout = traceOptions.clientOptions.timeout;
  }

  return {
    apiEndpoint: url,
    request: traceOptions.request,
    headers: safeHeaders,
    ...(Object.keys(safeOptions).length > 0 ? { options: safeOptions } : {}),
    ...(traceOptions.model ? { model: traceOptions.model } : {}),
  };
}

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
    if (typeof out[key] === 'object' && out[key] !== null) {
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

// ---------------------------------------------------------------------------
// Interactions API stream processing (Interactions-only).
//
// Kept separate from the generateContent stream processing below so that the
// generateContent path can be removed cleanly once those models are retired.
// ---------------------------------------------------------------------------

/**
 * Interactions API content-policy codes. These are reported in two ways:
 *  - prompt blocked before execution: HTTP 400 with `error.code` set;
 *  - response blocked after execution: HTTP 200, `status: 'failed'`, with the
 *    code in `interaction.errors[]`.
 *
 * In the second case the plugin returns `finishReason: 'blocked'` (matching
 * the generateContent path) rather than throwing. When thrown (first case) they
 * map to `FAILED_PRECONDITION`, which is not retryable: the same input will be
 * blocked again.
 */
const INTERACTION_CONTENT_BLOCK_CODES = new Set([
  'safety',
  'recitation',
  'language',
  'prohibited_content',
  'spii',
  'blocklist',
  'image_safety',
  'image_prohibited_content',
  'image_recitation',
  'image_other',
  'content_blocked',
  'jailbreak',
  'model_armor',
]);

/**
 * Reports whether an Interactions API error code is a content-policy block
 * (safety, recitation, blocklist, etc.).
 *
 * @param code The `code` field from an Interactions API error.
 */
export function isInteractionContentBlockCode(code: unknown): boolean {
  return (
    typeof code === 'string' &&
    INTERACTION_CONTENT_BLOCK_CODES.has(code.trim().toLowerCase())
  );
}

/**
 * Interactions API error codes mapped to Genkit statuses.
 *
 * The Interactions API reports errors as `{ code: string, message: string }`.
 * The public spec documents `code` only as "a URI that identifies the error
 * type" and does not enumerate values; this table is the full set of codes the
 * API returns (with the HTTP status the API pairs each one with).
 */
const INTERACTION_ERROR_CODE_TO_STATUS: Record<string, StatusName> = {
  // Generation failures reported via `status: 'failed'` + `errors[]` after the
  // server's own internal retries are exhausted. The server message advises
  // retrying, so these map to ABORTED ("retry at a higher level"), which the
  // retry middleware retries by default.
  malformed_function_call: 'ABORTED',
  unexpected_tool_call: 'ABORTED',
  no_image: 'ABORTED',
  // Content-policy blocks; see INTERACTION_CONTENT_BLOCK_CODES.
  ...Object.fromEntries(
    [...INTERACTION_CONTENT_BLOCK_CODES].map((code): [string, StatusName] => [
      code,
      'FAILED_PRECONDITION',
    ])
  ),
  service_unavailable: 'UNAVAILABLE', // 503: server or model overloaded
  rate_limit_exceeded: 'RESOURCE_EXHAUSTED', // 429: RPM/TPM limits
  quota_exceeded: 'RESOURCE_EXHAUSTED', // 429: RPD/daily limits
  invalid_request: 'INVALID_ARGUMENT', // 400: general bad request
  parameter_unknown: 'INVALID_ARGUMENT', // 400: unrecognized field
  failed_precondition: 'FAILED_PRECONDITION', // 400
  out_of_range: 'OUT_OF_RANGE', // 400
  // 400: exceeded the agent's max token limit. Deterministic for a given
  // request, so deliberately not RESOURCE_EXHAUSTED (which is retryable).
  agent_max_token_limit: 'INVALID_ARGUMENT',
  model_not_found: 'NOT_FOUND', // 404
  not_found: 'NOT_FOUND', // 404
  authentication: 'UNAUTHENTICATED', // 401
  permission_denied: 'PERMISSION_DENIED', // 403
  already_exists: 'ALREADY_EXISTS', // 409
  aborted: 'ABORTED', // 409
  cancelled: 'CANCELLED', // 499
  deadline_exceeded: 'DEADLINE_EXCEEDED', // 504
  api_error: 'INTERNAL', // 500
  unimplemented: 'UNIMPLEMENTED', // 501
};

/**
 * Maps an Interactions API error `code` to a Genkit `StatusName`.
 *
 * Looks the code up (case-insensitively) in the known Interactions error codes.
 * Also accepts numeric HTTP codes (e.g. `503` or `"503"`), in case the front
 * end returns a Google-style error body instead.
 *
 * @param code The `code` field from an Interactions API error.
 * @returns The matching `StatusName`, or `undefined` if it cannot be mapped.
 */
export function interactionErrorCodeToGenkitStatus(
  code: unknown
): StatusName | undefined {
  if (typeof code === 'number') {
    const status = httpStatusToGenkitStatus(code);
    return status === 'UNKNOWN' ? undefined : status;
  }
  if (typeof code !== 'string' || !code.trim()) {
    return undefined;
  }
  const normalized = code.trim().toLowerCase();
  if (/^\d+$/.test(normalized)) {
    return interactionErrorCodeToGenkitStatus(Number(normalized));
  }
  return Object.prototype.hasOwnProperty.call(
    INTERACTION_ERROR_CODE_TO_STATUS,
    normalized
  )
    ? INTERACTION_ERROR_CODE_TO_STATUS[normalized]
    : undefined;
}

/**
 * Builds an error for leftover, unparsed Interactions stream text. When the
 * model is overloaded (or otherwise fails), the API may return a plain JSON
 * error body (e.g. `{"error":{"code":"service_unavailable","message":"..."}}`)
 * instead of SSE `data:` frames. Surfacing a `GenkitError` with the correct
 * status lets downstream middleware (e.g. retry) react appropriately.
 *
 * @param text The leftover text that could not be parsed as an SSE event.
 * @returns A `GenkitError` if the text is a recognizable API error body,
 *  otherwise a generic `Error`.
 */
export function parseInteractionStreamErrorText(text: string): Error {
  try {
    const json = JSON.parse(text);
    const apiError = json?.error;
    if (
      apiError &&
      typeof apiError === 'object' &&
      (apiError.code || apiError.status)
    ) {
      // Prefer an explicit Google-style `status` if present, otherwise map the
      // Interactions `code`.
      const parsedStatus = StatusNameSchema.safeParse(apiError.status);
      const status: StatusName = parsedStatus.success
        ? parsedStatus.data
        : (interactionErrorCodeToGenkitStatus(apiError.code) ?? 'UNKNOWN');
      const message =
        typeof apiError.message === 'string'
          ? apiError.message
          : 'Error streaming from the model';
      return new GenkitError({
        status,
        message,
        detail: json,
      });
    }
  } catch (e) {
    // Not JSON or not a recognizable error body, fall through to generic error.
  }
  // Truncate to avoid memory/log bloat from large non-JSON payloads.
  const truncatedText =
    text.length > 500 ? text.substring(0, 500) + '...' : text;
  return new Error('Failed to parse stream: ' + truncatedText);
}

export function interactionProcessStream(
  response: Response
): InteractionStreamResult {
  if (!response.body) {
    throw new Error('Error processing stream because response.body not found');
  }
  const inputStream = response.body.pipeThrough(
    new TextDecoderStream('utf8', { fatal: true })
  );
  const responseStream = getInteractionResponseStream(inputStream);
  const [stream1, stream2] = responseStream.tee();
  const responsePromise = getInteractionResponsePromise(stream2);
  // See processStream: avoid an unhandled rejection if the caller only
  // consumes `stream` and it errors before `response` is ever awaited.
  responsePromise.catch(() => {});
  return {
    stream: generateInteractionResponseSequence(stream1),
    response: responsePromise,
  };
}

/**
 * Reports whether a parsed SSE payload is an Interactions stream event, i.e.
 * an object with a string `event_type`. Event payloads are not otherwise
 * validated; their fields are read defensively where they are used.
 */
function isInteractionSseEvent(value: unknown): value is InteractionSseEvent {
  return isObject(value) && typeof value.event_type === 'string';
}

function getInteractionResponseStream(
  inputStream: ReadableStream<string>
): ReadableStream<InteractionSseEvent> {
  const reader = inputStream.getReader();
  const stream = new ReadableStream<InteractionSseEvent>({
    start(controller) {
      let currentText = '';
      return pump();
      function pump(): Promise<(() => Promise<void>) | undefined> {
        return reader
          .read()
          .then(({ value, done }) => {
            if (done) {
              reader.releaseLock();
              if (currentText.trim()) {
                controller.error(parseInteractionStreamErrorText(currentText));
                return;
              }
              controller.close();
              return;
            }

            currentText += value;

            while (true) {
              const doubleNewline = currentText.indexOf('\n\n');
              const doubleReturn = currentText.indexOf('\r\r');
              const doubleReturnNewline = currentText.indexOf('\r\n\r\n');

              let endIndex = -1;
              let skip = 2;
              if (
                doubleReturnNewline !== -1 &&
                (endIndex === -1 || doubleReturnNewline < endIndex)
              ) {
                endIndex = doubleReturnNewline;
                skip = 4;
              }
              if (
                doubleNewline !== -1 &&
                (endIndex === -1 || doubleNewline < endIndex)
              ) {
                endIndex = doubleNewline;
                skip = 2;
              }
              if (
                doubleReturn !== -1 &&
                (endIndex === -1 || doubleReturn < endIndex)
              ) {
                endIndex = doubleReturn;
                skip = 2;
              }

              if (endIndex === -1) {
                break; // Need more data
              }

              const block = currentText.substring(0, endIndex);
              currentText = currentText.substring(endIndex + skip);

              const lines = block.split(/\r\n|\r|\n/);
              let dataText = '';

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  dataText += line.substring(6);
                } else if (line.startsWith('data:')) {
                  dataText += line.substring(5);
                }
              }

              if (dataText === '[DONE]') {
                continue;
              }

              if (dataText) {
                let parsed: unknown;
                try {
                  parsed = JSON.parse(dataText);
                } catch {
                  reader.releaseLock();
                  controller.error(
                    new Error(`Error parsing JSON response: "${dataText}"`)
                  );
                  return;
                }
                if (isInteractionSseEvent(parsed)) {
                  controller.enqueue(parsed);
                }
                // Unknown payloads (no `event_type`) are skipped, as the API
                // docs recommend for unrecognized events.
              }
            }
            return pump();
          })
          .catch((e: unknown) => {
            reader.releaseLock();
            if (e instanceof Error && e.name === 'AbortError') {
              throw new GenkitError({
                status: 'ABORTED',
                message: 'Request aborted when reading from the stream',
              });
            }
            // Keep the original error (e.g. a dropped connection) so the
            // failure can be diagnosed.
            const detail = e instanceof Error ? e.message : String(e);
            throw new Error(`Error reading from the stream: ${detail}`, {
              cause: e,
            });
          });
      }
    },
    cancel() {
      reader.cancel().catch(() => {});
    },
  });
  return stream;
}

async function* generateInteractionResponseSequence(
  stream: ReadableStream<InteractionSseEvent>
): AsyncGenerator<InteractionSseEvent> {
  const reader = stream.getReader();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}

async function getInteractionResponsePromise(
  stream: ReadableStream<InteractionSseEvent>
): Promise<GeminiInteraction> {
  let interaction: GeminiInteraction = {};
  const partialArgumentsMap = new Map<number, string>();
  const reader = stream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        return interaction;
      }

      if (value.event_type === 'error') {
        throw new GenkitError({
          // Map the error code so e.g. quota/overload errors get a retryable
          // status; fall back to INTERNAL for unrecognized codes.
          status:
            interactionErrorCodeToGenkitStatus(value.error?.code) ?? 'INTERNAL',
          message: `Interaction API returned an error: [${value.error?.code}] ${value.error?.message}`,
          detail: value,
        });
      }

      if (
        value.event_type === 'interaction.created' ||
        value.event_type === 'interaction.completed'
      ) {
        // Merge metadata (id, status, usage, errors, ...) only. Steps are
        // assembled from step.* events, and these events may carry partial or
        // no steps, which must not overwrite them. Use their steps only if
        // none were streamed.
        const { steps, ...rest } = value.interaction;
        Object.assign(interaction, rest);
        if (steps?.length && !interaction.steps?.length) {
          interaction.steps = steps;
        }
      } else if (value.event_type === 'interaction.status_update') {
        interaction.status = value.status;
      } else if (
        value.event_type === 'step.start' ||
        value.event_type === 'step.delta' ||
        value.event_type === 'step.stop'
      ) {
        if (!interaction.steps) interaction.steps = [];
        applyInteractionStepEvent(
          interaction.steps,
          partialArgumentsMap,
          value
        );
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Applies a `step.start`, `step.delta` or `step.stop` event to the steps
 * assembled so far (other events are ignored). Used for the final response
 * and for stream chunks (`createInteractionChunkConverter`), so both see the
 * same steps.
 *
 * @param steps The steps so far, by index; updated in place.
 * @param partialArguments Function call arguments received so far, by step
 *   index; parsed when the step stops.
 * @param event The stream event.
 */
export function applyInteractionStepEvent(
  steps: Step[],
  partialArguments: Map<number, string>,
  event: InteractionSseEvent
): void {
  if (event.event_type === 'step.start') {
    // A copy: both readers of the stream get the same event objects, and each
    // assembles its own steps.
    steps[event.index] = structuredClone(event.step);
  } else if (event.event_type === 'step.delta') {
    const step = steps[event.index];
    if (step) {
      if (
        step.type === 'model_output' ||
        step.type === 'user_input' ||
        step.type === 'thought'
      ) {
        const contentArray =
          step.type === 'thought' ? step.summary : step.content;
        if (!contentArray) {
          if (step.type === 'thought') step.summary = [];
          else step.content = [];
        }
        const arr = (step.type === 'thought' ? step.summary : step.content)!;

        if (event.delta.type === 'text') {
          // Append to the previous text block so a streamed answer becomes
          // a single block, matching the non-streamed response. This also
          // keeps annotation start/end indices meaningful.
          const last = arr[arr.length - 1];
          if (last?.type === 'text') {
            last.text = (last.text ?? '') + event.delta.text;
          } else {
            arr.push({ type: 'text', text: event.delta.text });
          }
        } else if (
          event.delta.type === 'image' ||
          event.delta.type === 'audio' ||
          event.delta.type === 'video' ||
          event.delta.type === 'document'
        ) {
          // Media deltas have the same shape as the matching content block.
          arr.push({ ...event.delta });
        } else if (event.delta.type === 'text_annotation_delta') {
          if (event.delta.annotations?.length) {
            let target = [...arr]
              .reverse()
              .find((c): c is TextContent => c.type === 'text');
            if (!target) {
              target = { type: 'text', text: '' };
              arr.push(target);
            }
            target.annotations = [
              ...(target.annotations ?? []),
              ...event.delta.annotations,
            ];
          }
        } else if (
          event.delta.type === 'thought_summary' &&
          event.delta.content
        ) {
          // A copy, like the step itself (see step.start).
          arr.push({ ...event.delta.content });
        } else if (event.delta.type === 'thought_signature') {
          if (step.type === 'thought') {
            step.signature = event.delta.signature;
          }
        }
      } else if (
        step.type === 'function_call' &&
        event.delta.type === 'arguments_delta'
      ) {
        const existing = partialArguments.get(event.index) || '';
        partialArguments.set(
          event.index,
          existing + (event.delta.arguments || '')
        );
      } else if (event.delta.type === step.type) {
        // Built-in tool steps (google_search_call/result,
        // code_execution_call/result, url_context_*, file_search_*,
        // google_maps_*, mcp_server_tool_*, function_result, ...): step.start
        // carries only ids and an empty signature; the arguments, result
        // and real signature arrive in a delta of the same type. Copy those
        // fields onto the step.
        const { type: _type, ...fields } = event.delta;
        Object.assign(
          step,
          Object.fromEntries(
            Object.entries(fields).filter(([, v]) => v !== undefined)
          )
        );
      }
    }
  } else if (event.event_type === 'step.stop') {
    // If we had partial arguments for this step, parse them and update the function call arguments
    if (partialArguments.has(event.index)) {
      const step = steps[event.index];
      if (step && step.type === 'function_call') {
        try {
          const argStr = partialArguments.get(event.index);
          if (argStr) {
            step.arguments = JSON.parse(argStr);
          }
        } catch (e) {
          logger.warn(
            'Failed to parse partial arguments JSON for function call:',
            e
          );
        }
      }
      partialArguments.delete(event.index);
    }
  }
}

/**
 * Maps an HTTP status code to the corresponding Genkit `StatusName`.
 *
 * @param code The HTTP status code.
 * @returns The matching `StatusName`, or `'UNKNOWN'` if there is no mapping.
 */
export function httpStatusToGenkitStatus(code?: number): StatusName {
  switch (code) {
    case 400:
      return 'INVALID_ARGUMENT';
    case 401:
      return 'UNAUTHENTICATED';
    case 403:
      return 'PERMISSION_DENIED';
    case 404:
      return 'NOT_FOUND';
    case 429:
      return 'RESOURCE_EXHAUSTED';
    case 499:
      return 'CANCELLED';
    case 500:
      return 'INTERNAL';
    case 503:
      return 'UNAVAILABLE';
    case 504:
      return 'DEADLINE_EXCEEDED';
    default:
      return 'UNKNOWN';
  }
}

/** Reports whether every element of an array is a string. */
export function isStringArray(values: unknown[]): values is string[] {
  return values.every((v) => typeof v === 'string');
}

/** Hints for config fields the Interactions API doesn't support. */
const DROPPED_CONFIG_FIELD_HINTS: Record<string, string> = {
  'thinkingConfig.thinkingBudget':
    " Use thinkingConfig.thinkingLevel ('LOW' | 'MEDIUM' | 'HIGH') instead.",
};

/** `model:field` pairs that have already been warned about. */
const warnedDroppedConfigFields = new Set<string>();

/**
 * Warns (once per model and field) that a config field isn't supported by the
 * model and was ignored.
 */
export function warnDroppedConfigField(model: string, field: string) {
  const key = `${model}:${field}`;
  if (warnedDroppedConfigFields.has(key)) return;
  warnedDroppedConfigFields.add(key);
  logger.warn(
    `${field} is not supported by '${model}' and was ignored.` +
      (DROPPED_CONFIG_FIELD_HINTS[field] ?? '')
  );
}

/**
 * Reports whether a URL targets the Interactions API, i.e. its path contains
 * an `interactions` segment.
 */
export function isInteractionUrl(url: string): boolean {
  let pathname: string;
  try {
    pathname = new URL(url).pathname;
  } catch {
    pathname = url.split('?')[0];
  }
  return pathname.split('/').includes('interactions');
}

/**
 * Builds an error for leftover, unparsed stream text. When the model API is
 * overloaded (or otherwise fails), it may return HTTP 200 with a plain JSON
 * error body instead of SSE `data:` frames. In that case we surface a proper
 * `GenkitError` with the correct status so downstream middleware (e.g. retry)
 * can react appropriately, rather than a generic parse error.
 *
 * @param text The leftover text that could not be parsed as a stream chunk.
 * @returns A `GenkitError` if the text is a recognizable API error body,
 *  otherwise a generic `Error`.
 */
export function parseStreamErrorText(text: string): Error {
  try {
    const json = JSON.parse(text);
    const apiError = json?.error;
    if (
      apiError &&
      typeof apiError === 'object' &&
      (apiError.code || apiError.status)
    ) {
      // Coerce `code` to a number so a stringified code (e.g. "503") still maps.
      const rawCode = Number(apiError.code);
      const status: StatusName = StatusNameSchema.safeParse(apiError.status)
        .success
        ? (apiError.status as StatusName)
        : httpStatusToGenkitStatus(isNaN(rawCode) ? undefined : rawCode);
      const message =
        typeof apiError.message === 'string'
          ? apiError.message
          : 'Error streaming from the model';
      return new GenkitError({
        status,
        message,
        detail: json,
      });
    }
  } catch (e) {
    // Not JSON or not a recognizable error body, fall through to generic error.
  }
  // Truncate to avoid memory/log bloat from large non-JSON payloads (e.g. an
  // HTML error page from an upstream proxy).
  const truncatedText =
    text.length > 500 ? text.substring(0, 500) + '...' : text;
  return new Error('Failed to parse stream: ' + truncatedText);
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
  const responsePromise = getResponsePromise(stream2);
  // Attach a no-op catch so that if the caller only consumes `stream` and it
  // errors before `response` is ever awaited, the teed response promise does
  // not become an unhandled rejection (which would crash the process). This
  // does not swallow the rejection for a real awaiter of `response`.
  responsePromise.catch(() => {});
  return {
    stream: generateResponseSequence(stream1),
    response: responsePromise,
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
              reader.releaseLock();
              if (currentText.trim()) {
                controller.error(parseStreamErrorText(currentText));
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
                reader.releaseLock();
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
            reader.releaseLock();
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
    cancel() {
      // Catch prevents unhandled promise rejection if the stream was already cleanly finalized
      reader.cancel().catch(() => {});
    },
  });
  return stream;
}

async function* generateResponseSequence(
  stream: ReadableStream<GenerateContentResponse>
): AsyncGenerator<GenerateContentResponse> {
  const reader = stream.getReader();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}

async function getResponsePromise(
  stream: ReadableStream<GenerateContentResponse>
): Promise<GenerateContentResponse> {
  const allResponses: GenerateContentResponse[] = [];
  const reader = stream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        return aggregateResponses(allResponses);
      }
      allResponses.push(value);
    }
  } finally {
    reader.releaseLock();
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

/**
 * Copies headers into a plain object, leaving out the named headers in any
 * letter case. Header names are case-insensitive, but object keys aren't:
 * without this, a custom `api-revision` would sit next to the plugin's
 * `Api-Revision` and fetch would send both values.
 *
 * @param headers The headers to copy (e.g. user-provided custom headers).
 * @param names The header names to leave out.
 */
export function withoutHeaders(
  headers: Record<string, string> | Headers | undefined,
  names: string[]
): Record<string, string> {
  const excluded = new Set(names.map((n) => n.toLowerCase()));
  const result: Record<string, string> = {};
  const add = (value: string, key: string) => {
    if (!excluded.has(key.toLowerCase())) {
      result[key] = value;
    }
  };
  if (headers instanceof Headers) {
    headers.forEach(add);
  } else if (headers) {
    for (const [key, value] of Object.entries(headers)) {
      add(value, key);
    }
  }
  return result;
}

export function isKnownKey<T extends object>(
  key: string | number | symbol,
  obj: T
): key is keyof T {
  return key in obj;
}

/**
 * Parses the value of a `Retry-After` HTTP header into milliseconds.
 * Supports both delay-seconds (e.g. "60") and HTTP-date formats
 * (e.g. "Mon, 19 May 2026 12:00:00 GMT") per RFC 7231 §7.1.3.
 *
 * @param value The raw Retry-After header value.
 * @returns The delay in milliseconds, or undefined if the value cannot be parsed.
 */
export function parseRetryAfterMs(value: string): number | undefined {
  if (!value || !value.trim()) {
    return undefined;
  }
  // Try as delay-seconds (e.g., "60")
  const seconds = Number(value);
  if (!isNaN(seconds) && seconds >= 0) {
    return seconds * 1000;
  }
  // Try as HTTP-date (e.g., "Mon, 19 May 2026 12:00:00 GMT")
  const date = new Date(value);
  if (!isNaN(date.getTime())) {
    return Math.max(0, date.getTime() - Date.now());
  }
  return undefined;
}

/**
 * Converts a string in camelCase to lowercase snake_case.
 */
export function camelToSnakeCase(str: string): string {
  return str.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

/**
 * Converts a string in snake_case to camelCase.
 */
export function snakeToCamelCase(str: string): string {
  return str.replace(/_([a-z0-9])/g, (_match, letter: string) =>
    letter.toUpperCase()
  );
}

export function convertObjectKeysToSnakeCase(obj: Array<unknown>): unknown[];
export function convertObjectKeysToSnakeCase(
  obj: Record<string, unknown>
): Record<string, unknown>;
export function convertObjectKeysToSnakeCase(obj: unknown): unknown;

export function convertObjectKeysToSnakeCase(obj: unknown): unknown {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => convertObjectKeysToSnakeCase(item));
  }

  const newObj: Record<string, unknown> = {};
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      const snakeCaseKey = camelToSnakeCase(key);
      newObj[snakeCaseKey] = convertObjectKeysToSnakeCase(obj[key]);
    }
  }
  return newObj;
}

export function toSnakeCaseObj(val: unknown): Record<string, unknown> {
  const snakeVal = convertObjectKeysToSnakeCase(val);
  return isObject(snakeVal) ? snakeVal : {};
}

/**
 * Inserts raw audio bytes at the end of a WAV file's `data` chunk and updates
 * the RIFF and `data` sizes. Chunks after `data` (the server sends one) stay
 * after the audio. Returns `undefined` if `wav` isn't a WAV file with a
 * `data` chunk.
 */
export function appendToWavData(
  wav: Buffer,
  audio: Buffer
): Buffer | undefined {
  if (
    wav.length < 12 ||
    wav.toString('ascii', 0, 4) !== 'RIFF' ||
    wav.toString('ascii', 8, 12) !== 'WAVE'
  ) {
    return undefined;
  }
  // Walk the chunks: each is a 4-byte id, a 4-byte little-endian size, then
  // the data, padded to an even length.
  let offset = 12;
  while (offset + 8 <= wav.length) {
    const id = wav.toString('ascii', offset, offset + 4);
    const size = wav.readUInt32LE(offset + 4);
    const dataStart = offset + 8;
    if (id === 'data') {
      const dataEnd = Math.min(dataStart + size, wav.length);
      const result = Buffer.concat([
        wav.subarray(0, dataEnd),
        audio,
        wav.subarray(dataEnd),
      ]);
      result.writeUInt32LE(size + audio.length, offset + 4);
      result.writeUInt32LE(result.length - 8, 4);
      return result;
    }
    offset = dataStart + size + (size % 2);
  }
  return undefined;
}

/**
 * Returns `bytes` without any ID3v2 tags at the start. A tag is only removed
 * if its header is valid ("ID3", version 2-4, size bytes below 0x80) and the
 * whole tag fits in `bytes`; otherwise `bytes` is returned unchanged.
 */
export function stripLeadingId3Tags(bytes: Buffer): Buffer {
  let start = 0;
  while (start + 10 <= bytes.length) {
    const b = bytes.subarray(start);
    const isTag =
      b[0] === 0x49 && // I
      b[1] === 0x44 && // D
      b[2] === 0x33 && // 3
      b[3] >= 2 &&
      b[3] <= 4 &&
      b[4] !== 0xff &&
      b[6] < 0x80 &&
      b[7] < 0x80 &&
      b[8] < 0x80 &&
      b[9] < 0x80;
    if (!isTag) break;
    // The size is "syncsafe" (7 bits per byte) and excludes the 10-byte
    // header and the optional 10-byte footer (flag 0x10).
    const size = (b[6] << 21) | (b[7] << 14) | (b[8] << 7) | b[9];
    const tagLength = 10 + size + (b[5] & 0x10 ? 10 : 0);
    if (tagLength > b.length) break;
    start += tagLength;
  }
  return start === 0 ? bytes : bytes.subarray(start);
}

export const TEST_ONLY = { aggregateResponses };
