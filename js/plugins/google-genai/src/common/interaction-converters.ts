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

import {
  GenerateResponseData,
  GenerationUsage,
  GenkitError,
  MessageData,
  Operation,
  Part,
  z,
} from 'genkit';
import { ToolDefinition } from 'genkit/model';
import {
  AudioContent,
  BuiltInToolStep,
  CodeExecutionCallStep,
  CodeExecutionResultStep,
  Content,
  DocumentContent,
  FunctionCallContent,
  FunctionCallStep,
  FunctionResultContent,
  FunctionResultStep,
  GeminiInteraction,
  GoogleSearchCallStep,
  GoogleSearchResultStep,
  ImageContent,
  InteractionFileSearchTool,
  InteractionFunctionTool,
  InteractionGoogleSearchTool,
  InteractionSafetySetting,
  InteractionSseEvent,
  InteractionTool,
  InteractionToolChoiceConfig,
  ModelGenerationConfig,
  SpeechAnnotation,
  SpeechConfig,
  Step,
  StepDeltaData,
  TextAnnotationSchema,
  TextContent,
  ThoughtContent,
  ThoughtStep,
  Usage,
  VideoContent,
} from './interaction-types.js';
import { isObject } from './types.js';
import {
  appendToWavData,
  applyInteractionStepEvent,
  camelToSnakeCase,
  cleanSchema,
  convertObjectKeysToSnakeCase,
  extractMimeType,
  interactionErrorCodeToGenkitStatus,
  isInteractionContentBlockCode,
  snakeToCamelCase,
  stripLeadingId3Tags,
  toSnakeCaseObj,
} from './utils.js';

/**
 * Ensures that all tool requests and responses in a list of messages have unique reference IDs.
 *
 * This function performs two passes:
 * 1. Assigns generated IDs to tool requests that lack a `ref`.
 * 2. Assigns matching IDs to tool responses that lack a `ref`, assuming they correspond
 *    sequentially to the requests. If a response has no matching request, it gets an orphaned ID.
 *
 * @param messages - The list of messages to process.
 * @returns A deep copy of the messages with tool IDs ensured.
 */
export function ensureToolIds(messages: MessageData[]): MessageData[] {
  const generatedIds: string[] = [];
  let nextIdCounter = 0;

  // Deep copy to avoid mutating original request messages
  const newMessages = structuredClone(messages) as MessageData[];

  // First pass: find ToolRequests without ref
  for (const message of newMessages) {
    for (const part of message.content) {
      if (part.toolRequest && !part.toolRequest.ref) {
        const newId = `genkit-auto-id-${nextIdCounter++}`;
        part.toolRequest.ref = newId;
        generatedIds.push(newId);
      }
    }
  }

  // Second pass: find ToolResponses without ref and assign from queue
  // Note: This assumes responses are in the same order as requests.
  for (const message of newMessages) {
    for (const part of message.content) {
      if (part.toolResponse && !part.toolResponse.ref) {
        const id = generatedIds.shift();
        if (id) {
          part.toolResponse.ref = id;
        } else {
          // No matching request found (or queue empty).
          // Generate unique one to avoid empty string rejection.
          part.toolResponse.ref = `genkit-orphan-id-${nextIdCounter++}`;
        }
      }
    }
  }

  return newMessages;
}

/**
 * Converts a Genkit ToolDefinition to an InteractionTool format.
 *
 * Maps the name, description, and input schema (cleaned) to the interaction tool structure.
 *
 * @param tool - The Genkit tool definition.
 * @returns The converted InteractionTool.
 */
export function toInteractionTool(tool: ToolDefinition): InteractionTool {
  const func: InteractionFunctionTool = {
    type: 'function',
    name: tool.name,
    description: tool.description,
  };
  if (tool.inputSchema) {
    func.parameters = cleanSchema(tool.inputSchema);
  }
  return func;
}

export function toInteractionConfigTool(toolRaw: unknown): InteractionTool {
  if (!isObject(toolRaw)) {
    throw new Error(
      `Invalid tool configuration: Expected an object, got ${typeof toolRaw}`
    );
  }
  const tool = toolRaw;

  if ('googleSearch' in tool || 'google_search' in tool) {
    return toInteractionGoogleSearch(tool.googleSearch || tool.google_search);
  }
  if ('codeExecution' in tool || 'code_execution' in tool) {
    const config = tool.codeExecution || tool.code_execution;
    if (config === true || config === undefined) {
      return { type: 'code_execution' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for codeExecution tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'code_execution',
      ...toSnakeCaseObj(config),
    };
  }
  if ('fileSearch' in tool || 'file_search' in tool) {
    const config = tool.fileSearch || tool.file_search;
    if (config === true || config === undefined) {
      return { type: 'file_search' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for fileSearch tool: Expected object, got ${typeof config}`
      );
    }
    const result: InteractionFileSearchTool = { type: 'file_search' };

    const fileSearchStoreNames =
      config.fileSearchStoreNames || config.file_search_store_names;
    const restFileSearch = { ...config };
    delete restFileSearch.fileSearchStoreNames;
    delete restFileSearch.file_search_store_names;

    if (fileSearchStoreNames !== undefined) {
      if (
        !Array.isArray(fileSearchStoreNames) ||
        !fileSearchStoreNames.every((n) => typeof n === 'string')
      ) {
        throw new Error('fileSearchStoreNames must be an array of strings.');
      }
      result.file_search_store_names = fileSearchStoreNames;
    }
    return {
      ...result,
      ...toSnakeCaseObj(restFileSearch),
    };
  }
  if ('urlContext' in tool) {
    const config = tool.urlContext;
    if (config === true || config === undefined) {
      return { type: 'url_context' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for urlContext tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'url_context',
      ...toSnakeCaseObj(config),
    };
  }
  if ('googleMaps' in tool) {
    const config = tool.googleMaps;
    if (config === true || config === undefined) {
      return { type: 'google_maps' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for googleMaps tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'google_maps',
      ...toSnakeCaseObj(config),
    };
  }
  if ('computerUse' in tool) {
    const config = tool.computerUse;
    if (config === true || config === undefined) {
      return { type: 'computer_use' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for computerUse tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'computer_use',
      ...toSnakeCaseObj(config),
    };
  }
  if ('retrieval' in tool) {
    const config = tool.retrieval;
    if (config === true || config === undefined) {
      return { type: 'retrieval' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for retrieval tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'retrieval',
      ...toSnakeCaseObj(config),
    };
  }
  if ('mcpServer' in tool) {
    const config = tool.mcpServer;
    if (config === true || config === undefined) {
      return { type: 'mcp_server' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for mcpServer tool: Expected object or true, got ${typeof config}`
      );
    }
    const snakeConfig = toSnakeCaseObj(config);
    if (
      Array.isArray(snakeConfig.allowed_tools) &&
      snakeConfig.allowed_tools.length > 0 &&
      typeof snakeConfig.allowed_tools[0] === 'string'
    ) {
      snakeConfig.allowed_tools = [{ tools: snakeConfig.allowed_tools }];
    }
    return {
      type: 'mcp_server',
      ...snakeConfig,
    };
  }

  // Pass through any other tool that names its own `type`, in snake_case.
  const { type, ...rest } = toSnakeCaseObj(tool);
  if (typeof type !== 'string') {
    throw new Error(
      `Unsupported tool configuration: ${JSON.stringify(tool)}. ` +
        'Use a built-in tool key (e.g. googleSearch, codeExecution) or set `type`.'
    );
  }
  return { type, ...rest };
}

/**
 * Copies a `retrievalConfig.latLng` user location onto the Interactions
 * `google_maps` tools, which take `latitude`/`longitude` directly (there is no
 * retrieval config on the Interactions path). A location already set on a tool
 * is kept.
 *
 * @returns Whether there was a `google_maps` tool to apply the location to.
 */
export function applyLatLngToGoogleMapsTools(
  tools: InteractionTool[],
  latLng: { latitude?: number; longitude?: number }
): boolean {
  let found = false;
  tools.forEach((tool, i) => {
    if (tool.type !== 'google_maps') return;
    found = true;
    if ('latitude' in tool || 'longitude' in tool) return;
    tools[i] = {
      ...tool,
      ...(latLng.latitude !== undefined ? { latitude: latLng.latitude } : {}),
      ...(latLng.longitude !== undefined
        ? { longitude: latLng.longitude }
        : {}),
    };
  });
  return found;
}

/**
 * Picks the messages to send to the Interactions API and the interaction to
 * continue, if any.
 *
 * Without `store`, the server keeps no history, so every message is sent.
 * With `store`, the last model message's `metadata.interactionId` is
 * continued and only the newer messages are sent. An explicit
 * `previousInteractionId` (which requires `store`) overrides the metadata;
 * messages up to the last model message are still left out.
 *
 * @throws GenkitError (INVALID_ARGUMENT) if `previousInteractionId` is set
 *  without `store`.
 */
export function selectInteractionInput(
  messages: MessageData[],
  store: boolean,
  previousInteractionIdFromConfig?: string
): { messages: MessageData[]; previousInteractionId?: string } {
  if (previousInteractionIdFromConfig && !store) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'store must be true when previousInteractionId is set.',
    });
  }

  if (previousInteractionIdFromConfig) {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'model') {
        return {
          messages: messages.slice(i + 1),
          previousInteractionId: previousInteractionIdFromConfig,
        };
      }
    }
    return { messages, previousInteractionId: previousInteractionIdFromConfig };
  }

  if (store) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const prevId = messages[i]?.metadata?.interactionId;
      if (
        messages[i].role === 'model' &&
        prevId &&
        typeof prevId === 'string'
      ) {
        return {
          messages: messages.slice(i + 1),
          previousInteractionId: prevId,
        };
      }
    }
  }
  return { messages };
}

/**
 * Converts `functionCallingConfig` (or, without it, the request's
 * `toolChoice`) to the Interactions `generation_config.tool_choice`.
 * Returns `undefined` when neither sets anything.
 */
export function toInteractionToolChoice(
  functionCallingConfig?: { mode?: string; allowedFunctionNames?: string[] },
  toolChoice?: string
): InteractionToolChoiceConfig | undefined {
  if (functionCallingConfig) {
    const mode = functionCallingConfig.mode?.toLowerCase();
    const validMode = mode && mode !== 'mode_unspecified' ? mode : undefined;
    if (!validMode && !functionCallingConfig.allowedFunctionNames?.length) {
      return undefined;
    }
    return {
      allowed_tools: {
        ...(validMode ? { mode: validMode } : {}),
        ...(functionCallingConfig.allowedFunctionNames
          ? { tools: functionCallingConfig.allowedFunctionNames }
          : {}),
      },
    };
  }
  if (toolChoice) {
    return {
      allowed_tools: { mode: toolChoice === 'required' ? 'any' : toolChoice },
    };
  }
  return undefined;
}

export function toInteractionGoogleSearch(
  gs: boolean | unknown
): InteractionGoogleSearchTool {
  const result: InteractionGoogleSearchTool = { type: 'google_search' };

  if (gs === true || gs === undefined) {
    return result;
  }

  if (!isObject(gs)) {
    throw new Error(
      `Invalid configuration for googleSearch tool: Expected object or true, got ${typeof gs}`
    );
  }

  const searchTypesObj = gs.searchTypes || gs.search_types;
  if (searchTypesObj !== undefined) {
    const searchTypes = new Set<string>();

    if (Array.isArray(searchTypesObj)) {
      for (const type of searchTypesObj) {
        if (typeof type === 'string') {
          if (type === 'webSearch' || type === 'web_search') {
            searchTypes.add('web_search');
          } else if (type === 'imageSearch' || type === 'image_search') {
            searchTypes.add('image_search');
          } else if (
            type === 'enterpriseWebSearch' ||
            type === 'enterprise_web_search'
          ) {
            searchTypes.add('enterprise_web_search');
          } else {
            // Passthrough for any unknown string elements
            searchTypes.add(camelToSnakeCase(type));
          }
        } else {
          throw new Error(
            `Invalid search type: Expected string, got ${typeof type}`
          );
        }
      }
    } else if (isObject(searchTypesObj)) {
      for (const [key, value] of Object.entries(searchTypesObj)) {
        if (value) {
          if (key === 'webSearch' || key === 'web_search') {
            searchTypes.add('web_search');
          } else if (key === 'imageSearch' || key === 'image_search') {
            searchTypes.add('image_search');
          } else if (
            key === 'enterpriseWebSearch' ||
            key === 'enterprise_web_search'
          ) {
            searchTypes.add('enterprise_web_search');
          } else {
            // Passthrough for any unknown properties
            searchTypes.add(camelToSnakeCase(key));
          }
        }
      }
    } else {
      throw new Error(
        `Invalid searchTypes configuration: Expected array or object, got ${typeof searchTypesObj}`
      );
    }

    if (searchTypes.size > 0) {
      result.search_types = Array.from(searchTypes);
    }
  }

  // Handle any other properties on gs (passthrough)
  const restConfig = { ...gs };
  delete restConfig.searchTypes;
  delete restConfig.search_types;

  Object.assign(result, toSnakeCaseObj(restConfig));

  return result;
}

/**
 * Converts generateContent-style safety settings to the Interactions API
 * shape: `type` without the `HARM_CATEGORY_` prefix, and lower-case values
 * (`HARM_CATEGORY_HATE_SPEECH` / `BLOCK_LOW_AND_ABOVE` -> `hate_speech` /
 * `block_low_and_above`). The server rejects the upper-case values with a
 * 400. `method` (`SEVERITY` / `PROBABILITY`) is lower-cased too.
 * `HARM_CATEGORY_UNSPECIFIED` has no equivalent and is left out. Other
 * fields are passed through.
 */
export function toInteractionSafetySettings(
  settings?: { category: string; threshold: string; method?: unknown }[]
): InteractionSafetySetting[] | undefined {
  if (!settings) return undefined;
  const result = settings
    .filter((s) => s.category !== 'HARM_CATEGORY_UNSPECIFIED')
    .map(({ category, threshold, method, ...rest }) => ({
      ...toSnakeCaseObj(rest),
      type: category.replace(/^HARM_CATEGORY_/, '').toLowerCase(),
      threshold: threshold.toLowerCase(),
      ...(typeof method === 'string' ? { method: method.toLowerCase() } : {}),
    }));
  return result.length ? result : undefined;
}

/**
 * Converts a Genkit/generateContent-style `speechConfig` to the Interactions
 * API `speech_config`.
 *
 * - `voiceConfig.prebuiltVoiceConfig.voiceName` becomes `[{ voice }]`.
 * - `multiSpeakerVoiceConfig.speakerVoiceConfigs[]` becomes
 *   `{ speakers: [{ speaker, voice }, ...] }`. Each text part must then name
 *   its speaker via `metadata.speechMetadata.speaker`.
 * - A top-level `languageCode` is applied to every entry as `language`.
 *
 * Returns `undefined` if no voice or language is set.
 */
export function toInteractionSpeechConfig(
  speechConfig: unknown
): ModelGenerationConfig['speech_config'] {
  if (!isObject(speechConfig)) {
    return undefined;
  }

  const language =
    typeof speechConfig.languageCode === 'string'
      ? speechConfig.languageCode
      : undefined;
  const voiceNameOf = (voiceConfig: unknown): string | undefined => {
    if (!isObject(voiceConfig) || !isObject(voiceConfig.prebuiltVoiceConfig)) {
      return undefined;
    }
    const voiceName = voiceConfig.prebuiltVoiceConfig.voiceName;
    return typeof voiceName === 'string' ? voiceName : undefined;
  };
  const entry = (voice?: string, speaker?: string): SpeechConfig => ({
    ...(voice ? { voice } : {}),
    ...(language ? { language } : {}),
    ...(speaker ? { speaker } : {}),
  });

  const multi = speechConfig.multiSpeakerVoiceConfig;
  if (isObject(multi) && Array.isArray(multi.speakerVoiceConfigs)) {
    return {
      speakers: multi.speakerVoiceConfigs
        .filter(isObject)
        .map((s) =>
          entry(
            voiceNameOf(s.voiceConfig),
            typeof s.speaker === 'string' ? s.speaker : undefined
          )
        ),
    };
  }

  const voice = voiceNameOf(speechConfig.voiceConfig);
  if (voice || language) {
    return [entry(voice)];
  }
  return undefined;
}

/**
 * Converts the remaining request config into the Interactions
 * `generation_config`.
 *
 * The API rejects unknown `generation_config` fields with a 400, so config
 * fields this plugin defines are mapped explicitly:
 *  - `speechConfig` is converted to `speech_config`;
 *  - `thinkingConfig.thinkingLevel` / `includeThoughts` become
 *    `thinking_level` / `thinking_summaries`. There is no `thinking_config`,
 *    so other thinking fields (e.g. `thinkingBudget`) are dropped;
 *  - `contextCache` (not implemented by this plugin) is dropped;
 *  - `topK` is dropped: there's no `top_k` in the Interactions
 *    `generation_config` (Gemini Enterprise Agent Platform rejects it with a
 *    400; the Gemini API accepts and ignores it);
 *  - `imageConfig`, `audioFormat` and `videoFormat` are left out: they're
 *    sent as `response_format` entries (see toInteractionResponseFormat).
 * Everything else, including keys the user passes through that aren't in the
 * schema, is forwarded in snake_case.
 *
 * @param config The request config, without fields handled elsewhere.
 * @param onDropped Called with the name of each config field that was dropped.
 */
export function toInteractionGenerationConfig(
  config: Record<string, unknown>,
  onDropped?: (field: string) => void
): ModelGenerationConfig {
  const {
    speechConfig,
    thinkingConfig,
    thinking_config,
    contextCache,
    context_cache,
    topK,
    top_k,
    // Sent as `response_format` entries (see toInteractionResponseFormat).
    audioFormat: _audioFormat,
    videoFormat: _videoFormat,
    imageConfig: _imageConfig,
    image_config: _image_config,
    ...rest
  } = config;
  const result = convertObjectKeysToSnakeCase(rest);

  const interactionSpeechConfig = toInteractionSpeechConfig(speechConfig);
  if (interactionSpeechConfig) {
    result.speech_config = interactionSpeechConfig;
  }

  const thinking = thinkingConfig ?? thinking_config;
  if (isObject(thinking)) {
    for (const [key, value] of Object.entries(thinking)) {
      if (key === 'thinkingLevel' || key === 'thinking_level') {
        result.thinking_level =
          typeof value === 'string' ? value.toLowerCase() : value;
      } else if (key === 'includeThoughts' || key === 'include_thoughts') {
        result.thinking_summaries =
          typeof value === 'boolean' ? (value ? 'auto' : 'none') : value;
      } else {
        onDropped?.(`thinkingConfig.${key}`);
      }
    }
  }

  if (contextCache !== undefined || context_cache !== undefined) {
    onDropped?.('contextCache');
  }

  if (topK !== undefined || top_k !== undefined) {
    onDropped?.('topK');
  }

  return result;
}

/**
 * Converts the `audioFormat` config (`{ mimeType, sampleRate }`) into an
 * Interactions audio `response_format`. Returns `undefined` if it isn't set,
 * so the server defaults apply.
 */
export function toInteractionAudioResponseFormat(
  audioFormat: unknown
): Record<string, unknown> | undefined {
  if (!isObject(audioFormat)) {
    return undefined;
  }
  const { mimeType, sampleRate, ...rest } = audioFormat;
  return {
    type: 'audio',
    ...toSnakeCaseObj(rest),
    ...(typeof mimeType === 'string' ? { mime_type: mimeType } : {}),
    ...(typeof sampleRate === 'number' ? { sample_rate: sampleRate } : {}),
  };
}

/**
 * Converts the `videoFormat` config (`{ aspectRatio, resolution }`) into an
 * Interactions video `response_format`. Returns `undefined` if it isn't set,
 * so the server defaults apply (16:9, 720p).
 */
export function toInteractionVideoResponseFormat(
  videoFormat: unknown
): Record<string, unknown> | undefined {
  if (!isObject(videoFormat)) {
    return undefined;
  }
  return { type: 'video', ...toSnakeCaseObj(videoFormat) };
}

/** What `toInteractionResponseFormat` builds the `response_format` from. */
export interface InteractionResponseFormatOptions {
  /**
   * The requested output types, e.g. `['TEXT', 'IMAGE']` (the config's
   * `responseModalities`, or the model's default). One entry each.
   */
  modalities?: string[];
  /** The `imageConfig` config (`{ aspectRatio, imageSize }`). */
  imageConfig?: unknown;
  /** The `audioFormat` config (`{ mimeType, sampleRate }`). */
  audioFormat?: unknown;
  /** The `videoFormat` config (`{ aspectRatio, resolution, ... }`). */
  videoFormat?: unknown;
  /** JSON output: the text entry gets `application/json` (and a schema). */
  json?: { schema?: unknown };
}

/**
 * Builds the Interactions `response_format`: one entry per output type
 * (`text`, `image`, `audio`, `video`), each with its settings. This replaces
 * the deprecated `response_modalities` and `generation_config.image_config`.
 *
 * - Each requested modality gets an entry.
 * - `imageConfig`, `audioFormat` and `videoFormat` go on the image, audio and
 *   video entries (adding the entry if needed).
 * - JSON output goes on the text entry (adding it if needed).
 *
 * Returns a single entry when there's one, an array when there are more, and
 * `undefined` when nothing is set (the server's defaults apply).
 */
export function toInteractionResponseFormat(
  options: InteractionResponseFormatOptions
): Record<string, unknown> | Record<string, unknown>[] | undefined {
  const entries = new Map<string, Record<string, unknown>>();
  const entry = (type: string) => {
    let existing = entries.get(type);
    if (!existing) {
      existing = { type };
      entries.set(type, existing);
    }
    return existing;
  };

  for (const modality of options.modalities ?? []) {
    entry(modality.toLowerCase());
  }
  if (isObject(options.imageConfig)) {
    Object.assign(entry('image'), toSnakeCaseObj(options.imageConfig));
  }
  const audio = toInteractionAudioResponseFormat(options.audioFormat);
  if (audio) {
    Object.assign(entry('audio'), audio);
  }
  const video = toInteractionVideoResponseFormat(options.videoFormat);
  if (video) {
    Object.assign(entry('video'), video);
  }
  if (options.json) {
    Object.assign(entry('text'), {
      mime_type: 'application/json',
      ...(options.json.schema !== undefined
        ? { schema: options.json.schema }
        : {}),
    });
  }

  const result = [...entries.values()];
  if (result.length === 0) return undefined;
  return result.length === 1 ? result[0] : result;
}

/**
 * Converts a Genkit Part to an Interaction Content object.
 *
 * Handles various part types including text, media, tool requests (mapped to function calls),
 * and tool responses (mapped to function results).
 *
 * @param part - The Genkit message part.
 * @returns The corresponding Interaction Content object.
 * @throws Error if the part type is unsupported.
 */
export function toInteractionContent(part: Part): Content | undefined {
  if (part.text !== undefined) {
    const speechMetadata = toSpeechMetadataAnnotation(
      part.metadata?.speechMetadata
    );
    return {
      type: 'text',
      text: part.text,
      ...(speechMetadata ? { annotations: [speechMetadata] } : {}),
    };
  }
  if (part.media) {
    return toInteractionMedia(part);
  }
  console.warn(
    `Unsupported part type for Interaction input: ${JSON.stringify(part)}`
  );
  return undefined;
}

/**
 * Converts a text part's `metadata.speechMetadata` (`{ speaker?, style? }`)
 * into an Interactions `speech_metadata` annotation, used by TTS models to
 * assign each turn to a speaker and set its delivery style.
 */
function toSpeechMetadataAnnotation(
  speechMetadata: unknown
): SpeechAnnotation | undefined {
  if (!isObject(speechMetadata)) {
    return undefined;
  }
  const speaker =
    typeof speechMetadata.speaker === 'string'
      ? speechMetadata.speaker
      : undefined;
  const style =
    typeof speechMetadata.style === 'string' ? speechMetadata.style : undefined;
  if (!speaker && !style) {
    return undefined;
  }
  return {
    type: 'speech_metadata',
    ...(speaker ? { speaker } : {}),
    ...(style ? { style } : {}),
  };
}

/** Reports whether a URL points at a YouTube video. */
function isYouTubeUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname.replace(/^(www\.|m\.)/, '');
    return host === 'youtube.com' || host === 'youtu.be';
  } catch {
    return false;
  }
}

/**
 * The `processing` mode (`static` or `agentic`) for a video part, from the
 * part's `metadata.processing`.
 */
function videoProcessingOf(part: Part): Pick<VideoContent, 'processing'> {
  const processing = part.metadata?.processing;
  return typeof processing === 'string' ? { processing } : {};
}

function toInteractionMedia(part: Part): Content {
  if (!part.media) throw new Error('Media part missing media');
  const { url } = part.media;
  const contentType = part.media.contentType || extractMimeType(url);

  if (!contentType) {
    if (url.startsWith('data:')) {
      throw new Error('Media part missing contentType');
    }
    // A URI whose type can't be determined (e.g. a YouTube link or a Files
    // API URI). The server works out the type itself, so send it without a
    // mime_type, as the generateContent path does. YouTube links are video;
    // anything else is sent as a document (verified live: the server still
    // treats a YouTube link sent as a document as video).
    if (isYouTubeUrl(url)) {
      return { type: 'video', uri: url, ...videoProcessingOf(part) };
    }
    return { type: 'document', uri: url };
  }

  // Inline data URLs are sent as `data`, everything else as a `uri`.
  const source = url.startsWith('data:')
    ? { data: url.substring(url.indexOf(',') + 1) }
    : { uri: url };
  const fields = { mime_type: contentType, ...source };

  if (contentType.startsWith('image/')) {
    return { type: 'image', ...fields };
  }
  if (contentType.startsWith('audio/')) {
    return { type: 'audio', ...fields };
  }
  if (contentType.startsWith('video/')) {
    return { type: 'video', ...fields, ...videoProcessingOf(part) };
  }
  // Everything else is a document (PDF, CSV, Word, Excel, ...). The server
  // decides which types it accepts and returns a clear error for the rest;
  // its list has grown beyond the PDF and CSV in the published spec.
  return { type: 'document', ...fields };
}

/**
 * Maps a Genkit message role to the corresponding Interaction API role.
 *
 * - 'user' -> 'user'
 * - 'model' -> 'model'
 * - 'tool' -> 'user' (Tool outputs are treated as user turns in this context)
 *
 * @param role - The Genkit message role.
 * @returns The mapped Interaction role string.
 * @throws Error if the role is 'system', as system instructions are handled separately.
 */
export function toInteractionRole(role: MessageData['role']): string {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'model';
    case 'tool':
      return 'user';
    case 'system':
      throw new Error(
        `System role should be handled as system_instruction, not part of turns.`
      );
    default:
      return 'user';
  }
}

const GoogleSearchArgsSchema = z
  .object({ queries: z.array(z.string()).optional() })
  .passthrough();
const RecordUnknownSchema = z.record(z.unknown());

const MediaResolutionSchema = z.enum(['low', 'medium', 'high', 'ultra_high']);

const TextContentSchema = z.object({
  type: z.literal('text'),
  text: z.string().optional(),
  annotations: z.array(TextAnnotationSchema).optional(),
});

const ImageContentSchema = z.object({
  type: z.literal('image'),
  data: z.string().optional(),
  uri: z.string().optional(),
  mime_type: z.string().optional(),
  resolution: MediaResolutionSchema.optional(),
});

const FunctionResultArraySchema = z.array(
  z.union([ImageContentSchema, TextContentSchema])
);

const RecordUnknownOrStringOrArraySchema = z.union([
  RecordUnknownSchema,
  z.string(),
  FunctionResultArraySchema,
]);

const OptionalStringSchema = z.string().optional();

// `arguments`, `searchType` and `result` are optional: the platform sometimes
// omits them (seen live on vertexAI), and they're only sent back when present.
const GoogleSearchCallSchema = z.object({
  id: z.string(),
  arguments: GoogleSearchArgsSchema.optional(),
  searchType: z.string().optional(),
});

// The API returns `result` as an array of items (e.g.
// `[{ search_suggestions }]`), and it must be sent back the same way when the
// history is resent (e.g. on the second turn of a tool loop). An object is
// still accepted for parts saved before this was fixed.
const GoogleSearchResultSchema = z.object({
  callId: z.string(),
  result: z
    .union([z.array(RecordUnknownSchema), RecordUnknownSchema])
    .optional(),
  isError: z.boolean().optional(),
});

const ExecutableCodeSchema = z.object({
  code: z.string(),
  language: z.string().default('PYTHON'),
});

const CodeExecutionResultSchema = z.object({
  output: z.string(),
  outcome: z.string().optional(),
});

const ProcessingCallSchema = z.object({ id: z.string() });

const ProcessingResultSchema = z.object({ callId: z.string() });

// Custom parts for the server-side built-in tool steps (url context, file
// search, Google Maps, MCP servers), in camelCase like
// `googleSearchCall`. `.passthrough()` keeps fields the server may add.
// Only the ids are required: the platform can omit other fields (as it does
// for Google Search steps), and missing fields are not sent back.
const UrlsArgsSchema = z
  .object({ urls: z.array(z.string()).optional() })
  .passthrough();
const QueriesArgsSchema = z
  .object({ queries: z.array(z.string()).optional() })
  .passthrough();
const UrlContextCallPartSchema = z.object({
  id: z.string(),
  arguments: UrlsArgsSchema.optional(),
});
const UrlContextResultPartSchema = z.object({
  callId: z.string(),
  result: z.array(RecordUnknownSchema).optional(),
  isError: z.boolean().optional(),
});
const FileSearchCallPartSchema = z.object({ id: z.string() });
const FileSearchResultPartSchema = z.object({
  callId: z.string(),
  result: z.array(RecordUnknownSchema).optional(),
});
const GoogleMapsCallPartSchema = z.object({
  id: z.string(),
  arguments: QueriesArgsSchema.optional(),
});
const GoogleMapsResultPartSchema = z.object({
  callId: z.string(),
  result: z.array(RecordUnknownSchema).optional(),
});
const McpServerToolCallPartSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  serverName: z.string().optional(),
  arguments: RecordUnknownSchema.optional(),
});
const McpServerToolResultPartSchema = z.object({
  callId: z.string(),
  name: z.string().optional(),
  serverName: z.string().optional(),
  result: RecordUnknownOrStringOrArraySchema.optional(),
});

/**
 * Converts a built-in tool custom part (see `fromBuiltInToolStep`) back into
 * its step, with its signature, so the tool call and result are resent in
 * the history. Returns `undefined` for any other part.
 */
function toBuiltInToolStep(part: Part): BuiltInToolStep | undefined {
  const custom = part.custom;
  if (!custom) return undefined;
  const signature = OptionalStringSchema.parse(part.metadata?.thoughtSignature);
  const withSignature = signature ? { signature } : {};

  if (custom.urlContextCall) {
    const call = UrlContextCallPartSchema.parse(custom.urlContextCall);
    return {
      type: 'url_context_call',
      id: call.id,
      ...(call.arguments ? { arguments: call.arguments } : {}),
      ...withSignature,
    };
  }
  if (custom.urlContextResult) {
    const result = UrlContextResultPartSchema.parse(custom.urlContextResult);
    return {
      type: 'url_context_result',
      call_id: result.callId,
      ...(result.result ? { result: result.result } : {}),
      ...(result.isError !== undefined ? { is_error: result.isError } : {}),
      ...withSignature,
    };
  }
  if (custom.fileSearchCall) {
    const call = FileSearchCallPartSchema.parse(custom.fileSearchCall);
    return { type: 'file_search_call', id: call.id, ...withSignature };
  }
  if (custom.fileSearchResult) {
    const result = FileSearchResultPartSchema.parse(custom.fileSearchResult);
    return {
      type: 'file_search_result',
      call_id: result.callId,
      ...(result.result ? { result: result.result } : {}),
      ...withSignature,
    };
  }
  if (custom.googleMapsCall) {
    const call = GoogleMapsCallPartSchema.parse(custom.googleMapsCall);
    return {
      type: 'google_maps_call',
      id: call.id,
      ...(call.arguments ? { arguments: call.arguments } : {}),
      ...withSignature,
    };
  }
  if (custom.googleMapsResult) {
    const result = GoogleMapsResultPartSchema.parse(custom.googleMapsResult);
    return {
      type: 'google_maps_result',
      call_id: result.callId,
      ...(result.result ? { result: result.result } : {}),
      ...withSignature,
    };
  }
  if (custom.mcpServerToolCall) {
    // MCP steps have no signature in the spec.
    const call = McpServerToolCallPartSchema.parse(custom.mcpServerToolCall);
    return {
      type: 'mcp_server_tool_call',
      id: call.id,
      ...(call.name ? { name: call.name } : {}),
      ...(call.serverName ? { server_name: call.serverName } : {}),
      ...(call.arguments ? { arguments: call.arguments } : {}),
    };
  }
  if (custom.mcpServerToolResult) {
    const result = McpServerToolResultPartSchema.parse(
      custom.mcpServerToolResult
    );
    return {
      type: 'mcp_server_tool_result',
      call_id: result.callId,
      ...(result.name ? { name: result.name } : {}),
      ...(result.serverName ? { server_name: result.serverName } : {}),
      ...(result.result !== undefined ? { result: result.result } : {}),
    };
  }
  return undefined;
}

/**
 * Converts an array of Genkit MessageData objects into an array of Interaction Steps.
 */
export function toInteractionSteps(messages: MessageData[]): Step[] {
  const steps: Step[] = [];

  for (const message of messages) {
    const normalContent: Content[] = [];

    for (const part of message.content) {
      const builtInToolStep = toBuiltInToolStep(part);
      if (builtInToolStep) {
        steps.push(builtInToolStep);
        continue;
      }
      if (part.toolRequest) {
        const signature = OptionalStringSchema.parse(
          part.metadata?.thoughtSignature
        );
        steps.push({
          type: 'function_call',
          name: part.toolRequest.name,
          arguments: RecordUnknownSchema.optional().parse(
            part.toolRequest.input
          ),
          id: part.toolRequest.ref || '',
          ...(signature ? { signature } : {}),
        });
      } else if (part.toolResponse) {
        let result: unknown = part.toolResponse.output;

        if (part.toolResponse.content && part.toolResponse.content.length > 0) {
          const contentParts: Content[] = [];
          if (result !== undefined) {
            const outputText =
              typeof result === 'string' ? result : JSON.stringify(result);
            contentParts.push({ type: 'text', text: outputText });
          }
          for (const p of part.toolResponse.content) {
            const mapped = toInteractionContent(p);
            if (mapped) {
              contentParts.push(mapped);
            }
          }
          result = contentParts;
        } else if (
          (typeof result !== 'object' && typeof result !== 'string') ||
          Array.isArray(result)
        ) {
          result = { result: result };
        }

        let parsedResult:
          | Record<string, unknown>
          | string
          | (ImageContent | TextContent)[];
        try {
          parsedResult = RecordUnknownOrStringOrArraySchema.parse(result ?? {});
        } catch {
          throw new GenkitError({
            status: 'INVALID_ARGUMENT',
            message: `Tool output for ${part.toolResponse.name} may only contain text or image content.`,
          });
        }
        steps.push({
          type: 'function_result',
          name: part.toolResponse.name,
          result: parsedResult,
          call_id: part.toolResponse.ref || '',
        });
      } else if (part.custom?.googleSearchCall) {
        const gsCall = GoogleSearchCallSchema.parse(
          part.custom.googleSearchCall
        );
        steps.push({
          type: 'google_search_call',
          id: gsCall.id,
          ...(gsCall.arguments ? { arguments: gsCall.arguments } : {}),
          ...(gsCall.searchType ? { search_type: gsCall.searchType } : {}),
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else if (part.custom?.googleSearchResult) {
        const gsResult = GoogleSearchResultSchema.parse(
          part.custom.googleSearchResult
        );
        steps.push({
          type: 'google_search_result',
          call_id: gsResult.callId,
          ...(gsResult.result ? { result: gsResult.result } : {}),
          ...(gsResult.isError !== undefined
            ? { is_error: gsResult.isError }
            : {}),
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else if (part.custom?.executableCode) {
        const execCode = ExecutableCodeSchema.parse(part.custom.executableCode);
        steps.push({
          type: 'code_execution_call',
          id: z.string().parse(part.metadata?.callId),
          arguments: {
            code: execCode.code,
            language: execCode.language,
          },
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else if (part.custom?.codeExecutionResult) {
        const execResult = CodeExecutionResultSchema.parse(
          part.custom.codeExecutionResult
        );
        steps.push({
          type: 'code_execution_result',
          call_id: z.string().parse(part.metadata?.callId),
          result: execResult.output,
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else if (part.custom?.processingCall) {
        // Agentic video steps are resent as they came back, with their
        // signatures, when the history is sent again.
        const call = ProcessingCallSchema.parse(part.custom.processingCall);
        const signature = OptionalStringSchema.parse(
          part.metadata?.thoughtSignature
        );
        steps.push({
          type: 'processing_call',
          id: call.id,
          ...(signature ? { signature } : {}),
        });
      } else if (part.custom?.processingResult) {
        const result = ProcessingResultSchema.parse(
          part.custom.processingResult
        );
        const signature = OptionalStringSchema.parse(
          part.metadata?.thoughtSignature
        );
        steps.push({
          type: 'processing_result',
          call_id: result.callId,
          ...(signature ? { signature } : {}),
        });
      } else if (
        part.reasoning !== undefined ||
        Boolean(part.custom?.thought)
      ) {
        const signature =
          part.metadata?.thoughtSignature ?? part.custom?.thought?.signature;
        const parsedSignature = OptionalStringSchema.parse(signature);

        let summary: (TextContent | ImageContent)[] | undefined = undefined;
        if (
          Array.isArray(part.custom?.thought?.summary) &&
          part.custom.thought.summary.length > 0
        ) {
          summary = part.custom.thought.summary;
        } else if (part.reasoning) {
          summary = [{ type: 'text', text: part.reasoning }];
        }

        const thoughtStep: ThoughtStep = {
          type: 'thought',
          ...(summary ? { summary } : {}),
          ...(parsedSignature ? { signature: parsedSignature } : {}),
        };
        steps.push(thoughtStep);
      } else {
        const content = toInteractionContent(part);
        if (content) {
          normalContent.push(content);
        }
      }
    }

    if (normalContent.length > 0) {
      if (message.role === 'model') {
        steps.push({
          type: 'model_output',
          content: normalContent,
        });
      } else {
        steps.push({
          type: 'user_input',
          content: normalContent,
        });
      }
    }
  }

  return steps;
}

/** Steps whose content streams as deltas (text, media, thought summaries). */
const STREAMED_CONTENT_STEP_TYPES = new Set(['model_output', 'thought']);

/** Steps streamed when the interaction completes (see below). */
const FUNCTION_STEP_TYPES = new Set(['function_call', 'function_result']);

/**
 * Creates a converter from stream events to the parts of stream chunks.
 *
 * Content (text, media, thought summaries and signatures) is emitted as each
 * delta arrives. Tool steps (Google Search, code execution, URL context,
 * ...) are emitted once complete, at `step.stop`: the server sends their ids
 * in `step.start` and everything else in a single delta, so waiting for the
 * end gives the same parts as the final response. Function calls (and
 * function results, to keep them in order) are emitted when the interaction
 * completes, because only then is it known whether a call is for Genkit to
 * run (a `toolRequest`) or was already run by the server.
 *
 * @returns A function that takes each stream event, in order, and returns the
 *   parts to stream for it (often none).
 */
export function createInteractionChunkConverter(): (
  event: InteractionSseEvent
) => Part[] {
  // Each event is applied to this converter's own copy of the steps, the same
  // way the final response is assembled.
  const steps: Step[] = [];
  const partialArguments = new Map<number, string>();
  let status: GeminiInteraction['status'];
  return (event) => {
    applyInteractionStepEvent(steps, partialArguments, event);
    if (event.event_type === 'interaction.status_update') {
      status = event.status;
    }
    if (event.event_type === 'step.delta') {
      return fromInteractionDelta(event.delta);
    }
    if (event.event_type === 'step.stop') {
      const step = steps[event.index];
      if (
        !step ||
        FUNCTION_STEP_TYPES.has(step.type) ||
        STREAMED_CONTENT_STEP_TYPES.has(step.type)
      ) {
        return [];
      }
      return withoutEmptyParts(fromInteractionStep(step));
    }
    if (event.event_type === 'interaction.completed') {
      const pendingIds = getPendingFunctionCallIds(
        steps,
        event.interaction.status ?? status
      );
      return withoutEmptyParts(
        steps.flatMap((step) => {
          if (!FUNCTION_STEP_TYPES.has(step.type)) return [];
          const isPending =
            step.type === 'function_call' && pendingIds.has(step.id);
          return fromInteractionStep(step, isPending);
        })
      );
    }
    return [];
  };
}

function withoutEmptyParts(parts: Part[]): Part[] {
  return parts.filter((p) => p && Object.keys(p).length > 0);
}

/**
 * Converts a streamed content delta (text, media, thought summary or
 * signature) into Genkit Parts for a stream chunk. Tool step deltas return no
 * parts: `createInteractionChunkConverter` emits those steps once complete.
 *
 * @param delta - The `delta` of a `step.delta` SSE event.
 * @returns The corresponding Genkit Parts (empty for deltas with nothing to
 *   emit on their own, such as function call arguments).
 */
export function fromInteractionDelta(delta: StepDeltaData): Part[] {
  switch (delta.type) {
    case 'text':
      return [{ text: delta.text }];
    case 'image':
    case 'audio':
    case 'document':
    case 'video': {
      let url = delta.uri;
      if (delta.data && delta.mime_type) {
        url = `data:${delta.mime_type};base64,${delta.data}`;
      }
      const part: Part = {
        media: {
          url: url || '',
          contentType: delta.mime_type,
        },
      };
      if (
        (delta.type === 'image' || delta.type === 'video') &&
        delta.resolution !== undefined
      ) {
        part.metadata = { resolution: delta.resolution };
      }
      return [part];
    }
    case 'thought_summary':
      if (!delta.content) return [];
      if (delta.content.type === 'text') {
        return [
          {
            reasoning: delta.content.text || '',
            ...(delta.content.annotations
              ? { metadata: { annotations: delta.content.annotations } }
              : {}),
          },
        ];
      }
      // Non-text thought content (e.g. a draft image) is part of the
      // reasoning, not the answer. Use the same placeholder as the final
      // response so streaming clients don't receive it as a regular part.
      return [{ reasoning: thoughtContentPlaceholder(delta.content) }];
    case 'thought_signature':
      return [{ metadata: { thoughtSignature: delta.signature } }];
    default:
      return [];
  }
}

/**
 * Converts an Interaction Content object back into a Genkit Part.
 *
 * Supports text, image, audio, document, video, thought, function calls, and
 * function results. Unknown content types are returned as a custom part.
 *
 * @param content - The Interaction Content object.
 * @returns The corresponding Genkit Part.
 */
export function fromInteractionContent(content: Content): Part {
  switch (content.type) {
    case 'text':
      return fromTextContent(content);
    case 'image':
      return fromImageContent(content);
    case 'audio':
    case 'document':
      return fromMediaContent(content);
    case 'video':
      return fromVideoContent(content);
    case 'thought':
      return fromThoughtContent(content);
    case 'function_call':
      return fromFunctionCallContent(content);
    case 'function_result':
      return fromFunctionResultContent(content);
    default:
      return {
        custom: {
          unknownContent: content,
        },
      };
  }
}

function maybeAddGeminiThoughtSignature(step: Step, part: Part): Part {
  let updatedPart = part;

  if ('signature' in step && step.signature) {
    updatedPart.metadata = {
      ...(part.metadata ?? {}),
      thoughtSignature: step.signature,
    };
  }
  return updatedPart;
}

export function fromGoogleSearchCall(step: GoogleSearchCallStep): Part {
  const part: Part = {
    custom: {
      googleSearchCall: {
        id: step.id,
        ...(step.arguments ? { arguments: step.arguments } : {}),
        ...(step.search_type ? { searchType: step.search_type } : {}),
      },
    },
  };
  return maybeAddGeminiThoughtSignature(step, part);
}

export function fromGoogleSearchResult(step: GoogleSearchResultStep): Part {
  const part: Part = {
    custom: {
      googleSearchResult: {
        callId: step.call_id,
        ...(step.result ? { result: step.result } : {}),
        ...(step.is_error !== undefined ? { isError: step.is_error } : {}),
      },
    },
  };
  return maybeAddGeminiThoughtSignature(step, part);
}

export function fromCodeExecutionCall(step: CodeExecutionCallStep): Part {
  const part: Part = {
    custom: {
      executableCode: {
        code: step.arguments.code,
        language: step.arguments.language || 'PYTHON',
      },
    },
  };
  part.metadata = { callId: step.id };
  return maybeAddGeminiThoughtSignature(step, part);
}

export function fromCodeExecutionResult(step: CodeExecutionResultStep): Part {
  const part: Part = {
    custom: {
      codeExecutionResult: {
        output:
          typeof step.result === 'string'
            ? step.result
            : JSON.stringify(step.result),
        outcome: 'OUTCOME_OK',
      },
    },
  };
  part.metadata = { callId: step.call_id };
  return maybeAddGeminiThoughtSignature(step, part);
}

export function fromPendingFunctionCall(
  step: FunctionCallContent | FunctionCallStep
): Part {
  // Keep the signature so it can be sent back with the function call when the
  // history is resent (stateless, the default).
  return maybeAddGeminiThoughtSignature(step, {
    toolRequest: {
      name: step.name,
      ref: step.id,
      input: step.arguments,
    },
  });
}

export function fromServerFunctionCall(
  step: FunctionCallContent | FunctionCallStep
): Part {
  return maybeAddGeminiThoughtSignature(step, {
    custom: {
      serverFunctionCall: {
        id: step.id,
        name: step.name,
        arguments: step.arguments,
      },
    },
  });
}

export function fromServerFunctionResult(
  step: FunctionResultContent | FunctionResultStep
): Part {
  return {
    custom: {
      serverFunctionResult: {
        callId: step.call_id,
        name: step.name || '',
        result: step.result,
        isError: step.is_error,
      },
    },
  };
}

export function fromInteractionStep(step: Step, isPending?: boolean): Part[] {
  switch (step.type) {
    case 'model_output':
      return (step.content ?? []).map(fromInteractionContent);
    case 'user_input':
      return [];
    case 'google_search_call':
      return [fromGoogleSearchCall(step)];
    case 'google_search_result':
      return [fromGoogleSearchResult(step)];
    case 'code_execution_call':
      return [fromCodeExecutionCall(step)];
    case 'code_execution_result':
      return [fromCodeExecutionResult(step)];
    case 'thought':
      return [fromThoughtContent(step)];
    case 'function_call':
      return isPending
        ? [fromPendingFunctionCall(step)]
        : [fromServerFunctionCall(step)];
    case 'function_result':
      return [fromServerFunctionResult(step)];
    case 'processing_call':
      return [
        maybeAddGeminiThoughtSignature(step, {
          custom: { processingCall: { id: step.id } },
        }),
      ];
    case 'processing_result':
      return [
        maybeAddGeminiThoughtSignature(step, {
          custom: { processingResult: { callId: step.call_id } },
        }),
      ];
    case 'url_context_call':
    case 'url_context_result':
    case 'file_search_call':
    case 'file_search_result':
    case 'google_maps_call':
    case 'google_maps_result':
    case 'mcp_server_tool_call':
    case 'mcp_server_tool_result':
      return [fromBuiltInToolStep(step)];
  }

  return [{ custom: { unknownStep: step } }];
}

/**
 * Converts a server-side built-in tool step (url context, file search,
 * Google Maps, MCP server) into a custom part, with the step's
 * `signature` in `metadata.thoughtSignature`. `toBuiltInToolStep` turns it
 * back into the step when the history is resent. These are never
 * `toolRequest` parts: the server ran the tool, so there's nothing for
 * Genkit to run.
 */
export function fromBuiltInToolStep(step: BuiltInToolStep): Part {
  switch (step.type) {
    case 'url_context_call':
      return maybeAddGeminiThoughtSignature(step, {
        custom: {
          urlContextCall: {
            id: step.id,
            ...(step.arguments ? { arguments: step.arguments } : {}),
          },
        },
      });
    case 'url_context_result':
      return maybeAddGeminiThoughtSignature(step, {
        custom: {
          urlContextResult: {
            callId: step.call_id,
            ...(step.result ? { result: step.result } : {}),
            ...(step.is_error !== undefined ? { isError: step.is_error } : {}),
          },
        },
      });
    case 'file_search_call':
      return maybeAddGeminiThoughtSignature(step, {
        custom: { fileSearchCall: { id: step.id } },
      });
    case 'file_search_result':
      return maybeAddGeminiThoughtSignature(step, {
        custom: {
          fileSearchResult: {
            callId: step.call_id,
            ...(step.result ? { result: step.result } : {}),
          },
        },
      });
    case 'google_maps_call':
      return maybeAddGeminiThoughtSignature(step, {
        custom: {
          googleMapsCall: {
            id: step.id,
            ...(step.arguments ? { arguments: step.arguments } : {}),
          },
        },
      });
    case 'google_maps_result':
      return maybeAddGeminiThoughtSignature(step, {
        custom: {
          googleMapsResult: {
            callId: step.call_id,
            ...(step.result ? { result: step.result } : {}),
          },
        },
      });
    case 'mcp_server_tool_call':
      return {
        custom: {
          mcpServerToolCall: {
            id: step.id,
            ...(step.name ? { name: step.name } : {}),
            ...(step.server_name ? { serverName: step.server_name } : {}),
            ...(step.arguments ? { arguments: step.arguments } : {}),
          },
        },
      };
    case 'mcp_server_tool_result':
      return {
        custom: {
          mcpServerToolResult: {
            callId: step.call_id,
            ...(step.name ? { name: step.name } : {}),
            ...(step.server_name ? { serverName: step.server_name } : {}),
            ...(step.result !== undefined ? { result: step.result } : {}),
          },
        },
      };
  }
}

function getPendingFunctionCallIds(
  steps: Step[],
  status?: string
): Set<string> {
  const pendingIds = new Set<string>();
  if (status !== 'requires_action') {
    return pendingIds;
  }

  const resultSet = new Set<string>();
  for (const step of steps) {
    if (step.type === 'function_result' && step.call_id) {
      resultSet.add(step.call_id);
    }
  }

  for (const step of steps) {
    if (step.type === 'function_call' && step.id) {
      if (!resultSet.has(step.id)) {
        pendingIds.add(step.id);
      }
    }
  }

  return pendingIds;
}

function fromMediaContent(
  content: ImageContent | AudioContent | VideoContent | DocumentContent
): Part {
  let url = content.uri;
  if (content.data && content.mime_type) {
    url = `data:${content.mime_type};base64,${content.data}`;
  }
  return {
    media: {
      url: url || '',
      contentType: content.mime_type,
    },
  };
}

function fromTextContent(content: TextContent): Part {
  return {
    text: content.text || '',
    metadata: {
      annotations: content.annotations,
    },
  };
}

function fromImageContent(content: ImageContent): Part {
  const part = fromMediaContent(content);
  if (content.resolution !== undefined) {
    part.metadata = { resolution: content.resolution };
  }
  return part;
}

function fromVideoContent(content: VideoContent): Part {
  const part = fromMediaContent(content);
  if (content.resolution !== undefined) {
    part.metadata = { resolution: content.resolution };
  }
  return part;
}

/**
 * Placeholder text for non-text content inside a thought summary, e.g.
 * `[Image]`, `[Audio]` or `[Function call]`.
 *
 * Thoughts are returned as reasoning text, so non-text items (such as a draft
 * image the model reviews before its final output) are shown as a label
 * instead of being dropped or returned as regular media parts. The label is
 * derived from the content type, so any type the server adds in the future
 * still shows up in the reasoning.
 */
function thoughtContentPlaceholder(content: Content): string {
  const label = content.type.replace(/_/g, ' ');
  return `[${label.charAt(0).toUpperCase()}${label.slice(1)}]`;
}

function fromThoughtContent(content: ThoughtContent): Part {
  let reasoning = '';
  if (content.summary) {
    // Separate summary blocks with a newline so distinct thought sections
    // don't run together. Each summary block is a whole section, both in
    // non-streamed responses and in streamed `thought_summary` deltas (one
    // delta per section, confirmed live), so this doesn't split sentences.
    reasoning = content.summary
      .map((c) => {
        if (c.type === 'text') return c.text;
        return thoughtContentPlaceholder(c);
      })
      .join('\n');
  }

  return {
    reasoning,
    metadata: {
      thoughtSignature: content.signature,
    },
    custom: {
      thought: content,
    },
  };
}

function fromFunctionCallContent(content: FunctionCallContent): Part {
  return {
    toolRequest: {
      name: content.name,
      input: content.arguments,
      ref: content.id,
    },
  };
}

function isContentArray(val: unknown): val is Content[] {
  return (
    Array.isArray(val) &&
    val.length > 0 &&
    val.every((item) => isObject(item) && typeof item.type === 'string')
  );
}

function fromFunctionResultContent(content: FunctionResultContent): Part {
  if (isContentArray(content.result)) {
    return {
      toolResponse: {
        name: content.name,
        content: content.result.map((c) => fromInteractionContent(c)),
        ref: content.call_id,
      },
    };
  }
  return {
    toolResponse: {
      name: content.name,
      output: content.result,
      ref: content.call_id,
    },
  };
}

/** Converts Interactions API token usage to Genkit usage. */
function fromInteractionUsage(usage: Usage): GenerationUsage {
  const result: GenerationUsage = {
    inputTokens: usage.total_input_tokens,
    outputTokens: usage.total_output_tokens,
    totalTokens: usage.total_tokens,
    cachedContentTokens: usage.total_cached_tokens,
    thoughtsTokens: usage.total_thought_tokens,
  };
  for (const modalityToken of usage.input_tokens_by_modality ?? []) {
    switch (modalityToken.modality) {
      case 'text':
        result.inputCharacters = modalityToken.tokens;
        break;
      case 'image':
        result.inputImages = modalityToken.tokens;
        break;
      case 'audio':
        result.inputAudioFiles = modalityToken.tokens;
        break;
      case 'video':
        result.inputVideos = modalityToken.tokens;
        break;
    }
  }
  for (const modalityToken of usage.output_tokens_by_modality ?? []) {
    switch (modalityToken.modality) {
      case 'text':
        result.outputCharacters = modalityToken.tokens;
        break;
      case 'image':
        result.outputImages = modalityToken.tokens;
        break;
      case 'audio':
        result.outputAudioFiles = modalityToken.tokens;
        break;
      case 'video':
        result.outputVideos = modalityToken.tokens;
        break;
    }
  }
  // Genkit has no standard fields for these.
  const custom: Record<string, number> = {};
  // Tokens loaded by tools, e.g. video frames read during agentic video
  // understanding.
  if (usage.total_tool_use_tokens) {
    custom.toolUseTokens = usage.total_tool_use_tokens;
  }
  // How many times each grounding tool was used, e.g. `googleSearchCount`.
  for (const { type, count } of usage.grounding_tool_count ?? []) {
    if (type && typeof count === 'number') {
      const key = `${snakeToCamelCase(type)}Count`;
      custom[key] = (custom[key] ?? 0) + count;
    }
  }
  if (Object.keys(custom).length > 0) {
    result.custom = custom;
  }
  return result;
}

const INTERACTION_FAILED_MESSAGE = 'Interaction failed';
const INTERACTION_INCOMPLETE_MESSAGE =
  'Interaction incomplete (truncated output)';

/** Builds a human-readable message from a failed interaction's errors. */
function interactionFailureMessage(interaction: GeminiInteraction): string {
  const details = (interaction.errors ?? [])
    .map((e) => [e.code ? `[${e.code}]` : '', e.message ?? ''].join(' ').trim())
    .filter(Boolean)
    .join('; ');
  return details
    ? `${INTERACTION_FAILED_MESSAGE}: ${details}`
    : INTERACTION_FAILED_MESSAGE;
}

/**
 * Converts a `status: 'failed'` interaction into a Genkit outcome:
 *  - content-policy blocks (safety, recitation, ...) return a response with
 *    `finishReason: 'blocked'`, matching the generateContent path;
 *  - everything else throws a `GenkitError` with the status mapped from the
 *    error code (`UNKNOWN` if unrecognized, which is not retried).
 */
function fromFailedInteraction(
  interaction: GeminiInteraction
): GenerateResponseData {
  const firstError = interaction.errors?.[0];
  if (isInteractionContentBlockCode(firstError?.code)) {
    return {
      finishReason: 'blocked',
      finishMessage:
        firstError?.message || interactionFailureMessage(interaction),
      message: {
        role: 'model',
        content: [],
        ...interactionMessageMetadata(interaction),
      },
      custom: interaction,
      raw: interaction,
      ...(interaction.usage
        ? { usage: fromInteractionUsage(interaction.usage) }
        : {}),
    };
  }
  throw new GenkitError({
    status: interactionErrorCodeToGenkitStatus(firstError?.code) ?? 'UNKNOWN',
    message: interactionFailureMessage(interaction),
    detail: interaction,
  });
}

/** Message metadata carrying the interaction and environment IDs, if any. */
function interactionMessageMetadata(
  interaction: GeminiInteraction
): Pick<MessageData, 'metadata'> {
  return interaction.id || interaction.environment_id
    ? {
        metadata: {
          ...(interaction.id ? { interactionId: interaction.id } : {}),
          ...(interaction.environment_id
            ? { environmentId: interaction.environment_id }
            : {}),
        },
      }
    : {};
}

export function fromInteractionSync(
  interaction: GeminiInteraction
): GenerateResponseData {
  if (interaction.status === 'failed') {
    return fromFailedInteraction(interaction);
  }

  const response: GenerateResponseData = {
    finishReason: 'stop',
    message: {
      role: 'model',
      content: [],
      ...(interaction.id || interaction.environment_id
        ? {
            metadata: {
              ...(interaction.id ? { interactionId: interaction.id } : {}),
              ...(interaction.environment_id
                ? { environmentId: interaction.environment_id }
                : {}),
            },
          }
        : {}),
    },
    custom: interaction,
    raw: interaction,
    ...(interaction.usage
      ? { usage: fromInteractionUsage(interaction.usage) }
      : {}),
  };

  if (interaction.status === 'cancelled') {
    response.finishReason = 'aborted';
    response.finishMessage = 'Operation cancelled';
    response.message!.content = [{ text: 'Operation cancelled.' }];
    return response;
  }

  if (interaction.status === 'incomplete') {
    // Token/step limit reached (MAX_TOKENS); partial content may be present.
    response.finishReason = 'length';
    response.finishMessage = INTERACTION_INCOMPLETE_MESSAGE;
  }

  const steps = interaction.steps;
  if (steps?.length) {
    const pendingIds = getPendingFunctionCallIds(steps, interaction.status);

    response.message!.content = steps
      .flatMap((step) => {
        const isPending =
          step.type === 'function_call' && step.id
            ? pendingIds.has(step.id)
            : false;
        return fromInteractionStep(step, isPending);
      })
      .filter((p) => p && Object.keys(p).length > 0);
  }
  return response;
}

/** Audio types without a container header, so chunks can be joined end to end. */
const HEADERLESS_AUDIO_TYPES = ['audio/l16', 'audio/mulaw', 'audio/alaw'];

function isHeaderlessAudio(part: Part): boolean {
  const contentType = part.media?.contentType?.toLowerCase();
  return (
    !!part.media?.url.startsWith('data:') &&
    !!contentType &&
    HEADERLESS_AUDIO_TYPES.some((type) => contentType.startsWith(type))
  );
}

function isWavAudio(part: Part): boolean {
  return (
    !!part.media?.url.startsWith('data:') &&
    !!part.media.contentType?.toLowerCase().startsWith('audio/wav')
  );
}

function mediaBytes(part: Part): Buffer {
  const url = part.media?.url ?? '';
  return Buffer.from(url.substring(url.indexOf(',') + 1), 'base64');
}

function mediaPart(contentType: string, bytes: Buffer): Part {
  return {
    media: {
      url: `data:${contentType};base64,${bytes.toString('base64')}`,
      contentType,
    },
  };
}

/**
 * Removes ID3 tags from the start of a headerless audio part.
 *
 * The server sometimes sends a C2PA manifest, in an ID3 tag, at the start of a
 * streamed `audio/l16` chunk (seen with gemini-3.8-flash-lite-tts and two
 * speakers). The tag isn't audio: played or joined as raw audio it is noise,
 * and its odd size shifts every sample after it by one byte. This has been
 * reported to the Gemini API team; until it's fixed, the tag is removed so
 * the audio plays correctly. That manifest is dropped. The tag has only been
 * seen whole at the start of a chunk; a tag that doesn't fit in the chunk is
 * left in, since bytes that can't be checked as a tag might be audio.
 */
export function removeId3FromAudio(part: Part): Part {
  if (!part.media || !isHeaderlessAudio(part)) {
    return part;
  }
  const bytes = mediaBytes(part);
  const audio = stripLeadingId3Tags(bytes);
  if (audio === bytes) {
    return part;
  }
  const contentType = part.media.contentType ?? 'audio/l16';
  return {
    ...part,
    media: {
      ...part.media,
      url: `data:${contentType};base64,${audio.toString('base64')}`,
    },
  };
}

/**
 * Joins streamed audio chunks in the final response into one part, in the
 * format the server returned:
 *  - consecutive headerless chunks (`audio/l16`, `audio/mulaw`, `audio/alaw`)
 *    of the same type are joined end to end;
 *  - a streamed `audio/wav` request starts with a small complete WAV file and
 *    continues with `audio/l16` chunks (seen live). Those chunks go into the
 *    WAV's `data` chunk, giving one WAV like a non-streamed response.
 * Other parts are kept as they are.
 */
export function mergeStreamedAudio(parts: Part[]): Part[] {
  const merged: Part[] = [];
  let i = 0;
  while (i < parts.length) {
    const part = parts[i];
    if (isWavAudio(part)) {
      let j = i + 1;
      while (j < parts.length && isHeaderlessAudio(parts[j])) j++;
      const audio = Buffer.concat(parts.slice(i + 1, j).map(mediaBytes));
      const wav =
        audio.length > 0 ? appendToWavData(mediaBytes(part), audio) : undefined;
      if (wav && part.media) {
        merged.push(mediaPart(part.media.contentType ?? 'audio/wav', wav));
        i = j;
        continue;
      }
      merged.push(part);
      i++;
      continue;
    }
    if (isHeaderlessAudio(part)) {
      const contentType = part.media?.contentType;
      let j = i + 1;
      while (
        j < parts.length &&
        isHeaderlessAudio(parts[j]) &&
        parts[j].media?.contentType === contentType
      ) {
        j++;
      }
      if (j - i > 1 && contentType) {
        merged.push(
          mediaPart(
            contentType,
            Buffer.concat(parts.slice(i, j).map(mediaBytes))
          )
        );
      } else {
        merged.push(...parts.slice(i, j));
      }
      i = j;
      continue;
    }
    merged.push(part);
    i++;
  }
  return merged;
}

export function fromInteraction(
  interaction: GeminiInteraction
): Operation<GenerateResponseData> {
  const op = { id: interaction.id } as Operation<GenerateResponseData>;
  if (interaction.status === 'in_progress' || interaction.status === 'queued') {
    // Still running, or waiting for capacity (e.g. flex/deferred tiers).
    op.done = false;
  } else if (interaction.status === 'failed') {
    // Always finish the operation on failure; leaving `done` unset would make
    // callers poll forever.
    op.done = true;
    if (isInteractionContentBlockCode(interaction.errors?.[0]?.code)) {
      op.output = fromFailedInteraction(interaction);
    } else {
      op.error = {
        message: interactionFailureMessage(interaction),
        ...(interaction.errors?.[0]?.code
          ? { code: interaction.errors[0].code }
          : {}),
      };
    }
  } else if (interaction.status === 'cancelled') {
    op.done = true;
    op.output = {
      ...(interaction.usage
        ? { usage: fromInteractionUsage(interaction.usage) }
        : {}),
      finishReason: 'aborted',
      finishMessage: 'Operation cancelled',
      message: {
        role: 'model',
        content: [{ text: 'Operation cancelled.' }],
        ...(interaction.id || interaction.environment_id
          ? {
              metadata: {
                ...(interaction.id ? { interactionId: interaction.id } : {}),
                ...(interaction.environment_id
                  ? { environmentId: interaction.environment_id }
                  : {}),
              },
            }
          : {}),
      },
    };
  } else if (
    interaction.status === 'completed' ||
    interaction.status === 'incomplete' ||
    interaction.status === 'requires_action'
  ) {
    // `requires_action` is a deliberate pause, not a failure: e.g. a
    // collaborative-planning plan awaiting approval, a client-side function
    // call, or an elicitation. The turn is over, so finish the operation and
    // surface what was produced; the caller continues with
    // `previousInteractionId`. Pending client function calls become
    // `toolRequest` parts.
    op.done = true;
    const steps = interaction.steps;
    if (steps?.length) {
      const pendingIds = getPendingFunctionCallIds(steps, interaction.status);
      const content = steps
        .flatMap((step) =>
          fromInteractionStep(
            step,
            step.type === 'function_call' && step.id
              ? pendingIds.has(step.id)
              : false
          )
        )
        .filter((p) => p && Object.keys(p).length > 0);
      op.output = {
        ...(interaction.status === 'incomplete'
          ? {
              finishReason: 'length' as const,
              finishMessage: INTERACTION_INCOMPLETE_MESSAGE,
            }
          : { finishReason: 'stop' as const }),
        message: {
          role: 'model',
          content,
          ...(interaction.id || interaction.environment_id
            ? {
                metadata: {
                  ...(interaction.id ? { interactionId: interaction.id } : {}),
                  ...(interaction.environment_id
                    ? { environmentId: interaction.environment_id }
                    : {}),
                  ...(interaction.status === 'requires_action'
                    ? { interactionStatus: interaction.status }
                    : {}),
                },
              }
            : {}),
        },
        custom: interaction,
        raw: interaction,
        ...(interaction.usage
          ? { usage: fromInteractionUsage(interaction.usage) }
          : {}),
      };
    }
  } else if (interaction.status) {
    // A status this plugin doesn't know. Finish the operation rather than
    // leaving `done` unset, which would make callers poll forever.
    op.done = true;
    op.error = {
      message: `Unknown interaction status: ${interaction.status}`,
    };
  }
  return op;
}
