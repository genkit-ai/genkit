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
import { z } from 'genkit';
import { FunctionDeclaration } from '../common/types.js';

/**
 * A tool that can be used by the model.
 * See {@link https://ai.google.dev/gemini-api/docs/function-calling}
 */
export declare interface InteractionFunctionTool extends FunctionDeclaration {
  type: 'function';
}

/**
 * A tool that can be used by the model to search Google.
 * See {@link https://ai.google.dev/gemini-api/docs/grounding}
 */
export declare interface InteractionGoogleSearchTool {
  type: 'google_search';
  search_types?: ('web_search' | 'image_search' | 'enterprise_web_search')[];
}

/**
 * A tool that can be used by the model to execute code.
 * See {@link https://ai.google.dev/gemini-api/docs/code-execution}
 */
export declare interface InteractionCodeExecutionTool {
  type: 'code_execution';
}

/**
 * A tool that can be used by the model to read and summarize web page content.
 */
export declare interface InteractionUrlContextTool {
  type: 'url_context';
}

/**
 * A tool that can be used by the model to search uploaded document corpora.
 */
export declare interface InteractionFileSearchTool {
  type: 'file_search';
  file_search_store_names?: string[];
  metadata_filter?: string;
  top_k?: number;
}

export declare interface InteractionAllowedTools {
  mode?: string;
  tools?: string[];
}

export declare interface InteractionToolChoiceConfig {
  allowed_tools?: InteractionAllowedTools;
}

/**
 * A tool that can be used by the model to connect to remote MCP servers.
 */
export declare interface InteractionMcpServerTool {
  type: 'mcp_server';
  name?: string;
  url?: string;
  headers?: Record<string, string>;
  allowed_tools?: InteractionAllowedTools[];
}

export declare interface InteractionGoogleMapsTool {
  type: 'google_maps';
  enable_widget?: boolean;
  latitude?: number;
  longitude?: number;
}

export declare interface InteractionComputerUseTool {
  type: 'computer_use';
  disabled_safety_policies?: string[];
  enable_prompt_injection_detection?: boolean;
  environment?: string;
  excluded_predefined_functions?: string[];
}

export declare interface InteractionRetrievalTool {
  type: 'retrieval';
  exa_ai_search_config?: Record<string, unknown>;
  parallel_ai_search_config?: Record<string, unknown>;
  rag_store_config?: Record<string, unknown>;
  retrieval_types?: string[];
  vertex_ai_search_config?: Record<string, unknown>;
}

export declare interface InteractionDynamicTool {
  type: string;
  [key: string]: unknown;
}

/**
 * A tool that can be used by the model.
 */
export declare type InteractionTool =
  | InteractionFunctionTool
  | InteractionGoogleSearchTool
  | InteractionCodeExecutionTool
  | InteractionUrlContextTool
  | InteractionFileSearchTool
  | InteractionMcpServerTool
  | InteractionGoogleMapsTool
  | InteractionComputerUseTool
  | InteractionRetrievalTool
  | InteractionDynamicTool;

// Text annotations are defined as Zod schemas (used for runtime validation)
// with their TypeScript types derived via z.infer, so the shape is written
// once. `.passthrough()` keeps fields not listed here.

/** Fields shared by all text annotations. */
const AnnotationSpanShape = {
  /** Start of the annotated segment (in bytes). */
  start_index: z.number().optional(),
  /** End of the annotated segment, exclusive. */
  end_index: z.number().optional(),
};

/** A URL citation annotation. */
export const UrlCitationSchema = z
  .object({
    type: z.literal('url_citation'),
    ...AnnotationSpanShape,
    /** The URL. */
    url: z.string().optional(),
    /** The title of the URL. */
    title: z.string().optional(),
  })
  .passthrough();
export type UrlCitation = z.infer<typeof UrlCitationSchema>;

/** A file citation annotation. */
export const FileCitationSchema = z
  .object({
    type: z.literal('file_citation'),
    ...AnnotationSpanShape,
    /** The URI of the file. */
    document_uri: z.string().optional(),
    /** The name of the file. */
    file_name: z.string().optional(),
    /** Source attributed for a portion of the text. */
    source: z.string().optional(),
    /** Page number of the cited document, if applicable. */
    page_number: z.number().optional(),
    /** Media ID in case of image citations, if applicable. */
    media_id: z.string().optional(),
    /** User-provided metadata about the retrieved context. */
    custom_metadata: z.record(z.unknown()).optional(),
  })
  .passthrough();
export type FileCitation = z.infer<typeof FileCitationSchema>;

/** A place (Google Maps) citation annotation. */
export const PlaceCitationSchema = z
  .object({
    type: z.literal('place_citation'),
    ...AnnotationSpanShape,
    /** Title of the place. */
    name: z.string().optional(),
    /** The ID of the place, in `places/{place_id}` format. */
    place_id: z.string().optional(),
    /** URI reference of the place. */
    url: z.string().optional(),
    /** Review snippets about features of the place. */
    review_snippets: z
      .array(
        z.object({
          review_id: z.string().optional(),
          title: z.string().optional(),
          url: z.string().optional(),
        })
      )
      .optional(),
  })
  .passthrough();
export type PlaceCitation = z.infer<typeof PlaceCitationSchema>;

/**
 * Speech annotation for text content. Used as input to TTS models to assign a
 * turn to a speaker and set its delivery style.
 */
export const SpeechAnnotationSchema = z
  .object({
    type: z.literal('speech_metadata'),
    ...AnnotationSpanShape,
    /** The speaker to associate with this turn. */
    speaker: z.string().optional(),
    /** Style instruction for the speech synthesis. */
    style: z.string().optional(),
  })
  .passthrough();
export type SpeechAnnotation = z.infer<typeof SpeechAnnotationSchema>;

/** Word-level speech recognition annotation for transcription output. */
export const WordInfoSchema = z
  .object({
    type: z.literal('word_info'),
    ...AnnotationSpanShape,
    /** The transcribed word. */
    text: z.string().optional(),
    /** Start offset in time of the word relative to the start of the audio. */
    start_offset: z.string().optional(),
    /** End offset in time of the word relative to the start of the audio. */
    end_offset: z.string().optional(),
    /** Speaker label for this word (e.g. "spk_1"). */
    speaker: z.string().optional(),
  })
  .passthrough();
export type WordInfo = z.infer<typeof WordInfoSchema>;

/** An annotation on text content, discriminated by `type`. */
export const TextAnnotationSchema = z.discriminatedUnion('type', [
  UrlCitationSchema,
  FileCitationSchema,
  PlaceCitationSchema,
  SpeechAnnotationSchema,
  WordInfoSchema,
]);
export type TextAnnotation = z.infer<typeof TextAnnotationSchema>;

/**
 * A text content block.
 */
export declare interface TextContent {
  type: 'text';
  /** The text content. */
  text?: string;
  /**
   * Annotations on the text: citations on model output, or speech metadata on
   * TTS input.
   */
  annotations?: TextAnnotation[];
}

/**
 * The resolution of the media.
 */
export declare type MediaResolution = 'low' | 'medium' | 'high' | 'ultra_high';

/**
 * An image content block.
 */
export declare interface ImageContent {
  type: 'image';
  /** The image content. */
  data?: string;
  /** The URI of the image. */
  uri?: string;
  /** The mime type of the image. */
  mime_type?: string;
  /** The resolution of the media. */
  resolution?: MediaResolution;
}

/**
 * An audio content block.
 */
export declare interface AudioContent {
  type: 'audio';
  /** The audio content. */
  data?: string;
  /** The URI of the audio. */
  uri?: string;
  /** The mime type of the audio. */
  mime_type?: string;
}

/**
 * A document content block.
 */
export declare interface DocumentContent {
  type: 'document';
  /** The document content. */
  data?: string;
  /** The URI of the document. */
  uri?: string;
  /** The mime type of the document. */
  mime_type?: string;
}

/**
 * A video content block.
 */
export declare interface VideoContent {
  type: 'video';
  /** The video content. */
  data?: string;
  /** The URI of the video. */
  uri?: string;
  /** The mime type of the video. */
  mime_type?: string;
  /** The resolution of the media. */
  resolution?: MediaResolution;
}

/**
 * A thought content block.
 */
export declare interface ThoughtContent {
  type: 'thought';
  /** Signature to match the backend source to be part of the generation. */
  signature?: string;
  /** A summary of the thought. */
  summary?: (TextContent | ImageContent)[];
}

/**
 * A function tool call content block.
 */
export declare interface FunctionCallContent {
  type: 'function_call';
  /** The name of the tool to call. */
  name: string;
  /** The arguments to pass to the function. */
  arguments?: Record<string, unknown>;
  /** A unique ID for this specific tool call. */
  id: string;
}

/**
 * A function tool result content block.
 */
export declare interface FunctionResultContent {
  type: 'function_result';
  /** The name of the tool that was called. */
  name: string;
  /** Whether the tool call resulted in an error. */
  is_error?: boolean;
  /** The result of the tool call. */
  result?: Record<string, unknown> | string;
  /** ID to match the ID from the function call block. */
  call_id: string;
}

/**
 * The content of the response.
 */
export type Content =
  | TextContent
  | ImageContent
  | AudioContent
  | DocumentContent
  | VideoContent
  | ThoughtContent
  | FunctionCallContent
  | FunctionResultContent;

export declare interface ModelOutputStep {
  type: 'model_output';
  content: Content[];
}

export declare interface UserInputStep {
  type: 'user_input';
  content: Content[];
}

export declare interface GoogleSearchCallStep {
  type: 'google_search_call';
  id: string;
  arguments: { queries: string[] };
  signature?: string;
}

export declare interface GoogleSearchResultStep {
  type: 'google_search_result';
  call_id: string;
  result: Record<string, unknown>;
  signature?: string;
}

export declare interface CodeExecutionCallStep {
  type: 'code_execution_call';
  id: string;
  arguments: { code: string; language?: string; [key: string]: unknown };
  signature?: string;
}

export declare interface CodeExecutionResultStep {
  type: 'code_execution_result';
  call_id: string;
  result: Record<string, unknown> | string;
  signature?: string;
}

export declare interface FunctionCallStep {
  type: 'function_call';
  name: string;
  arguments?: Record<string, unknown>;
  id: string;
  signature?: string;
}

export declare interface FunctionResultStep {
  type: 'function_result';
  name?: string;
  call_id: string;
  is_error?: boolean;
  result: Record<string, unknown> | string | (ImageContent | TextContent)[];
  signature?: string;
}

export declare interface ThoughtStep {
  type: 'thought';
  signature?: string;
  summary?: (TextContent | ImageContent)[];
}

export type Step =
  | ModelOutputStep
  | UserInputStep
  | Content
  | GoogleSearchCallStep
  | GoogleSearchResultStep
  | CodeExecutionCallStep
  | CodeExecutionResultStep
  | FunctionCallStep
  | FunctionResultStep
  | ThoughtStep;

/**
 * A turn in a conversation.
 */
export declare interface Turn {
  /** The originator of this turn. Must be user for input or model for model output. */
  role: string;
  /** The content of the turn. */
  content: string | Content[];
}

/**
 * The token count for a single response modality.
 */
export declare interface ModalityTokens {
  /** The modality associated with the token count. */
  modality?: ResponseModality;
  /** Number of tokens for the modality. */
  tokens?: number;
}

/**
 * Statistics on the interaction request's token usage.
 */
// These match what comes back from the REST API
// Unfortunately they are forced snake case at the API level.
export declare interface Usage {
  /** Number of tokens in the prompt (context). */
  total_input_tokens?: number;
  /** A breakdown of input token usage by modality. */
  input_tokens_by_modality?: ModalityTokens[];
  /** Number of tokens in the cached part of the prompt (the cached content). */
  total_cached_tokens?: number;
  /** A breakdown of cached token usage by modality. */
  cached_tokens_by_modality?: ModalityTokens[];
  /** Total number of tokens across all the generated responses. */
  total_output_tokens?: number;
  /** A breakdown of output token usage by modality. */
  output_tokens_by_modality?: ModalityTokens[];
  /** Number of tokens present in tool-use prompt(s). */
  total_tool_use_tokens?: number;
  /** A breakdown of tool-use token usage by modality. */
  tool_use_by_modality?: ModalityTokens[];
  /** Number of tokens of thoughts for thinking models. */
  total_thought_tokens?: number;
  /** Total token count for the interaction request (prompt + responses + other internal tokens). */
  total_tokens?: number;
}

/**
 * The configuration for speech interaction.
 */
export declare interface SpeechConfig {
  /** The voice of the speaker. */
  voice?: string;
  /** The language of the speech. */
  language?: string;
  /** The speaker's name, it should match the speaker name given in the prompt. */
  speaker?: string;
}

/**
 * The configuration for image interaction.
 */
export declare interface ImageConfig {
  /** The aspect ratio of the image. */
  aspect_ratio?: string;
  /** The size of the image. */
  image_size?: string;
}

/**
 * Configuration parameters for model interactions.
 */
export declare interface ModelGenerationConfig {
  /** Controls the randomness of the output. */
  temperature?: number;
  /** The maximum cumulative probability of tokens to consider when sampling. */
  top_p?: number;
  /** Seed used in decoding for reproducibility. */
  seed?: number;
  /** A list of character sequences that will stop output interaction. */
  stop_sequences?: string[];
  /** The tool choice for the interaction. */
  tool_choice?: string | InteractionToolChoiceConfig;
  /** The level of thought tokens that the model should generate. */
  thinking_level?: 'minimal' | 'low' | 'medium' | 'high';
  /** Whether to include thought summaries in the response. */
  thinking_summaries?: 'auto' | 'none';
  /** The maximum number of tokens to include in the response. */
  max_output_tokens?: number;
  /**
   * Speech configuration: an array for a single speaker, or
   * `{ speakers: [...] }` for multi-speaker.
   */
  speech_config?: SpeechConfig[] | { speakers: SpeechConfig[] };
  /** Configuration for image interaction. */
  image_config?: ImageConfig;
}

/**
 * Configuration for dynamic agents.
 */
export declare interface DynamicAgentConfig {
  type: 'dynamic';
}

/**
 * Configuration for the Deep Research agent.
 */
export declare interface DeepResearchAgentConfig {
  type: 'deep-research';
  /** Whether to include thought summaries in the response. */
  thinking_summaries?: 'auto' | 'none';
  /** Visualization allows the agent to generate charts and graphs to support its findings. */
  visualization?: 'auto' | 'off';
  /** Collaborative planning allows you to review and refine the research plan before execution. */
  collaborative_planning?: boolean;
}

/**
 * Service Tier
 */
export declare type ServiceTier =
  | 'flex'
  | 'standard'
  | 'priority'
  | (string & {});

/**
 * Configuration for the agent.
 */
export type InteractionsAgentConfig =
  | DynamicAgentConfig
  | DeepResearchAgentConfig;

/**
 * Indicates the model should return text, images, or audio.
 */
export declare type ResponseModality =
  | 'text'
  | 'image'
  | 'audio'
  | (string & {});

/**
 * Parameters for creating interactions.
 */
export declare interface CreateInteractionRequest {
  /** The ID of the previous interaction, if any. */
  previous_interaction_id?: string;

  /** The model to use for this request (mutually exclusive with agent) */
  model?: string;
  /** The agent to use for this request (mutually exclusive with model) */
  agent?: string;

  /** The environment configuration for the sandbox. */
  environment?: string | Record<string, unknown>;

  /** The inputs for the interaction. */
  input: string | Step[] | Step;

  /** System instruction for the interaction. */
  system_instruction?: string;

  /** A list of tool declarations the model may call during interaction. */
  tools?: InteractionTool[];

  /** Enforces that the generated response is a JSON object that complies with the JSON schema specified in this field */
  response_format?: Record<string, unknown> | Record<string, unknown>[];

  /** The requested modalities of the response (TEXT, IMAGE, AUDIO). */
  response_modalities?: ResponseModality[];

  /** Whether the interaction will be streamed. */
  stream?: boolean;
  /** Whether to store the response and request for later retrieval. */
  store?: boolean;
  /** Whether to run the model interaction in the background. */
  background?: boolean;

  /** Configuration parameters for the model interaction. */
  generation_config?: ModelGenerationConfig;
  /** Configuration for the agent. */
  agent_config?: InteractionsAgentConfig;
  /** */
  service_tier?: ServiceTier;
}

export interface TextDelta {
  type: 'text';
  text: string;
}

export interface ImageDelta {
  type: 'image';
  data?: string;
  uri?: string;
  mime_type?: string;
  resolution?: MediaResolution;
}

export interface AudioDelta {
  type: 'audio';
  data?: string;
  uri?: string;
  mime_type?: string;
  sample_rate?: number;
  channels?: number;
}

export interface DocumentDelta {
  type: 'document';
  data?: string;
  uri?: string;
  mime_type?: string;
}

export interface VideoDelta {
  type: 'video';
  data?: string;
  uri?: string;
  mime_type?: string;
  resolution?: MediaResolution;
}

export interface ThoughtSummaryDelta {
  type: 'thought_summary';
  content?: Content;
}

export interface ThoughtSignatureDelta {
  type: 'thought_signature';
  signature?: string;
}

export interface FunctionCallDelta {
  type: 'function_call';
  name: string;
  arguments: Record<string, unknown>;
  id: string;
}

export interface ArgumentsDelta {
  type: 'arguments_delta';
  arguments?: string;
}

export interface CodeExecutionCallDelta {
  type: 'code_execution_call';
  arguments: { code?: string; language?: string; [key: string]: unknown };
  signature?: string;
}

export interface UrlContextCallDelta {
  type: 'url_context_call';
  arguments: { urls?: string[] };
  signature?: string;
}

export interface GoogleSearchCallDelta {
  type: 'google_search_call';
  arguments: { queries?: string[] };
  signature?: string;
}

export interface McpServerToolCallDelta {
  type: 'mcp_server_tool_call';
  name: string;
  server_name: string;
  arguments: Record<string, unknown>;
}

export interface FileSearchCallDelta {
  type: 'file_search_call';
  signature?: string;
}

export interface GoogleMapsCallDelta {
  type: 'google_maps_call';
  arguments?: { queries?: string[] };
  signature?: string;
}

export interface FunctionResultDelta {
  type: 'function_result';
  name?: string;
  call_id: string;
  is_error?: boolean;
  result: Record<string, unknown> | string | (ImageContent | TextContent)[];
}

export interface CodeExecutionResultDelta {
  type: 'code_execution_result';
  result: string;
  is_error?: boolean;
  signature?: string;
}

export interface UrlContextResultDelta {
  type: 'url_context_result';
  result: Record<string, unknown>[];
  is_error?: boolean;
  signature?: string;
}

export interface GoogleSearchResultDelta {
  type: 'google_search_result';
  result: Record<string, unknown>[];
  is_error?: boolean;
  signature?: string;
}

export interface McpServerToolResultDelta {
  type: 'mcp_server_tool_result';
  name?: string;
  server_name?: string;
  result: Record<string, unknown> | string | (ImageContent | TextContent)[];
}

export interface FileSearchResultDelta {
  type: 'file_search_result';
  result: Record<string, unknown>[];
  signature?: string;
}

export interface GoogleMapsResultDelta {
  type: 'google_maps_result';
  result?: Record<string, unknown>[];
  signature?: string;
}

export interface TextAnnotationDelta {
  type: 'text_annotation_delta';
  annotations?: TextAnnotation[];
}

export type StepDeltaData =
  | TextDelta
  | ImageDelta
  | AudioDelta
  | DocumentDelta
  | VideoDelta
  | ThoughtSummaryDelta
  | ThoughtSignatureDelta
  | FunctionCallDelta
  | ArgumentsDelta
  | CodeExecutionCallDelta
  | UrlContextCallDelta
  | GoogleSearchCallDelta
  | McpServerToolCallDelta
  | FileSearchCallDelta
  | GoogleMapsCallDelta
  | FunctionResultDelta
  | CodeExecutionResultDelta
  | UrlContextResultDelta
  | GoogleSearchResultDelta
  | McpServerToolResultDelta
  | FileSearchResultDelta
  | GoogleMapsResultDelta
  | TextAnnotationDelta;

export type InteractionSseEvent =
  | {
      event_type: 'interaction.created';
      interaction: Partial<GeminiInteraction>;
      event_id?: string;
    }
  | {
      event_type: 'interaction.completed';
      interaction: Partial<GeminiInteraction>;
      event_id?: string;
    }
  | {
      event_type: 'interaction.status_update';
      /** Only present when the interaction is stored (`store: true`). */
      interaction_id?: string;
      status: GeminiInteraction['status'];
      event_id?: string;
    }
  | {
      event_type: 'error';
      error: { code: string; message: string };
      event_id?: string;
    }
  | { event_type: 'step.start'; index: number; step: Step; event_id?: string }
  | {
      event_type: 'step.delta';
      index: number;
      delta: StepDeltaData;
      event_id?: string;
    }
  | { event_type: 'step.stop'; index: number; event_id?: string };

export interface InteractionStreamResult {
  stream: AsyncGenerator<InteractionSseEvent>;
  response: Promise<GeminiInteraction>;
}

export declare interface GeminiInteraction {
  /** The name of the Model used for generating the interaction. */
  model?: string;
  /** The name of the Agent used for generating the interaction. */
  agent?: string;
  /** The environment ID for the sandbox, if used. */
  environment_id?: string;
  /** The unique identifier for the interaction completion. */
  id?: string; // The interactionId to be used in subsequent interactions
  /** The ID of the previous interaction, if any. */
  previous_interaction_id?: string;
  /** The status of the interaction. */
  status?:
    | 'in_progress'
    | 'queued'
    | 'requires_action'
    | 'completed'
    | 'incomplete'
    | 'failed'
    | 'cancelled';
  /** The time at which the response was created in ISO 8601 format. */
  created?: string;
  /** The time at which the response was last updated in ISO 8601 format. */
  updated?: string;
  /** The role of the interaction. */
  role?: string;
  /** Steps comprising the interaction timeline. */
  steps?: Step[];
  /** Statistics on the interaction request's token usage. */
  usage?: Usage;
  /**
   * Output only. Errors recorded on the interaction. Populated when `status`
   * is `failed` (e.g. a safety block or an unrecoverable tool-call error).
   */
  errors?: InteractionError[];
}

/** An error recorded on an interaction or sent in an SSE `error` event. */
export declare interface InteractionError {
  /** Error code, e.g. `safety` or `malformed_function_call`. */
  code?: string;
  /** Human-readable error message. */
  message?: string;
}
