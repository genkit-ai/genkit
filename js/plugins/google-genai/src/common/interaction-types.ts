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
import { ToolChoice } from 'genkit';
import { FunctionDeclaration, ToolConfig } from './types';

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
}

/**
 * A tool that can be used by the model to execute code.
 * See {@link https://ai.google.dev/gemini-api/docs/code-execution}
 */
export declare interface InteractionCodeExecutionTool {
  type: 'code_execution';
}

/**
 * A tool that can be used by the model.
 */
export declare type InteractionTool =
  | InteractionFunctionTool
  | InteractionGoogleSearchTool
  | InteractionCodeExecutionTool;

/**
 * Citation information for model-generated content.
 */
declare interface TextAnnotation {
  /** Start of segment of the response that is attributed to this source. */
  start_index?: number;
  /** End of the attributed segment, exclusive. */
  end_index?: number;
  /** Source attributed for a portion of the text. Could be a URL, title, or other identifier. */
  source?: string;
}

/**
 * A text content block.
 */
export declare interface TextContent {
  type: 'text';
  /** The text content. */
  text?: string;
  /** Citation information for model-generated content. */
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
  arguments?: Record<string, any>;
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
  result?: Record<string, any> | string;
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
  tool_choice?: ToolChoice | ToolConfig;
  /** The level of thought tokens that the model should generate. */
  thinking_level?: 'minimal' | 'low' | 'medium' | 'high';
  /** Whether to include thought summaries in the response. */
  thinking_summaries?: 'auto' | 'none';
  /** The maximum number of tokens to include in the response. */
  max_output_tokens?: number;
  /** Configuration for speech interaction. */
  speech_config?: SpeechConfig;
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
}

/**
 * Configuration for the agent.
 */
export type InteractionsAgentConfig =
  | DynamicAgentConfig
  | DeepResearchAgentConfig;

/**
 * Indicates the model should return text, images, or audio.
 */
export declare type ResponseModality = 'text' | 'image' | 'audio';

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

  /** The inputs for the interaction. */
  input: string | Content[] | Turn[] | Content;

  /** System instruction for the interaction. */
  system_instruction?: string;

  /** A list of tool declarations the model may call during interaction. */
  tools?: InteractionTool[];

  /** Enforces that the generated response is a JSON object that complies with the JSON schema specified in this field */
  response_format?: Record<string, any>;
  /** Required if responseFormat is set. */
  response_mime_type?: string;

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
}

/**
 * Response from creating an interaction.
 */
export declare interface GeminiInteraction {
  /** The name of the Model used for generating the interaction. */
  model?: string;
  /** The name of the Agent used for generating the interaction. */
  agent?: string;
  /** The unique identifier for the interaction completion. */
  id?: string; // The interactionId to be used in subsequent interactions
  /** The ID of the previous interaction, if any. */
  previous_interaction_id?: string;
  /** The status of the interaction. */
  status?:
    | 'in_progress'
    | 'requires_action'
    | 'completed'
    | 'failed'
    | 'cancelled';
  /** The time at which the response was created in ISO 8601 format. */
  created?: string;
  /** The time at which the response was last updated in ISO 8601 format. */
  updated?: string;
  /** The role of the interaction. */
  role?: string;
  /** Responses from the model. */
  outputs?: Content[];
  /** Statistics on the interaction request's token usage. */
  usage?: Usage;
}
