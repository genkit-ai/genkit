/**
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

import { OperationSchema, z } from '@genkit-ai/core';
import { DocumentDataSchema } from './document.js';
import {
  CustomPartSchema,
  DataPartSchema,
  MediaPartSchema,
  ReasoningPartSchema,
  ResourcePartSchema,
  TextPartSchema,
  ToolRequestPartSchema,
  ToolResponsePartSchema,
} from './parts.js';

//
// IMPORTANT: Please keep type definitions in sync with
//   genkit-tools/src/types/model.ts
//

/**
 * Zod schema of message part.
 */
export const PartSchema = z.union([
  TextPartSchema,
  MediaPartSchema,
  ToolRequestPartSchema,
  ToolResponsePartSchema,
  DataPartSchema,
  CustomPartSchema,
  ReasoningPartSchema,
  ResourcePartSchema,
]);

/**
 * Message part.
 */
export type Part = z.infer<typeof PartSchema>;

/**
 * Zod schema of a message role.
 */
export const RoleSchema = z.enum(['system', 'user', 'model', 'tool']);

/**
 * Message role.
 */
export type Role = z.infer<typeof RoleSchema>;

/**
 * Zod schema of a message.
 */
export const MessageSchema = z.object({
  role: RoleSchema,
  content: z.array(PartSchema),
  metadata: z.record(z.unknown()).optional(),
});

/**
 * Model message data.
 */
export type MessageData = z.infer<typeof MessageSchema>;

/**
 * Zod schema of model info metadata.
 */
export const ModelInfoSchema = z.object({
  /** Acceptable names for this model (e.g. different versions). */
  versions: z.array(z.string()).optional(),
  /** Friendly label for this model (e.g. "Google AI - Gemini Pro") */
  label: z.string().optional(),
  /** Model Specific configuration. */
  configSchema: z.record(z.any()).optional(),
  /** Supported model capabilities. */
  supports: z
    .object({
      /** Model can process historical messages passed with a prompt. */
      multiturn: z.boolean().optional(),
      /** Model can process media as part of the prompt (multimodal input). */
      media: z.boolean().optional(),
      /** Model can perform tool calls. */
      tools: z.boolean().optional(),
      /** Model can accept messages with role "system". */
      systemRole: z.boolean().optional(),
      /** Model can output this type of data. */
      output: z.array(z.string()).optional(),
      /** Model supports output in these content types. */
      contentType: z.array(z.string()).optional(),
      /** Model can natively support document-based context grounding. */
      context: z.boolean().optional(),
      /** Model can natively support constrained generation. */
      constrained: z.enum(['none', 'all', 'no-tools']).optional(),
      /** Model supports controlling tool choice, e.g. forced tool calling. */
      toolChoice: z.boolean().optional(),
    })
    .optional(),
  /** At which stage of development this model is.
   * - `featured` models are recommended for general use.
   * - `stable` models are well-tested and reliable.
   * - `unstable` models are experimental and may change.
   * - `legacy` models are no longer recommended for new projects.
   * - `deprecated` models are deprecated by the provider and may be removed in future versions.
   */
  stage: z
    .enum(['featured', 'stable', 'unstable', 'legacy', 'deprecated'])
    .optional(),
});

/**
 * Model info metadata.
 */
export type ModelInfo = z.infer<typeof ModelInfoSchema>;

/**
 * Zod schema of a tool definition.
 */
export const ToolDefinitionSchema = z.object({
  name: z.string(),
  description: z.string(),
  inputSchema: z
    .record(z.any())
    .describe('Valid JSON Schema representing the input of the tool.')
    .nullish(),
  outputSchema: z
    .record(z.any())
    .describe('Valid JSON Schema describing the output of the tool.')
    .nullish(),
  metadata: z
    .record(z.any())
    .describe('additional metadata for this tool definition')
    .optional(),
});

/**
 * Tool definition.
 */
export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;

/**
 * Configuration parameter descriptions.
 */
export const GenerationCommonConfigDescriptions = {
  temperature:
    'Controls the degree of randomness in token selection. A lower value is ' +
    'good for a more predictable response. A higher value leads to more ' +
    'diverse or unexpected results.',
  maxOutputTokens: 'The maximum number of tokens to include in the response.',
  topK: 'The maximum number of tokens to consider when sampling.',
  topP:
    'Decides how many possible words to consider. A higher value means ' +
    'that the model looks at more possible words, even the less likely ' +
    'ones, which makes the generated text more diverse.',
};

/**
 * Zod schema of a common config object.
 */
export const GenerationCommonConfigSchema = z
  .object({
    version: z
      .string()
      .describe(
        'A specific version of a model family, e.g. `gemini-2.5-flash` ' +
          'for the `googleai` family.'
      )
      .optional(),
    temperature: z
      .number()
      .describe(GenerationCommonConfigDescriptions.temperature)
      .optional(),
    maxOutputTokens: z
      .number()
      .describe(GenerationCommonConfigDescriptions.maxOutputTokens)
      .optional(),
    topK: z
      .number()
      .describe(GenerationCommonConfigDescriptions.topK)
      .optional(),
    topP: z
      .number()
      .describe(GenerationCommonConfigDescriptions.topP)
      .optional(),
    stopSequences: z
      .array(z.string())
      .max(5)
      .describe(
        'Set of character sequences (up to 5) that will stop output generation.'
      )
      .optional(),
    apiKey: z
      .string()
      .describe(
        'API Key to use for the model call, overrides API key provided in plugin config.'
      )
      .optional(),
  })
  .passthrough();

/**
 * Common config object.
 */
export type GenerationCommonConfig = typeof GenerationCommonConfigSchema;

/**
 * Zod schema of output config.
 */
export const OutputConfigSchema = z.object({
  format: z.string().optional(),
  schema: z.record(z.any()).optional(),
  constrained: z.boolean().optional(),
  contentType: z.string().optional(),
});

/**
 * Output config.
 */
export type OutputConfig = z.infer<typeof OutputConfigSchema>;

/** ModelRequestSchema represents the parameters that are passed to a model when generating content. */
export const ModelRequestSchema = z.object({
  messages: z.array(MessageSchema),
  config: z.any().optional(),
  tools: z.array(ToolDefinitionSchema).optional(),
  toolChoice: z.enum(['auto', 'required', 'none']).optional(),
  output: OutputConfigSchema.optional(),
  docs: z.array(DocumentDataSchema).optional(),
});
/** ModelRequest represents the parameters that are passed to a model when generating content. */
export interface ModelRequest<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
> extends z.infer<typeof ModelRequestSchema> {
  config?: z.infer<CustomOptionsSchema>;
}
/**
 * Zod schema of a generate request.
 */
export const GenerateRequestSchema = ModelRequestSchema.extend({
  /** @deprecated All responses now return a single candidate. This will always be `undefined`. */
  candidates: z.number().optional(),
});

/**
 * Generate request data.
 */
export type GenerateRequestData = z.infer<typeof GenerateRequestSchema>;

/**
 * Generate request.
 */
export interface GenerateRequest<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
> extends z.infer<typeof GenerateRequestSchema> {
  config?: z.infer<CustomOptionsSchema>;
}

/**
 * Zod schema of usage info from a generate request.
 */
export const GenerationUsageSchema = z.object({
  inputTokens: z.number().optional(),
  outputTokens: z.number().optional(),
  totalTokens: z.number().optional(),
  inputCharacters: z.number().optional(),
  outputCharacters: z.number().optional(),
  inputImages: z.number().optional(),
  outputImages: z.number().optional(),
  inputVideos: z.number().optional(),
  outputVideos: z.number().optional(),
  inputAudioFiles: z.number().optional(),
  outputAudioFiles: z.number().optional(),
  custom: z.record(z.number()).optional(),
  thoughtsTokens: z.number().optional(),
  cachedContentTokens: z.number().optional(),
});

/**
 * Usage info from a generate request.
 */
export type GenerationUsage = z.infer<typeof GenerationUsageSchema>;

/** Model response finish reason enum. */
export const FinishReasonSchema = z.enum([
  'stop',
  'length',
  'blocked',
  'interrupted',
  'other',
  'unknown',
]);

/** @deprecated All responses now return a single candidate. Only the first candidate will be used if supplied. */
export const CandidateSchema = z.object({
  index: z.number(),
  message: MessageSchema,
  usage: GenerationUsageSchema.optional(),
  finishReason: FinishReasonSchema,
  finishMessage: z.string().optional(),
  custom: z.unknown(),
});
/** @deprecated All responses now return a single candidate. Only the first candidate will be used if supplied. */
export type CandidateData = z.infer<typeof CandidateSchema>;

/** @deprecated All responses now return a single candidate. Only the first candidate will be used if supplied. */
export const CandidateErrorSchema = z.object({
  index: z.number(),
  code: z.enum(['blocked', 'other', 'unknown']),
  message: z.string().optional(),
});
/** @deprecated All responses now return a single candidate. Only the first candidate will be used if supplied. */
export type CandidateError = z.infer<typeof CandidateErrorSchema>;

/**
 * Zod schema of a model response.
 */
export const ModelResponseSchema = z.object({
  message: MessageSchema.optional(),
  finishReason: FinishReasonSchema,
  finishMessage: z.string().optional(),
  latencyMs: z.number().optional(),
  usage: GenerationUsageSchema.optional(),
  /** @deprecated use `raw` instead */
  custom: z.unknown(),
  raw: z.unknown(),
  request: GenerateRequestSchema.optional(),
  operation: OperationSchema.optional(),
});

/**
 * Model response data.
 */
export type ModelResponseData = z.infer<typeof ModelResponseSchema>;

/**
 * Zod schema of generaete response.
 */
export const GenerateResponseSchema = ModelResponseSchema.extend({
  /** @deprecated All responses now return a single candidate. Only the first candidate will be used if supplied. Return `message`, `finishReason`, and `finishMessage` instead. */
  candidates: z.array(CandidateSchema).optional(),
  finishReason: FinishReasonSchema.optional(),
});

/**
 * Generate response data.
 */
export type GenerateResponseData = z.infer<typeof GenerateResponseSchema>;

/** ModelResponseChunkSchema represents a chunk of content to stream to the client. */
export const ModelResponseChunkSchema = z.object({
  role: RoleSchema.optional(),
  /** index of the message this chunk belongs to. */
  index: z.number().optional(),
  /** The chunk of content to stream right now. */
  content: z.array(PartSchema),
  /** Model-specific extra information attached to this chunk. */
  custom: z.unknown().optional(),
  /** If true, the chunk includes all data from previous chunks. Otherwise, considered to be incremental. */
  aggregated: z.boolean().optional(),
});
export type ModelResponseChunkData = z.infer<typeof ModelResponseChunkSchema>;

export const GenerateResponseChunkSchema = ModelResponseChunkSchema;
export type GenerateResponseChunkData = z.infer<
  typeof GenerateResponseChunkSchema
>;

export const GenerateActionOutputConfig = z.object({
  format: z.string().optional(),
  contentType: z.string().optional(),
  instructions: z.union([z.boolean(), z.string()]).optional(),
  jsonSchema: z.any().optional(),
  constrained: z.boolean().optional(),
});

export const GenerateActionOptionsSchema = z.object({
  /** A model name (e.g. `vertexai/gemini-1.0-pro`). */
  model: z.string().optional(),
  /** Retrieved documents to be used as context for this generation. */
  docs: z.array(DocumentDataSchema).optional(),
  /** Conversation history for multi-turn prompting when supported by the underlying model. */
  messages: z.array(MessageSchema),
  /** List of registered tool names for this generation if supported by the underlying model. */
  tools: z.array(z.string()).optional(),
  /** List of registered resource names for this generation if supported by the underlying model. */
  resources: z.array(z.string()).optional(),
  /** Tool calling mode. `auto` lets the model decide whether to use tools, `required` forces the model to choose a tool, and `none` forces the model not to use any tools. Defaults to `auto`.  */
  toolChoice: z.enum(['auto', 'required', 'none']).optional(),
  /** Configuration for the generation request. */
  config: z.any().optional(),
  /** Configuration for the desired output of the request. Defaults to the model's default output if unspecified. */
  output: GenerateActionOutputConfig.optional(),
  /** Options for resuming an interrupted generation. */
  resume: z
    .object({
      respond: z.array(ToolResponsePartSchema).optional(),
      restart: z.array(ToolRequestPartSchema).optional(),
      metadata: z.record(z.any()).optional(),
    })
    .optional(),
  /** When true, return tool calls for manual processing instead of automatically resolving them. */
  returnToolRequests: z.boolean().optional(),
  /** Maximum number of tool call iterations that can be performed in a single generate call (default 5). */
  maxTurns: z.number().optional(),
  /** Custom step name for this generate call to display in trace views. Defaults to "generate". */
  stepName: z.string().optional(),
});
export type GenerateActionOptions = z.infer<typeof GenerateActionOptionsSchema>;
