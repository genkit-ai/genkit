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

import { z } from 'genkit';
import type { ModelInfo, ModelReference } from 'genkit/model';
import { compatOaiModelRef } from '../../model';

/**
 * Citations produced by Responses API built-in tools, attached to text
 * Parts via `metadata.citations`.
 *
 * Discriminated by `type`:
 *  - `url_citation` — produced by `web_search_preview`. Has UTF-16
 *    `startIndex`/`endIndex` offsets into the text the citation refers to.
 *  - `file_citation` — produced by `file_search`. Identifies a file in
 *    the retrieval set; `fileIndex` is the file's position in the call's
 *    `results[]` array (NOT a character offset).
 *
 * The schema is intentionally a structural type rather than a Genkit Part
 * subtype: it can be enriched later (e.g. with a first-class citation Part)
 * without breaking consumers that read `metadata.citations`.
 */
export const CitationSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('url_citation'),
    url: z.string(),
    title: z.string().optional(),
    /** UTF-16 code-unit offsets relative to the text Part the citation belongs to. */
    startIndex: z.number().int().nonnegative().optional(),
    endIndex: z.number().int().nonnegative().optional(),
  }),
  z.object({
    type: z.literal('file_citation'),
    fileId: z.string(),
    /** Index of the file in the originating file_search_call results array. */
    fileIndex: z.number().int().nonnegative().optional(),
  }),
]);
export type Citation = z.infer<typeof CitationSchema>;

/**
 * Built-in tool spec passed via {@link OpenAIResponsesConfigSchema.builtInTools}.
 *
 * Built-in tools are server-side and execute on OpenAI infrastructure (web
 * search, retrieval over user-uploaded vector stores, sandboxed Python
 * execution). They are distinct from function-call tools — function tools
 * are still passed through the standard Genkit `tools` field.
 */
export const BuiltInToolSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('web_search_preview'),
    userLocation: z
      .object({
        type: z.literal('approximate'),
        country: z.string().optional(),
        region: z.string().optional(),
        city: z.string().optional(),
        timezone: z.string().optional(),
      })
      .optional(),
    searchContextSize: z.enum(['low', 'medium', 'high']).optional(),
  }),
  z.object({
    type: z.literal('file_search'),
    vectorStoreIds: z.array(z.string()).min(1),
    maxNumResults: z.number().int().positive().optional(),
    ranker: z
      .object({
        ranker: z.enum(['auto', 'default-2024-11-15']).optional(),
        scoreThreshold: z.number().min(0).max(1).optional(),
      })
      .optional(),
  }),
  z.object({
    type: z.literal('code_interpreter'),
    /**
     * Container in which the interpreter runs. Two forms (matching the
     * SDK's `Tool.CodeInterpreter`):
     *  - omitted ⇒ defaults to `{ type: 'auto' }`
     *  - `{ type: 'auto', fileIds }` — auto-provisioned sandbox seeded
     *    with these files
     *  - `string` — explicit container id from a prior call
     */
    container: z
      .union([
        z.string(),
        z.object({
          type: z.literal('auto'),
          fileIds: z.array(z.string()).optional(),
        }),
      ])
      .optional(),
  }),
]);
export type BuiltInToolSpec = z.infer<typeof BuiltInToolSchema>;

/**
 * Configuration schema for OpenAI Responses API models.
 *
 * Notably distinct from {@link OpenAIChatCompletionConfigSchema} (Chat
 * Completions) — Responses-only fields like `previousResponseId`,
 * `builtInTools`, `reasoning`, `include`, `text.format` live here. The two
 * schemas are intentionally separated so that consumers of `openAI.model()`
 * and downstream compat providers (deepseek, xai, …) do not see Responses-
 * specific configuration in their typings.
 */
export const OpenAIResponsesConfigSchema = z.object({
  /** Sampling temperature; ignored by reasoning models that do not support it. */
  temperature: z.number().min(0).max(2).optional(),
  /** Top-p nucleus sampling. */
  topP: z.number().min(0).max(1).optional(),
  /** Max tokens to generate (mapped to `max_output_tokens` server-side). */
  maxOutputTokens: z.number().int().positive().optional(),
  /** End-user identifier for OpenAI usage attribution. */
  user: z.string().optional(),

  /**
   * Stateful chaining: pass the `responseId` returned in
   * `response.custom.responseId` from a previous turn to continue a
   * server-side conversation. Mutually informative with `store`.
   */
  previousResponseId: z.string().optional(),

  /**
   * Whether OpenAI persists the response on its servers for later
   * retrieval / chaining via `previous_response_id`.
   *
   * Defaults to `false` in this plugin (stateless-by-default), which is
   * the safer choice for ZDR-aware deployments. OpenAI's own API default
   * is `true`; the discrepancy is documented in the README.
   */
  store: z.boolean().optional(),

  /**
   * Server-side data to include in the response. Notable values:
   *  - `'reasoning.encrypted_content'` — opaque encrypted reasoning blob
   *    suitable for ZDR-compliant chaining.
   *  - `'file_search_call.results'` — retrieved document chunks.
   *  - `'message.input_image.image_url'` — surface input image URLs back.
   *  - `'computer_call_output.output.image_url'` — surface computer-use
   *    screenshot URLs back.
   *
   * (Web search results are surfaced through `metadata.citations` on the
   * generated text Parts, not via `include`.)
   */
  include: z
    .array(
      z.enum([
        'reasoning.encrypted_content',
        'file_search_call.results',
        'message.input_image.image_url',
        'computer_call_output.output.image_url',
      ])
    )
    .optional(),

  /**
   * Reasoning controls for o1/o3/gpt-5* family. `effort` trades latency
   * for depth; `summary` controls whether to surface a reasoning summary.
   */
  reasoning: z
    .object({
      effort: z.enum(['low', 'medium', 'high']).optional(),
      summary: z.enum(['auto', 'concise', 'detailed']).optional(),
    })
    .optional(),

  /** Server-side tools (web_search, file_search, code_interpreter). */
  builtInTools: z.array(BuiltInToolSchema).optional(),

  /**
   * `text.format` lets callers request structured / verbose output.
   * Genkit's `output.format='json'` + `output.schema` already wires
   * `text.format = { type: 'json_schema', strict: true, schema }`
   * automatically — set this only for advanced overrides.
   */
  text: z
    .object({
      verbosity: z.enum(['low', 'medium', 'high']).optional(),
      format: z
        .union([
          z.object({ type: z.literal('text') }),
          z.object({ type: z.literal('json_object') }),
          z.object({
            type: z.literal('json_schema'),
            name: z.string().optional(),
            schema: z.record(z.any()),
            strict: z.boolean().optional(),
            description: z.string().optional(),
          }),
        ])
        .optional(),
    })
    .optional(),

  /** OpenAI deployment routing tier. */
  serviceTier: z.enum(['auto', 'default', 'flex']).optional(),

  /** System-message replacement for models with `systemRole=false` (o1/o3). */
  instructions: z.string().optional(),

  /** Per-call metadata (16 KV pairs max, surfaced in the response). */
  metadata: z.record(z.string()).optional(),

  /** Whether multiple tool calls may be issued in parallel (default: true). */
  parallelToolCalls: z.boolean().optional(),

  /** Truncation strategy when context exceeds model window. */
  truncation: z.enum(['auto', 'disabled']).optional(),

  /** Cap on tool invocations within a single response. */
  maxToolCalls: z.number().int().positive().optional(),

  /** Override the underlying OpenAI model id (e.g. for snapshotted versions). */
  version: z.string().optional(),

  /** Per-request scoped API key (mirrors compat-oai pattern). */
  apiKey: z.string().optional(),
});
export type OpenAIResponsesConfig = z.infer<typeof OpenAIResponsesConfigSchema>;

/**
 * Helper to build a {@link ModelReference} for the OpenAI Responses API.
 * Uses the dedicated `'openai-responses'` namespace so it never collides
 * with `openai/...` Chat Completions registrations.
 *
 * @param params.name Bare model id (e.g. `'gpt-5-mini'`).
 * @param params.info Capability advertisement (`tools`, `media`, …).
 */
export function openAIResponsesModelRef(params: {
  name: string;
  info?: ModelInfo;
  config?: Partial<z.infer<typeof OpenAIResponsesConfigSchema>>;
}): ModelReference<typeof OpenAIResponsesConfigSchema> {
  return compatOaiModelRef({
    ...params,
    configSchema: OpenAIResponsesConfigSchema,
    namespace: 'openai-responses',
  });
}

const REASONING_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: true,
    // The OpenAI Responses API itself rejects `system` role messages
    // on o1/o3/gpt-5 reasoning models, but this plugin handles that
    // internally: `toResponsesRequestBody` lifts text-only system
    // messages into the top-level `instructions` field before the
    // request goes out. From Genkit's perspective the model accepts
    // system role, so we advertise `systemRole: true`.
    systemRole: true,
    output: ['text', 'json'],
  },
};

const STANDARD_RESPONSES_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: true,
    systemRole: true,
    output: ['text', 'json'],
  },
};

/**
 * Models known to be served via the Responses API.
 *
 * OpenAI also serves several of these names through Chat Completions (under
 * the existing `openai/...` namespace), so users can pick either path. New
 * capabilities (built-in tools, reasoning summaries, structured citations)
 * are only available via this namespace.
 */
export const SUPPORTED_RESPONSES_MODELS = {
  'gpt-5': openAIResponsesModelRef({
    name: 'gpt-5',
    info: STANDARD_RESPONSES_MODEL_INFO,
  }),
  'gpt-5-mini': openAIResponsesModelRef({
    name: 'gpt-5-mini',
    info: STANDARD_RESPONSES_MODEL_INFO,
  }),
  'gpt-5-nano': openAIResponsesModelRef({
    name: 'gpt-5-nano',
    info: STANDARD_RESPONSES_MODEL_INFO,
  }),
  'gpt-5.1': openAIResponsesModelRef({
    name: 'gpt-5.1',
    info: STANDARD_RESPONSES_MODEL_INFO,
  }),
  o1: openAIResponsesModelRef({
    name: 'o1',
    info: REASONING_MODEL_INFO,
  }),
  o3: openAIResponsesModelRef({
    name: 'o3',
    info: REASONING_MODEL_INFO,
  }),
  'o3-mini': openAIResponsesModelRef({
    name: 'o3-mini',
    info: REASONING_MODEL_INFO,
  }),
  'o4-mini': openAIResponsesModelRef({
    name: 'o4-mini',
    info: REASONING_MODEL_INFO,
  }),
} as const;
