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
  GenkitError,
  z,
  type GenerateRequest,
  type GenerateResponseData,
  type GenerationUsage,
} from 'genkit';
import { logger } from 'genkit/logging';
import type { ModelAction, ModelInfo } from 'genkit/model';
import { model } from 'genkit/plugin';
import { RESERVED_FIELDS, type Client } from './client.js';
import {
  answersText,
  compileQuestions,
  enumQuestion,
  wireQuestions,
} from './protocol.js';
import { INFO_MARKER, type Info } from './questions.js';
import { buildState, systemPreamble, type Reads } from './state.js';

/** The per-request configuration of a decision model. */
export const SystemOneConfigSchema = z.object({
  /**
   * Parses each text part of the state as JSON, so a prompt template that
   * renders a JSON document produces an object state rather than a string
   * one. A text that is not JSON is an error. A fallback model reference
   * that carries a config of its own replaces this one, so set it there too,
   * or send the state as a data part, which needs no parsing.
   */
  stateJSON: z
    .boolean()
    .optional()
    .describe(
      'Parse each text part of the state as JSON, so a template that renders JSON produces an object state.'
    ),
  /**
   * Merged into the top-level fields of the request body. It reaches fields
   * the plugin does not model, such as a gateway's session_id or trace. It
   * cannot name a field the request builds itself: model, state, questions,
   * images, audio, or videos.
   */
  extra: z
    .record(z.any())
    .optional()
    .describe(
      "Extra top-level request fields, such as a gateway's session_id. Cannot replace model, state, questions, images, audio, or videos."
    ),
  /**
   * The API key for this request, in place of the plugin's. A plugin created
   * with `apiKey: false` takes its key here only. It is recorded in traces and the
   * Dev UI like any other config field, so where traces are kept, prefer the
   * environment variable or the plugin's `apiKey`.
   */
  apiKey: z
    .string()
    .optional()
    .describe("The API key for this request, in place of the plugin's."),
});

/** The per-request configuration of a decision model. */
export type SystemOneConfig = z.infer<typeof SystemOneConfigSchema>;

/** Describes one model to {@link newModel}. */
export interface ModelDescription {
  /** Names the model in the Dev UI. */
  label: string;
  /**
   * The media requests may carry. A kind the model does not read is refused
   * before a request is sent.
   */
  reads: Reads;
  /**
   * Whether each request takes its key from `config.apiKey`, for a plugin
   * created with `apiKey: false`.
   */
  keyPerRequest?: boolean;
}

/**
 * What a decision model supports. It claims constrained output, so the loop
 * hands it the output schema untouched, and claims the system role and
 * context so the loop does not rewrite either into instruction text that
 * would land in the state.
 */
export function modelInfo(desc: ModelDescription): ModelInfo {
  return {
    label: desc.label,
    supports: {
      multiturn: true,
      systemRole: true,
      context: true,
      tools: false,
      media: desc.reads.images || desc.reads.audio || desc.reads.video,
      constrained: 'all',
      output: ['json', 'enum'],
      contentType: ['application/json', 'text/enum'],
    },
  };
}

/** Builds the model action for one model ID under its registered name. */
export function newModel(
  client: Client,
  name: string,
  id: string,
  desc: ModelDescription
): ModelAction<typeof SystemOneConfigSchema> {
  return model(
    {
      name,
      ...modelInfo(desc),
      configSchema: SystemOneConfigSchema,
    },
    (req, { abortSignal }) =>
      generate(
        client,
        id,
        desc,
        req as GenerateRequest<typeof SystemOneConfigSchema>,
        abortSignal
      )
  );
}

/** Answers the questions the request's output schema encodes. */
async function generate(
  client: Client,
  id: string,
  desc: ModelDescription,
  req: GenerateRequest<typeof SystemOneConfigSchema>,
  abortSignal?: AbortSignal
): Promise<GenerateResponseData> {
  const isEnum = req.output?.format === 'enum';
  if (!req.output?.schema && !isEnum) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'systemone: the model answers questions encoded in an output schema; pass a decision schema as output.schema, or output.jsonSchema = systemOne.schema(questions) for questions built at run time',
    });
  }
  if (req.tools?.length) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'systemone: the model calls no tools; leave tools out',
    });
  }
  const config = req.config ?? {};
  if (
    config.extra !== undefined &&
    (config.extra === null ||
      typeof config.extra !== 'object' ||
      Array.isArray(config.extra))
  ) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'systemone: config.extra must be an object of request fields',
    });
  }
  // A plugin created with apiKey: false takes its key from each request.
  const apiKey = config.apiKey || undefined;
  if (desc.keyPerRequest && !apiKey) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: `${client.endpoint.name}: the plugin was created with apiKey: false, so pass the API key as config.apiKey`,
    });
  }
  for (const field of RESERVED_FIELDS) {
    if (config.extra && field in config.extra) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `systemone: extra field "${field}" is one the request builds itself`,
      });
    }
  }
  const preamble = systemPreamble(req.messages);
  const questions = isEnum
    ? enumQuestion(req.output?.schema, preamble)
    : compileQuestions(req.output?.schema, preamble);
  const { state, media } = buildState(req, !!config.stateJSON, desc.reads);

  const resp = await client.decide(
    id,
    { state, media, questions: wireQuestions(questions), extra: config.extra },
    { signal: abortSignal, apiKey: apiKey || undefined }
  );
  // A count the server leaves out stays out, rather than reading as zero.
  // A count is a whole number from 0 (JSON's 3.0 is 3); anything else,
  // such as 3.5, -1, or 1e400, which parses as Infinity, is left out.
  const count = (n: unknown) =>
    typeof n === 'number' && Number.isSafeInteger(n) && n >= 0 ? n : undefined;
  const inputTokens = count(resp.usage?.input_tokens);
  const outputTokens = count(resp.usage?.output_tokens);
  logger.debug(
    `systemone: answered: endpoint=${client.endpoint.name} model=${resp.model} questions=${Object.keys(questions).length} inputTokens=${inputTokens} outputTokens=${outputTokens}`
  );

  const text = answersText(resp.answers, questions, isEnum);
  const info: Info = {
    model: typeof resp.model === 'string' ? resp.model : '',
  };
  if (typeof resp.provider === 'string' && resp.provider) {
    info.provider = resp.provider;
  }
  if (typeof resp.id === 'string' && resp.id) info.id = resp.id;
  info.answers = resp.answers;
  const usage: GenerationUsage = {};
  if (inputTokens !== undefined) usage.inputTokens = inputTokens;
  if (outputTokens !== undefined) usage.outputTokens = outputTokens;
  if (inputTokens !== undefined && outputTokens !== undefined) {
    usage.totalTokens = inputTokens + outputTokens;
  }
  const cost = resp.usage?.cost;
  if (typeof cost === 'number' && Number.isFinite(cost)) {
    usage.custom = { cost };
  }
  // The generate loop serializes a response's custom field and not its raw
  // one, and a GenerateResponse reads raw from custom, so the info goes in
  // both: response.raw has it after generate as well as from the model. The
  // marker tells responseInfo the info is this plugin's.
  const raw = { ...info, [INFO_MARKER]: true };
  return {
    message: { role: 'model', content: [{ text }] },
    finishReason: 'stop',
    ...(Object.keys(usage).length > 0 ? { usage } : {}),
    custom: raw,
    raw,
  };
}
