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
import type { Info } from './questions.js';
import { buildState, systemPreamble, type Reads } from './state.js';

/** The per-request configuration of a decision model. */
export const SystemOneConfigSchema = z.object({
  /**
   * Parses each text part of the state as JSON, so a prompt template that
   * renders a JSON document produces an object state rather than a string
   * one. A text that is not JSON is an error. A fallback to another model
   * takes that model's config, so set it there too, or send the state as a
   * data part, which needs no parsing.
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
        desc.reads,
        req as GenerateRequest<typeof SystemOneConfigSchema>,
        abortSignal
      )
  );
}

/** Answers the questions the request's output schema encodes. */
async function generate(
  client: Client,
  id: string,
  reads: Reads,
  req: GenerateRequest<typeof SystemOneConfigSchema>,
  abortSignal?: AbortSignal
): Promise<GenerateResponseData> {
  const isEnum = req.output?.format === 'enum';
  if (!req.output?.schema && !isEnum) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'systemone: the model answers questions encoded in an output schema; pass a decision schema as output.schema, or output.jsonSchema = schema(questions) for questions built at run time',
    });
  }
  if (req.tools?.length) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'systemone: the model calls no tools; leave tools out',
    });
  }
  const config = req.config ?? {};
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
  const { state, media } = buildState(req, !!config.stateJSON, reads);

  const resp = await client.decide(
    id,
    { state, media, questions: wireQuestions(questions), extra: config.extra },
    abortSignal
  );
  const inputTokens = resp.usage?.input_tokens ?? 0;
  const outputTokens = resp.usage?.output_tokens ?? 0;
  logger.debug(
    `systemone: answered: endpoint=${client.endpoint.name} model=${resp.model} questions=${Object.keys(questions).length} inputTokens=${inputTokens} outputTokens=${outputTokens}`
  );

  const text = answersText(resp.answers, questions, isEnum);
  const info: Info = { model: resp.model };
  if (resp.provider) info.provider = resp.provider;
  if (resp.id) info.id = resp.id;
  info.answers = resp.answers;
  return {
    message: { role: 'model', content: [{ text }] },
    finishReason: 'stop',
    usage: {
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
      ...(typeof resp.usage?.cost === 'number'
        ? { custom: { cost: resp.usage.cost } }
        : {}),
    },
    // The generate loop serializes a response's custom field and not its raw
    // one, and a GenerateResponse reads raw from custom, so the info goes in
    // both: response.raw has it after generate as well as from the model.
    custom: info,
    raw: info,
  };
}
