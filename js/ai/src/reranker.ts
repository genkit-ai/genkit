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

import { action, z, type Action } from '@genkit-ai/core';
import type { Registry } from '@genkit-ai/core/registry';
import { toJsonSchema } from '@genkit-ai/core/schema';
import { PartSchema, type Part } from './parts.js';
import {
  Document,
  DocumentDataSchema,
  type DocumentData,
} from './retriever.js';

export type RerankerFn<RerankerOptions extends z.ZodTypeAny> = (
  query: Document,
  documents: Document[],
  queryOpts: z.infer<RerankerOptions>
) => Promise<RerankerResponse>;

/**
 * Zod schema for a reranked document metadata.
 */
export const RankedDocumentMetadataSchema = z
  .object({
    score: z.number(), // Enforces that 'score' must be a number
  })
  .passthrough(); // Allows other properties in 'metadata' with any type

export const RankedDocumentDataSchema = z.object({
  content: z.array(PartSchema),
  metadata: RankedDocumentMetadataSchema,
});

export type RankedDocumentData = z.infer<typeof RankedDocumentDataSchema>;

export class RankedDocument extends Document implements RankedDocumentData {
  content: Part[];
  metadata: { score: number } & Record<string, any>;

  constructor(data: RankedDocumentData) {
    super(data);
    this.content = data.content;
    this.metadata = data.metadata;
  }
  /**
   * Returns the score of the document.
   * @returns The score of the document.
   */
  score(): number {
    return this.metadata.score;
  }
}

const RerankerRequestSchema = z.object({
  query: DocumentDataSchema,
  documents: z.array(DocumentDataSchema),
  options: z.any().optional(),
});

const RerankerResponseSchema = z.object({
  documents: z.array(RankedDocumentDataSchema),
});
type RerankerResponse = z.infer<typeof RerankerResponseSchema>;

export const RerankerInfoSchema = z.object({
  label: z.string().optional(),
  /** Supported model capabilities. */
  supports: z
    .object({
      /** Model can process media as part of the prompt (multimodal input). */
      media: z.boolean().optional(),
    })
    .optional(),
});
export type RerankerInfo = z.infer<typeof RerankerInfoSchema>;

export type RerankerAction<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> =
  Action<typeof RerankerRequestSchema, typeof RerankerResponseSchema> & {
    __configSchema?: CustomOptions;
  };

function rerankerWithMetadata<
  RerankerOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  reranker: Action<typeof RerankerRequestSchema, typeof RerankerResponseSchema>,
  configSchema?: RerankerOptions
): RerankerAction<RerankerOptions> {
  const withMeta = reranker as RerankerAction<RerankerOptions>;
  withMeta.__configSchema = configSchema;
  return withMeta;
}

/**
 *  Creates a reranker action for the provided {@link RerankerFn} implementation and registers it in the registry.
 */
export function defineReranker<OptionsType extends z.ZodTypeAny = z.ZodTypeAny>(
  registry: Registry,
  options: {
    name: string;
    configSchema?: OptionsType;
    info?: RerankerInfo;
  },
  runner: RerankerFn<OptionsType>
): RerankerAction<OptionsType> {
  const act = reranker(options, runner);

  registry.registerAction('reranker', act);

  return act;
}

/**
 *  Creates a reranker action for the provided {@link RerankerFn} implementation.
 */
export function reranker<OptionsType extends z.ZodTypeAny = z.ZodTypeAny>(
  options: {
    name: string;
    configSchema?: OptionsType;
    info?: RerankerInfo;
  },
  runner: RerankerFn<OptionsType>
): RerankerAction<OptionsType> {
  const reranker = action(
    {
      actionType: 'reranker',
      name: options.name,
      inputSchema: options.configSchema
        ? RerankerRequestSchema.extend({
            options: options.configSchema.optional(),
          })
        : RerankerRequestSchema,
      outputSchema: RerankerResponseSchema,
      metadata: {
        type: 'reranker',
        info: options.info,
        reranker: {
          customOptions: options.configSchema
            ? toJsonSchema({ schema: options.configSchema })
            : undefined,
        },
      },
    },
    (i) =>
      runner(
        new Document(i.query),
        i.documents.map((d) => new Document(d)),
        i.options
      )
  );
  const rwm = rerankerWithMetadata(
    reranker as Action<
      typeof RerankerRequestSchema,
      typeof RerankerResponseSchema
    >,
    options.configSchema
  );
  return rwm;
}

export interface RerankerParams<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  reranker: RerankerArgument<CustomOptions>;
  query: string | DocumentData;
  documents: DocumentData[];
  options?: z.infer<CustomOptions>;
}

export type RerankerArgument<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = RerankerAction<CustomOptions> | RerankerReference<CustomOptions> | string;

/**
 * Reranks documents from a {@link RerankerArgument} based on the provided query.
 */
export async function rerank<CustomOptions extends z.ZodTypeAny>(
  registry: Registry,
  params: RerankerParams<CustomOptions>
): Promise<Array<RankedDocument>> {
  let reranker: RerankerAction<CustomOptions>;
  if (typeof params.reranker === 'string') {
    reranker = await registry.lookupAction(`/reranker/${params.reranker}`);
  } else if (Object.hasOwnProperty.call(params.reranker, 'info')) {
    reranker = await registry.lookupAction(`/reranker/${params.reranker.name}`);
  } else {
    reranker = params.reranker as RerankerAction<CustomOptions>;
  }
  if (!reranker) {
    throw new Error('Unable to resolve the reranker');
  }
  const response = await reranker({
    query:
      typeof params.query === 'string'
        ? Document.fromText(params.query)
        : params.query,
    documents: params.documents,
    options: params.options,
  });

  return response.documents.map((d) => new RankedDocument(d));
}

export const CommonRerankerOptionsSchema = z.object({
  k: z.number().describe('Number of documents to rerank').optional(),
});

export interface RerankerReference<CustomOptions extends z.ZodTypeAny> {
  name: string;
  configSchema?: CustomOptions;
  info?: RerankerInfo;
}

/**
 * Helper method to configure a {@link RerankerReference} to a plugin.
 */
export function rerankerRef<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: RerankerReference<CustomOptionsSchema>
): RerankerReference<CustomOptionsSchema> {
  return { ...options };
}
