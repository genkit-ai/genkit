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

import { GenkitError, action, z, type Action } from '@genkit-ai/core';
import type { Registry } from '@genkit-ai/core/registry';
import { toJsonSchema } from '@genkit-ai/core/schema';
import { Document, DocumentDataSchema, type DocumentData } from './document.js';
import type { EmbedderInfo } from './embedder.js';

export { Document, DocumentDataSchema, type DocumentData } from './document.js';
export { type MediaPart, type Part, type TextPart } from './parts.js';

/**
 * Retriever implementation function signature.
 */
export type RetrieverFn<RetrieverOptions extends z.ZodTypeAny> = (
  query: Document,
  queryOpts: z.infer<RetrieverOptions>
) => Promise<RetrieverResponse>;

/**
 * Indexer implementation function signature.
 */
export type IndexerFn<IndexerOptions extends z.ZodTypeAny> = (
  docs: Array<Document>,
  indexerOpts: z.infer<IndexerOptions>
) => Promise<void>;

const RetrieverRequestSchema = z.object({
  query: DocumentDataSchema,
  options: z.any().optional(),
});

const RetrieverResponseSchema = z.object({
  documents: z.array(DocumentDataSchema),
  // TODO: stats, etc.
});
type RetrieverResponse = z.infer<typeof RetrieverResponseSchema>;

const IndexerRequestSchema = z.object({
  documents: z.array(DocumentDataSchema),
  options: z.any().optional(),
});

/**
 * Zod schema of retriever info metadata.
 */
export const RetrieverInfoSchema = z.object({
  label: z.string().optional(),
  /** Supported model capabilities. */
  supports: z
    .object({
      /** Model can process media as part of the prompt (multimodal input). */
      media: z.boolean().optional(),
    })
    .optional(),
});
export type RetrieverInfo = z.infer<typeof RetrieverInfoSchema>;

/**
 * A retriever action type.
 */
export type RetrieverAction<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> =
  Action<typeof RetrieverRequestSchema, typeof RetrieverResponseSchema> & {
    __configSchema?: CustomOptions;
  };

/**
 * An indexer action type.
 */
export type IndexerAction<IndexerOptions extends z.ZodTypeAny = z.ZodTypeAny> =
  Action<typeof IndexerRequestSchema, z.ZodVoid> & {
    __configSchema?: IndexerOptions;
  };

function retrieverWithMetadata<
  RetrieverOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  retriever: Action<
    typeof RetrieverRequestSchema,
    typeof RetrieverResponseSchema
  >,
  configSchema?: RetrieverOptions
): RetrieverAction<RetrieverOptions> {
  const withMeta = retriever as RetrieverAction<RetrieverOptions>;
  withMeta.__configSchema = configSchema;
  return withMeta;
}

function indexerWithMetadata<
  IndexerOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  indexer: Action<typeof IndexerRequestSchema, z.ZodVoid>,
  configSchema?: IndexerOptions
): IndexerAction<IndexerOptions> {
  const withMeta = indexer as IndexerAction<IndexerOptions>;
  withMeta.__configSchema = configSchema;
  return withMeta;
}

/**
 *  Creates a retriever action for the provided {@link RetrieverFn} implementation.
 */
export function defineRetriever<
  OptionsType extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  options: {
    name: string;
    configSchema?: OptionsType;
    info?: RetrieverInfo;
  },
  runner: RetrieverFn<OptionsType>
): RetrieverAction<OptionsType> {
  const r = retriever(options, runner);
  registry.registerAction('retriever', r);
  return r;
}

/**
 *  Creates a retriever action for the provided {@link RetrieverFn} implementation.
 */
export function retriever<OptionsType extends z.ZodTypeAny = z.ZodTypeAny>(
  options: {
    name: string;
    configSchema?: OptionsType;
    info?: RetrieverInfo;
  },
  runner: RetrieverFn<OptionsType>
): RetrieverAction<OptionsType> {
  const retriever = action(
    {
      actionType: 'retriever',
      name: options.name,
      inputSchema: options.configSchema
        ? RetrieverRequestSchema.extend({
            options: options.configSchema.optional(),
          })
        : RetrieverRequestSchema,
      outputSchema: RetrieverResponseSchema,
      metadata: {
        type: 'retriever',
        info: options.info,
        retriever: {
          customOptions: options.configSchema
            ? toJsonSchema({ schema: options.configSchema })
            : undefined,
        },
      },
    },
    (i) => runner(new Document(i.query), i.options)
  );
  const rwm = retrieverWithMetadata(
    retriever as Action<
      typeof RetrieverRequestSchema,
      typeof RetrieverResponseSchema
    >,
    options.configSchema
  );
  return rwm;
}

/**
 *  Creates an indexer action for the provided {@link IndexerFn} implementation.
 */
export function defineIndexer<IndexerOptions extends z.ZodTypeAny>(
  registry: Registry,
  options: {
    name: string;
    embedderInfo?: EmbedderInfo;
    configSchema?: IndexerOptions;
  },
  runner: IndexerFn<IndexerOptions>
): IndexerAction<IndexerOptions> {
  const r = indexer(options, runner);
  registry.registerAction('indexer', r);
  return r;
}

/**
 *  Creates an indexer action for the provided {@link IndexerFn} implementation.
 */
export function indexer<IndexerOptions extends z.ZodTypeAny>(
  options: {
    name: string;
    embedderInfo?: EmbedderInfo;
    configSchema?: IndexerOptions;
  },
  runner: IndexerFn<IndexerOptions>
): IndexerAction<IndexerOptions> {
  const indexer = action(
    {
      actionType: 'indexer',
      name: options.name,
      inputSchema: options.configSchema
        ? IndexerRequestSchema.extend({
            options: options.configSchema.optional(),
          })
        : IndexerRequestSchema,
      outputSchema: z.void(),
      metadata: {
        type: 'indexer',
        embedderInfo: options.embedderInfo,
        indexer: {
          customOptions: options.configSchema
            ? toJsonSchema({ schema: options.configSchema })
            : undefined,
        },
      },
    },
    (i) =>
      runner(
        i.documents.map((dd) => new Document(dd)),
        i.options
      )
  );
  const iwm = indexerWithMetadata(
    indexer as Action<typeof IndexerRequestSchema, z.ZodVoid>,
    options.configSchema
  );
  return iwm;
}

export interface RetrieverParams<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  retriever: RetrieverArgument<CustomOptions>;
  query: string | DocumentData;
  options?: z.infer<CustomOptions>;
}

/**
 * A type that can be used to pass a retriever as an argument, either using a reference or an action.
 */
export type RetrieverArgument<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = RetrieverAction<CustomOptions> | RetrieverReference<CustomOptions> | string;

/**
 * Retrieves documents from a {@link RetrieverArgument} based on the provided query.
 */
export async function retrieve<CustomOptions extends z.ZodTypeAny>(
  registry: Registry,
  params: RetrieverParams<CustomOptions>
): Promise<Array<Document>> {
  let retriever: RetrieverAction<CustomOptions>;
  if (typeof params.retriever === 'string') {
    retriever = await registry.lookupAction(`/retriever/${params.retriever}`);
  } else if (Object.hasOwnProperty.call(params.retriever, 'info')) {
    retriever = await registry.lookupAction(
      `/retriever/${params.retriever.name}`
    );
  } else {
    retriever = params.retriever as RetrieverAction<CustomOptions>;
  }
  if (!retriever) {
    throw new Error('Unable to resolve the retriever');
  }
  const response = await retriever({
    query:
      typeof params.query === 'string'
        ? Document.fromText(params.query)
        : params.query,
    options: params.options,
  });

  return response.documents.map((d) => new Document(d));
}

/**
 * A type that can be used to pass an indexer as an argument, either using a reference or an action.
 */
export type IndexerArgument<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> =
  IndexerReference<CustomOptions> | IndexerAction<CustomOptions> | string;

/**
 * Options passed to the index function.
 */
export interface IndexerParams<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  indexer: IndexerArgument<CustomOptions>;
  documents: Array<DocumentData>;
  options?: z.infer<CustomOptions>;
}

/**
 * Indexes documents using a {@link IndexerArgument}.
 */
export async function index<CustomOptions extends z.ZodTypeAny>(
  registry: Registry,
  params: IndexerParams<CustomOptions>
): Promise<void> {
  let indexer: IndexerAction<CustomOptions>;
  if (typeof params.indexer === 'string') {
    indexer = await registry.lookupAction(`/indexer/${params.indexer}`);
  } else if (Object.hasOwnProperty.call(params.indexer, 'info')) {
    indexer = await registry.lookupAction(`/indexer/${params.indexer.name}`);
  } else {
    indexer = params.indexer as IndexerAction<CustomOptions>;
  }
  if (!indexer) {
    throw new Error('Unable to utilize the provided indexer');
  }
  return await indexer({
    documents: params.documents,
    options: params.options,
  });
}

/**
 * Zod schema of common retriever options.
 */
export const CommonRetrieverOptionsSchema = z.object({
  k: z.number().describe('Number of documents to retrieve').optional(),
});

/**
 * A retriver reference object.
 */
export interface RetrieverReference<CustomOptions extends z.ZodTypeAny> {
  name: string;
  configSchema?: CustomOptions;
  info?: RetrieverInfo;
}

/**
 * Helper method to configure a {@link RetrieverReference} to a plugin.
 */
export function retrieverRef<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: RetrieverReference<CustomOptionsSchema>
): RetrieverReference<CustomOptionsSchema> {
  return { ...options };
}

// Reuse the same schema for both indexers and retrievers -- for now.
export const IndexerInfoSchema = RetrieverInfoSchema;

/**
 * Indexer metadata.
 */
export type IndexerInfo = z.infer<typeof IndexerInfoSchema>;

export interface IndexerReference<CustomOptions extends z.ZodTypeAny> {
  name: string;
  configSchema?: CustomOptions;
  info?: IndexerInfo;
}

/**
 * Helper method to configure a {@link IndexerReference} to a plugin.
 */
export function indexerRef<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: IndexerReference<CustomOptionsSchema>
): IndexerReference<CustomOptionsSchema> {
  return { ...options };
}

function itemToDocument<R>(
  item: any,
  options: SimpleRetrieverOptions
): Document {
  if (!item)
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Items returned from simple retriever must be non-null.`,
    });
  if (typeof item === 'string') return Document.fromText(item);
  if (typeof options.content === 'function') {
    const transformed = options.content(item);
    return typeof transformed === 'string'
      ? Document.fromText(transformed)
      : new Document({ content: transformed });
  }
  if (typeof options.content === 'string' && typeof item === 'object')
    return Document.fromText(item[options.content]);
  throw new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: `Cannot convert item to document without content option. Item: ${JSON.stringify(item)}`,
  });
}

function itemToMetadata(
  item: any,
  options: SimpleRetrieverOptions
): Document['metadata'] {
  if (typeof item === 'string') return undefined;
  if (Array.isArray(options.metadata) && typeof item === 'object') {
    const out: Record<string, any> = {};
    options.metadata.forEach((key) => (out[key] = item[key]));
    return out;
  }
  if (typeof options.metadata === 'function') return options.metadata(item);
  if (!options.metadata && typeof item === 'object') {
    const out = { ...item };
    if (typeof options.content === 'string') delete out[options.content];
    return out;
  }
  throw new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: `Unable to extract metadata from item with supplied options. Item: ${JSON.stringify(item)}`,
  });
}

/**
 * Simple retriever options.
 */
export interface SimpleRetrieverOptions<
  C extends z.ZodTypeAny = z.ZodTypeAny,
  R = any,
> {
  /** The name of the retriever you're creating. */
  name: string;
  /** A Zod schema containing any configuration info available beyond the query. */
  configSchema?: C;
  /**
   * Specifies how to extract content from the returned items.
   *
   * - If a string, specifies the key of the returned item to extract as content.
   * - If a function, allows you to extract content as text or a document part.
   **/
  content?: string | ((item: R) => Document['content'] | string);
  /**
   * Specifies how to extract metadata from the returned items.
   *
   * - If an array of strings, specifies list of keys to extract from returned objects.
   * - If a function, allows you to use custom behavior to extract metadata from returned items.
   */
  metadata?: string[] | ((item: R) => Document['metadata']);
}

/**
 * defineSimpleRetriever makes it easy to map existing data into documents that
 * can be used for prompt augmentation.
 *
 * @param options Configuration options for the retriever.
 * @param handler A function that queries a datastore and returns items from which to extract documents.
 * @returns A Genkit retriever.
 */
export function defineSimpleRetriever<
  C extends z.ZodTypeAny = z.ZodTypeAny,
  R = any,
>(
  registry: Registry,
  options: SimpleRetrieverOptions<C, R>,
  handler: (query: Document, config: z.infer<C>) => Promise<R[]>
) {
  return defineRetriever(
    registry,
    {
      name: options.name,
      configSchema: options.configSchema,
    },
    async (query, config) => {
      const result = await handler(query, config);
      return {
        documents: result.map((item) => {
          const doc = itemToDocument(item, options);
          if (typeof item !== 'string')
            doc.metadata = itemToMetadata(item, options);
          return doc;
        }),
      };
    }
  );
}
