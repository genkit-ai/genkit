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
import { z, type Document } from 'genkit';
import {
  embedderRef,
  type EmbedderAction,
  type Embedding,
} from 'genkit/embedder';
import { embedder as pluginEmbedder } from 'genkit/plugin';
import type {
  MultimodalEmbedRequestInputsItem,
  MultimodalEmbedRequestInputsItemContentItem,
  VoyageAIClient,
} from 'voyageai';
import { inputCosts, planBatches, type TokenCounter } from './batching.js';
import {
  CONTEXTUALIZED_EMBEDDING_MODELS,
  MULTIMODAL_EMBEDDING_MODELS,
  TEXT_EMBEDDING_MODELS,
  type ContextualizedEmbeddingModelName,
  type EmbedderModelName,
  type EmbeddingModelSpec,
  type MultimodalEmbeddingModelName,
  type TextEmbeddingModelName,
} from './catalog.js';
import {
  callVoyage,
  checkOutputDimension,
  invalidArgument,
  parseConfig,
  textOnly,
} from './shared.js';

const NAMESPACE = 'voyageai';

const inputTypeSchema = z
  .enum(['document', 'query'])
  .describe(
    'Whether the inputs are documents to index or queries to search with. Voyage prepends a retrieval prompt for the chosen type.'
  );

/** Options of the Voyage text embedders (`voyage-4`, `voyage-3.5`, `voyage-code-3`, ...). */
export const TextEmbedderConfigSchema = z
  .object({
    inputType: inputTypeSchema.optional(),
    outputDimension: z
      .number()
      .int()
      .positive()
      .optional()
      .describe(
        'Number of dimensions of the returned vectors (Matryoshka models: 256, 512, 1024, 2048). Only sent when set.'
      ),
    truncation: z
      .boolean()
      .optional()
      .describe(
        'Truncate inputs that exceed the model context length. Voyage truncates by default; false makes over-length inputs an error.'
      ),
  })
  .strict();
export type TextEmbedderConfig = z.infer<typeof TextEmbedderConfigSchema>;

/** Options of the Voyage contextualized chunk embedders (`voyage-context-4`, `voyage-context-3`). */
export const ContextualizedEmbedderConfigSchema = z
  .object({
    inputType: inputTypeSchema.describe(
      'Required. "document": each input is a whole document that Voyage chunks (32,000-token chunks) and embeds with document context. "query": each input is a query, embedded without chunking.'
    ),
    outputDimension: z
      .number()
      .int()
      .positive()
      .optional()
      .describe(
        'Number of dimensions of the returned vectors (256, 512, 1024, 2048). Only sent when set.'
      ),
  })
  .strict();
export type ContextualizedEmbedderConfig = z.infer<
  typeof ContextualizedEmbedderConfigSchema
>;

/** Options of the Voyage multimodal embedders (`voyage-multimodal-3.5`, `voyage-multimodal-3`). */
export const MultimodalEmbedderConfigSchema = z
  .object({
    inputType: inputTypeSchema.optional(),
    truncation: z
      .boolean()
      .optional()
      .describe(
        'Truncate inputs that exceed the model context length. Voyage truncates by default; false makes over-length inputs an error.'
      ),
  })
  .strict();
export type MultimodalEmbedderConfig = z.infer<
  typeof MultimodalEmbedderConfigSchema
>;

/** Metadata attached to every embedding returned by a contextualized embedder. */
export interface ContextualizedChunkMetadata {
  /** Index of the input document (or query) the vector belongs to. */
  documentIndex: number;
  /** Index of the chunk within its document. */
  chunkIndex: number;
  /** The chunk text. Present when Voyage chunked the document itself (not for queries). */
  text?: string;
}

/** Auto-chunking target size: the largest chunk Voyage accepts, in tokens. */
export const CONTEXTUALIZED_CHUNK_SIZE = 32_000;

function checkEmbeddingCount(
  actionName: string,
  vectors: (number[] | undefined)[],
  expected: number
): number[][] {
  if (vectors.length !== expected || vectors.some((v) => !v?.length)) {
    throw new Error(
      `${actionName}: Voyage returned ${vectors.length} embeddings for ${expected} inputs, or an empty vector.`
    );
  }
  return vectors as number[][];
}

/** Returns the embedder reference for a text embedding model. */
export function textEmbedderRef(name: TextEmbeddingModelName) {
  const spec = TEXT_EMBEDDING_MODELS[name];
  return embedderRef({
    name,
    namespace: NAMESPACE,
    configSchema: TextEmbedderConfigSchema,
    info: {
      label: `Voyage AI - ${name}`,
      dimensions: spec.dimensions[0],
      supports: { input: ['text'] },
    },
  });
}

/** Returns the embedder reference for a contextualized embedding model. */
export function contextualizedEmbedderRef(
  name: ContextualizedEmbeddingModelName
) {
  const spec = CONTEXTUALIZED_EMBEDDING_MODELS[name];
  return embedderRef({
    name,
    namespace: NAMESPACE,
    configSchema: ContextualizedEmbedderConfigSchema,
    info: {
      label: `Voyage AI - ${name}`,
      dimensions: spec.dimensions[0],
      supports: { input: ['text'] },
    },
  });
}

/** Returns the embedder reference for a multimodal embedding model. */
export function multimodalEmbedderRef(name: MultimodalEmbeddingModelName) {
  const spec = MULTIMODAL_EMBEDDING_MODELS[name];
  return embedderRef({
    name,
    namespace: NAMESPACE,
    configSchema: MultimodalEmbedderConfigSchema,
    info: {
      label: `Voyage AI - ${name}`,
      dimensions: spec.dimensions[0],
      supports: {
        input: spec.video ? ['text', 'image', 'video'] : ['text', 'image'],
      },
    },
  });
}

/** Returns the embedder reference for any known Voyage embedding model, or throws for an unknown one. */
export function embedderReference(name: string) {
  if (name in TEXT_EMBEDDING_MODELS) {
    return textEmbedderRef(name as TextEmbeddingModelName);
  }
  if (name in CONTEXTUALIZED_EMBEDDING_MODELS) {
    return contextualizedEmbedderRef(name as ContextualizedEmbeddingModelName);
  }
  if (name in MULTIMODAL_EMBEDDING_MODELS) {
    return multimodalEmbedderRef(name as MultimodalEmbeddingModelName);
  }
  throw invalidArgument(
    `Unknown Voyage AI embedding model '${name}'. Known models: ${allEmbedderModelNames().join(', ')}.`
  );
}

/** Names of every embedding model the plugin defines. */
export function allEmbedderModelNames(): EmbedderModelName[] {
  return [
    ...(Object.keys(TEXT_EMBEDDING_MODELS) as TextEmbeddingModelName[]),
    ...(Object.keys(
      CONTEXTUALIZED_EMBEDDING_MODELS
    ) as ContextualizedEmbeddingModelName[]),
    ...(Object.keys(
      MULTIMODAL_EMBEDDING_MODELS
    ) as MultimodalEmbeddingModelName[]),
  ];
}

/** Defines the embedder action of a text embedding model. */
export function defineTextEmbedder(
  name: TextEmbeddingModelName,
  client: VoyageAIClient,
  countTokens: TokenCounter
): EmbedderAction {
  const ref = textEmbedderRef(name);
  const spec: EmbeddingModelSpec = TEXT_EMBEDDING_MODELS[name];
  return pluginEmbedder(
    { name: ref.name, configSchema: ref.configSchema, info: ref.info },
    async (request) => {
      const config = parseConfig(
        TextEmbedderConfigSchema,
        request.options,
        ref.name
      );
      const outputDimension = checkOutputDimension(
        name,
        spec.dimensions,
        config.outputDimension
      );
      if (request.input.length === 0) return { embeddings: [] };
      const input = request.input.map((doc, i) => textOnly(doc, ref.name, i));
      const batches = planBatches(
        input.length,
        await inputCosts(
          input.map((text) => ({ texts: [text] })),
          {
            model: name,
            contextTokens: spec.contextTokens,
            truncates: config.truncation !== false,
            maxTokens: spec.maxTokensPerRequest,
            countTokens,
          }
        ),
        spec.maxTokensPerRequest,
        ref.name
      );
      const vectors: number[][] = [];
      for (const batch of batches) {
        const part = input.slice(batch.start, batch.end);
        const response = await callVoyage('embeddings request', () =>
          client.embed({
            input: part,
            model: name,
            inputType: config.inputType,
            outputDimension,
            truncation: config.truncation,
          })
        );
        const ordered = [...(response.data ?? [])].sort(
          (a, b) => (a.index ?? 0) - (b.index ?? 0)
        );
        vectors.push(
          ...checkEmbeddingCount(
            ref.name,
            ordered.map((d) => d.embedding),
            part.length
          )
        );
      }
      return { embeddings: vectors.map((embedding) => ({ embedding })) };
    }
  ) as EmbedderAction;
}

/** The single text of a document that is embedded as a whole by a contextualized model. */
function wholeDocumentText(doc: Document, actionName: string, index: number) {
  const text = textOnly(doc, actionName, index);
  const textParts = doc.content.filter((p) => p.text !== undefined);
  if (textParts.length !== 1) {
    throw invalidArgument(
      `${actionName}: document ${index} has ${textParts.length} text parts. Pass each document as ONE text part; Voyage chunks it itself (pre-chunked documents are not supported with auto-chunking).`
    );
  }
  return text;
}

/**
 * Defines the embedder action of a contextualized embedding model.
 *
 * In `document` mode each Genkit document is one whole document that Voyage
 * chunks (`enable_auto_chunking`, `chunk_size` 32,000) and the result holds one
 * embedding per chunk, in (document, chunk) order, with the chunk text in the
 * metadata. In `query` mode there is no chunking: one embedding per query.
 */
export function defineContextualizedEmbedder(
  name: ContextualizedEmbeddingModelName,
  client: VoyageAIClient,
  countTokens: TokenCounter
): EmbedderAction {
  const ref = contextualizedEmbedderRef(name);
  const spec = CONTEXTUALIZED_EMBEDDING_MODELS[name];
  return pluginEmbedder(
    { name: ref.name, configSchema: ref.configSchema, info: ref.info },
    async (request) => {
      const config = parseConfig(
        ContextualizedEmbedderConfigSchema,
        request.options,
        ref.name
      );
      const outputDimension = checkOutputDimension(
        name,
        spec.dimensions,
        config.outputDimension
      );
      if (request.input.length === 0) return { embeddings: [] };
      const inputs = request.input.map((doc, i) =>
        wholeDocumentText(doc, ref.name, i)
      );
      const isDocument = config.inputType === 'document';
      // Documents are chunked by Voyage, so the context length does not cap
      // their cost: a document that cannot fit one request is an error.
      const batches = planBatches(
        inputs.length,
        await inputCosts(
          inputs.map((text) => ({ texts: [text] })),
          {
            model: name,
            contextTokens: spec.contextTokens,
            truncates: false,
            maxTokens: spec.maxTokensPerRequest,
            countTokens,
          }
        ),
        spec.maxTokensPerRequest,
        ref.name
      );
      const embeddings: Embedding[] = [];
      for (const batch of batches) {
        const part = inputs.slice(batch.start, batch.end);
        const result = await callVoyage(
          'contextualized embeddings request',
          () =>
            client.contextualizedEmbed({
              inputs: part,
              model: name,
              inputType: config.inputType,
              outputDimension,
              ...(isDocument
                ? {
                    enableAutoChunking: true,
                    chunkSize: CONTEXTUALIZED_CHUNK_SIZE,
                  }
                : {}),
            })
        );
        if (result.results.length !== part.length) {
          throw new Error(
            `${ref.name}: Voyage returned ${result.results.length} results for ${part.length} inputs.`
          );
        }
        for (const r of [...result.results].sort((a, b) => a.index - b.index)) {
          const documentIndex = batch.start + r.index;
          if (
            r.embeddings.length === 0 ||
            r.embeddings.some((v) => !v.length)
          ) {
            throw new Error(
              `${ref.name}: Voyage returned no embedding, or an empty one, for input ${documentIndex}.`
            );
          }
          r.embeddings.forEach((embedding, chunkIndex) => {
            const metadata: ContextualizedChunkMetadata = {
              documentIndex,
              chunkIndex,
              ...(r.chunkTexts ? { text: r.chunkTexts[chunkIndex] } : {}),
            };
            embeddings.push({ embedding, metadata: { ...metadata } });
          });
        }
      }
      return { embeddings };
    }
  ) as EmbedderAction;
}

const IMAGE_MEDIA_TYPES = [
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
];
const VIDEO_MEDIA_TYPES = ['video/mp4'];
const EXTENSION_MEDIA_TYPES: Record<string, string> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
  gif: 'image/gif',
  mp4: 'video/mp4',
};

/** Splits a `data:<mediatype>;base64,<data>` URL; returns undefined for other URLs. */
function parseDataUrl(url: string): string | undefined {
  const match = /^data:([^;,]+);base64,/i.exec(url);
  return match ? match[1].toLowerCase() : undefined;
}

function mediaTypeOf(url: string, contentType: string | undefined): string {
  const fromDataUrl = parseDataUrl(url);
  if (fromDataUrl) return fromDataUrl;
  if (contentType) return contentType.toLowerCase();
  const path = url.split(/[?#]/)[0];
  const extension = path.split('.').pop()?.toLowerCase() ?? '';
  const inferred = EXTENSION_MEDIA_TYPES[extension];
  if (!inferred) {
    throw invalidArgument(
      `Cannot tell the media type of ${url.slice(0, 80)}. Set the media contentType (${[...IMAGE_MEDIA_TYPES, ...VIDEO_MEDIA_TYPES].join(', ')}).`
    );
  }
  return inferred;
}

type ContentItem = MultimodalEmbedRequestInputsItemContentItem;

/**
 * Converts a document into a Voyage multimodal input, keeping the order of its
 * parts. Data URLs are sent as base64 items, http(s) URLs as URL items.
 */
function toMultimodalInput(
  doc: Document,
  index: number,
  actionName: string,
  videoAllowed: boolean
): {
  item: MultimodalEmbedRequestInputsItem;
  usesBase64: boolean[];
  texts: string[];
} {
  const content: ContentItem[] = [];
  const usesBase64: boolean[] = [];
  for (const part of doc.content) {
    if (part.text !== undefined) {
      content.push({ type: 'text', text: part.text });
    } else if (part.media) {
      const url = part.media.url;
      if (!url) {
        throw invalidArgument(
          `${actionName}: document ${index} has a media part without a URL.`
        );
      }
      const mediaType = mediaTypeOf(url, part.media.contentType);
      const isImage = IMAGE_MEDIA_TYPES.includes(mediaType);
      const isVideo = VIDEO_MEDIA_TYPES.includes(mediaType);
      if (!isImage && !isVideo) {
        throw invalidArgument(
          `${actionName}: document ${index} has unsupported media type '${mediaType}'. Supported: ${[...IMAGE_MEDIA_TYPES, ...(videoAllowed ? VIDEO_MEDIA_TYPES : [])].join(', ')}.`
        );
      }
      if (isVideo && !videoAllowed) {
        throw invalidArgument(
          `${actionName}: document ${index} contains a video, which this model does not accept. Use voyage-multimodal-3.5.`
        );
      }
      const base64 = parseDataUrl(url) !== undefined;
      if (!base64 && !/^https?:\/\//i.test(url)) {
        throw invalidArgument(
          `${actionName}: document ${index} has media that is neither an http(s) URL nor a base64 data URL.`
        );
      }
      usesBase64.push(base64);
      if (isImage) {
        content.push(
          base64
            ? { type: 'image_base64', imageBase64: url }
            : { type: 'image_url', imageUrl: url }
        );
      } else {
        content.push(
          base64
            ? { type: 'video_base64', videoBase64: url }
            : { type: 'video_url', videoUrl: url }
        );
      }
    } else {
      throw invalidArgument(
        `${actionName}: document ${index} has a part that is neither text nor media.`
      );
    }
  }
  if (content.length === 0) {
    throw invalidArgument(`${actionName}: document ${index} is empty.`);
  }
  return {
    item: { content },
    usesBase64,
    texts: content.flatMap((c) => (c.text === undefined ? [] : [c.text])),
  };
}

/**
 * Defines the embedder action of a multimodal embedding model. Each Genkit
 * document (text and media parts, in order) becomes one interleaved input and
 * yields one embedding.
 */
export function defineMultimodalEmbedder(
  name: MultimodalEmbeddingModelName,
  client: VoyageAIClient,
  countTokens: TokenCounter
): EmbedderAction {
  const ref = multimodalEmbedderRef(name);
  const spec = MULTIMODAL_EMBEDDING_MODELS[name];
  return pluginEmbedder(
    { name: ref.name, configSchema: ref.configSchema, info: ref.info },
    async (request) => {
      if (
        typeof request.options === 'object' &&
        request.options !== null &&
        (request.options as Record<string, unknown>).outputDimension !==
          undefined
      ) {
        throw invalidArgument(
          `${ref.name}: outputDimension is not supported for multimodal embedders (the official Voyage SDK does not send it for this endpoint).`
        );
      }
      const config = parseConfig(
        MultimodalEmbedderConfigSchema,
        request.options,
        ref.name
      );
      if (request.input.length === 0) return { embeddings: [] };
      const converted = request.input.map((doc, i) =>
        toMultimodalInput(doc, i, ref.name, spec.video)
      );
      const flags = new Set(converted.flatMap((c) => c.usesBase64));
      if (flags.size > 1) {
        throw invalidArgument(
          `${ref.name}: one request must use either only media URLs or only base64 data URLs for images and videos, not both.`
        );
      }
      // Image and video tokens depend on pixels that are not known here, so an
      // input with media is charged the most one input can cost.
      const batches = planBatches(
        converted.length,
        await inputCosts(
          converted.map((c) => ({
            texts: c.texts,
            hasMedia: c.usesBase64.length > 0,
          })),
          {
            model: name,
            contextTokens: spec.contextTokens,
            truncates: config.truncation !== false,
            maxTokens: spec.maxTokensPerRequest,
            countTokens,
          }
        ),
        spec.maxTokensPerRequest,
        ref.name
      );
      const vectors: number[][] = [];
      for (const batch of batches) {
        const part = converted.slice(batch.start, batch.end);
        const response = await callVoyage('multimodal embeddings request', () =>
          client.multimodalEmbed({
            inputs: part.map((c) => c.item),
            model: name,
            inputType: config.inputType,
            truncation: config.truncation,
          })
        );
        const ordered = [...(response.data ?? [])].sort(
          (a, b) => (a.index ?? 0) - (b.index ?? 0)
        );
        vectors.push(
          ...checkEmbeddingCount(
            ref.name,
            ordered.map((d) => d.embedding),
            part.length
          )
        );
      }
      return { embeddings: vectors.map((embedding) => ({ embedding })) };
    }
  ) as EmbedderAction;
}
