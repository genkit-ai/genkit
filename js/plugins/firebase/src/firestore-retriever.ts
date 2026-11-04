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

import type {
  Firestore,
  Query,
  QueryDocumentSnapshot,
  VectorQuerySnapshot,
} from '@google-cloud/firestore';
import {
  z,
  type EmbedderArgument,
  type Genkit,
  type RetrieverAction,
} from 'genkit';
import type { DocumentData, Part } from 'genkit/retriever';

function toContent(
  d: QueryDocumentSnapshot,
  contentField: string | ((snap: QueryDocumentSnapshot) => Part[])
): Part[] {
  if (typeof contentField === 'function') {
    return contentField(d);
  }

  return [{ text: d.get(contentField) }];
}

function toDocuments(
  result: VectorQuerySnapshot,
  vectorField: string,
  contentField: string | ((snap: QueryDocumentSnapshot) => Part[]),
  metadataFields?:
    | string[]
    | ((snap: QueryDocumentSnapshot) => Record<string, any>)
): DocumentData[] {
  return result.docs.map((d) => {
    const out: DocumentData = { content: toContent(d, contentField) };
    if (typeof metadataFields === 'function') {
      out.metadata = metadataFields(d);
      return out;
    }

    out.metadata = { id: d.id };
    if (metadataFields) {
      for (const field of metadataFields) {
        out.metadata[field] = d.get(field);
      }
      return out;
    }

    out.metadata = d.data();
    delete out.metadata[vectorField];
    if (typeof contentField === 'string') delete out.metadata[contentField];
    return out;
  });
}

/**
 * Define a retriever that uses vector similarity search to retrieve documents from Firestore.
 * You must create a vector index on the associated field before you can perform nearest-neighbor
 * search.
 **/
export function defineFirestoreRetriever(
  ai: Genkit,
  config: {
    /** The name of the retriever. */
    name: string;
    /** Optional label for display in Developer UI. */
    label?: string;
    /** The Firestore database instance from which to query. */
    firestore: Firestore;
    /** The name of the collection from which to query. */
    collection?: string;
    /** The embedder to use with this retriever. */
    embedder: EmbedderArgument;
    /** The name of the field within the collection containing the vector data. */
    vectorField: string;
    /** The name of the field containing the document content you wish to return. */
    contentField: string | ((snap: QueryDocumentSnapshot) => Part[]);
    /** The distance measure to use when comparing vectors. Defaults to 'COSINE'. */
    distanceMeasure?: 'EUCLIDEAN' | 'COSINE' | 'DOT_PRODUCT';
    /**
     * Specifies a threshold for which no less similar documents will be returned. The behavior
     * of the specified `distanceMeasure` will affect the meaning of the distance threshold.
     *
     *  - For `distanceMeasure: "EUCLIDEAN"`, the meaning of `distanceThreshold` is:
     *     SELECT docs WHERE euclidean_distance <= distanceThreshold
     *  - For `distanceMeasure: "COSINE"`, the meaning of `distanceThreshold` is:
     *     SELECT docs WHERE cosine_distance <= distanceThreshold
     *  - For `distanceMeasure: "DOT_PRODUCT"`, the meaning of `distanceThreshold` is:
     *     SELECT docs WHERE dot_product_distance >= distanceThreshold
     */
    distanceThreshold?: number;
    /**
     * Optionally specifies the name of a metadata field that will be set on each returned Document,
     * which will contain the computed distance for the document.
     */
    distanceResultField?: string;
    /**
     * A list of fields to include in the returned document metadata. If not supplied, all fields other
     * than the vector are included. Alternatively, provide a transform function to extract the desired
     * metadata fields from a snapshot.
     **/
    metadataFields?:
      | string[]
      | ((snap: QueryDocumentSnapshot) => Record<string, any>);
  }
): RetrieverAction {
  const {
    name,
    label,
    firestore,
    embedder,
    collection,
    vectorField,
    metadataFields,
    contentField,
    distanceMeasure,
    distanceThreshold,
    distanceResultField,
  } = config;
  return ai.defineRetriever(
    {
      name,
      info: {
        label: label || `Firestore - ${name}`,
      },
      configSchema: z.object({
        where: z.record(z.string(), z.any()).optional(),
        /** Max number of results to return. Defaults to 10. */
        limit: z.number().optional(),
        /* Supply or override the distanceMeasure */
        distanceMeasure: z
          .enum(['COSINE', 'DOT_PRODUCT', 'EUCLIDEAN'])
          .optional(),
        /* Supply or override the distanceThreshold */
        distanceThreshold: z.number().optional(),
        /* Supply or override the metadata field where distances are stored. */
        distanceResultField: z.string().optional(),
        /* Supply or override the collection for retrieval. */
        collection: z.string().optional(),
      }),
    },
    async (content, options) => {
      options = options || {};
      if (!options.collection && !collection) {
        throw new Error(
          'Must specify a collection to query in Firestore retriever.'
        );
      }
      let query: Query = firestore.collection(
        options.collection || collection!
      );
      for (const field in options.where || {}) {
        query = query.where(field, '==', options.where![field]);
      }
      // Single embedding for text input
      const queryVector = (await ai.embed({ embedder, content }))[0].embedding;

      const result = await query
        .findNearest({
          vectorField,
          queryVector,
          limit: options.limit || 10,
          distanceMeasure:
            options.distanceMeasure || distanceMeasure || 'COSINE',
          distanceResultField:
            options.distanceResultField || distanceResultField,
          distanceThreshold: options.distanceThreshold || distanceThreshold,
        })
        .get();

      return {
        documents: toDocuments(
          result,
          vectorField,
          contentField,
          metadataFields
        ),
      };
    }
  );
}
