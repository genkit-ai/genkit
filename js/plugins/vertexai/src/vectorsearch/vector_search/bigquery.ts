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

import type { BigQuery, QueryRowsResponse } from '@google-cloud/bigquery';
import { z } from 'genkit';
import { logger } from 'genkit/logging';
import { Document, DocumentDataSchema } from 'genkit/retriever';
import type { DocumentIndexer, DocumentRetriever, Neighbor } from './types.js';

/**
 * Creates a BigQuery Document Retriever.
 *
 * This function returns a DocumentRetriever function that retrieves documents
 * from a BigQuery table based on the provided neighbors.
 *
 * @param {BigQuery} bq - The BigQuery instance.
 * @param {string} tableId - The ID of the BigQuery table.
 * @param {string} datasetId - The ID of the BigQuery dataset.
 * @returns {DocumentRetriever} - The DocumentRetriever function.
 */
export const getBigQueryDocumentRetriever = (
  bq: BigQuery,
  tableId: string,
  datasetId: string
): DocumentRetriever => {
  const bigQueryRetriever: DocumentRetriever = async (
    neighbors: Neighbor[]
  ): Promise<Document[]> => {
    const ids: string[] = neighbors
      .map((neighbor) => neighbor.datapoint?.datapointId)
      .filter(Boolean) as string[];

    const query = `
      SELECT * FROM \`${datasetId}.${tableId}\`
      WHERE id IN UNNEST(@ids)
    `;

    const options = {
      query,
      params: { ids },
    };

    let rows: QueryRowsResponse[0];

    try {
      [rows] = await bq.query(options);
    } catch (queryError) {
      logger.error('Failed to execute BigQuery query:', queryError);
      return [];
    }

    const documents: Document[] = [];

    for (const row of rows) {
      try {
        const docData: { content: any; metadata?: any } = {
          content: JSON.parse(row.content),
        };

        if (row.metadata) {
          docData.metadata = JSON.parse(row.metadata);
        }

        const parsedDocData = DocumentDataSchema.parse(docData);
        documents.push(new Document(parsedDocData));
      } catch (error) {
        const id = row.id;
        const errorPrefix = `Failed to parse document data for document with ID ${id}:`;

        if (error instanceof z.ZodError || error instanceof Error) {
          logger.warn(`${errorPrefix} ${error.message}`);
        } else {
          logger.warn(errorPrefix);
        }
      }
    }

    return documents;
  };

  return bigQueryRetriever;
};

/**
 * Creates a BigQuery Document Indexer.
 *
 * This function returns a DocumentIndexer function that indexes documents
 * into a BigQuery table. Note this indexer does not handle duplicate
 * documents.
 *
 * @param {BigQuery} bq - The BigQuery instance.
 * @param {string} tableId - The ID of the BigQuery table.
 * @param {string} datasetId - The ID of the BigQuery dataset.
 * @returns {DocumentIndexer} - The DocumentIndexer function.
 */
export const getBigQueryDocumentIndexer = (
  bq: BigQuery,
  tableId: string,
  datasetId: string
): DocumentIndexer => {
  const bigQueryIndexer: DocumentIndexer = async (
    docs: Document[]
  ): Promise<string[]> => {
    const ids: string[] = [];
    const rows = docs.map((doc) => {
      const id = Math.random().toString(36).substring(7);
      ids.push(id);
      return {
        id,
        content: JSON.stringify(doc.content),
        metadata: JSON.stringify(doc.metadata),
      };
    });
    await bq.dataset(datasetId).table(tableId).insert(rows);
    return ids;
  };
  return bigQueryIndexer;
};
