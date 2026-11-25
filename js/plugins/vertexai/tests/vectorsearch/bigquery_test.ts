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

import type { BigQuery } from '@google-cloud/bigquery';
import * as assert from 'assert';
import { Document } from 'genkit/retriever';
import { describe, it } from 'node:test';
import { getBigQueryDocumentRetriever } from '../../src/vectorsearch/index.js';

class MockBigQuery {
  query: Function;

  constructor({
    mockRows,
    shouldThrowError = false,
  }: {
    mockRows: any[];
    shouldThrowError?: boolean;
  }) {
    this.query = async (_options: {
      query: string;
      params: { ids: string[] };
    }) => {
      if (shouldThrowError) {
        throw new Error('Query failed');
      }
      return [mockRows];
    };
  }
}

describe('getBigQueryDocumentRetriever', () => {
  it('returns a function that retrieves documents from BigQuery', async () => {
    const doc1 = Document.fromText('content1');
    const doc2Metadata = {
      restricts: [{ namespace: 'color', allowList: 'red' }],
    };
    const doc2 = Document.fromText('content2', doc2Metadata);

    const mockRows = [
      {
        id: '1',
        content: JSON.stringify(doc1.content),
        metadata: null,
      },
      {
        id: '2',
        content: JSON.stringify(doc2.content),
        metadata: JSON.stringify(doc2.metadata),
      },
    ];

    const mockBigQuery = new MockBigQuery({ mockRows }) as unknown as BigQuery;
    const documentRetriever = getBigQueryDocumentRetriever(
      mockBigQuery,
      'test-table',
      'test-dataset'
    );

    const documents = await documentRetriever([
      { datapoint: { datapointId: '1' } },
      { datapoint: { datapointId: '2' } },
    ]);

    assert.deepStrictEqual(documents, [doc1, doc2]);
  });

  it('returns an empty array when no documents match', async () => {
    const mockRows: any[] = [];

    const mockBigQuery = new MockBigQuery({ mockRows }) as unknown as BigQuery;
    const documentRetriever = getBigQueryDocumentRetriever(
      mockBigQuery,
      'test-table',
      'test-dataset'
    );

    const documents = await documentRetriever([
      { datapoint: { datapointId: '3' } },
    ]);

    assert.deepStrictEqual(documents, []);
  });

  it('handles BigQuery query errors', async () => {
    const mockBigQuery = new MockBigQuery({
      mockRows: [],
      shouldThrowError: true,
    }) as unknown as BigQuery;
    const documentRetriever = getBigQueryDocumentRetriever(
      mockBigQuery,
      'test-table',
      'test-dataset'
    );
    //  no need to assert the error, just make sure it doesn't throw
    await documentRetriever([{ datapoint: { datapointId: '1' } }]);
  });

  it('filters out invalid documents', async () => {
    const validDoc = Document.fromText('valid content');
    const mockRows = [
      {
        id: '1',
        content: JSON.stringify(validDoc.content),
        metadata: null,
      },
      {
        id: '2',
        content: 'invalid JSON',
        metadata: null,
      },
    ];

    const mockBigQuery = new MockBigQuery({ mockRows }) as unknown as BigQuery;
    const documentRetriever = getBigQueryDocumentRetriever(
      mockBigQuery,
      'test-table',
      'test-dataset'
    );

    const documents = await documentRetriever([
      { datapoint: { datapointId: '1' } },
      { datapoint: { datapointId: '2' } },
    ]);

    assert.deepStrictEqual(documents, [validDoc]);
  });

  it('handles missing content in documents', async () => {
    const validDoc = Document.fromText('valid content');
    const mockRows = [
      {
        id: '1',
        content: JSON.stringify(validDoc.content),
        metadata: null,
      },
      {
        id: '2',
        content: null,
        metadata: null,
      },
    ];

    const mockBigQuery = new MockBigQuery({ mockRows }) as unknown as BigQuery;
    const documentRetriever = getBigQueryDocumentRetriever(
      mockBigQuery,
      'test-table',
      'test-dataset'
    );

    const documents = await documentRetriever([
      { datapoint: { datapointId: '1' } },
      { datapoint: { datapointId: '2' } },
    ]);

    assert.deepStrictEqual(documents, [validDoc]);
  });
});
