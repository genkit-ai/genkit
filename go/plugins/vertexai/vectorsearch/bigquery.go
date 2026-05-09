// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vectorsearch

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"time"

	"cloud.google.com/go/bigquery"
	"github.com/firebase/genkit/go/ai"
	"google.golang.org/api/iterator"
)

// BigQueryDocumentRow defines the structure of a row in the BigQuery table.
type BigQueryDocumentRow struct {
	ID       string `bigquery:"id"`
	Content  string `bigquery:"content"`  // Stored as JSON string
	Metadata string `bigquery:"metadata"` // Stored as JSON string
}

// GetBigQueryDocumentRetriever creates a BigQuery Document Retriever.
// This function returns a DocumentRetriever function that retrieves documents
// from a BigQuery table based on the provided neighbors' IDs.
func GetBigQueryDocumentRetriever(bqClient *bigquery.Client, datasetID, tableID string) DocumentRetriever {
	return func(ctx context.Context, neighbors []Neighbor, options any) ([]*ai.Document, error) {
		var ids []string
		for _, neighbor := range neighbors {
			if neighbor.Datapoint.DatapointId != "" {
				ids = append(ids, neighbor.Datapoint.DatapointId)
			}
		}

		if len(ids) == 0 {
			return []*ai.Document{}, nil
		}

		// Constructing the query with UNNEST for array parameters
		// BigQuery expects parameters for IN clauses with UNNEST to be arrays.
		query := fmt.Sprintf("SELECT id, content, metadata FROM `%s.%s.%s` WHERE id IN UNNEST(@ids)", bqClient.Project(), datasetID, tableID)

		q := bqClient.Query(query)
		q.Parameters = []bigquery.QueryParameter{
			{Name: "ids", Value: ids},
		}

		it, err := q.Read(ctx)
		if err != nil {
			log.Printf("Failed to execute BigQuery query: %v", err)
			return nil, fmt.Errorf("failed to query BigQuery: %w", err)
		}

		var documents []*ai.Document
		for {
			var row BigQueryDocumentRow
			err := it.Next(&row)
			if err == iterator.Done {
				break
			}
			if err != nil {
				log.Printf("Error reading BigQuery row: %v", err)
				return nil, fmt.Errorf("error reading BigQuery row: %w", err)
			}

			var doc ai.Document

			if err := json.Unmarshal([]byte(row.Content), &doc.Content); err != nil {
				log.Printf("Failed to parse content for document ID %s: %v", row.ID, err)
			}
			if err := json.Unmarshal([]byte(row.Metadata), &doc.Metadata); err != nil {
				log.Printf("Failed to parse metadata for document ID %s: %v", row.ID, err)
			}
			documents = append(documents, &doc)
		}

		return documents, nil
	}
}

// GetBigQueryDocumentIndexer creates a BigQuery Document Indexer.
// This function returns a DocumentIndexer function that indexes documents
// into a BigQuery table. It generates a random ID for each document and
// stores the content and metadata as JSON strings.
func GetBigQueryDocumentIndexer(bqClient *bigquery.Client, datasetID, tableID string) func(ctx context.Context, docs []*ai.Document) ([]string, error) {
	return func(ctx context.Context, docs []*ai.Document) ([]string, error) {
		var ids []string
		var rows []*BigQueryDocumentRow

		// Seed the random number generator for generating unique IDs.
		rand.Seed(time.Now().UnixNano())

		for _, doc := range docs {
			id := fmt.Sprintf("%x", rand.Int63()) // Generate a random ID.
			ids = append(ids, id)

			content, err := json.Marshal(doc.Content)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal document content: %w", err)
			}

			metadata, err := json.Marshal(doc.Metadata)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal document metadata: %w", err)
			}

			row := &BigQueryDocumentRow{
				ID:       id,
				Content:  string(content),
				Metadata: string(metadata),
			}
			rows = append(rows, row)
		}

		// Log rows for debugging.
		for _, row := range rows {
			log.Printf("Inserting row: %+v", row)
		}

		// Insert rows into the BigQuery table.
		inserter := bqClient.Dataset(datasetID).Table(tableID).Inserter()
		if err := inserter.Put(ctx, rows); err != nil {
			return nil, fmt.Errorf("failed to insert rows into BigQuery: %w", err)
		}

		return ids, nil
	}
}
