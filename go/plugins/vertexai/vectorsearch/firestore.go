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
	"fmt"
	"log"

	"cloud.google.com/go/firestore"
	"github.com/firebase/genkit/go/ai"
	"github.com/googleapis/gax-go/v2/apierror"
)

// GetFirestoreDocumentRetriever creates a Firestore Document Retriever.
// This function returns a DocumentRetriever function that retrieves documents
// from a Firestore collection based on the provided Vertex AI Vector Search neighbors' IDs.
func GetFirestoreDocumentRetriever(db *firestore.Client, collectionName string) DocumentRetriever {
	return func(ctx context.Context, neighbors []Neighbor, options any) ([]*ai.Document, error) {
		docs := []*ai.Document{}
		for _, neighbor := range neighbors {
			if neighbor.Datapoint.DatapointId == "" {
				log.Printf("Skipping neighbor with empty or nil DatapointId: %+v", neighbor)
				continue
			}

			docRef := db.Collection(collectionName).Doc(neighbor.Datapoint.DatapointId)
			docSnapshot, err := docRef.Get(ctx)
			if err != nil {
				// Log the error but continue to try other neighbors.
				log.Printf("Failed to get document %s from Firestore: %v", neighbor.Datapoint.DatapointId, err)
				continue
			}

			if !docSnapshot.Exists() {
				log.Printf("Document %s does not exist in collection %s. Skipping.", neighbor.Datapoint.DatapointId, collectionName)
				continue
			}

			var firestoreData ai.Document
			if err := docSnapshot.DataTo(&firestoreData); err != nil {
				log.Printf("Failed to unmarshal document data for ID %s: %v", neighbor.Datapoint.DatapointId, err)
				continue
			}

			docs = append(docs, &firestoreData)
		}
		return docs, nil
	}
}

// GetFirestoreDocumentIndexer creates a Firestore Document Indexer.
// This function returns a DocumentIndexer function that indexes documents
// into a Firestore collection.
func GetFirestoreDocumentIndexer(db *firestore.Client, collectionName string) DocumentIndexer {
	return func(ctx context.Context, docs []*ai.Document) ([]string, error) {
		batch := db.Batch()
		var ids []string

		for _, doc := range docs {
			docRef := db.Collection(collectionName).NewDoc() // Generate a new document reference.
			batch.Set(docRef, map[string]interface{}{
				"content":  doc.Content,
				"metadata": doc.Metadata,
			})
			ids = append(ids, docRef.ID)
		}

		// Commit the batch operation.
		if _, err := batch.Commit(ctx); err != nil {
			if apiErr, ok := err.(*apierror.APIError); ok {
				log.Printf("Firestore API Error: %v, DebugInfo: %v", apiErr, apiErr.Details())
			}
			return nil, fmt.Errorf("failed to commit Firestore batch: %w", err)
		}

		return ids, nil
	}
}
