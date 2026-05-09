// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package firebase

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"cloud.google.com/go/firestore"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const firestoreCollectionEnv = "FIRESTORE_COLLECTION"

type VectorType int

// TODO: in retriever options add field that controls the 32/64

// RetrieverOptions struct for retriever configuration
type RetrieverOptions struct {
	Name            string                    // Name of the retriever
	Label           string                    // Label for the retriever
	Collection      string                    // Firestore collection name
	Embedder        ai.Embedder               // Embedder instance for generating embeddings
	VectorField     string                    // Field name for storing vectors
	MetadataFields  []string                  // List of metadata fields to retrieve
	ContentField    string                    // Field name for storing content
	Limit           int                       // Limit on the number of results
	DistanceMeasure firestore.DistanceMeasure // Distance measure for vector similarity
}

// Convert a Firestore document snapshot to a Genkit Document object.
func convertToDoc(docSnapshots []*firestore.DocumentSnapshot, contentField string, metadataFields []string) []*ai.Document {
	var documents []*ai.Document // Prepare the documents to return in the response

	for _, result := range docSnapshots {
		data := result.Data() // Retrieve document data

		// Ensure content field exists and is of type string
		content, ok := data[contentField].(string)
		if !ok {
			fmt.Printf("Content field %s missing or not a string in document %s", contentField, result.Ref.ID)
			continue
		}

		// Extract metadata fields
		metadata := make(map[string]any)
		for _, field := range metadataFields {
			if value, ok := data[field]; ok {
				metadata[field] = value
			}
		}

		// Create a Genkit Document object
		doc := ai.DocumentFromText(content, metadata)
		documents = append(documents, doc)
	}
	return documents
}

// defineFirestoreRetriever defines and registers a retriever for Firestore.
func defineFirestoreRetriever(g *genkit.Genkit, cfg RetrieverOptions, client *firestore.Client) (ai.Retriever, error) {

	if client == nil {
		return nil, fmt.Errorf("defineFirestoreRetriever: Firestore client is not provided")
	}

	// Resolve the Firestore collection name
	collection, err := resolveFirestoreCollection(cfg.Collection)
	if err != nil {
		return nil, err
	}

	// Define the retriever function
	retrieve := func(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
		if req.Query == nil {
			return nil, fmt.Errorf("defineFirestoreRetriever: Request document is nil")
		}

		// Generate query embedding using the Embedder
		embedRequest := &ai.EmbedRequest{Input: []*ai.Document{req.Query}}
		embedResponse, err := cfg.Embedder.Embed(ctx, embedRequest)
		if err != nil {
			return nil, fmt.Errorf("defineFirestoreRetriever: Embedding failed: %v", err)
		}

		if len(embedResponse.Embeddings) == 0 {
			return nil, fmt.Errorf("defineFirestoreRetriever: No embeddings returned")
		}

		queryEmbedding := embedResponse.Embeddings[0].Embedding
		if len(queryEmbedding) == 0 {
			return nil, fmt.Errorf("defineFirestoreRetriever: Generated embedding is empty")
		}

		// Perform the FindNearest query
		vectorQuery := client.Collection(collection).FindNearest(
			cfg.VectorField,
			firestore.Vector32(queryEmbedding),
			cfg.Limit,
			cfg.DistanceMeasure,
			nil,
		)
		iter := vectorQuery.Documents(ctx)

		results, err := iter.GetAll()
		if err != nil {
			return nil, fmt.Errorf("defineFirestoreRetriever: FindNearest query failed: %v", err)
		}

		// Convert Firestore documents to Genkit documents
		documents := convertToDoc(results, cfg.ContentField, cfg.MetadataFields)
		return &ai.RetrieverResponse{Documents: documents}, nil
	}

	retOpts := &ai.RetrieverOptions{
		ConfigSchema: core.InferSchemaMap(cfg),
		Label:        cfg.Name,
		Supports: &ai.RetrieverSupports{
			Media: false,
		},
	}

	return genkit.DefineRetriever(g, api.NewName(provider, cfg.Name), retOpts, retrieve), nil
}

func resolveFirestoreCollection(collectionName string) (string, error) {
	if collectionName != "" {
		return collectionName, nil
	}
	collectionName = os.Getenv(firestoreCollectionEnv)
	if collectionName == "" {
		return "", fmt.Errorf("firebase: no Firestore collection provided. " +
			"Pass the collection in RetrieverOptions: RetrieverOptions{Collection: \"my-collection\"}")
	}
	slog.Warn("Using FIRESTORE_COLLECTION environment variable is deprecated for retriever configuration. "+
		"Use RetrieverOptions{Collection: \"my-collection\"} instead.",
		"collection", collectionName)
	return collectionName, nil
}
