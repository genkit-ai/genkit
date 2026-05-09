// Copyright 2024 Google LLC
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

package firebase

import (
	"context"
	"flag"
	"testing"

	"cloud.google.com/go/firestore"
	firebasev4 "firebase.google.com/go/v4"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"google.golang.org/api/iterator"
)

var (
	testProjectID   = flag.String("test-project-id", "", "GCP Project ID to use for tests")
	testCollection  = flag.String("test-collection", "samplecollection", "Firestore collection to use for tests")
	testVectorField = flag.String("test-vector-field", "embedding", "Field name for vector embeddings")
)

/*
 * Pre-requisites to run this test:
 *
 * 1. **Set Up Firebase Project and Firestore:**
 *    You must create a Firebase project and ensure Firestore is enabled for that project. To do so:
 *
 *    - Visit the Firebase Console: https://console.firebase.google.com/
 *    - Create a new project (or use an existing one).
 *    - Enable Firestore in your project from the "Build" section > "Firestore Database".
 *
 * 2. **Create a Firestore Collection and Composite Index:**
 *    This test assumes you have a Firestore collection set up for storing documents with vector embeddings.
 *    Additionally, you need to create a vector index for the embedding field. You can do this via the Firestore API with the following `curl` command:
 *
 *    ```bash
 *    curl -X POST \
 *      "https://firestore.googleapis.com/v1/projects/<YOUR_PROJECT_ID>/databases/(default)/collectionGroups/<YOUR_COLLECTION>/indexes" \
 *      -H "Authorization: Bearer $(gcloud auth print-access-token)" \
 *      -H "Content-Type: application/json" \
 *      -d '{
 *        "fields": [
 *          {
 *            "fieldPath": "embedding",
 *            "vectorConfig": {
 *              "dimension": 3,
 *              "flat": {}
 *            }
 *          }
 *        ],
 *        "queryScope": "COLLECTION"
 *      }'
 *    ```
 *    Replace `<YOUR_PROJECT_ID>` and `<YOUR_COLLECTION>` with your project and collection names.
 *
 * 3. **Authentication & Credentials:**
 *    Ensure you have access to the project and Firestore using Google Cloud CLI. You can authenticate using the following commands:
 *
 *    ```bash
 *    gcloud auth login
 *    gcloud config set project <YOUR_PROJECT_ID>
 *    gcloud auth application-default login
 *    ```
 *
 *    This authenticates your local environment with your GCP project and ensures the Go SDK can access Firestore.
 *
 * 4. **Running the Test:**
 *    After setting up Firestore and the index, you can run the test by passing in the required flags for the project, collection, and vector field:
 *
 *    ```bash
 *    go test \
 *      -test-project-id=<YOUR_PROJECT_ID> \
 *      -test-collection=<YOUR_COLLECTION> \
 *      -test-vector-field=embedding
 *    ```
 */

// MockEmbedder implements the Embedder interface for testing purposes
type MockEmbedder struct{}

func (e *MockEmbedder) Name() string {
	return "MockEmbedder"
}

func (e *MockEmbedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	var embeddings []*ai.Embedding
	for _, doc := range req.Input {
		var embedding []float32
		switch doc.Content[0].Text {
		case "This is document one":
			// Embedding for document one is the closest to the query
			embedding = []float32{0.9, 0.1, 0.0}
		case "This is document two":
			// Embedding for document two is less close to the query
			embedding = []float32{0.7, 0.2, 0.1}
		case "This is document three":
			// Embedding for document three is even further from the query
			embedding = []float32{0.4, 0.3, 0.3}
		case "This is input query":
			// Embedding for the input query
			embedding = []float32{0.9, 0.1, 0.0}
		default:
			// Default embedding for any other documents
			embedding = []float32{0.0, 0.0, 0.0}
		}

		embeddings = append(embeddings, &ai.Embedding{Embedding: embedding})
	}
	return &ai.EmbedResponse{Embeddings: embeddings}, nil
}

func (e *MockEmbedder) Register(r api.Registry) {}

// To run this test you must have a Firestore database initialized in a GCP project, with a vector indexed collection (of dimension 3).
// Warning: This test will delete all documents in the collection in cleanup.

func TestFirestoreRetriever(t *testing.T) {
	//  skip if flags aren't defined
	if *testProjectID == "" {
		t.Skip("Skipping test due to missing flags")
	}
	if *testCollection == "" {
		t.Skip("Skipping test due to missing flags")
	}
	if *testVectorField == "" {
		t.Skip("Skipping test due to missing flags")
	}

	ctx := context.Background()
	g := genkit.Init(ctx)

	// Initialize Firebase app
	app, err := firebasev4.NewApp(ctx, &firebasev4.Config{ProjectID: *testProjectID})
	if err != nil {
		t.Fatalf("Failed to create Firebase app: %v", err)
	}

	// Initialize Firestore client
	client, err := app.Firestore(ctx)
	if err != nil {
		t.Fatalf("Failed to create Firestore client: %v", err)
	}
	defer client.Close()

	// Clean up the collection before the test
	defer deleteCollection(ctx, client, *testCollection, t)

	// Initialize the embedder
	embedder := &MockEmbedder{}

	// Insert test documents with embeddings generated by the embedder
	testDocs := []struct {
		ID   string
		Text string
		Data map[string]any
	}{
		{"doc1", "This is document one", map[string]any{"metadata": "meta1"}},
		{"doc2", "This is document two", map[string]any{"metadata": "meta2"}},
		{"doc3", "This is document three", map[string]any{"metadata": "meta3"}},
	}

	// Expected document text content in order of relevance for the query
	expectedTexts := []string{
		"This is document one",
		"This is document two",
	}

	for _, doc := range testDocs {
		// Create an ai.Document
		aiDoc := ai.DocumentFromText(doc.Text, doc.Data)

		// Generate embedding
		embedRequest := &ai.EmbedRequest{Input: []*ai.Document{aiDoc}}
		embedResponse, err := embedder.Embed(ctx, embedRequest)
		if err != nil {
			t.Fatalf("Failed to generate embedding for document %s: %v", doc.ID, err)
		}

		if len(embedResponse.Embeddings) == 0 {
			t.Fatalf("No embeddings returned for document %s", doc.ID)
		}

		embedding := embedResponse.Embeddings[0].Embedding
		if len(embedding) == 0 {
			t.Fatalf("Generated embedding is empty for document %s", doc.ID)
		}

		// Store in Firestore
		_, err = client.Collection(*testCollection).Doc(doc.ID).Set(ctx, map[string]any{
			"text":           doc.Text,
			"metadata":       doc.Data["metadata"],
			*testVectorField: firestore.Vector32(embedding),
		})
		if err != nil {
			t.Fatalf("Failed to insert document %s: %v", doc.ID, err)
		}
		t.Logf("Inserted document: %s with embedding: %v", doc.ID, embedding)
	}

	// Define retriever options
	retrieverOptions := RetrieverOptions{
		Name:            "test-retriever",
		Label:           "Test Retriever",
		Collection:      *testCollection,
		Embedder:        embedder,
		VectorField:     *testVectorField,
		MetadataFields:  []string{"metadata"},
		ContentField:    "text",
		Limit:           2,
		DistanceMeasure: firestore.DistanceMeasureEuclidean,
	}

	// Define the retriever
	retriever, err := defineFirestoreRetriever(g, retrieverOptions, client)
	if err != nil {
		t.Fatalf("Failed to define retriever: %v", err)
	}

	// Create a retriever request with the input document
	queryText := "This is input query"
	inputDocument := ai.DocumentFromText(queryText, nil)

	req := &ai.RetrieverRequest{
		Query: inputDocument,
	}

	// Perform the retrieval
	resp, err := retriever.Retrieve(ctx, req)
	if err != nil {
		t.Fatalf("Retriever failed: %v", err)
	}

	// Check the retrieved documents
	if len(resp.Documents) == 0 {
		t.Fatalf("No documents retrieved")
	}

	// Verify the content of all retrieved documents against the expected list
	for i, doc := range resp.Documents {
		if i >= len(expectedTexts) {
			t.Errorf("More documents retrieved than expected. Retrieved: %d, Expected: %d", len(resp.Documents), len(expectedTexts))
			break
		}

		if doc.Content[0].Text != expectedTexts[i] {
			t.Errorf("Mismatch in document %d content. Expected: '%s', Got: '%s'", i+1, expectedTexts[i], doc.Content[0].Text)
		} else {
			t.Logf("Retrieved Document %d matches expected content: '%s'", i+1, expectedTexts[i])
		}
	}
}

func deleteCollection(ctx context.Context, client *firestore.Client, collectionName string, t *testing.T) {
	// Get all documents in the collection
	iter := client.Collection(collectionName).Documents(ctx)
	for {
		doc, err := iter.Next()
		if err == iterator.Done {
			break // No more documents
		}
		if err != nil {
			t.Fatalf("Failed to iterate documents for deletion: %v", err)
		}

		// Delete each document
		_, err = doc.Ref.Delete(ctx)
		if err != nil {
			t.Errorf("Failed to delete document %s: %v", doc.Ref.ID, err)
		} else {
			t.Logf("Deleted document: %s", doc.Ref.ID)
		}
	}
}
