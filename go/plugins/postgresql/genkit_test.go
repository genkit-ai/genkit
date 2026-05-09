// Copyright 2025 Google LLC
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

package postgresql

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/fakeembedder"
)

const (
	TestTable             = "test_embeddings"
	SchemaName            = "test"
	CustomContentColumn   = "my_content"
	CustomEmbeddingColumn = "my_embedding"
	CustomIdColumn        = "custom_id"
	CustomMetadataColumn  = "custom_metadata"
	DIM                   = 768
)

var testUsername = flag.String("test-postgres-user", "", "postgres username for tests")
var testPassword = flag.String("test-postgres-password", "", "postgres password  for tests")
var testDatabase = flag.String("test-postgres-database", "test_database", "postgres database")
var testProjectID = flag.String("test-postgres-project-id", "", "postgres project id  for tests")
var testRegion = flag.String("test-postgres-region", "", "postgres region for tests")
var testInstance = flag.String("test-postgres-instance", "", "postgres instance for tests")
var testIAMEmail = flag.String("test-postgres-iam-email", "", "postgres instance for tests")

func TestPostgres(t *testing.T) {
	if !areValidFlags() {
		t.Skip("no valid postgres flags")
	}

	ctx := context.Background()

	pEngine, err := NewPostgresEngine(ctx,
		WithUser(*testUsername),
		WithPassword(*testPassword),
		WithCloudSQLInstance(*testProjectID, *testRegion, *testInstance),
		WithDatabase(*testDatabase),
		WithIAMAccountEmail(*testIAMEmail))

	if err != nil {
		t.Fatal(err)
	}

	postgres := &Postgres{
		Engine: pEngine,
	}

	g := genkit.Init(ctx, genkit.WithPlugins(postgres))

	// Create test schema and table
	_, err = pEngine.Pool.Exec(ctx, fmt.Sprintf("CREATE SCHEMA IF NOT EXISTS %s", SchemaName))
	if err != nil {
		t.Fatal(err.Error())
	}

	_, err = pEngine.Pool.Exec(ctx, fmt.Sprintf("DROP TABLE IF EXISTS %s.%s", SchemaName, TestTable))
	if err != nil {
		t.Fatal(err.Error())
	}

	// Initialize the vectorstore table
	err = pEngine.InitVectorstoreTable(ctx, VectorstoreTableOptions{
		TableName:          TestTable,
		VectorSize:         DIM,
		SchemaName:         SchemaName,
		ContentColumnName:  CustomContentColumn,
		EmbeddingColumn:    CustomEmbeddingColumn,
		MetadataJSONColumn: CustomMetadataColumn,
		IDColumn: Column{
			Name:     CustomIdColumn,
			Nullable: false,
		},
		MetadataColumns: []Column{
			{
				Name:     "source",
				DataType: "text",
				Nullable: true,
			},
			{
				Name:     "name",
				DataType: "text",
				Nullable: true,
			},
		},
		OverwriteExisting: true,
		StoreMetadata:     true,
	})
	if err != nil {
		t.Fatal(err.Error())
	}

	d1 := ai.DocumentFromText("hello1", map[string]any{"source": "test1", "name": "some_value1", "custom_metadata": "{\"key\":\"value1\"}"})
	d2 := ai.DocumentFromText("hello2", map[string]any{"source": "test2", "name": "some_value2", "custom_metadata": "{\"key\":\"value2\"}"})
	d3 := ai.DocumentFromText("goodbye", map[string]any{"source": "test3", "name": "some_value3", "custom_metadata": "{\"key\": { \"subKey\":\"value3\"}}"})

	embedder := newFakeEmbedder([3]*ai.Document{d1, d2, d3})

	cfg := &Config{
		TableName:             TestTable,
		SchemaName:            SchemaName,
		ContentColumn:         CustomContentColumn,
		EmbeddingColumn:       CustomEmbeddingColumn,
		MetadataColumns:       []string{"source", "name"},
		IDColumn:              CustomIdColumn,
		MetadataJSONColumn:    CustomMetadataColumn,
		IgnoreMetadataColumns: []string{"created_at", "updated_at"},
		Embedder:              genkit.DefineEmbedder(g, "fake/embedder3", nil, embedder.Embed),
		EmbedderOptions:       nil,
	}

	ds, retriever, err := DefineRetriever(ctx, g, postgres, cfg)
	if err != nil {
		t.Fatal(err)
	}

	docs := []*ai.Document{d1, d2, d3}

	err = ds.Index(ctx, docs)
	if err != nil {
		t.Fatal(err.Error())
	}

	rows, err := pEngine.Pool.Query(ctx, fmt.Sprintf("SELECT * FROM %s.%s", SchemaName, TestTable))
	if err != nil {
		t.Fatal(err.Error())
	}

	if !rows.Next() {
		t.Fatal("must have a single document")
	}

	resp, err := retriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query: d1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Documents) != 3 {
		t.Fatalf("expected 3 documents, got %d", len(resp.Documents))
	}
	if len(resp.Documents[0].Content) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(resp.Documents[0].Content))
	}
	if got, want := resp.Documents[0].Content[0].Text, "hello1"; got != want {
		t.Errorf("got content %q, want %q", got, want)
	}

	resp, err = retriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query: d1,
		Options: &RetrieverOptions{
			Filter: "name='some_value2'",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(resp.Documents) != 1 {
		t.Fatalf("expected 1 document, got %d", len(resp.Documents))
	}
	if len(resp.Documents[0].Content) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(resp.Documents[0].Content))
	}
	if got, want := resp.Documents[0].Content[0].Text, "hello2"; got != want {
		t.Errorf("got content %q, want %q", got, want)
	}

}

/** ***************
Helper functions
************** */

func newFakeEmbedder(docs [3]*ai.Document) *fakeembedder.Embedder {

	v1 := make([]float32, DIM)
	v2 := make([]float32, DIM)
	v3 := make([]float32, DIM)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i)
		v3[i] = float32(DIM - i)
	}
	v2[0] = 1

	embedder := fakeembedder.New()
	embedder.Register(docs[0], v1)
	embedder.Register(docs[1], v2)
	embedder.Register(docs[2], v3)

	return embedder
}

func areValidFlags() bool {
	isBasicAuth := testUsername != nil && strings.TrimSpace(*testUsername) != "" &&
		testPassword != nil && strings.TrimSpace(*testPassword) != ""

	isIAMAuth := testIAMEmail != nil && strings.TrimSpace(*testIAMEmail) != ""

	isPostgresEnv := testDatabase != nil && strings.TrimSpace(*testDatabase) != "" &&
		testProjectID != nil && strings.TrimSpace(*testProjectID) != "" &&
		testRegion != nil && strings.TrimSpace(*testRegion) != "" &&
		testInstance != nil && strings.TrimSpace(*testInstance) != ""

	return (isBasicAuth || isIAMAuth) && isPostgresEnv
}
