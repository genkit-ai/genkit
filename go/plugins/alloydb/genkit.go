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

package alloydb

import (
	"context"
	"fmt"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// The provider used in the registry.
const provider = "postgres"

// Postgres holds the current plugin state.
type Postgres struct {
	mu      sync.Mutex
	initted bool
	engine  *PostgresEngine
}

func (p *Postgres) Name() string {
	return provider
}

// Init initialize the PostgreSQL
func (p *Postgres) Init(ctx context.Context) []api.Action {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.initted {
		panic("postgres.Init already initted")
	}

	if p.engine == nil {
		panic("postgres.Init engine is nil")
	}

	if p.engine.Pool == nil {
		panic("postgres.Init engine has no pool")
	}
	p.initted = true
	return []api.Action{}
}

// Config provides configuration options for [DefineRetriever].
type Config struct {
	// TableName the table name in which documents will be stored and searched.
	TableName string
	// SchemaName schema name in which documents will be stored and searched.
	SchemaName string
	// ContentColumn column name which contains content of the document
	ContentColumn string
	// EmbeddingColumn column name which contains the vector
	EmbeddingColumn string
	// MetadataColumns a list of columns to create for custom metadata
	MetadataColumns []string
	// IDColumn column name which represents the identifier of the table
	IDColumn string
	// MetadataJSONColumn the column to store extra metadata in JSON format
	MetadataJSONColumn string
	// IgnoreMetadataColumns column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns.
	IgnoreMetadataColumns []string
	// Embedder to use. Required.
	Embedder ai.Embedder
	// EmbedderOptions options to pass to the Embedder.
	EmbedderOptions any
}

// DefineRetriever defines a Retriever with the given configuration.
func DefineRetriever(ctx context.Context, g *genkit.Genkit, p *Postgres, cfg *Config) (*DocStore, ai.Retriever, error) {
	ds, err := newDocStore(ctx, p, cfg)
	if err != nil {
		return nil, nil, err
	}

	return ds, genkit.DefineRetriever(g, provider+"/"+ds.config.TableName, &ai.RetrieverOptions{}, ds.Retrieve), nil
}

// Retriever returns the retriever with the given index name.
func Retriever(g *genkit.Genkit, name string) ai.Retriever {
	return genkit.LookupRetriever(g, name)
}

/* +++++++++++++++++++++++
		Advance methods
 +++++++++++++++++++++++++ */

// ApplyVectorIndex creates an index in the table of the embeddings.
func (p *Postgres) ApplyVectorIndex(ctx context.Context, config *Config, index BaseIndex, name string, concurrently bool) error {
	if !p.initted {
		panic("postgres.ApplyVectorIndex not initted")
	}

	if index.indexType == "exactnearestneighbor" {
		return p.DropVectorIndex(ctx, config, name)
	}

	filter := ""
	if len(index.partialIndexes) > 0 {
		filter = fmt.Sprintf("WHERE %s", index.partialIndexes)
	}
	optsString := index.indexOptions()
	params := fmt.Sprintf("WITH %s", optsString)

	if name == "" {
		if index.name == "" {
			index.name = config.TableName + defaultIndexNameSuffix
		}
		name = index.name
	}

	concurrentlyStr := ""
	if concurrently {
		concurrentlyStr = "CONCURRENTLY"
	}

	function := index.distanceStrategy.searchFunction()
	stmt := fmt.Sprintf(`CREATE INDEX %s %s ON "%s"."%s" USING %s (%s %s) %s %s`,
		concurrentlyStr, name, config.SchemaName, config.TableName, index.indexType, config.EmbeddingColumn, function, params, filter)

	_, err := p.engine.Pool.Exec(ctx, stmt)
	if err != nil {
		return fmt.Errorf("failed to execute creation of index: %w", err)
	}

	return nil
}

// ReIndex recreates the index on the table.
func (p *Postgres) ReIndex(ctx context.Context, config *Config) error {
	if !p.initted {
		panic("postgres.ApplyVectorIndex not initted")
	}
	indexName := config.TableName + defaultIndexNameSuffix
	return p.ReIndexWithName(ctx, indexName)
}

// ReIndexWithName recreates the index on the table by name.
func (p *Postgres) ReIndexWithName(ctx context.Context, indexName string) error {
	if !p.initted {
		panic("postgres.ApplyVectorIndex not initted")
	}
	query := fmt.Sprintf("REINDEX INDEX %s;", indexName)
	_, err := p.engine.Pool.Exec(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to reindex: %w", err)
	}

	return nil
}

// DropVectorIndex drops the vector index from the table.
func (p *Postgres) DropVectorIndex(ctx context.Context, config *Config, indexName string) error {
	if !p.initted {
		panic("postgres.ApplyVectorIndex not initted")
	}
	if indexName == "" {
		indexName = config.TableName + defaultIndexNameSuffix
	}
	query := fmt.Sprintf("DROP INDEX IF EXISTS %s;", indexName)
	_, err := p.engine.Pool.Exec(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to drop vector index: %w", err)
	}

	return nil
}

// IsValidIndex checks if index exists in the table.
func (p *Postgres) IsValidIndex(ctx context.Context, config *Config, indexName string) (bool, error) {
	if !p.initted {
		panic("postgres.ApplyVectorIndex not initted")
	}
	if indexName == "" {
		indexName = config.TableName + defaultIndexNameSuffix
	}
	query := fmt.Sprintf("SELECT tablename, indexname  FROM pg_indexes WHERE tablename = '%s' AND schemaname = '%s' AND indexname = '%s';",
		config.TableName, config.SchemaName, indexName)
	var tableName, indexNameFromDB string
	if err := p.engine.Pool.QueryRow(ctx, query).Scan(&tableName, &indexNameFromDB); err != nil {
		return false, fmt.Errorf("failed to check if index exists: %w", err)
	}

	return indexNameFromDB == indexName, nil
}
