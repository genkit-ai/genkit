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
	"fmt"
	"strings"
)

type DocStore struct {
	engine *PostgresEngine
	config *Config
}

// newDocStore instantiate a DocStore
func newDocStore(ctx context.Context, p *Postgres, cfg *Config) (*DocStore, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.initted {
		panic("postgres.Init not called")
	}

	ds := &DocStore{
		engine: p.Engine,
		config: cfg,
	}

	if strings.TrimSpace(ds.config.TableName) == "" {
		return nil, fmt.Errorf("table name must be defined")
	}

	if ds.config.SchemaName == "" {
		ds.config.SchemaName = defaultSchemaName
	}
	if ds.config.IDColumn == "" {
		ds.config.IDColumn = defaultIDColumn
	}
	if ds.config.MetadataJSONColumn == "" {
		ds.config.MetadataJSONColumn = defaultMetadataJsonColumn
	}
	if ds.config.ContentColumn == "" {
		ds.config.ContentColumn = defaultContentColumn
	}
	if ds.config.EmbeddingColumn == "" {
		ds.config.EmbeddingColumn = defaultEmbeddingColumn
	}

	if ds.config.Embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}

	if err := ds.validateConfiguration(ctx); err != nil {
		return nil, err
	}

	return ds, nil
}

func (ds *DocStore) validateConfiguration(ctx context.Context) error {
	stmt := fmt.Sprintf("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '%s' AND table_schema = '%s'", ds.config.TableName, ds.config.SchemaName)
	rows, err := ds.engine.Pool.Query(ctx, stmt)
	if err != nil {
		return err
	}

	mapColumnNameDataType := make(map[string]string)

	for rows.Next() {
		var columnName, dataType string
		if err := rows.Scan(&columnName, &dataType); err != nil {
			return err
		}
		mapColumnNameDataType[columnName] = dataType
	}

	if _, ok := mapColumnNameDataType[ds.config.IDColumn]; !ok {
		return fmt.Errorf("id column '%s' does not exist", ds.config.IDColumn)
	}

	ccdt, ok := mapColumnNameDataType[ds.config.ContentColumn]
	if !ok {
		return fmt.Errorf("content column '%s' does not exist", ds.config.ContentColumn)
	}

	if ccdt != "text" && strings.Contains(ccdt, "char") {
		return fmt.Errorf("content column '%s' is type '%s'. must be a type of character string", ds.config.ContentColumn, ccdt)
	}

	ecdt, ok := mapColumnNameDataType[ds.config.EmbeddingColumn]
	if !ok {
		return fmt.Errorf("embeddig column '%s' does not exist", ds.config.EmbeddingColumn)
	}

	if ecdt != "USER-DEFINED" {
		return fmt.Errorf("embeddig column '%s' must be a type vector", ds.config.EmbeddingColumn)
	}

	for _, mc := range ds.config.MetadataColumns {
		if _, ok = mapColumnNameDataType[mc]; !ok {
			return fmt.Errorf("metadata column '%s' does not exist", mc)
		}
	}

	// If using IgnoreMetadataColumns, filter out known columns and set known metadata columns
	if len(ds.config.IgnoreMetadataColumns) > 0 {
		delete(mapColumnNameDataType, ds.config.IDColumn)
		delete(mapColumnNameDataType, ds.config.ContentColumn)
		delete(mapColumnNameDataType, ds.config.EmbeddingColumn)

		for _, col := range ds.config.IgnoreMetadataColumns {
			delete(mapColumnNameDataType, col)
		}

		var filteredColumns []string

		for col, _ := range mapColumnNameDataType {
			filteredColumns = append(filteredColumns, col)
		}
		ds.config.MetadataColumns = filteredColumns
	}

	return nil

}
