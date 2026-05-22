// Copyright 2026 Google LLC
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
//
// SPDX-License-Identifier: Apache-2.0

// Package valkey provides a Genkit plugin for Valkey vector store using
// valkey-glide. It supports indexing documents as Hashes with HNSW vector
// fields and retrieving them via FT.SEARCH KNN queries.
package valkey

import (
	"context"
	"crypto/md5"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	glide "github.com/valkey-io/valkey-glide/go/v2"
	"github.com/valkey-io/valkey-glide/go/v2/config"
	"github.com/valkey-io/valkey-glide/go/v2/constants"
	glideopts "github.com/valkey-io/valkey-glide/go/v2/options"
	"github.com/valkey-io/valkey-glide/go/v2/pipeline"
	"github.com/valkey-io/valkey-glide/go/v2/servermodules/glideft"
)

const provider = "valkey"

// MetadataFieldType describes how a metadata field is indexed in the FT schema.
type MetadataFieldType string

const (
	// MetadataFieldTypeTag indexes the field as a TAG (exact-match filter).
	MetadataFieldTypeTag MetadataFieldType = "TAG"
	// MetadataFieldTypeNumeric indexes the field as a NUMERIC (range filter).
	MetadataFieldTypeNumeric MetadataFieldType = "NUMERIC"
)

// MetadataFieldConfig declares a metadata key to be indexed for query-time filtering.
type MetadataFieldConfig struct {
	// Name is the metadata key (must match a key in the document's Metadata map).
	Name string
	// Type determines whether the field is indexed as TAG or NUMERIC.
	Type MetadataFieldType
}

// Valkey implements the Genkit plugin interface for Valkey vector store.
type Valkey struct {
	// Addresses is the list of Valkey server addresses.
	Addresses []config.NodeAddress

	mu      sync.Mutex
	client  *glide.Client
	initted bool
	initErr error
}

// Close closes the underlying Glide client connection.
func (v *Valkey) Close() {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.client != nil {
		v.client.Close()
		v.client = nil
	}
}

// Client returns the underlying Glide client. Useful for administrative
// operations such as dropping indexes in tests.
func (v *Valkey) Client() *glide.Client {
	return v.client
}

// Name returns the plugin name.
func (v *Valkey) Name() string {
	return provider
}

// Init initializes the Valkey plugin by creating a Glide client connection.
// If initialization fails, the error is stored and surfaced on first use via
// newDocstore. This avoids panicking on recoverable errors like transient
// network failures.
func (v *Valkey) Init(ctx context.Context) []api.Action {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.initted {
		v.initErr = errors.New("valkey.Init already called")
		return []api.Action{}
	}

	clientConfig := config.NewClientConfiguration()
	for i := range v.Addresses {
		clientConfig.WithAddress(&v.Addresses[i])
	}
	clientConfig.WithClientName("genkit_vector_store_client")

	client, err := glide.NewClient(clientConfig)
	if err != nil {
		v.initErr = fmt.Errorf("valkey.Init: failed to create client: %w", err)
		v.initted = true
		return []api.Action{}
	}
	v.client = client
	v.initted = true
	return []api.Action{}
}

// Config provides configuration options for DefineRetriever.
type Config struct {
	// IndexName is the name of the FT index in Valkey.
	IndexName string
	// Embedder to use for embedding documents and queries.
	Embedder ai.Embedder
	// EmbedderOptions are options passed to the embedder.
	EmbedderOptions any
	// Dimension is the vector dimension for the HNSW index.
	Dimension int
	// Prefix is the key prefix for stored documents. Defaults to IndexName.
	Prefix string
	// DistanceMetric is the distance metric for the HNSW index.
	// Defaults to constants.DistanceMetricCosine.
	DistanceMetric constants.DistanceMetric
	// MetadataFields declares which document metadata keys to index for filtering.
	MetadataFields []MetadataFieldConfig
}

// RetrieverOptions may be passed in the Options field of ai.RetrieverRequest.
type RetrieverOptions struct {
	K int `json:"k,omitempty"` // Number of results to return; defaults to 10.
	// Filter is a raw FT.SEARCH pre-filter expression interpolated directly
	// into the query. Callers must ensure it does not contain untrusted user
	// input — a malformed value can alter query semantics.
	Filter string `json:"filter,omitempty"`
}

// Docstore holds the Valkey client and configuration for indexing and retrieval.
type Docstore struct {
	Client          *glide.Client
	IndexName       string
	Prefix          string
	Dimension       int
	Embedder        ai.Embedder
	EmbedderOptions any
	MetadataFields  []MetadataFieldConfig
}

// IndexerRequest is the input type for the indexer action.
type IndexerRequest struct {
	Documents []*ai.Document `json:"documents"`
}

// IndexerResponse is the output type for the indexer action (empty on success).
type IndexerResponse struct{}

// DefineRetriever defines a Retriever and Indexer backed by Valkey vector search.
// It ensures the FT index exists and registers both actions with the Genkit registry.
func DefineRetriever(ctx context.Context, g *genkit.Genkit, cfg Config, opts *ai.RetrieverOptions) (*Docstore, ai.Retriever, error) {
	plugin := genkit.LookupPlugin(g, provider)
	if plugin == nil {
		return nil, nil, errors.New("valkey plugin not found; did you call genkit.Init with the valkey plugin")
	}
	p := plugin.(*Valkey)

	ds, err := p.newDocstore(ctx, cfg)
	if err != nil {
		return nil, nil, err
	}

	// Register the indexer action so it appears in the Genkit Dev UI and can
	// be invoked via the reflection API, matching JS and Python behaviour.
	indexerAction := core.NewAction(
		api.NewName(provider, cfg.IndexName),
		api.ActionTypeIndexer,
		map[string]any{
			"type": api.ActionTypeIndexer,
			"indexer": map[string]any{
				"label": fmt.Sprintf("Valkey - %s", cfg.IndexName),
			},
		},
		nil,
		func(ctx context.Context, req *IndexerRequest) (*IndexerResponse, error) {
			if err := ds.Index(ctx, req.Documents); err != nil {
				return nil, err
			}
			return &IndexerResponse{}, nil
		},
	)
	genkit.RegisterAction(g, indexerAction)

	return ds, genkit.DefineRetriever(g, api.NewName(provider, cfg.IndexName), opts, ds.Retrieve), nil
}

// Retriever returns the retriever with the given index name.
func Retriever(g *genkit.Genkit, name string) ai.Retriever {
	return genkit.LookupRetriever(g, api.NewName(provider, name))
}

// IsDefinedRetriever reports whether the named Retriever is defined by this plugin.
func IsDefinedRetriever(g *genkit.Genkit, name string) bool {
	return genkit.LookupRetriever(g, api.NewName(provider, name)) != nil
}

func (v *Valkey) newDocstore(ctx context.Context, cfg Config) (*Docstore, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.initted {
		return nil, errors.New("valkey.Init not called")
	}
	if v.initErr != nil {
		return nil, v.initErr
	}
	if cfg.IndexName == "" {
		return nil, errors.New("valkey: IndexName required")
	}
	if cfg.Embedder == nil {
		return nil, errors.New("valkey: Embedder required")
	}
	if cfg.Dimension <= 0 {
		return nil, errors.New("valkey: Dimension must be positive")
	}

	prefix := cfg.Prefix
	if prefix == "" {
		prefix = cfg.IndexName
	}

	metric := cfg.DistanceMetric
	if metric == "" {
		metric = constants.DistanceMetricCosine
	}

	if err := ensureIndex(ctx, v.client, cfg.IndexName, cfg.Dimension, prefix, metric, cfg.MetadataFields); err != nil {
		return nil, fmt.Errorf("valkey: failed to ensure index: %w", err)
	}

	return &Docstore{
		Client:          v.client,
		IndexName:       cfg.IndexName,
		Prefix:          prefix,
		Dimension:       cfg.Dimension,
		Embedder:        cfg.Embedder,
		EmbedderOptions: cfg.EmbedderOptions,
		MetadataFields:  cfg.MetadataFields,
	}, nil
}

// ensureIndex creates the FT index if it does not already exist.
func ensureIndex(
	ctx context.Context,
	client *glide.Client,
	indexName string,
	dimension int,
	prefix string,
	metric constants.DistanceMetric,
	metadataFields []MetadataFieldConfig,
) error {
	schema := []glideopts.Field{
		glideopts.NewVectorFieldHNSW("embedding", metric, dimension),
		glideopts.NewTextField("_content"),
		glideopts.NewTextField("_metadata"),
		glideopts.NewTextField("_dataType"),
	}

	for _, mf := range metadataFields {
		switch mf.Type {
		case MetadataFieldTypeNumeric:
			schema = append(schema, glideopts.NewNumericField(mf.Name))
		default:
			schema = append(schema, glideopts.NewTagField(mf.Name))
		}
	}

	ftOpts := &glideopts.FtCreateOptions{
		DataType: constants.IndexDataTypeHash,
		Prefixes: []string{prefix + ":"},
	}

	_, err := glideft.FtCreate(ctx, client, indexName, schema, ftOpts)
	if err != nil {
		if strings.Contains(err.Error(), "already exists") {
			return nil
		}
		return err
	}
	return nil
}

// Index stores documents in Valkey as Hashes with vector embeddings.
func (ds *Docstore) Index(ctx context.Context, docs []*ai.Document) error {
	if len(docs) == 0 {
		return nil
	}

	ereq := &ai.EmbedRequest{
		Input:   docs,
		Options: ds.EmbedderOptions,
	}
	eres, err := ds.Embedder.Embed(ctx, ereq)
	if err != nil {
		return fmt.Errorf("valkey index embedding failed: %w", err)
	}

	if len(eres.Embeddings) != len(docs) {
		return fmt.Errorf("valkey: embedder returned %d embeddings for %d docs", len(eres.Embeddings), len(docs))
	}

	// Pre-validate all embeddings before writing anything.
	for i, de := range eres.Embeddings {
		if len(de.Embedding) != ds.Dimension {
			return fmt.Errorf("valkey: embedder returned %d-dim vector for doc %d, expected %d", len(de.Embedding), i, ds.Dimension)
		}
	}

	// Build all HSet commands into a non-atomic batch (pipeline) and execute
	// in a single round-trip.
	// Note: NewStandaloneBatch works only with glide.Client (standalone mode).
	// Cluster deployments require GlideClusterClient + pipeline.NewClusterBatch,
	// which is planned for a future release.
	//
	// Large document sets are chunked into batches of indexBatchSize to avoid
	// unbounded pipeline sizes that could cause OOM or timeouts.
	const indexBatchSize = 1000

	type docEntry struct {
		key    string
		fields map[string]string
	}
	entries := make([]docEntry, 0, len(docs))

	for i, de := range eres.Embeddings {
		doc := docs[i]
		id, err := docID(doc)
		if err != nil {
			return err
		}

		var sb strings.Builder
		for _, p := range doc.Content {
			sb.WriteString(p.Text)
		}

		metadataJSON, err := json.Marshal(doc.Metadata)
		if err != nil {
			return fmt.Errorf("valkey: error marshaling metadata: %w", err)
		}

		embeddingBytes := float32SliceToBytes(de.Embedding)

		key := fmt.Sprintf("%s:%s", ds.Prefix, id)
		fields := map[string]string{
			"embedding": string(embeddingBytes),
			"_content":  sb.String(),
			"_metadata": string(metadataJSON),
			"_dataType": "text", // Go SDK Document lacks DataType field; always "text" for now
		}

		// Store declared metadata keys as top-level HASH fields for filtering.
		if doc.Metadata != nil {
			for _, mf := range ds.MetadataFields {
				if val, ok := doc.Metadata[mf.Name]; ok {
					fields[mf.Name] = metadataValueToString(val, mf.Type)
				}
			}
		}

		entries = append(entries, docEntry{key: key, fields: fields})
	}

	for start := 0; start < len(entries); start += indexBatchSize {
		end := min(start+indexBatchSize, len(entries))
		chunk := entries[start:end]

		batch := pipeline.NewStandaloneBatch(false)
		for _, e := range chunk {
			batch.HSet(e.key, e.fields)
		}

		results, err := ds.Client.Exec(ctx, *batch, true)
		if err != nil {
			return fmt.Errorf("valkey: batch index failed (%d/%d commands executed, chunk offset %d): %w", len(results), len(chunk), start, err)
		}
	}

	return nil
}

// Retrieve performs a KNN vector search against the Valkey FT index.
func (ds *Docstore) Retrieve(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	k := 10
	var filter string
	if req.Options != nil {
		ropt, ok := req.Options.(*RetrieverOptions)
		if !ok {
			return nil, fmt.Errorf("valkey.Retrieve options have type %T, want %T", req.Options, &RetrieverOptions{})
		}
		if ropt.K > 0 {
			k = ropt.K
		}
		if k > 1000 {
			return nil, errors.New("valkey: K must not exceed 1000")
		}
		filter = ropt.Filter
	}

	if filter != "" {
		if err := validateFilter(filter); err != nil {
			return nil, err
		}
	}

	ereq := &ai.EmbedRequest{
		Input:   []*ai.Document{req.Query},
		Options: ds.EmbedderOptions,
	}
	eres, err := ds.Embedder.Embed(ctx, ereq)
	if err != nil {
		return nil, fmt.Errorf("valkey retrieve embedding failed: %w", err)
	}

	if len(eres.Embeddings) == 0 {
		return nil, errors.New("valkey: embedder returned no embeddings")
	}
	queryVec := float32SliceToBytes(eres.Embeddings[0].Embedding)

	searchOpts := &glideopts.FtSearchOptions{
		Params: []glideopts.FtSearchParam{
			{Key: "k", Value: fmt.Sprintf("%d", k)},
			{Key: "query_vec", Value: string(queryVec)},
		},
		ReturnFields: []glideopts.FtSearchReturnField{
			{FieldIdentifier: "_content"},
			{FieldIdentifier: "_metadata"},
			{FieldIdentifier: "_dataType"},
		},
	}

	// Build KNN query with optional pre-filter expression.
	var query string
	if filter != "" {
		query = fmt.Sprintf("(%s)=>[KNN $k @embedding $query_vec]", filter)
	} else {
		query = "*=>[KNN $k @embedding $query_vec]"
	}

	result, err := glideft.FtSearch(ctx, ds.Client, ds.IndexName, query, searchOpts)
	if err != nil {
		return nil, fmt.Errorf("valkey retrieve search failed: %w", err)
	}

	docs := make([]*ai.Document, 0, len(result.Documents))
	for _, doc := range result.Documents {
		content := fieldToString(doc.Fields["_content"])
		if content == "" {
			continue
		}
		metadataStr := fieldToString(doc.Fields["_metadata"])
		// Read _dataType for cross-language consistency. The Go SDK Document
		// struct does not expose a DataType field, so we cannot reconstruct
		// non-text documents yet. The value is preserved in storage for
		// interop with JS/Python retrievers.
		_ = fieldToString(doc.Fields["_dataType"])

		var meta map[string]any
		if metadataStr != "" && metadataStr != "{}" {
			if err := json.Unmarshal([]byte(metadataStr), &meta); err != nil {
				slog.Warn("valkey: failed to parse document metadata", "key", doc.Key, "error", err)
				meta = map[string]any{"_metadata_parse_error": true}
			}
		}

		d := ai.DocumentFromText(content, meta)
		docs = append(docs, d)
	}

	return &ai.RetrieverResponse{
		Documents: docs,
	}, nil
}

// float32SliceToBytes converts a float32 slice to a little-endian byte slice,
// matching the JS Buffer.from(new Float32Array(...).buffer) encoding.
func float32SliceToBytes(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// docID returns the ID to use for a Document.
// Uses a canonical serialization format (sorted JSON of {content, metadata, dataType})
// that matches the JS and Python implementations for cross-language interop.
func docID(doc *ai.Document) (string, error) {
	var sb strings.Builder
	for _, p := range doc.Content {
		sb.WriteString(p.Text)
	}

	canonical := map[string]any{
		"data":     sb.String(),
		"metadata": doc.Metadata,
		"dataType": "text",
	}
	b, err := json.Marshal(canonical)
	if err != nil {
		return "", fmt.Errorf("valkey: error marshaling document: %w", err)
	}
	return fmt.Sprintf("%02x", md5.Sum(b)), nil
}

// fieldToString extracts a string from a field value that may be returned as
// string or []byte depending on the Valkey client version.
func fieldToString(v any) string {
	switch val := v.(type) {
	case string:
		return val
	case []byte:
		return string(val)
	default:
		return ""
	}
}

// filterDisallowedChars contains characters that could alter FT.SEARCH query
// semantics if injected into a filter expression.
const filterDisallowedChars = ";|`$\\"

// maxFilterLength is the maximum allowed length for a filter expression to
// prevent DoS via query amplification.
const maxFilterLength = 2048

// validateFilter checks that a filter expression does not contain characters
// or sequences that could break out of the filter context.
func validateFilter(filter string) error {
	if len(filter) > maxFilterLength {
		return errors.New("valkey: filter expression too long")
	}
	if strings.ContainsAny(filter, filterDisallowedChars) {
		return errors.New("valkey: filter expression contains disallowed characters; do not pass untrusted user input as a filter")
	}
	if strings.Contains(filter, "=>") {
		return errors.New("valkey: filter expression contains disallowed sequence '=>'; do not pass untrusted user input as a filter")
	}
	return nil
}

// metadataValueToString converts a metadata value to a string suitable for
// storage in a Valkey HASH field. Numeric fields are formatted to preserve
// parseability as 64-bit floats.
func metadataValueToString(val any, fieldType MetadataFieldType) string {
	if fieldType == MetadataFieldTypeNumeric {
		switch v := val.(type) {
		case float64:
			return fmt.Sprintf("%g", v)
		case float32:
			return fmt.Sprintf("%g", v)
		case int:
			return fmt.Sprintf("%d", v)
		case int64:
			return fmt.Sprintf("%d", v)
		case json.Number:
			return v.String()
		}
	}
	return fmt.Sprintf("%v", val)
}
