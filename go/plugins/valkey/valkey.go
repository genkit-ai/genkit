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
	"math"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	glide "github.com/valkey-io/valkey-glide/go/v2"
	"github.com/valkey-io/valkey-glide/go/v2/config"
	"github.com/valkey-io/valkey-glide/go/v2/constants"
	glideopts "github.com/valkey-io/valkey-glide/go/v2/options"
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
func (v *Valkey) Init(ctx context.Context) []api.Action {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.initted {
		panic("valkey.Init already called")
	}

	clientConfig := config.NewClientConfiguration()
	for i := range v.Addresses {
		clientConfig.WithAddress(&v.Addresses[i])
	}

	client, err := glide.NewClient(clientConfig)
	if err != nil {
		panic(fmt.Errorf("valkey.Init: failed to create client: %w", err))
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

// DefineRetriever defines a Retriever backed by Valkey vector search.
// It ensures the FT index exists and registers the retriever action.
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
		panic("valkey.Init not called")
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
		return nil, fmt.Errorf("valkey: failed to ensure index: %v", err)
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
func Index(ctx context.Context, docs []*ai.Document, ds *Docstore) error {
	if len(docs) == 0 {
		return nil
	}

	ereq := &ai.EmbedRequest{
		Input:   docs,
		Options: ds.EmbedderOptions,
	}
	eres, err := ds.Embedder.Embed(ctx, ereq)
	if err != nil {
		return fmt.Errorf("valkey index embedding failed: %v", err)
	}

	if len(eres.Embeddings) != len(docs) {
		return fmt.Errorf("valkey: embedder returned %d embeddings for %d docs", len(eres.Embeddings), len(docs))
	}
	for i, de := range eres.Embeddings {
		if len(de.Embedding) != ds.Dimension {
			return fmt.Errorf("valkey: embedder returned %d-dim vector for doc %d, expected %d", len(de.Embedding), i, ds.Dimension)
		}
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
			return fmt.Errorf("valkey: error marshaling metadata: %v", err)
		}

		// Go strings can hold arbitrary bytes; this is the correct way to pass
		// binary vector data to HSet which expects string-typed field values.
		embeddingBytes := float32SliceToBytes(de.Embedding)

		key := fmt.Sprintf("%s:%s", ds.Prefix, id)
		fields := map[string]string{
			"embedding": string(embeddingBytes), // Go strings can hold arbitrary bytes for HSet
			"_content":  sb.String(),
			"_metadata": string(metadataJSON),
			"_dataType": "text",
		}

		// Store declared metadata keys as top-level HASH fields for filtering.
		if doc.Metadata != nil {
			for _, mf := range ds.MetadataFields {
				if val, ok := doc.Metadata[mf.Name]; ok {
					fields[mf.Name] = fmt.Sprintf("%v", val)
				}
			}
		}

		_, err = ds.Client.HSet(ctx, key, fields)
		if err != nil {
			return fmt.Errorf("valkey: error storing document %s: %v", key, err)
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
		filter = ropt.Filter
	}

	ereq := &ai.EmbedRequest{
		Input:   []*ai.Document{req.Query},
		Options: ds.EmbedderOptions,
	}
	eres, err := ds.Embedder.Embed(ctx, ereq)
	if err != nil {
		return nil, fmt.Errorf("valkey retrieve embedding failed: %v", err)
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
		return nil, fmt.Errorf("valkey retrieve search failed: %v", err)
	}

	docs := make([]*ai.Document, 0, len(result.Documents))
	for _, doc := range result.Documents {
		content, _ := doc.Fields["_content"].(string)
		if content == "" {
			continue
		}
		metadataStr, _ := doc.Fields["_metadata"].(string)

		var meta map[string]any
		if metadataStr != "" && metadataStr != "{}" {
			if err := json.Unmarshal([]byte(metadataStr), &meta); err != nil {
				meta = nil
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
// Go's encoding/json sorts map keys, so this produces deterministic output.
func docID(doc *ai.Document) (string, error) {
	b, err := json.Marshal(doc)
	if err != nil {
		return "", fmt.Errorf("valkey: error marshaling document: %v", err)
	}
	return fmt.Sprintf("%02x", md5.Sum(b)), nil
}
