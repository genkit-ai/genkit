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
//
// SPDX-License-Identifier: Apache-2.0

// Package localvec is a local vector database for development and testing.
// The database is stored in a file in the local file system.
// Production code should use a real vector database.
package localvec

import (
	"cmp"
	"context"
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "devLocalVectorStore"

type Config struct {
	// Where to store the data. Defaults to os.TempDir.
	Dir             string
	Embedder        ai.Embedder
	EmbedderOptions any
}

// Init initializes the plugin.
func Init() error { return nil }

// DefineRetriever defines a Retriever and docStore which is also used by the retriever.
// The name uniquely identifies the Retriever in the registry.
func DefineRetriever(g *genkit.Genkit, name string, cfg Config, opts *ai.RetrieverOptions) (*DocStore, ai.Retriever, error) {
	ds, err := newDocStore(cfg.Dir, name, cfg.Embedder, cfg.EmbedderOptions)
	if err != nil {
		return nil, nil, err
	}

	if opts != nil && opts.ConfigSchema == nil {
		opts.ConfigSchema = core.InferSchemaMap(RetrieverOptions{})
	}

	return ds, genkit.DefineRetriever(g, api.NewName(provider, name), opts, ds.retrieve), nil
}

// IsDefinedRetriever reports whether the named [Retriever] is defined by this plugin.
func IsDefinedRetriever(g *genkit.Genkit, name string) bool {
	return genkit.LookupRetriever(g, api.NewName(provider, name)) != nil
}

// Retriever returns the retriever with the given name.
// The name must match the [Config.Name] value passed to [Init].
func Retriever(g *genkit.Genkit, name string) ai.Retriever {
	return genkit.LookupRetriever(g, api.NewName(provider, name))
}

// DocStore implements a local vector database.
// This is based on js/plugins/dev-local-vectorstore/src/index.ts.
type DocStore struct {
	Filename        string
	Embedder        ai.Embedder
	EmbedderOptions any
	Data            map[string]DbValue
}

// DbValue is the type of a document stored in the database.
type DbValue struct {
	Doc       *ai.Document `json:"doc"`
	Embedding []float32    `json:"embedding"`
}

// newDocStore returns a new ai.DocumentStore to register.
func newDocStore(dir, name string, embedder ai.Embedder, embedderOptions any) (*DocStore, error) {
	if dir == "" {
		dir = os.TempDir()
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	dbname := "__db_" + name + ".json"
	filename := filepath.Join(dir, dbname)
	f, err := os.Open(filename)
	var data map[string]DbValue
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			return nil, err
		}
	} else {
		defer f.Close()
		decoder := json.NewDecoder(f)
		if err := decoder.Decode(&data); err != nil {
			return nil, err
		}
	}

	ds := &DocStore{
		Filename:        filename,
		Embedder:        embedder,
		EmbedderOptions: embedderOptions,
		Data:            data,
	}
	return ds, nil
}

// RetrieverOptions may be passed in the Options field
// of [ai.RetrieverRequest] to pass options to the retriever.
// The Options field should be either nil or a value of type *RetrieverOptions.
type RetrieverOptions struct {
	K int `json:"k,omitempty"` // number of entries to return
}

// retrieve retrieves documents close to the argument.
func (ds *DocStore) retrieve(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	// Use the embedder to convert the document we want to
	// retrieve into a vector.
	ereq := &ai.EmbedRequest{
		Input:   []*ai.Document{req.Query},
		Options: ds.EmbedderOptions,
	}
	eres, err := ds.Embedder.Embed(ctx, ereq)
	if err != nil {
		return nil, fmt.Errorf("localvec retrieve embedding failed: %v", err)
	}
	vals := eres.Embeddings[0].Embedding

	type scoredDoc struct {
		score float64
		doc   *ai.Document
	}
	scoredDocs := make([]scoredDoc, 0, len(ds.Data))
	for _, dbv := range ds.Data {
		score := similarity(vals, dbv.Embedding)
		scoredDocs = append(scoredDocs, scoredDoc{
			score: score,
			doc:   dbv.Doc,
		})
	}

	slices.SortFunc(scoredDocs, func(a, b scoredDoc) int {
		// We want to sort by descending score,
		// so pass b.score first to reverse the default ordering.
		return cmp.Compare(b.score, a.score)
	})

	k := 3
	if options, _ := req.Options.(*RetrieverOptions); options != nil {
		k = options.K
	}
	k = min(k, len(scoredDocs))

	docs := make([]*ai.Document, 0, k)
	for i := range k {
		docs = append(docs, scoredDocs[i].doc)
	}

	resp := &ai.RetrieverResponse{
		Documents: docs,
	}
	return resp, nil
}

// similarity computes the [cosine similarity] between two vectors.
//
// [cosine similarity]: https://en.wikipedia.org/wiki/Cosine_similarity
func similarity(vals1, vals2 []float32) float64 {
	l2norm := func(v float64, s, t float64) (float64, float64) {
		if v == 0 {
			return s, t
		}
		a := math.Abs(v)
		if a > t {
			r := t / v
			s = 1 + s*r*r
			t = a
		} else {
			r := v / t
			s = s + r*r
		}
		return s, t
	}

	dot := float64(0)
	s1 := float64(1)
	t1 := float64(0)
	s2 := float64(1)
	t2 := float64(0)

	for i, v1f := range vals1 {
		v1 := float64(v1f)
		v2 := float64(vals2[i])
		dot += v1 * v2
		s1, t1 = l2norm(v1, s1, t1)
		s2, t2 = l2norm(v2, s2, t2)
	}

	l1 := t1 * math.Sqrt(s1)
	l2 := t2 * math.Sqrt(s2)

	return dot / (l1 * l2)
}

// Helper function to get started with indexing
func Index(ctx context.Context, docs []*ai.Document, ds *DocStore) error {
	ereq := &ai.EmbedRequest{
		Input:   docs,
		Options: ds.EmbedderOptions,
	}
	eres, err := ds.Embedder.Embed(ctx, ereq)
	if err != nil {
		return fmt.Errorf("localvec index embedding failed: %v", err)
	}
	for i, de := range eres.Embeddings {
		id, err := docID(docs[i])
		if err != nil {
			return err
		}
		if _, ok := ds.Data[id]; ok {
			logger.FromContext(ctx).Debug("localvec skipping document because already present", "id", id)
			continue
		}

		if ds.Data == nil {
			ds.Data = make(map[string]DbValue)
		}

		ds.Data[id] = DbValue{
			Doc:       docs[i],
			Embedding: de.Embedding,
		}
	}

	// Update the file every time we add documents.
	// We use a temporary file to avoid losing the original
	// file, in case of a crash.
	tmpname := ds.Filename + ".tmp"
	f, err := os.Create(tmpname)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(f)
	if err := encoder.Encode(ds.Data); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpname, ds.Filename); err != nil {
		return err
	}

	return nil
}

// docID returns the ID to use for a Document.
// This is intended to be the same as the genkit Typescript computation.
func docID(doc *ai.Document) (string, error) {
	b, err := json.Marshal(doc)
	if err != nil {
		return "", fmt.Errorf("localvec: error marshaling document: %v", err)
	}
	return fmt.Sprintf("%02x", md5.Sum(b)), nil
}
