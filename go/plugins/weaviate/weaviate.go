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

package weaviate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/auth"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
)

// The provider used in the registry.
const provider = "weaviate"

// The metadata key used to hold document text.
const textKey = "text"

// The metadata key to hold document metadata.
const metadataKey = "metadata"

// Weaviate passes configuration options to the plugin.
type Weaviate struct {
	// The hostname:port to use to contact the Weaviate database.
	// If not set, the default is read from the WEAVIATE_URL environment variable.
	Addr string
	// The scheme to use to contact the Weaviate database.
	// Typically http or https.
	// If not set, the default is https.
	Scheme string
	// The API key to use with the Weaviate database.
	// If not set, the default is read from the WEAVIATE_API_KEY environment variable.
	APIKey string

	client  *weaviate.Client // Client for the Weaviate database.
	mu      sync.Mutex       // Mutex to control access.
	initted bool             // Whether the plugin has been initialized.
}

// Name returns the name of the plugin.
func (w *Weaviate) Name() string {
	return provider
}

// Init initializes the Weaviate plugin.
func (w *Weaviate) Init(ctx context.Context) []api.Action {
	if w == nil {
		w = &Weaviate{}
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.initted {
		panic("plugin already initialized")
	}

	var host string
	if w.Addr != "" {
		host = w.Addr
	}
	if host == "" {
		host = os.Getenv("WEAVIATE_URL")
	}

	var scheme string
	if w.Scheme != "" {
		scheme = w.Scheme
	}
	if scheme == "" {
		scheme = "https"
	}

	var apiKey string
	if w.APIKey != "" {
		apiKey = w.APIKey
	}
	if apiKey == "" {
		apiKey = os.Getenv("WEAVIATE_API_KEY")
	}

	config := weaviate.Config{
		Host:       host,
		Scheme:     scheme,
		AuthConfig: auth.ApiKey{Value: apiKey},
	}

	client, err := weaviate.NewClient(config)
	if err != nil {
		panic(fmt.Errorf("weaviate.Init: initialization failed: %v", err))
	}

	live, err := client.Misc().LiveChecker().Do(ctx)
	if err != nil {
		panic(fmt.Errorf("weaviate.Init: initialization failed: %v", err))
	}
	if !live {
		panic("weaviate instance not alive")
	}

	w.client = client
	w.initted = true

	return []api.Action{}
}

// ClassConfig holds configuration options for a retriever.
// Weaviate stores data in collections, and each collection has a class name.
// Use a separate genkit Retriever for each different class.
type ClassConfig struct {
	// The weaviate class name. May not be the empty string.
	Class string

	// The Embedder and options to use to embed documents.
	// Embedder may not be nil.
	Embedder        ai.Embedder
	EmbedderOptions any
}

// DefineRetriever defines [ai.Retriever]
// that use the same class.
// The name uniquely identifies the Retriever in the registry.
func DefineRetriever(ctx context.Context, g *genkit.Genkit, cfg ClassConfig, opts *ai.RetrieverOptions) (*Docstore, ai.Retriever, error) {
	if cfg.Embedder == nil {
		return nil, nil, errors.New("weaviate: Embedder required")
	}
	if cfg.Class == "" {
		return nil, nil, errors.New("weaviate: class required")
	}

	w := genkit.LookupPlugin(g, provider).(*Weaviate)
	if w == nil {
		return nil, nil, errors.New("weaviate plugin not found; did you call genkit.Init with the weaviate plugin?")
	}

	ds, err := w.newDocstore(ctx, &cfg)
	if err != nil {
		return nil, nil, err
	}
	retriever := genkit.DefineRetriever(g, api.NewName(provider, cfg.Class), opts, ds.Retrieve)
	return ds, retriever, nil
}

// Docstore defines a Retriever.
type Docstore struct {
	Client          *weaviate.Client
	Class           string
	Embedder        ai.Embedder
	EmbedderOptions any
}

// newDocstore creates a Docstore.
func (w *Weaviate) newDocstore(ctx context.Context, cfg *ClassConfig) (*Docstore, error) {
	if w.client == nil {
		return nil, errors.New("weaviate.Init not called")
	}

	// Create the class if it doesn't already exist.
	exists, err := w.client.Schema().ClassExistenceChecker().WithClassName(cfg.Class).Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("weaviate class check failed for %q: %v", cfg.Class, err)
	}
	if !exists {
		cls := &models.Class{
			Class:      cfg.Class,
			Vectorizer: "none",
		}

		err := w.client.Schema().ClassCreator().WithClass(cls).Do(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to create weaviate class %q: %v", cfg.Class, err)
		}
	}

	ds := &Docstore{
		Client:          w.client,
		Class:           cfg.Class,
		Embedder:        cfg.Embedder,
		EmbedderOptions: cfg.EmbedderOptions,
	}
	return ds, nil
}

// Retriever returns the retriever for the given class.
func Retriever(g *genkit.Genkit, class string) ai.Retriever {
	return genkit.LookupRetriever(g, api.NewName(provider, class))
}

// RetrieverOptions may be passed in the Options field
// [ai.RetrieverRequest] to pass options to Weaviate.
// The options field should be either nil or
// a value of type *RetrieverOptions.
type RetrieverOptions struct {
	// Maximum number of values to retrieve.
	Count int `json:"count,omitempty"`
	// Keys to retrieve from document metadata.
	// There doesn't seem to be a way to ask for all of the metadata.
	MetadataKeys []string
}

// Retrieve implements the genkit Retriever.Retrieve method.
func (ds *Docstore) Retrieve(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	count := 3 // by default we fetch 3 documents
	var metadataKeys []string
	if req.Options != nil {
		ropt, ok := req.Options.(*RetrieverOptions)
		if !ok {
			return nil, fmt.Errorf("weaviate.Retrieve options have type %T, want %T", req.Options, &RetrieverOptions{})
		}
		count = ropt.Count
		metadataKeys = ropt.MetadataKeys
	}

	// Use the embedder to convert the document to a vector.
	ereq := &ai.EmbedRequest{
		Input:   []*ai.Document{req.Query},
		Options: ds.EmbedderOptions,
	}
	eres, err := ds.Embedder.Embed(ctx, ereq)
	if err != nil {
		return nil, fmt.Errorf("weaviate retrieve embedding failed: %v", err)
	}

	gql := ds.Client.GraphQL()
	fields := []graphql.Field{
		{Name: textKey},
	}
	if len(metadataKeys) > 0 {
		mfields := make([]graphql.Field, 0, len(metadataKeys))
		for _, k := range metadataKeys {
			mfields = append(mfields,
				graphql.Field{
					Name: k,
				},
			)
		}
		fields = append(fields,
			graphql.Field{
				Name:   metadataKey,
				Fields: mfields,
			},
		)
	}
	res, err := gql.Get().
		WithNearVector(
			gql.NearVectorArgBuilder().WithVector(eres.Embeddings[0].Embedding)).
		WithClassName(ds.Class).
		WithFields(fields...).
		WithLimit(count).
		Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("weaviate retrieve failed: %v", err)
	}
	if len(res.Errors) != 0 {
		ss := make([]string, 0, len(res.Errors))
		for _, e := range res.Errors {
			ss = append(ss, e.Message)
		}
		return nil, fmt.Errorf("weaviate retrieve failed: %v", ss)
	}

	data, ok := res.Data["Get"]
	if !ok {
		return nil, errors.New("weaviate retrieve did not return Get key")
	}

	doc, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("weaviate retrieve returned type %T, expected %T", data, map[string]any{})
	}

	docValAny, ok := doc[ds.Class]
	if !ok {
		return nil, fmt.Errorf("weaviate retrieve did not return %q key: %v", ds.Class, doc)
	}
	docVal, ok := docValAny.([]any)
	if !ok {
		return nil, fmt.Errorf("weaviate retrieve returned document type %T, expected %T", docValAny, []any{})
	}

	var docs []*ai.Document
	for _, dv := range docVal {
		dvMap, ok := dv.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("weaviate retrieve doc has type %T, expected %T", dv, map[string]any{})
		}
		t, ok := dvMap[textKey]
		if !ok {
			return nil, fmt.Errorf("weaviate doc missing key %q", textKey)
		}
		s, ok := t.(string)
		if !ok {
			return nil, fmt.Errorf("weaviate text is type %T, want %T", t, "")
		}
		props, _ := dvMap[metadataKey].(map[string]any)

		d := ai.DocumentFromText(s, props)
		docs = append(docs, d)
	}

	ret := &ai.RetrieverResponse{
		Documents: docs,
	}
	return ret, nil
}

// Helper function to get started with indexing
func Index(ctx context.Context, docs []*ai.Document, ds *Docstore) error {
	if len(docs) == 0 {
		return nil
	}

	// Use the embedder to convert each Document into a vector.
	ereq := &ai.EmbedRequest{
		Input:   docs,
		Options: ds.EmbedderOptions,
	}
	eres, err := ds.Embedder.Embed(ctx, ereq)
	if err != nil {
		return fmt.Errorf("weaviate index embedding failed: %v", err)
	}

	objects := make([]*models.Object, 0, len(eres.Embeddings))
	for i, de := range eres.Embeddings {
		doc := docs[i]

		var sb strings.Builder
		for _, p := range doc.Content {
			sb.WriteString(p.Text)
		}

		metadata := make(map[string]any)
		metadata[textKey] = sb.String()

		if doc.Metadata != nil {
			metadata[metadataKey] = doc.Metadata
		}

		obj := &models.Object{
			Class:      ds.Class,
			Properties: metadata,
			Vector:     de.Embedding,
		}
		objects = append(objects, obj)
	}

	_, err = ds.Client.Batch().ObjectsBatcher().WithObjects(objects...).Do(ctx)
	if err != nil {
		return fmt.Errorf("weaviate insert failed: %v", err)
	}

	return nil
}
