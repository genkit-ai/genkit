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

package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	openaiGo "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func initPlugin(t *testing.T) (*OpenAI, *genkit.Genkit) {
	t.Helper()
	t.Setenv("OPENAI_API_KEY", "test-key")
	o := &OpenAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(o))
	return o, g
}

// TestInitAdvertisesSDKConfigSchema pins that the models the plugin registers
// advertise the OpenAI SDK's own config schema, which the framework validates
// every request against.
func TestInitAdvertisesSDKConfigSchema(t *testing.T) {
	_, g := initPlugin(t)

	m := genkit.LookupModel(g, "openai/gpt-4o")
	if m == nil {
		t.Fatal("gpt-4o not registered by Init")
	}
	model, ok := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing")
	}
	if got := model["label"]; got != "OpenAI GPT-4o" {
		t.Errorf("label = %v, want %q", got, "OpenAI GPT-4o")
	}
	schema, ok := model["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions missing, got %v", model["customOptions"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok || props["max_tokens"] == nil || props["temperature"] == nil {
		t.Errorf("config schema is not the OpenAI chat completion params schema, got %v", schema)
	}
}

// TestDefineModelNilOptions covers the nil ModelOptions path: a model the
// plugin does not curate gets the generic multimodal defaults with a
// name-derived label, and registration makes the lookup helpers find it. The
// curated models are all registered by Init, which is why the test defines a
// new name; redefining a registered one panics (see [OpenAI.DefineModel]).
func TestDefineModelNilOptions(t *testing.T) {
	o, g := initPlugin(t)

	m, err := o.RegisterModel(g, "brand-new-model", nil)
	if err != nil {
		t.Fatalf("RegisterModel() error = %v", err)
	}
	if !IsDefinedModel(g, "brand-new-model") {
		t.Error("IsDefinedModel() = false after RegisterModel(), want the model registered")
	}

	model, ok := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing")
	}
	if want := "openai - brand-new-model"; model["label"] != want {
		t.Errorf("label = %v, want %q", model["label"], want)
	}
	supports, ok := model["supports"].(map[string]any)
	if !ok {
		t.Fatalf("supports metadata missing")
	}
	if supports["tools"] != true || supports["media"] != true {
		t.Errorf("supports = %v, want the generic multimodal defaults", supports)
	}
}

// TestPrefixedNamesAreEquivalent pins that the exported entry points take a
// model ID either bare or provider-prefixed.
func TestPrefixedNamesAreEquivalent(t *testing.T) {
	o, g := initPlugin(t)

	if _, err := o.RegisterModel(g, "openai/custom-model", nil); err != nil {
		t.Fatalf("RegisterModel() error = %v", err)
	}
	for _, name := range []string{"custom-model", "openai/custom-model"} {
		if !IsDefinedModel(g, name) {
			t.Errorf("IsDefinedModel(%q) = false, want the model defined under either form", name)
		}
		if o.Model(g, name) == nil {
			t.Errorf("Model(%q) = nil, want the model defined under either form", name)
		}
	}
}

// TestModelRef pins the name a ref carries and that the typed SDK config
// rides along, since the ref is how an application supplies config at the
// call site.
func TestModelRef(t *testing.T) {
	cfg := &openaiGo.ChatCompletionNewParams{Temperature: openaiGo.Float(0.7)}

	for _, name := range []string{"gpt-4o", "openai/gpt-4o"} {
		ref := ModelRef(name, cfg)
		if want := "openai/gpt-4o"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	if got := ModelRef("gpt-4o", nil).Config(); got != (*openaiGo.ChatCompletionNewParams)(nil) {
		t.Errorf("Config() = %v for a nil config, want a typed nil", got)
	}
}

// TestNewEmbedderRef pins the embedder ref contract and that the registered
// embedders advertise the typed embedding config's camelCase schema.
func TestNewEmbedderRef(t *testing.T) {
	_, g := initPlugin(t)

	cfg := &TextEmbeddingConfig{Dimensions: 256}
	for _, name := range []string{"text-embedding-3-small", "openai/text-embedding-3-small"} {
		ref := NewEmbedderRef(name, cfg)
		if want := "openai/text-embedding-3-small"; ref.Name() != want {
			t.Errorf("NewEmbedderRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("NewEmbedderRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	if !IsDefinedEmbedder(g, "text-embedding-3-small") {
		t.Fatal("IsDefinedEmbedder() = false, want the embedder registered by Init")
	}
	e := genkit.LookupEmbedder(g, "openai/text-embedding-3-small")
	if e == nil {
		t.Fatal("embedder not registered by Init")
	}
	embedder, ok := e.(*ai.EmbedderAction).Desc().Metadata["embedder"].(map[string]any)
	if !ok {
		t.Fatalf("embedder metadata missing")
	}
	schema, ok := embedder["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions missing, got %v", embedder["customOptions"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok || props["dimensions"] == nil || props["encodingFormat"] == nil {
		t.Errorf("embedder config schema is not the embedding config schema, got %v", schema)
	}
}

// TestDefineEmbedderNilOptions pins the nil EmbedderOptions path: a known
// embedder gets its curated dimensions and label.
func TestDefineEmbedderNilOptions(t *testing.T) {
	o, g := initPlugin(t)

	e, err := o.RegisterEmbedder(g, "openai/custom-embedding", nil)
	if err != nil {
		t.Fatalf("RegisterEmbedder() error = %v", err)
	}
	if e == nil {
		t.Fatal("RegisterEmbedder() = nil")
	}
	if !IsDefinedEmbedder(g, "custom-embedding") {
		t.Error("IsDefinedEmbedder() = false after RegisterEmbedder(), want the embedder registered")
	}
}

// TestEmbedderPerRequestAPIKey pins the embedder credential override: an
// ref whose config carries an APIKey authenticates that request with
// the override while the config's other fields reach the request body, and
// the key stays out of the body.
func TestEmbedderPerRequestAPIKey(t *testing.T) {
	var auth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		if strings.Contains(string(body), "override-key") || strings.Contains(string(body), "apiKey") {
			t.Errorf("request body leaks the API key: %s", body)
		}
		if !strings.Contains(string(body), `"dimensions":256`) {
			t.Errorf("request body is missing the config's dimensions: %s", body)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list","model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`)
	}))
	defer server.Close()

	t.Setenv("OPENAI_API_KEY", "plugin-key")
	o := &OpenAI{Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(context.Background(), genkit.WithPlugins(o))

	resp, err := genkit.Embed(context.Background(), g,
		ai.WithEmbedder(NewEmbedderRef("text-embedding-3-small", &TextEmbeddingConfig{
			APIKey:     "override-key",
			Dimensions: 256,
		})),
		ai.WithTextDocs("hello"),
	)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("embeddings = %d, want 1", len(resp.Embeddings))
	}
	if auth != "Bearer override-key" {
		t.Fatalf("Authorization = %q, want the request-scoped key", auth)
	}
}

// TestIsDefinedModelDoesNotResolve pins the guard semantics: checking whether
// a model is defined must not itself resolve and register one, which would
// make the guard answer true for any name and the subsequent DefineModel
// panic.
func TestIsDefinedModelDoesNotResolve(t *testing.T) {
	o, g := initPlugin(t)

	if IsDefinedModel(g, "never-defined-model") {
		t.Fatal("IsDefinedModel() = true for a model that was never defined")
	}
	if _, err := o.RegisterModel(g, "never-defined-model", nil); err != nil {
		t.Fatalf("RegisterModel() after the guard error = %v", err)
	}
	if !IsDefinedModel(g, "never-defined-model") {
		t.Error("IsDefinedModel() = false after RegisterModel()")
	}
}

// TestEmbedderBase64EncodingFormat pins that the base64 encoding the config
// advertises actually works: the API returns the vector as a base64 string of
// little-endian float32s, which the plugin decodes.
func TestEmbedderBase64EncodingFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"encoding_format":"base64"`) {
			t.Errorf("request body is missing the base64 encoding format: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		// base64 of little-endian float32s 1.0, 2.0.
		_, _ = io.WriteString(w, `{
			"object":"list","model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":0,"embedding":"AACAPwAAAEA="}],
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`)
	}))
	defer server.Close()

	t.Setenv("OPENAI_API_KEY", "test-key")
	o := &OpenAI{Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(context.Background(), genkit.WithPlugins(o))

	resp, err := genkit.Embed(context.Background(), g,
		ai.WithEmbedder(NewEmbedderRef("text-embedding-3-small", &TextEmbeddingConfig{
			EncodingFormat: openaiGo.EmbeddingNewParamsEncodingFormatBase64,
		})),
		ai.WithTextDocs("hello"),
	)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("embeddings = %d, want 1", len(resp.Embeddings))
	}
	got := resp.Embeddings[0].Embedding
	if len(got) != 2 || got[0] != 1.0 || got[1] != 2.0 {
		t.Fatalf("embedding = %v, want [1 2] decoded from base64", got)
	}
}

// TestDeprecatedBuildersDoNotRegister pins the released entry points this
// plugin keeps: they build a model or embedder and hand it back without
// touching the registry, which is what separates them from RegisterModel and
// RegisterEmbedder.
func TestDeprecatedBuildersDoNotRegister(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")

	o := &OpenAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(o))

	const model = "gpt-legacy-define"
	if m := o.DefineModel(model, ai.ModelOptions{Label: "Legacy"}); m == nil {
		t.Fatal("DefineModel() = nil, want the built model")
	}
	if IsDefinedModel(g, model) {
		t.Errorf("IsDefinedModel(%q) = true after DefineModel(), want the deprecated builder to leave the registry alone", model)
	}

	const embedder = "text-embedding-legacy-define"
	if e := o.DefineEmbedder(embedder, &ai.EmbedderOptions{Label: "Legacy"}); e == nil {
		t.Fatal("DefineEmbedder() = nil, want the built embedder")
	}
	if IsDefinedEmbedder(g, embedder) {
		t.Errorf("IsDefinedEmbedder(%q) = true after DefineEmbedder(), want the deprecated builder to leave the registry alone", embedder)
	}
}

// TestSupportedModelCatalog guards the hand-maintained model tables against the
// mistakes hand-maintaining them invites: a snapshot pasted under the wrong
// model, an alias that does not match its key, a missing capability set. It
// cannot tell whether the catalog still matches OpenAI's, only that what is
// written here is internally consistent.
func TestSupportedModelCatalog(t *testing.T) {
	seen := map[string]string{}
	for name, opts := range supportedModels {
		if opts.Label == "" {
			t.Errorf("%s has no label, so the Dev UI would list it blank", name)
		}
		if opts.Supports == nil {
			t.Errorf("%s declares no capabilities, so generation cannot check them", name)
		}
		if len(opts.Versions) == 0 {
			t.Errorf("%s lists no versions", name)
			continue
		}
		if got := opts.Versions[0]; got != name {
			t.Errorf("%s lists %q first, want the bare model ID before its snapshots", name, got)
		}
		for _, v := range opts.Versions {
			if !strings.HasPrefix(v, name) {
				t.Errorf("%s lists version %q, which is not a snapshot of it", name, v)
			}
			if other, dup := seen[v]; dup {
				t.Errorf("version %q is listed under both %s and %s", v, other, name)
			}
			seen[v] = name
		}
	}

	for name, opts := range supportedEmbeddingModels {
		if opts.Label == "" {
			t.Errorf("%s has no label", name)
		}
		if opts.Dimensions == 0 {
			t.Errorf("%s declares no dimensions", name)
		}
	}
}

// TestConstrainedSupport pins which models advertise native structured output.
// OpenAI gates response_format json_schema on the gpt-4o-mini and
// gpt-4o-2024-08-06 snapshots and later, so the three models predating it must
// stay unset: claiming support there would drop the schema instructions Genkit
// injects into the prompt and leave nothing enforcing the schema.
func TestConstrainedSupport(t *testing.T) {
	// Models OpenAI released before Structured Outputs.
	legacy := map[string]bool{"gpt-4-turbo": true, "gpt-4": true, "gpt-3.5-turbo": true}

	for id, opts := range supportedModels {
		got := opts.Supports.Constrained
		want := ai.ConstrainedSupportAll
		if legacy[id] {
			want = ""
		}
		if got != want {
			t.Errorf("%s constrained = %q, want %q", id, got, want)
		}
	}

	for id := range legacy {
		if _, ok := supportedModels[id]; !ok {
			t.Errorf("%s is no longer in the catalog; drop it from this test", id)
		}
	}

	if got := dynamicModelOptions.Supports.Constrained; got != ai.ConstrainedSupportAll {
		t.Errorf("dynamic constrained = %q, want %q", got, ai.ConstrainedSupportAll)
	}
}
