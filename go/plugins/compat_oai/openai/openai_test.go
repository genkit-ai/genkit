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
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	openaiGo "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
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

// TestUncuratedModelDefaults covers the fallback path: a model the plugin
// does not curate is described with the generic multimodal defaults and a
// label derived from its name. Nothing has to register it first, which is what
// retired the old ordering problem: Init registers every curated model, so a
// registration call could never describe one of those.
func TestUncuratedModelDefaults(t *testing.T) {
	o, _ := initPlugin(t)

	m := o.ResolveAction(api.ActionTypeModel, "brand-new-model")
	if m == nil {
		t.Fatal("ResolveAction(brand-new-model) = nil")
	}
	model, ok := m.Desc().Metadata["model"].(map[string]any)
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

// TestPrefixedNamesAreEquivalent pins that a [OpenAI.Models] key is taken
// either bare or provider-prefixed. The ID reaching ResolveAction is always
// bare, since the registry splits the provider off the action key before
// calling the plugin, so the key is the form an application controls.
func TestPrefixedNamesAreEquivalent(t *testing.T) {
	for _, key := range []string{"custom-model", "openai/custom-model"} {
		t.Setenv("OPENAI_API_KEY", "test-key")
		o := &OpenAI{Models: map[string]ai.ModelOptions{key: {Label: "Custom"}}}
		g := genkit.Init(context.Background(), genkit.WithPlugins(o))

		m := genkit.LookupModel(g, "openai/custom-model")
		if m == nil {
			t.Fatalf("Models key %q: custom-model did not resolve", key)
		}
		if got := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)["label"]; got != "Custom" {
			t.Errorf("Models key %q: label = %v, want the override's", key, got)
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

// TestDefineEmbedderNilOptions pins the nil EmbedderOptions path on the
// released builder: it returns an embedder rather than failing.
func TestDefineEmbedderNilOptions(t *testing.T) {
	o, _ := initPlugin(t)

	if e := o.DefineEmbedder("openai/custom-embedding", nil); e == nil {
		t.Fatal("DefineEmbedder(nil opts) = nil, want the built embedder")
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

// TestInitSendsOpenAIEnvIdentity pins that this plugin, alone among the
// compat_oai plugins, still speaks OpenAI's environment. The base plugin drops
// the identity the SDK reads from OPENAI_API_KEY, OPENAI_ORG_ID, and
// OPENAI_PROJECT_ID so no plugin serving another provider forwards it; this
// plugin serves OpenAI, so it applies the same three itself and they must
// reach the wire unchanged. Opts still override them.
func TestInitSendsOpenAIEnvIdentity(t *testing.T) {
	tests := []struct {
		name        string
		opts        []option.RequestOption
		wantOrg     string
		wantProject string
	}{
		{
			name:        "from the environment",
			wantOrg:     "org-from-env",
			wantProject: "proj-from-env",
		},
		{
			name: "opts win",
			opts: []option.RequestOption{
				option.WithOrganization("org-from-opts"),
				option.WithProject("proj-from-opts"),
			},
			wantOrg:     "org-from-opts",
			wantProject: "proj-from-opts",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var header http.Header
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				header = r.Header.Clone()
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{
					"id":"c1","object":"chat.completion","created":1,"model":"gpt-4o",
					"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
					"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
				}`)
			}))
			defer server.Close()

			t.Setenv("OPENAI_API_KEY", "plugin-key")
			t.Setenv("OPENAI_ORG_ID", "org-from-env")
			t.Setenv("OPENAI_PROJECT_ID", "proj-from-env")

			opts := append([]option.RequestOption{option.WithBaseURL(server.URL)}, tt.opts...)
			g := genkit.Init(context.Background(), genkit.WithPlugins(&OpenAI{Opts: opts}))
			if _, err := genkit.Generate(context.Background(), g,
				ai.WithModelName("openai/gpt-4o"), ai.WithPrompt("hi")); err != nil {
				t.Fatalf("Generate() error = %v", err)
			}

			if got := header.Get("Authorization"); got != "Bearer plugin-key" {
				t.Errorf("Authorization = %q, want the environment's key", got)
			}
			if got := header.Get("OpenAI-Organization"); got != tt.wantOrg {
				t.Errorf("OpenAI-Organization = %q, want %q", got, tt.wantOrg)
			}
			if got := header.Get("OpenAI-Project"); got != tt.wantProject {
				t.Errorf("OpenAI-Project = %q, want %q", got, tt.wantProject)
			}
		})
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
// touching the registry, so the capabilities they carry never reach a request.
// [OpenAI.Models] and [OpenAI.Embedders] are what describe one.
func TestDeprecatedBuildersDoNotRegister(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")

	o := &OpenAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(o))

	const model = "gpt-legacy-define"
	if m := o.DefineModel(model, ai.ModelOptions{Label: "Legacy"}); m == nil {
		t.Fatal("DefineModel() = nil, want the built model")
	}
	if genkit.LookupAction(g, "/model/openai/"+model) != nil {
		t.Errorf("%q was registered by DefineModel(), want the deprecated builder to leave the registry alone", model)
	}

	const embedder = "text-embedding-legacy-define"
	if e := o.DefineEmbedder(embedder, &ai.EmbedderOptions{Label: "Legacy"}); e == nil {
		t.Fatal("DefineEmbedder() = nil, want the built embedder")
	}
	if genkit.LookupAction(g, "/embedder/openai/"+embedder) != nil {
		t.Errorf("%q was registered by DefineEmbedder(), want the deprecated builder to leave the registry alone", embedder)
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

// TestManagedConfigFieldsRejected pins that the SDK config cannot carry the
// request fields Genkit builds. The schema the model advertises is the one the
// framework validates against, so naming one of them fails the request instead
// of being dropped on the way to the provider, which is what used to happen:
// a tool set here and nowhere else reached the model, and the model could
// answer with a call Genkit has no handler for.
func TestManagedConfigFieldsRejected(t *testing.T) {
	_, g := initPlugin(t)

	_, err := genkit.Generate(context.Background(), g,
		ai.WithModel(ModelRef("gpt-4o", &openaiGo.ChatCompletionNewParams{
			Tools: []openaiGo.ChatCompletionToolParam{{
				Function: shared.FunctionDefinitionParam{Name: "smuggled_tool"},
			}},
		})),
		ai.WithPrompt("hello"),
	)
	if err == nil {
		t.Fatal("Generate() error = nil, want the config rejected for naming a Genkit-managed field")
	}
	if !strings.Contains(err.Error(), "ai.WithTools()") {
		t.Errorf("error = %v, want the rejection to name the Genkit option to use", err)
	}

	// The rest of the SDK config is untouched by the pruning.
	if _, err := genkit.Generate(context.Background(), g,
		ai.WithModel(ModelRef("gpt-4o", &openaiGo.ChatCompletionNewParams{
			Temperature: openaiGo.Float(0.5),
		})),
		ai.WithPrompt("hello"),
	); err != nil && strings.Contains(err.Error(), "did not match expected schema") {
		t.Errorf("a plain config was rejected: %v", err)
	}
}

// TestInvalidConfigTypeRejected pins the boundary rejection for a config the
// SDK schema cannot describe at all: the request fails validation before any
// plugin code runs or anything is sent.
func TestInvalidConfigTypeRejected(t *testing.T) {
	_, g := initPlugin(t)

	_, err := genkit.Generate(context.Background(), g,
		ai.WithModelName("openai/gpt-4o-mini"),
		ai.WithPrompt("hello"),
		ai.WithConfig("not a config"),
	)
	if err == nil {
		t.Fatal("Generate() error = nil, want the boundary schema rejection")
	}
	if !strings.Contains(err.Error(), "did not match expected schema") {
		t.Errorf("error = %v, want the boundary schema rejection", err)
	}
}

// TestModelsOverride pins that a caller's entry reaches a curated model and an
// uncurated one alike, through every path that describes a model. It is the
// only way to describe a curated model differently: Init has already
// registered gpt-4o by the time an application could call anything.
func TestModelsOverride(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")
	o := &OpenAI{
		Models: map[string]ai.ModelOptions{
			// A curated model, described differently.
			"gpt-4o": {Supports: &ai.ModelSupports{Multiturn: true, Media: false}},
			// One the plugin does not curate, keyed provider-prefixed.
			"openai/proxy-model": {Label: "Proxy", Supports: &ai.ModelSupports{Multiturn: true}},
		},
		Embedders: map[string]ai.EmbedderOptions{
			"text-embedding-3-small": {Dimensions: 64},
		},
	}
	g := genkit.Init(context.Background(), genkit.WithPlugins(o))

	// Registered by Init, and carrying the override.
	model := genkit.LookupModel(g, "openai/gpt-4o").(*ai.ModelAction).Desc().
		Metadata["model"].(map[string]any)
	if supports := model["supports"].(map[string]any); supports["media"] != false {
		t.Errorf("gpt-4o media = %v, want the override's false", supports["media"])
	}
	// Overlaid, not replaced: the entry says nothing about the label or the
	// versions, so the curated ones stay.
	if got := model["label"]; got != "OpenAI GPT-4o" {
		t.Errorf("gpt-4o label = %v, want the curated label kept", got)
	}
	if versions, _ := model["versions"].([]string); len(versions) == 0 {
		t.Error("gpt-4o versions were dropped, want the curated list kept")
	}

	// Resolved on demand, and carrying the override.
	resolved := o.ResolveAction(api.ActionTypeModel, "proxy-model")
	if resolved == nil {
		t.Fatal("ResolveAction(proxy-model) = nil")
	}
	if got := resolved.Desc().Metadata["model"].(map[string]any)["label"]; got != "Proxy" {
		t.Errorf("proxy-model label = %v, want the override's", got)
	}

	// Embedders take the same treatment.
	embedder := genkit.LookupEmbedder(g, "openai/text-embedding-3-small")
	if embedder == nil {
		t.Fatal("text-embedding-3-small not registered by Init")
	}
	info := embedder.(*ai.EmbedderAction).Desc().Metadata["info"].(map[string]any)
	if got := info["dimensions"]; got != 64 {
		t.Errorf("dimensions = %v (%T), want the override's 64", got, got)
	}
}
