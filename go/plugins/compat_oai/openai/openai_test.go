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
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
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

// TestConstrainedSupport pins that every curated model advertises native
// structured output. Every model left in the catalog postdates the
// gpt-4o-2024-08-06 snapshot that introduced response_format json_schema.
func TestConstrainedSupport(t *testing.T) {
	for id, opts := range supportedModels {
		if got := opts.Supports.Constrained; got != ai.ConstrainedSupportAll {
			t.Errorf("%s constrained = %q, want %q", id, got, ai.ConstrainedSupportAll)
		}
	}
	if got := dynamicModelOptions.Supports.Constrained; got != ai.ConstrainedSupportAll {
		t.Errorf("dynamic constrained = %q, want %q", got, ai.ConstrainedSupportAll)
	}
}

// TestCatalogOmitsShutDownModels pins that no shut-down model or snapshot is registered.
func TestCatalogOmitsShutDownModels(t *testing.T) {
	registered := map[string]string{}
	for name, opts := range supportedModels {
		registered[name] = name
		for _, v := range opts.Versions {
			registered[v] = name
		}
	}
	shutDown := []string{
		"gpt-4.1-nano", "gpt-4.1-nano-2025-04-14", "gpt-4o-2024-05-13",
		"o4-mini", "o4-mini-2025-04-16", "o3-mini", "o3-mini-2025-01-31",
		"o1", "o1-2024-12-17", "gpt-4-turbo", "gpt-4-turbo-2024-04-09",
		"gpt-4", "gpt-4-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-0125",
		"gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct",
	}
	for _, id := range shutDown {
		if under, ok := registered[id]; ok {
			t.Errorf("%s is shut down but still registered under %s", id, under)
		}
	}
}

// TestModelStagesAndTools pins the two per-model facts the catalog states
// beyond the shared capability sets: which models are deprecated, and which
// cannot call tools through Chat Completions.
func TestModelStagesAndTools(t *testing.T) {
	// Every model with an announced shutdown date.
	deprecated := map[string]bool{
		"gpt-5": true, "gpt-5-mini": true, "gpt-5-nano": true, "o3": true, // 2026-12-11
		"gpt-4.1": true, "gpt-4.1-mini": true, "gpt-4o": true, "gpt-4o-mini": true, // 2027-01-14
		"gpt-5.1": true, "gpt-5.4-nano": true, // 2027-04-01
	}
	// The first three call tools only through the Responses API; the search
	// model rejects tools outright.
	noTools := map[string]bool{
		"gpt-6-astra": true, "gpt-6.1-sol": true, "gpt-6.1-terra": true,
		"gpt-5-search-api": true,
	}

	for _, id := range []string{
		"gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra",
		"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5-search-api",
	} {
		if _, ok := supportedModels[id]; !ok {
			t.Errorf("%s is missing from the catalog", id)
		}
	}
	for id, opts := range supportedModels {
		want := ai.ModelStageStable
		if deprecated[id] {
			want = ai.ModelStageDeprecated
		}
		if opts.Stage != want {
			t.Errorf("%s stage = %q, want %q", id, opts.Stage, want)
		}
		if got := opts.Supports.Tools; got == noTools[id] {
			t.Errorf("%s tools = %v, want %v", id, got, !noTools[id])
		}
		if opts.Supports.ToolChoice && !opts.Supports.Tools {
			t.Errorf("%s advertises tool choice without tools", id)
		}
	}
}

// TestSnapshotsTakeTheirModelsOptions pins that a dated snapshot resolved by
// name is described like the model it is a version of, deprecation included,
// rather than with the dynamic defaults.
func TestSnapshotsTakeTheirModelsOptions(t *testing.T) {
	o, g := initPlugin(t)

	for snapshot, parent := range map[string]string{
		"gpt-5-2025-08-07":   "gpt-5",
		"o3-2025-04-16":      "o3",
		"gpt-5.5-2026-04-23": "gpt-5.5",
	} {
		got := o.modelOptions(snapshot)
		want := supportedModels[parent]
		if got.Stage != want.Stage || got.Supports != want.Supports {
			t.Errorf("%s: stage %q supports %p, want %s's stage %q supports %p", snapshot, got.Stage, got.Supports, parent, want.Stage, want.Supports)
		}
		if !slices.Equal(got.Versions, []string{snapshot}) {
			t.Errorf("%s: versions = %v, want just the snapshot", snapshot, got.Versions)
		}
	}
	if got := o.modelOptions("gpt-5-2025-08-07").Label; got != "OpenAI GPT-5 2025-08-07" {
		t.Errorf("snapshot label = %q, want %q", got, "OpenAI GPT-5 2025-08-07")
	}
	m := genkit.LookupModel(g, "openai/gpt-5-2025-08-07")
	if m == nil {
		t.Fatal("gpt-5-2025-08-07 did not resolve")
	}
	if got := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)["stage"]; got != ai.ModelStageDeprecated {
		t.Errorf("resolved snapshot stage = %v, want %q", got, ai.ModelStageDeprecated)
	}
}

// TestFineTunesTakeTheirBaseModelsOptions pins that a fine-tuned model is
// described like its base model, deprecation included, and that a fine-tune
// of an unknown base takes the dynamic defaults.
func TestFineTunesTakeTheirBaseModelsOptions(t *testing.T) {
	o, g := initPlugin(t)

	for ft, base := range map[string]string{
		"ft:gpt-4o-mini-2024-07-18:acme::abc123": "gpt-4o-mini",
		"ft:gpt-4.1:acme:custom:def456":          "gpt-4.1",
		"ft:gpt-5.4-mini-2026-03-17:acme::ghi":   "gpt-5.4-mini",
	} {
		got := o.modelOptions(ft)
		want := supportedModels[base]
		if got.Stage != want.Stage || got.Supports != want.Supports {
			t.Errorf("%s: stage %q supports %p, want %s's stage %q supports %p", ft, got.Stage, got.Supports, base, want.Stage, want.Supports)
		}
		if got.Label != "" || len(got.Versions) != 0 {
			t.Errorf("%s: label %q versions %v, want neither", ft, got.Label, got.Versions)
		}
	}
	if got := o.modelOptions("ft:my-base:acme::xyz"); got.Stage != dynamicModelOptions.Stage || got.Supports != dynamicModelOptions.Supports {
		t.Errorf("unknown-base fine-tune = stage %q supports %p, want the dynamic defaults", got.Stage, got.Supports)
	}

	m := genkit.LookupModel(g, "openai/ft:gpt-4o-mini-2024-07-18:acme::abc123")
	if m == nil {
		t.Fatal("fine-tune did not resolve")
	}
	if got := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)["stage"]; got != ai.ModelStageDeprecated {
		t.Errorf("resolved fine-tune stage = %v, want %q", got, ai.ModelStageDeprecated)
	}
}

// TestIsChatModel pins which IDs from OpenAI's models list are taken for chat
// models. The list mixes in every other kind of model the account can use.
func TestIsChatModel(t *testing.T) {
	chat := []string{
		"gpt-6-astra", "gpt-6.1-terra", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.5-2026-04-23",
		"gpt-5-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini-2024-07-18",
		"o3", "o3-2025-04-16", "gpt-6-cyber", "gpt-5-search-api", "chat-latest",
		"ft:gpt-4o-mini-2024-07-18:acme::abc123",
		// IDs outside OpenAI's naming, from a proxy or self-hosted server,
		// pass even when they carry a word that marks an OpenAI non-chat model.
		"claude-sonnet-4-5", "meta-llama/Llama-3.1-8B-Instruct", "qwen2.5-7b-instruct", "acme-pro",
	}
	notChat := []string{
		// Image, speech, transcription, realtime and Live.
		"dall-e-3", "gpt-image-2", "gpt-image-2.5-flare", "chatgpt-image-latest",
		"tts-1", "gpt-4o-mini-tts", "whisper-1", "gpt-transcribe", "gpt-4o-transcribe-diarize",
		"gpt-realtime-2.1", "gpt-4o-realtime-preview", "gpt-live-1", "gpt-live-transcribe",
		"gpt-realtime-whisper", "gpt-audio-1.5", "gpt-audio-mini", "gpt-4o-audio-preview",
		// Embeddings, moderation, video, legacy completions.
		"text-embedding-3-small", "omni-moderation-latest", "sora-2",
		"babbage-002", "davinci-002", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct-0914",
		"curie:ft-acme-2023-06-07-10-19-42", "davinci:ft-acme-2023-01-01-00-00-00",
		"babbage:ft-acme-2023-01-01-00-00-00", "ada:ft-acme-2023-01-01-00-00-00",
		// Responses only.
		"gpt-5.5-pro", "gpt-5.5-pro-2026-04-23", "gpt-5.2-pro", "o3-pro", "gpt-5-pro-2025-10-06",
		"gpt-5.1-codex-max", "codex-mini-latest", "o3-deep-research", "computer-use-preview",
		"gpt-5.6-cyber", "gpt-rosalind-research", "gpt-6-rosalind", "gpt-daybreak-blue-latest",
		"ft:gpt-5.5-pro:acme::abc123",
	}
	for _, id := range chat {
		if !isChatModel(id) {
			t.Errorf("isChatModel(%q) = false, want true", id)
		}
	}
	for _, id := range notChat {
		if isChatModel(id) {
			t.Errorf("isChatModel(%q) = true, want false", id)
		}
	}
}

// TestListActionsListsChatModelsOnly pins that the listing the Dev UI shows
// holds only the models Chat Completions serves, not every model the
// endpoint reports.
func TestListActionsListsChatModelsOnly(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("request path = %q, want /models", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[
			{"id":"gpt-6-sol","object":"model","created":1,"owned_by":"openai"},
			{"id":"o3","object":"model","created":1,"owned_by":"openai"},
			{"id":"ft:gpt-4o-mini-2024-07-18:acme::abc123","object":"model","created":1,"owned_by":"acme"},
			{"id":"dall-e-3","object":"model","created":1,"owned_by":"system"},
			{"id":"gpt-image-2","object":"model","created":1,"owned_by":"system"},
			{"id":"tts-1","object":"model","created":1,"owned_by":"system"},
			{"id":"whisper-1","object":"model","created":1,"owned_by":"system"},
			{"id":"text-embedding-3-small","object":"model","created":1,"owned_by":"system"},
			{"id":"gpt-5.5-pro","object":"model","created":1,"owned_by":"system"},
			{"id":"gpt-realtime-2.1","object":"model","created":1,"owned_by":"system"},
			{"id":"meta-llama/Llama-3.1-8B-Instruct","object":"model","created":1,"owned_by":"vllm"}
		]}`)
	}))
	defer server.Close()

	t.Setenv("OPENAI_API_KEY", "test-key")
	o := &OpenAI{Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	genkit.Init(context.Background(), genkit.WithPlugins(o))

	var got []string
	for _, d := range o.ListActions(context.Background()) {
		got = append(got, d.Name)
	}
	slices.Sort(got)
	want := []string{
		"openai/ft:gpt-4o-mini-2024-07-18:acme::abc123",
		"openai/gpt-6-sol",
		"openai/meta-llama/Llama-3.1-8B-Instruct",
		"openai/o3",
	}
	if !slices.Equal(got, want) {
		t.Errorf("ListActions() names = %v, want %v", got, want)
	}
}

// listingFixture is the OpenAI discovery fixture every Genkit SDK tests its
// listing against.
type listingFixture struct {
	Chat   []string `json:"chat"`
	Hidden []string `json:"hidden"`
	Other  []string `json:"other"`
	Tools  struct {
		NeverOnChatCompletions []string `json:"never_on_chat_completions"`
	} `json:"tools"`
}

// TestListActionsMatchesFixture pins, against a models list holding every ID
// in the shared fixture, that each chat ID is listed and nothing else is, and
// which listed models advertise tools.
func TestListActionsMatchesFixture(t *testing.T) {
	raw, err := os.ReadFile("testdata/openai-listing-fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx listingFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}

	type entry struct {
		ID string `json:"id"`
	}
	var data []entry
	for _, ids := range [][]string{fx.Chat, fx.Hidden, fx.Other} {
		for _, id := range ids {
			data = append(data, entry{id})
		}
	}
	body, err := json.Marshal(map[string]any{"object": "list", "data": data})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	defer server.Close()

	t.Setenv("OPENAI_API_KEY", "test-key")
	o := &OpenAI{Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	genkit.Init(context.Background(), genkit.WithPlugins(o))

	listed := map[string]map[string]any{}
	for _, d := range o.ListActions(context.Background()) {
		listed[strings.TrimPrefix(d.Name, "openai/")] = d.Metadata["model"].(map[string]any)["supports"].(map[string]any)
	}

	noTools := map[string]bool{}
	for _, id := range fx.Tools.NeverOnChatCompletions {
		noTools[id] = true
	}
	for _, tc := range []struct {
		class  string
		ids    []string
		listed bool
	}{
		{"chat", fx.Chat, true},
		{"hidden", fx.Hidden, false},
		{"other", fx.Other, false},
	} {
		t.Run(tc.class, func(t *testing.T) {
			for _, id := range tc.ids {
				supports, ok := listed[id]
				if ok != tc.listed {
					t.Errorf("%s listed = %v, want %v", id, ok, tc.listed)
					continue
				}
				if ok && supports["tools"] != !noTools[id] {
					t.Errorf("%s tools = %v, want %v", id, supports["tools"], !noTools[id])
				}
			}
		})
	}
	if len(listed) != len(fx.Chat) {
		t.Errorf("listed %d models, want the %d chat IDs", len(listed), len(fx.Chat))
	}
}

// TestListActionsLogsListError pins that a failed listing is reported rather
// than swallowed: ListActions has no error to return, so it lists nothing and
// logs why.
func TestListActionsLogsListError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}}`)
	}))
	defer server.Close()

	t.Setenv("OPENAI_API_KEY", "test-key")
	o := &OpenAI{Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	genkit.Init(context.Background(), genkit.WithPlugins(o))

	var logged bytes.Buffer
	ctx := logger.WithContext(context.Background(), slog.New(slog.NewTextHandler(&logged, nil)))
	if got := o.ListActions(ctx); len(got) != 0 {
		t.Errorf("ListActions() = %v, want nothing listed", got)
	}
	out := logged.String()
	if !strings.Contains(out, "level=WARN") || !strings.Contains(out, "provider=openai") || !strings.Contains(out, "Incorrect API key") {
		t.Errorf("log = %q, want a warning naming the provider and the API's error", out)
	}

	// Reflection polls the list, so a failure that persists is warned
	// about once; the repeats go to debug, below the handler's level.
	logged.Reset()
	o.ListActions(ctx)
	if out := logged.String(); out != "" {
		t.Errorf("second failing ListActions() logged %q, want nothing at warn level", out)
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
