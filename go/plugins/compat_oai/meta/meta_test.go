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

package meta_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/compat_oai/meta"
	"github.com/openai/openai-go/option"
)

func TestCatalogConfigAndDynamicPathsAgree(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/models")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[
				{"id":"muse-spark-1.2","object":"model","created":1,"owned_by":"meta"},
				{"id":"muse-spark-future","object":"model","created":1,"owned_by":"meta"}
			]
		}`)
	}))
	defer server.Close()

	plugin := &meta.Meta{
		APIKey: "test-key",
		Opts:   []option.RequestOption{option.WithBaseURL(server.URL + "/v1")},
		Models: map[string]ai.ModelOptions{
			"meta/muse-spark-1.2": {Label: "Overridden Muse Spark 1.2"},
			"muse-spark-future":   {Label: "Future Muse Spark"},
		},
	}
	actions := plugin.Init(context.Background())

	registered := map[string]api.Action{}
	for _, action := range actions {
		registered[action.Desc().Name] = action
	}
	for _, name := range []string{
		"meta/muse-spark-1.1",
		"meta/muse-spark-1.2",
		"meta/muse-spark-1.2-contributor",
	} {
		if registered[name] == nil {
			t.Errorf("Init() did not register %q", name)
		}
	}
	if got := len(registered); got != 3 {
		t.Fatalf("Init() registered %d models, want 3", got)
	}

	assertModel := func(t *testing.T, desc api.ActionDesc, wantLabel string) {
		t.Helper()
		model := desc.Metadata["model"].(map[string]any)
		if got := model["label"]; got != wantLabel {
			t.Errorf("label = %v, want %q", got, wantLabel)
		}
		if versions, _ := model["versions"].([]string); len(versions) != 0 {
			t.Errorf("versions = %v, want none for complete model IDs", versions)
		}
		supports := model["supports"].(map[string]any)
		if got := supports["constrained"]; got != ai.ConstrainedSupportAll {
			t.Errorf("constrained = %v, want %q", got, ai.ConstrainedSupportAll)
		}
		if got := supports["output"]; !reflect.DeepEqual(got, []string{"text", "json"}) {
			t.Errorf("output = %#v, want text and json", got)
		}
		schema, ok := model["customOptions"].(map[string]any)
		if !ok {
			t.Fatalf("customOptions missing: %v", model)
		}
		props := schema["properties"].(map[string]any)
		for _, key := range []string{
			"temperature", "topP", "maxOutputTokens", "stopSequences",
			"reasoningEffort", "version", "extra",
		} {
			if props[key] == nil {
				t.Errorf("config schema is missing %q", key)
			}
		}
	}

	assertModel(t, registered["meta/muse-spark-1.2"].Desc(), "Overridden Muse Spark 1.2")

	listed := map[string]api.ActionDesc{}
	for _, desc := range plugin.ListActions(context.Background()) {
		listed[desc.Name] = desc
	}
	if desc, ok := listed["meta/muse-spark-1.2"]; !ok {
		t.Error("ListActions() omitted muse-spark-1.2")
	} else {
		assertModel(t, desc, "Overridden Muse Spark 1.2")
	}
	if desc, ok := listed["meta/muse-spark-future"]; !ok {
		t.Error("ListActions() omitted the dynamic model")
	} else {
		assertModel(t, desc, "Future Muse Spark")
	}

	resolved := plugin.ResolveAction(api.ActionTypeModel, "meta/muse-spark-future")
	if resolved == nil {
		t.Fatal("ResolveAction() returned nil for a dynamic model")
	}
	assertModel(t, resolved.Desc(), "Future Muse Spark")
}

func TestTypedConfigGenerationAndStreaming(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer request-key" {
			t.Errorf("Authorization = %q, want the per-request key", got)
		}
		for _, header := range []string{"OpenAI-Organization", "OpenAI-Project"} {
			if _, ok := r.Header[http.CanonicalHeaderKey(header)]; ok {
				t.Errorf("%s leaked to Meta: %q", header, r.Header.Values(header))
			}
		}

		var body struct {
			Model                string   `json:"model"`
			MaxCompletionTokens  int      `json:"max_completion_tokens"`
			PromptCacheRetention string   `json:"prompt_cache_retention"`
			ReasoningEffort      string   `json:"reasoning_effort"`
			Stop                 []string `json:"stop"`
			Stream               bool     `json:"stream"`
			StreamOptions        struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "muse-spark-1.2" {
			t.Errorf("model = %q, want muse-spark-1.2", body.Model)
		}
		if body.MaxCompletionTokens != 128 {
			t.Errorf("max_completion_tokens = %d, want 128", body.MaxCompletionTokens)
		}
		if body.ReasoningEffort != "low" {
			t.Errorf("reasoning_effort = %q, want low", body.ReasoningEffort)
		}
		if body.PromptCacheRetention != "24h" {
			t.Errorf("prompt_cache_retention = %q, want 24h", body.PromptCacheRetention)
		}
		if !reflect.DeepEqual(body.Stop, []string{"END"}) {
			t.Errorf("stop = %v, want [END]", body.Stop)
		}

		w.Header().Set("Content-Type", "application/json")
		if !body.Stream {
			_, _ = io.WriteString(w, `{
				"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"muse-spark-1.2",
				"choices":[{"index":0,"message":{"role":"assistant","content":"Muse completion works"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
			}`)
			return
		}
		if !body.StreamOptions.IncludeUsage {
			t.Error("stream_options.include_usage = false, want usage requested")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("test server does not support flushing")
		}
		for _, event := range []string{
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"muse-spark-1.2","choices":[{"index":0,"delta":{"role":"assistant","content":"Muse "},"finish_reason":null}]}`,
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"muse-spark-1.2","choices":[{"index":0,"delta":{"content":"streaming works"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"muse-spark-1.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"muse-spark-1.2","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`,
		} {
			_, _ = io.WriteString(w, "data: "+event+"\n\n")
			flusher.Flush()
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	t.Setenv("OPENAI_API_KEY", "openai-key-must-not-leak")
	t.Setenv("OPENAI_ORG_ID", "org-must-not-leak")
	t.Setenv("OPENAI_PROJECT_ID", "project-must-not-leak")

	ctx := context.Background()
	plugin := &meta.Meta{
		APIKey: "plugin-key",
		Opts:   []option.RequestOption{option.WithBaseURL(server.URL + "/v1")},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	model := meta.ModelRef("meta/muse-spark-1.1", &meta.ChatConfig{
		RequestConfig: compat_oai.RequestConfig{
			APIKey:  "request-key",
			Version: "muse-spark-1.2",
			Extra:   map[string]any{"prompt_cache_retention": "24h"},
		},
		MaxOutputTokens: 128,
		StopSequences:   []string{"END"},
		ReasoningEffort: meta.ReasoningEffortLow,
	})
	if got := model.Name(); got != "meta/muse-spark-1.1" {
		t.Fatalf("ModelRef().Name() = %q, want provider prefix exactly once", got)
	}

	t.Run("complete", func(t *testing.T) {
		before := requests.Load()
		resp, err := genkit.Generate(ctx, g, ai.WithModel(model), ai.WithPrompt("Reply briefly."))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Text(); got != "Muse completion works" {
			t.Errorf("Text() = %q, want Muse completion works", got)
		}
		if resp.Usage == nil || resp.Usage.TotalTokens != 5 {
			t.Errorf("Usage = %#v, want 5 total tokens", resp.Usage)
		}
		if got := requests.Load() - before; got != 1 {
			t.Errorf("requests = %d, want 1 for this subtest", got)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		before := requests.Load()
		var streamed strings.Builder
		chunks := 0
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(model),
			ai.WithPrompt("Reply briefly."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				chunks++
				streamed.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if chunks < 2 {
			t.Errorf("stream delivered %d chunks, want multiple flushed chunks", chunks)
		}
		if got := streamed.String(); got != "Muse streaming works" {
			t.Errorf("streamed text = %q, want Muse streaming works", got)
		}
		if got := resp.Text(); got != streamed.String() {
			t.Errorf("final text = %q, want streamed %q", got, streamed.String())
		}
		if resp.Usage == nil || resp.Usage.TotalTokens != 5 {
			t.Errorf("stream Usage = %#v, want 5 total tokens", resp.Usage)
		}
		if got := requests.Load() - before; got != 1 {
			t.Errorf("requests = %d, want 1 for this subtest", got)
		}
	})
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("MODEL_API_KEY", "")
	defer func() {
		got := recover()
		if got != "meta plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()
	(&meta.Meta{}).Init(context.Background())
}
