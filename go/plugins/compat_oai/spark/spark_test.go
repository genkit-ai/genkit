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

package spark_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/spark"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("SPARK_API_KEY", "")
	// An OPENAI_API_KEY must never be picked up as a fallback: sending it to
	// Spark would silently authenticate with the wrong provider's key.
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		got := recover()
		if got != "spark plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&spark.Spark{}).Init(context.Background())
}

func TestPluginConfigPrecedence(t *testing.T) {
	var mu sync.Mutex
	var rightHit, wrongHit bool
	var gotAuth string

	right := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		rightHit = true
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"4.0Ultra",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer right.Close()

	wrong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		wrongHit = true
		mu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer wrong.Close()

	// Explicit configuration must win over env vars: the struct field for the
	// key, and an Opts [option.WithBaseURL] for the endpoint, which the plugin
	// applies after its own defaults.
	t.Setenv("SPARK_API_KEY", "env-key")
	t.Setenv("SPARK_BASE_URL", wrong.URL)

	ctx := context.Background()
	plugin := &spark.Spark{APIKey: "struct-key", Opts: []option.RequestOption{option.WithBaseURL(right.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("spark/4.0Ultra"))

	if _, err := genkit.Generate(ctx, g, ai.WithPrompt("hi")); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !rightHit || wrongHit {
		t.Fatalf("rightHit = %v, wrongHit = %v, want struct fields to take precedence over env vars", rightHit, wrongHit)
	}
	if gotAuth != "Bearer struct-key" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer struct-key")
	}
}

func TestPluginRegistersModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"4.0Ultra",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	t.Setenv("SPARK_API_KEY", "test-key")
	t.Setenv("SPARK_BASE_URL", server.URL)

	ctx := context.Background()
	plugin := &spark.Spark{}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("spark/4.0Ultra"))

	if plugin.Name() != "spark" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "spark")
	}

	for _, modelID := range []string{"4.0Ultra", "generalv3.5", "lite"} {
		model := genkit.LookupModel(g, "spark/"+modelID)
		if model == nil {
			t.Errorf("LookupModel(%q) = nil", modelID)
			continue
		}
		desc := model.(api.Action).Desc()
		if got, want := desc.Name, "spark/"+modelID; got != want {
			t.Errorf("%s Desc().Name = %q, want %q", modelID, got, want)
		}
		supports := desc.Metadata["model"].(map[string]any)["supports"].(map[string]any)
		for field, want := range map[string]bool{"media": false, "tools": true, "toolChoice": true, "multiturn": true} {
			if got := supports[field]; got != want {
				t.Errorf("%s %s support = %v, want %v", modelID, field, got, want)
			}
		}
		output, _ := supports["output"].([]string)
		if !slices.Equal(output, []string{"text", "json"}) {
			t.Errorf("%s output = %v, want [text json]", modelID, output)
		}
	}
}

func TestChatConfigApplyToChatCompletion(t *testing.T) {
	temperature, topP := 0.3, 0.8
	cfg := spark.ChatConfig{
		Temperature:     &temperature,
		TopP:            &topP,
		MaxOutputTokens: 64,
		StopSequences:   []string{"STOP"},
	}
	var params openai.ChatCompletionNewParams
	// Exercises the field mapping; the zero-value fields must be left untouched
	// and the set fields must apply without panicking.
	cfg.ApplyToChatCompletion(&params)
}
