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
	"slices"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/meta"
	"github.com/openai/openai-go/option"
)

// completion is a minimal chat completion as Meta answers it: reasoning_content
// present but redacted to empty for external callers.
const completion = `{
	"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"muse-wonder-1.0",
	"choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"","content":"ok"},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8,
		"prompt_tokens_details":{"cached_tokens":2},
		"completion_tokens_details":{"reasoning_tokens":4}}
}`

// newServer records each request body and answers with [completion].
func newServer(t *testing.T) (*httptest.Server, func() []map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
	t.Cleanup(server.Close)
	return server, func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return bodies
	}
}

func newGenkit(t *testing.T, server *httptest.Server) *genkit.Genkit {
	t.Helper()
	return genkit.Init(context.Background(), genkit.WithPlugins(&meta.Meta{
		APIKey: "test-key",
		Opts:   []option.RequestOption{option.WithBaseURL(server.URL + "/v1")},
	}))
}

// TestConfigRidesTheWire pins where each Meta field lands: the SDK-modeled
// ones on their chat completion counterparts and the rest as Meta's extra
// request fields, under the wire names Meta documents.
func TestConfigRidesTheWire(t *testing.T) {
	server, bodies := newServer(t)
	g := newGenkit(t, server)

	parallel := false
	resp, err := genkit.Generate(context.Background(), g,
		ai.WithModel(meta.ModelRef("muse-wonder-1.0", &meta.ChatConfig{
			MaxOutputTokens:      4000,
			ReasoningEffort:      meta.ReasoningEffortMax,
			ParallelToolCalls:    &parallel,
			PromptCacheKey:       "tenant-1",
			PromptCacheRetention: meta.PromptCacheRetention24h,
			SafetyIdentifier:     "hashed-user",
		})),
		ai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Text(); got != "ok" {
		t.Errorf("Text() = %q, want %q", got, "ok")
	}
	// Meta redacts reasoning_content to empty, which must not become an
	// empty reasoning part.
	for _, part := range resp.Message.Content {
		if part.IsReasoning() {
			t.Errorf("content has a reasoning part %#v, want none from a redacted reasoning_content", part)
		}
	}
	if got := resp.Usage.CachedContentTokens; got != 2 {
		t.Errorf("CachedContentTokens = %d, want 2", got)
	}

	got := bodies()
	if len(got) != 1 {
		t.Fatalf("requests = %d, want 1", len(got))
	}
	body := got[0]
	for key, want := range map[string]any{
		"model":                  "muse-wonder-1.0",
		"max_completion_tokens":  4000.0,
		"reasoning_effort":       "max",
		"parallel_tool_calls":    false,
		"prompt_cache_key":       "tenant-1",
		"prompt_cache_retention": "24h",
		"safety_identifier":      "hashed-user",
	} {
		if body[key] != want {
			t.Errorf("%s = %#v, want %#v", key, body[key], want)
		}
	}
	// Meta rejects max_tokens' Responses-era sibling and deprecated user, so
	// neither may ride along.
	for _, key := range []string{"max_output_tokens", "max_tokens", "user"} {
		if _, ok := body[key]; ok {
			t.Errorf("request carries %q, want it absent", key)
		}
	}
}

// TestMediaParts pins the media Meta's chat completions read and the part each
// rides as, and that video, which the compat_oai base does not send, fails
// before the wire rather than as a billed 400.
func TestMediaParts(t *testing.T) {
	server, bodies := newServer(t)
	g := newGenkit(t, server)
	ctx := context.Background()

	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("meta/muse-wonder-1.0"),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Describe these."),
			ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgo="),
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,UklGRg=="),
			ai.NewMediaPart("audio/mpeg", "data:audio/mpeg;base64,SUQz"),
			ai.NewMediaPart("application/pdf", "data:application/pdf;base64,JVBERi0="),
		)),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	messages, _ := bodies()[0]["messages"].([]any)
	user, _ := messages[len(messages)-1].(map[string]any)
	parts, _ := user["content"].([]any)
	var types []string
	for _, p := range parts {
		part, _ := p.(map[string]any)
		types = append(types, part["type"].(string))
		if part["type"] == "input_audio" {
			audio, _ := part["input_audio"].(map[string]any)
			if f := audio["format"]; f != "wav" && f != "mp3" {
				t.Errorf("input_audio format = %v, want wav or mp3, the two Meta reads", f)
			}
		}
	}
	if want := []string{"text", "image_url", "input_audio", "input_audio", "file"}; !slices.Equal(types, want) {
		t.Errorf("part types = %v, want %v", types, want)
	}

	if _, err := genkit.Generate(ctx, g,
		ai.WithModelName("meta/muse-wonder-1.0"),
		ai.WithMessages(ai.NewUserMessage(ai.NewMediaPart("video/mp4", "data:video/mp4;base64,AAAA"))),
	); err == nil {
		t.Error("Generate() with video error = nil, want it rejected before the wire")
	}
	if n := len(bodies()); n != 1 {
		t.Errorf("requests = %d, want the video request kept off the wire", n)
	}
}

// TestToolChoiceRejected pins the capability: Meta takes tool_choice auto
// only and answers anything else with a 400, so the models advertise no
// tool-choice steering and Genkit fails the request before the wire.
func TestToolChoiceRejected(t *testing.T) {
	server, bodies := newServer(t)
	g := newGenkit(t, server)
	lookup := genkit.DefineTool(g, "lookup", "Looks up a value.",
		func(_ *ai.ToolContext, input struct {
			Value string `json:"value"`
		}) (string, error) {
			return "result for " + input.Value, nil
		})

	if _, err := genkit.Generate(context.Background(), g,
		ai.WithModelName("meta/muse-wonder-1.0"),
		ai.WithPrompt("Use the lookup tool."),
		ai.WithTools(lookup),
		ai.WithToolChoice(ai.ToolChoiceRequired),
	); err == nil {
		t.Error("Generate() error = nil, want the tool choice rejected")
	}
	if n := len(bodies()); n != 0 {
		t.Errorf("requests = %d, want none", n)
	}
}

// TestPluginRequiresAPIKey pins that only META_API_KEY is read: Meta's docs
// name MODEL_API_KEY, which is generic enough to hold another provider's key.
func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("META_API_KEY", "")
	t.Setenv("MODEL_API_KEY", "someone-elses-key")

	defer func() {
		if got := recover(); got != "meta plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()
	(&meta.Meta{}).Init(context.Background())
}

// TestListActionsSkipsNonChatModels pins that the image, transcription and
// segmentation models the Model API lists are not offered as chat models.
func TestListActionsSkipsNonChatModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/models")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[
			{"id":"muse-wonder-1.0","object":"model","created":1,"owned_by":"meta"},
			{"id":"muse-spark-1.3","object":"model","created":1,"owned_by":"meta"},
			{"id":"muse-image-1.0","object":"model","created":1,"owned_by":"meta"},
			{"id":"muse-voice-transcribe-1.0","object":"model","created":1,"owned_by":"meta"},
			{"id":"sam-3.1","object":"model","created":1,"owned_by":"meta"}
		]}`)
	}))
	defer server.Close()

	plugin := &meta.Meta{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	genkit.Init(context.Background(), genkit.WithPlugins(plugin))

	var names []string
	for _, desc := range plugin.ListActions(context.Background()) {
		names = append(names, desc.Name)
	}
	slices.Sort(names)
	if want := []string{"meta/muse-spark-1.3", "meta/muse-wonder-1.0"}; !slices.Equal(names, want) {
		t.Errorf("listed = %v, want %v", names, want)
	}
}

// TestModelRefAndConfigSchema pins the call-site surface and the config
// contract the Dev UI and request validation read.
func TestModelRefAndConfigSchema(t *testing.T) {
	cfg := &meta.ChatConfig{ReasoningEffort: meta.ReasoningEffortHigh}
	for _, name := range []string{"muse-wonder-1.0", "meta/muse-wonder-1.0"} {
		ref := meta.ModelRef(name, cfg)
		if want := "meta/muse-wonder-1.0"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	g := genkit.Init(context.Background(), genkit.WithPlugins(&meta.Meta{APIKey: "test-key"}))
	m := genkit.LookupModel(g, "meta/muse-wonder-1.0")
	if m == nil {
		t.Fatal("muse-wonder-1.0 not registered by Init")
	}
	model := m.(api.Action).Desc().Metadata["model"].(map[string]any)
	if got := model["stage"]; got != ai.ModelStageUnstable {
		t.Errorf("stage = %v, want %q while Meta labels Wonder a preview", got, ai.ModelStageUnstable)
	}
	props := model["customOptions"].(map[string]any)["properties"].(map[string]any)

	// Fields Meta answers with a 400 must not be offered.
	for _, key := range []string{"stopSequences", "logProbs", "topLogProbs", "user"} {
		if props[key] != nil {
			t.Errorf("config schema advertises %q, which Meta rejects", key)
		}
	}
	for field, want := range map[string]map[string]any{
		"temperature":      {"minimum": 0.0, "maximum": 2.0},
		"topP":             {"exclusiveMinimum": 0.0, "maximum": 1.0},
		"safetyIdentifier": {"maxLength": 64.0},
		"reasoningEffort":  {"enum": []any{"minimal", "low", "medium", "high", "xhigh", "max"}},
		"promptCacheRetention": {"enum": []any{
			string(meta.PromptCacheRetentionInMemory), string(meta.PromptCacheRetention24h)}},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}
}
