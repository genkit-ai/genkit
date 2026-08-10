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

package anthropic

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
)

// TestModelOptionsKnownModels verifies the curated Claude models resolve through
// the shared modelOptions helper (used by both ListActions and ResolveAction)
// with JS ADVANCED_MODEL_INFO-equivalent supports (JSON output) and a stable
// stage. The set mirrors the JS plugin's ADVANCED entries in KNOWN_MODELS.
func TestModelOptionsKnownModels(t *testing.T) {
	advancedModels := []string{
		"claude-fable-5",
		"claude-opus-5",
		"claude-sonnet-5",
		"claude-opus-4-8",
		"claude-opus-4-7",
		"claude-opus-4-6",
		"claude-opus-4-5",
		"claude-sonnet-4-6",
		"claude-sonnet-4-5",
		"claude-haiku-4-5",
	}
	for _, name := range advancedModels {
		opts := modelOptions(name)
		if opts.Supports == nil {
			t.Errorf("modelOptions(%q): Supports is nil", name)
			continue
		}
		if !slices.Contains(opts.Supports.Output, "json") {
			t.Errorf("modelOptions(%q): Output = %v, want it to include \"json\"", name, opts.Supports.Output)
		}
		if !opts.Supports.Tools || !opts.Supports.SystemRole {
			t.Errorf("modelOptions(%q): expected Tools and SystemRole supported, got %+v", name, opts.Supports)
		}
		if opts.Stage != ai.ModelStageStable {
			t.Errorf("modelOptions(%q): Stage = %q, want Stable", name, opts.Stage)
		}
		if opts.Label == "" {
			t.Errorf("modelOptions(%q): Label is empty", name)
		}
	}
}

func TestModelOptionsKnownVersionedModels(t *testing.T) {
	advancedModels := []string{
		"claude-opus-4-5-20251101",
		"claude-sonnet-4-5-20250929",
		"claude-haiku-4-5-20251001",
	}
	for _, name := range advancedModels {
		opts := modelOptions(name)
		if opts.Supports == nil {
			t.Errorf("modelOptions(%q): Supports is nil", name)
			continue
		}
		if !slices.Contains(opts.Supports.Output, "json") {
			t.Errorf("modelOptions(%q): Output = %v, want it to include \"json\"", name, opts.Supports.Output)
		}
		if !opts.Supports.Tools || !opts.Supports.SystemRole {
			t.Errorf("modelOptions(%q): expected Tools and SystemRole supported, got %+v", name, opts.Supports)
		}
	}
}

// TestModelOptionsUnknownFallback verifies models not in supportedModels fall back
// to dynamicModelOptions (no JSON output).
func TestModelOptionsUnknownFallback(t *testing.T) {
	const name = "claude-something-unreleased"
	opts := modelOptions(name)

	if opts.Supports == nil {
		t.Fatalf("modelOptions(%q): Supports is nil", name)
	}
	if slices.Contains(opts.Supports.Output, "json") {
		t.Errorf("modelOptions(%q): unknown model should use default supports without JSON output, got %v", name, opts.Supports.Output)
	}
}

// TestNewModelDescriptor covers what a built model advertises: a curated label
// for known models and a name-derived one for the rest, plus the config schema
// the framework validates every request against.
func TestNewModelDescriptor(t *testing.T) {
	tests := []struct {
		name      string
		wantLabel string
	}{
		{"claude-opus-4-5", anthropicLabelPrefix + " - Claude Opus 4.5"},
		{"claude-opus-4-5-20251101", anthropicLabelPrefix + " - Claude Opus 4.5"},
		{"claude-something-unreleased", anthropicLabelPrefix + " - claude-something-unreleased"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			desc := newModel(anthropic.Client{}, tt.name, tt.name, modelOptions(tt.name)).Desc()

			model, ok := desc.Metadata["model"].(map[string]any)
			if !ok {
				t.Fatalf("model metadata missing, got %v", desc.Metadata)
			}
			if got := model["label"]; got != tt.wantLabel {
				t.Errorf("label = %v, want %q", got, tt.wantLabel)
			}

			schema, ok := model["customOptions"].(map[string]any)
			if !ok {
				t.Fatalf("customOptions missing, got %v", model["customOptions"])
			}
			props, ok := schema["properties"].(map[string]any)
			if !ok || props["max_tokens"] == nil {
				t.Errorf("config schema is not the Anthropic message params schema, got %v", schema)
			}
		})
	}
}

// TestRegisterModelNilOptions covers the nil ModelOptions path: the model gets
// the capabilities the plugin resolves for its name rather than panicking or
// advertising a model that supports nothing. It also pins that RegisterModel
// registers, so the lookup helpers can find what it defined.
func TestRegisterModelNilOptions(t *testing.T) {
	a := &Anthropic{}
	g := genkit.Init(context.Background())

	m, err := a.RegisterModel(g, "claude-opus-4-5", nil)
	if err != nil {
		t.Fatalf("RegisterModel() error = %v", err)
	}

	if !IsDefinedModel(g, "claude-opus-4-5") {
		t.Error("IsDefinedModel() = false after RegisterModel(), want the model registered")
	}
	if Model(g, "claude-opus-4-5") == nil {
		t.Error("Model() = nil after RegisterModel(), want the registered model")
	}

	model, ok := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing")
	}
	if want := anthropicLabelPrefix + " - Claude Opus 4.5"; model["label"] != want {
		t.Errorf("label = %v, want %q", model["label"], want)
	}
	supports, ok := model["supports"].(map[string]any)
	if !ok {
		t.Fatalf("supports metadata missing")
	}
	if supports["tools"] != true || supports["multiturn"] != true {
		t.Errorf("supports = %v, want the curated Claude capabilities", supports)
	}
}

// TestModelConfigIsValidated pins that the config schema reaches the request
// input schema, so the framework rejects a config the SDK type cannot hold
// before it reaches the model function.
func TestModelConfigIsValidated(t *testing.T) {
	const name = "claude-opus-4-5"
	inputSchema := newModel(anthropic.Client{}, name, name, modelOptions(name)).Desc().InputSchema

	req := func(config any) *ai.ModelRequest {
		return &ai.ModelRequest{
			Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("hello"))},
			Config:   config,
		}
	}

	if err := base.ValidateValue(req(map[string]any{"max_tokens": 100, "temperature": 0.4}), inputSchema); err != nil {
		t.Errorf("config rejected at the action boundary: %v", err)
	}
	if err := base.ValidateValue(req(map[string]any{"max_tokens": "lots"}), inputSchema); err == nil {
		t.Error("expected a mistyped max_tokens to be rejected")
	}
}

func TestResolveModelID(t *testing.T) {
	availableModels := []string{
		"claude-opus-4-6",
		"claude-opus-4-5-20251101",
		"claude-opus-4-1-20250805",
		"claude-opus-4-20250514",
		"claude-sonnet-4-5-20250929",
		"claude-sonnet-4-20250514",
		"claude-haiku-4-5-20251001",
	}

	tests := []struct {
		input    string
		expected string
		found    bool
	}{
		// Exact matches
		{"claude-opus-4-6", "claude-opus-4-6", true},
		{"claude-opus-4-1-20250805", "claude-opus-4-1-20250805", true},
		{"claude-opus-4-20250514", "claude-opus-4-20250514", true},

		// Aliases
		{"claude-opus-4-5", "claude-opus-4-5-20251101", true},
		{"claude-sonnet-4-5", "claude-sonnet-4-5-20250929", true},
		{"claude-sonnet-4", "claude-sonnet-4-20250514", true},
		{"claude-opus-4", "claude-opus-4-20250514", true},
		{"claude-haiku-4-5", "claude-haiku-4-5-20251001", true},

		// Non-existent
		{"claude-2", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, found := resolveModelID(tt.input, availableModels)
			if found != tt.found {
				t.Errorf("found = %v, want %v", found, tt.found)
			}
			if got != tt.expected {
				t.Errorf("got = %q, want %q", got, tt.expected)
			}
		})
	}
}

// TestRegisterModelThenResolve pins the ordinary path: a model defined before
// its first use is the one generation resolves, and defining it does not make
// the plugin resolve it a second time. Generate turns a name into an action
// through the same resolving lookup Model uses, and the plugin's ResolveAction
// would call the Anthropic API, which this fake key cannot reach, so the
// caller's own capabilities coming back prove the registry short-circuited to
// the defined model instead of resolving again.
func TestRegisterModelThenResolve(t *testing.T) {
	a := &Anthropic{APIKey: "test-key"}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	const name = "claude-opus-4-5"
	opts := ai.ModelOptions{
		Label:    "Custom Claude",
		Supports: &ai.ModelSupports{Multiturn: true, Tools: true},
	}
	if _, err := a.RegisterModel(g, name, &opts); err != nil {
		t.Fatalf("RegisterModel() error = %v", err)
	}

	m := Model(g, name)
	if m == nil {
		t.Fatal("Model() = nil, want the model RegisterModel registered")
	}
	model, ok := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing")
	}
	if model["label"] != "Custom Claude" {
		t.Errorf("label = %v, want the capabilities RegisterModel was given", model["label"])
	}
}

// TestRegisterModelTwicePanics pins what registering costs: the registry rejects
// a duplicate key, so a name can only be defined once and [IsDefinedModel] is
// the guard.
func TestRegisterModelTwicePanics(t *testing.T) {
	a := &Anthropic{APIKey: "test-key"}
	g := genkit.Init(context.Background())

	const name = "claude-haiku-4-5"
	if _, err := a.RegisterModel(g, name, nil); err != nil {
		t.Fatalf("RegisterModel() error = %v", err)
	}

	defer func() {
		if recover() == nil {
			t.Error("defining the same name twice did not panic, want the registry to reject it")
		}
	}()
	a.RegisterModel(g, name, nil)
}

// TestModelRef pins the name a ref carries and that the typed config rides
// along, since the ref is how an application supplies config at the call site.
// Both the bare ID and the already-prefixed name resolve to the same model,
// so passing the name a sibling plugin would take is not a silent miss.
func TestModelRef(t *testing.T) {
	cfg := &anthropic.MessageNewParams{MaxTokens: 1024}

	for _, name := range []string{"claude-opus-4-5", "anthropic/claude-opus-4-5"} {
		ref := ModelRef(name, cfg)
		if want := "anthropic/claude-opus-4-5"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	// A nil config rides along as a typed nil rather than an untyped one. The
	// config slot tolerates that: it marshals to JSON null and deserializes to
	// the zero MessageNewParams, the same as googlegenai's refs.
	if got := ModelRef("claude-opus-4-5", nil).Config(); got != (*anthropic.MessageNewParams)(nil) {
		t.Errorf("Config() = %v for a nil config, want a typed nil", got)
	}
}

// TestPrefixedNamesAreEquivalent pins that the exported entry points take a
// model ID either bare or provider-prefixed. The prefix is applied by
// concatenation, so an untrimmed name would double up: RegisterModel would
// register a key IsDefinedModel never checks, breaking the guard the
// RegisterModel godoc points at.
func TestPrefixedNamesAreEquivalent(t *testing.T) {
	a := &Anthropic{APIKey: "test-key"}
	g := genkit.Init(context.Background())

	if _, err := a.RegisterModel(g, "anthropic/claude-opus-4-5", nil); err != nil {
		t.Fatalf("RegisterModel() error = %v", err)
	}

	for _, name := range []string{"claude-opus-4-5", "anthropic/claude-opus-4-5"} {
		if !IsDefinedModel(g, name) {
			t.Errorf("IsDefinedModel(%q) = false, want the model defined under either form", name)
		}
		if Model(g, name) == nil {
			t.Errorf("Model(%q) = nil, want the model defined under either form", name)
		}
	}

	// Defining by the prefixed name must resolve the curated capabilities, not
	// the unknown-model defaults, which is why the trim precedes the lookup.
	m := Model(g, "claude-opus-4-5")
	model, ok := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing")
	}
	if want := anthropicLabelPrefix + " - Claude Opus 4.5"; model["label"] != want {
		t.Errorf("label = %v, want %q", model["label"], want)
	}
}

// TestIsDefinedModelDoesNotResolve pins the guard semantics: checking whether
// a model is defined must not itself resolve and register one. The plugin
// resolves any name the Anthropic API can serve, so a resolving lookup would
// answer true for every name and make the guarded RegisterModel panic. The fake
// endpoint serves the models list to make resolution reachable, which is
// exactly what the guard must not trigger.
func TestIsDefinedModelDoesNotResolve(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"id":"claude-opus-4-5-20251101","type":"model"}],"has_more":false}`)
	}))
	defer server.Close()

	a := &Anthropic{APIKey: "test-key", BaseURL: server.URL}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	if IsDefinedModel(g, "claude-opus-4-5") {
		t.Fatal("IsDefinedModel() = true for a model that was never defined")
	}
	if _, err := a.RegisterModel(g, "claude-opus-4-5", nil); err != nil {
		t.Fatalf("RegisterModel() after the guard error = %v", err)
	}
	if !IsDefinedModel(g, "claude-opus-4-5") {
		t.Error("IsDefinedModel() = false after RegisterModel()")
	}
}

// TestDefineModelDoesNotRegister pins the deprecated builder: it hands back a
// model without touching the registry, which is what it has always done and
// what separates it from RegisterModel.
func TestDefineModelDoesNotRegister(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")

	a := &Anthropic{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	const name = "claude-opus-4-5"
	m, err := a.DefineModel(g, name, nil)
	if err != nil {
		t.Fatalf("DefineModel() error = %v", err)
	}
	if m == nil {
		t.Fatal("DefineModel() = nil, want the built model")
	}
	if IsDefinedModel(g, name) {
		t.Errorf("IsDefinedModel(%q) = true after DefineModel(), want the deprecated builder to leave the registry alone", name)
	}
}
