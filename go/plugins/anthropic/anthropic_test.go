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
	"slices"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

// TestModelOptionsKnownModels verifies the curated Claude models resolve through
// the shared modelOptions helper (used by both ListActions and ResolveAction)
// with JS ADVANCED_MODEL_INFO-equivalent supports (JSON output) and a stable
// stage. The set mirrors the JS plugin's ADVANCED entries in KNOWN_MODELS.
func TestModelOptionsKnownModels(t *testing.T) {
	advancedModels := []string{
		"claude-opus-4-8",
		"claude-opus-4-7",
		"claude-opus-4-6",
		"claude-opus-4-5",
		"claude-opus-4-1",
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
		"claude-opus-4-1-20250805",
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

// TestModelOptionsUnknownFallback verifies models not in knownModels fall back
// to defaultClaudeOpts (no JSON output).
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

// TestDefineModelNilOptions covers the nil ModelOptions path: the model gets
// the capabilities the plugin resolves for its name rather than panicking or
// advertising a model that supports nothing.
func TestDefineModelNilOptions(t *testing.T) {
	a := &Anthropic{}

	m, err := a.DefineModel(nil, "claude-opus-4-5", nil)
	if err != nil {
		t.Fatalf("DefineModel() error = %v", err)
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
