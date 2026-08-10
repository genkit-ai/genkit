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

	"github.com/firebase/genkit/go/ai"
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

// TestModelOptionsUnknownFallback verifies models not in knownModels fall back to
// defaultClaudeOpts (no JSON output) but still get a provider-prefixed label.
func TestModelOptionsUnknownFallback(t *testing.T) {
	const name = "claude-something-unreleased"
	opts := modelOptions(name)

	if opts.Supports == nil {
		t.Fatalf("modelOptions(%q): Supports is nil", name)
	}
	if slices.Contains(opts.Supports.Output, "json") {
		t.Errorf("modelOptions(%q): unknown model should use default supports without JSON output, got %v", name, opts.Supports.Output)
	}
	if want := anthropicLabelPrefix + " - " + name; opts.Label != want {
		t.Errorf("modelOptions(%q): Label = %q, want %q", name, opts.Label, want)
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
