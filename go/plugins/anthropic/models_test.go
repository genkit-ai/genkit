// Copyright 2026 Google LLC
//
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// TestConstrainedSupport pins which curated models advertise native structured
// output. Anthropic's Structured Outputs list is narrower than the model
// catalog, and the request path sends output_config only when the model claims
// support, so claiming it on a model absent from that list would have the
// request rejected after Genkit had already dropped the schema instructions
// from the prompt.
// See https://platform.claude.com/docs/en/build-with-claude/structured-outputs.
func TestConstrainedSupport(t *testing.T) {
	// Curated models absent from Anthropic's Structured Outputs list.
	withoutStructuredOutputs := map[string]bool{"claude-opus-4-1": true}

	for id, opts := range supportedModels {
		want := ai.ConstrainedSupportAll
		if withoutStructuredOutputs[id] {
			want = ""
		}
		if got := opts.Supports.Constrained; got != want {
			t.Errorf("%s constrained = %q, want %q", id, got, want)
		}
		// Every curated model is JSON-capable either way.
		if len(opts.Supports.Output) == 0 {
			t.Errorf("%s declares no output formats", id)
		}
	}

	for id := range withoutStructuredOutputs {
		if _, ok := supportedModels[id]; !ok {
			t.Errorf("%s is no longer curated; drop it from this test", id)
		}
	}

	// An unknown model may predate Structured Outputs, so the fallback must not
	// claim them.
	if got := dynamicModelOptions.Supports.Constrained; got != "" {
		t.Errorf("dynamic constrained = %q, want unset", got)
	}
}
