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
	// Every curated model is on the list, so a future entry that is not must
	// come with its own capability set rather than reusing structuredModel.
	for id, opts := range supportedModels {
		if got := opts.Supports.Constrained; got != ai.ConstrainedSupportAll {
			t.Errorf("%s constrained = %q, want %q", id, got, ai.ConstrainedSupportAll)
		}
		if len(opts.Supports.Output) == 0 {
			t.Errorf("%s declares no output formats", id)
		}
	}

	// An unknown model may predate Structured Outputs, so the fallback must not
	// claim them.
	if got := dynamicModelOptions.Supports.Constrained; got != "" {
		t.Errorf("dynamic constrained = %q, want unset", got)
	}
}

// TestNoRetiredModels guards the curated list against IDs the API no longer
// serves. A retired ID resolves and registers like any other, then fails every
// request with a 404 that names the model rather than the list that offered it.
//
// Retirements: https://platform.claude.com/docs/en/about-claude/model-deprecations
func TestNoRetiredModels(t *testing.T) {
	retired := []string{
		"claude-opus-4-1",   // retired 2026-08-05
		"claude-opus-4",     // retired 2026-06-15
		"claude-sonnet-4",   // retired 2026-06-15
		"claude-3-7-sonnet", // retired 2026-02-19
		"claude-3-5-haiku",  // retired 2026-02-19
		"claude-3-haiku",    // retired 2026-04-20
		"claude-3-opus",     // retired 2026-01-05
		"claude-3-5-sonnet", // retired 2025-10-28
	}
	for _, id := range retired {
		if _, ok := supportedModels[id]; ok {
			t.Errorf("%s is retired but still curated", id)
		}
	}
}
