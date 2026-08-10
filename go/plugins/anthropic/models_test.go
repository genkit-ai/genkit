// Copyright 2026 Google LLC
//
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
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

// TestSupportedModelsAreServableLive sends one small request to every curated
// model, so a retirement is caught here rather than by an application.
//
// A curated ID is not checked against anything at build time: a model
// Anthropic has retired still resolves, registers, and advertises curated
// capabilities, and only fails when a request reaches the API and comes back
// 404 naming the model rather than the list that offered it. Nothing offline
// can tell the difference, which is how claude-opus-4-1 stayed on the list
// past its retirement.
//
// Reads supportedModels directly so a model added to the catalog is covered
// without touching this test.
//
// Retirements: https://platform.claude.com/docs/en/about-claude/model-deprecations
func TestSupportedModelsAreServableLive(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not found in the environment")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Anthropic{}))

	for id := range supportedModels {
		t.Run(id, func(t *testing.T) {
			t.Parallel()
			resp, err := genkit.Generate(ctx, g,
				ai.WithModelName(provider+"/"+id),
				// No temperature: it is deprecated on Claude 4.7 and later and
				// returns 400 when set, so the request must not carry one.
				ai.WithConfig(&anthropic.MessageNewParams{MaxTokens: 16}),
				ai.WithPrompt("Reply with the single word: ok"))
			if err != nil {
				t.Fatalf("%s is curated but not servable: %v", id, err)
			}
			if strings.TrimSpace(resp.Text()) == "" {
				t.Errorf("%s returned an empty response", id)
			}
		})
	}
}
