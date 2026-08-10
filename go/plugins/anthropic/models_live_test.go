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
