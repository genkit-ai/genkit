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

package anthropic

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

// TestCuratedModels pins the current lineup: each curated entry has the
// structured-output capabilities, and only Sonnet 4.5 is marked deprecated.
//
// Catalog: https://platform.claude.com/docs/en/models/overview
func TestCuratedModels(t *testing.T) {
	current := []string{
		"claude-fable-5-2",
		"claude-fable-5-1",
		"claude-fable-5",
		"claude-opus-5-5",
		"claude-opus-5",
		"claude-opus-4-8",
		"claude-opus-4-7",
		"claude-opus-4-6",
		"claude-opus-4-5-20251101",
		"claude-sonnet-5-5",
		"claude-sonnet-5",
		"claude-sonnet-4-6",
		"claude-sonnet-4-5-20250929",
		"claude-haiku-5-5",
		"claude-haiku-4-5-20251001",
	}
	for _, id := range current {
		opts, ok := supportedModels[id]
		if !ok {
			t.Errorf("%s is current but not curated", id)
			continue
		}
		if opts.Label == "" {
			t.Errorf("%s has no label", id)
		}
		// Versions lists the values the version config field accepts. It
		// includes the key with any date suffix removed, which for a
		// dateless key is the key itself.
		alias := id
		if i := strings.LastIndex(id, "-20"); i > 0 {
			alias = id[:i]
		}
		if !slices.Contains(opts.Versions, alias) {
			t.Errorf("%s versions = %v, want them to include %s", id, opts.Versions, alias)
		}
		if opts.Supports == nil {
			t.Errorf("%s has no supports", id)
		} else if opts.Supports.Constrained != ai.ConstrainedSupportAll || !opts.Supports.SystemRole || !opts.Supports.Tools || !opts.Supports.Media {
			t.Errorf("%s supports = %+v, want the structured-output set", id, opts.Supports)
		}
		// Anthropic deprecated Sonnet 4.5 on September 30, 2026.
		wantDeprecated := id == "claude-sonnet-4-5-20250929"
		if got := opts.Stage == ai.ModelStageDeprecated; got != wantDeprecated {
			t.Errorf("%s deprecated = %v, want %v", id, got, wantDeprecated)
		}
	}
	if len(supportedModels) != len(current) {
		t.Errorf("supportedModels has %d entries, want %d: update this test with the catalog", len(supportedModels), len(current))
	}
}

// TestNoRetiredModels guards the curated list against IDs the API no longer
// serves. Init registers every curated model, so a retired one would be
// offered and then fail each request with a 404.
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
		"claude-3-sonnet",   // retired 2025-07-21
	}
	for id, opts := range supportedModels {
		for _, name := range append([]string{id}, opts.Versions...) {
			for _, r := range retired {
				// A retired family prefix on its own, followed by a date, or
				// followed by -latest. Matching only those suffixes keeps
				// claude-opus-4 from matching claude-opus-4-5.
				if name == r || strings.HasPrefix(name, r+"-20") || strings.HasPrefix(name, r+"-latest") {
					t.Errorf("%s (curated as %s) is retired", name, id)
				}
			}
		}
	}
}

// TestDynamicModelOptions pins what an uncurated ID resolves with: the full
// Claude set without native constrained generation, since the base URL can
// point at a gateway serving models that predate Structured Outputs.
func TestDynamicModelOptions(t *testing.T) {
	opts := (&Anthropic{}).modelOptions("claude-something-unreleased")
	if opts.Supports.Constrained != "" {
		t.Errorf("dynamic constrained = %q, want unset", opts.Supports.Constrained)
	}
	if !opts.Supports.Tools || !opts.Supports.SystemRole || !opts.Supports.Media || !opts.Supports.Multiturn {
		t.Errorf("dynamic supports = %+v, want the full Claude set", opts.Supports)
	}
	if opts.Stage != ai.ModelStageStable {
		t.Errorf("dynamic stage = %q, want %q", opts.Stage, ai.ModelStageStable)
	}
}

// TestSupportedModelsAreServableLive sends one small request to every curated
// model through the OpenAI-compatible endpoint, so a retired model still
// marked stable is caught here rather than by an application. Deprecated
// entries are skipped. A retired model left in supportedModels still resolves
// and registers with curated capabilities, and only fails when a request
// comes back 404.
//
// Reads supportedModels directly so a model added to the catalog is covered
// without touching this test.
//
// Retirements: https://platform.claude.com/docs/en/about-claude/model-deprecations
func TestSupportedModelsAreServableLive(t *testing.T) {
	livetest.Env(t, "ANTHROPIC_API_KEY")
	livetest.Expensive(t)

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Anthropic{}))

	for id := range supportedModels {
		t.Run(id, func(t *testing.T) {
			if supportedModels[id].Stage == ai.ModelStageDeprecated {
				t.Skip("deprecated: an account can lose access to it before its retirement date")
			}
			t.Parallel()
			resp, err := genkit.Generate(ctx, g,
				// Temperature stays unset: it is deprecated on Claude 4.7 and
				// later and returns 400 when set. The budget leaves room for a
				// model that thinks by default to finish thinking and still
				// reach a text block.
				ai.WithModel(ModelRef(id, &ChatConfig{MaxOutputTokens: 1024})),
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
