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

package openrouter_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
	"github.com/firebase/genkit/go/plugins/compat_oai/openrouter"
	"github.com/openai/openai-go"
)

// The models the live checks spend on. They are ordinary catalog entries
// rather than anything the plugin knows about, so swap in whatever the key
// has credit for; the plugin resolves any ID the gateway serves.
const (
	chatModel      = "openai/gpt-5-mini"
	visionModel    = "anthropic/claude-haiku-4.5"
	reasoningModel = "deepseek/deepseek-r1"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&openrouter.OpenRouter{}),
		genkit.WithDefaultModel("openrouter/"+chatModel),
	)

	livetest.Run(t, g, livetest.Suite{
		Model: openrouter.ModelRef(chatModel, nil),
		// OpenRouter normalizes each vendor's thinking onto the response's
		// reasoning field, so the content reaches the caller.
		ReasoningModel: openrouter.ModelRef(reasoningModel, &openrouter.ChatConfig{
			MaxOutputTokens: 1024,
			Reasoning:       &openrouter.ReasoningConfig{Effort: openrouter.ReasoningEffortLow},
		}),
		ReasoningContent: true,
		VisionModel:      openrouter.ModelRef(visionModel, nil),
		ToolChoice:       true,
		ExtraConfig: map[string]any{
			"extra": map[string]any{"user": "genkit-livetest"},
		},
	})
}

// TestCostReportedLive pins that OpenRouter prices a request and reports what
// it charged, with no request field asking for it. The field that used to turn
// this on is deprecated and does nothing, so the only check that the
// accounting still arrives is against the real API.
func TestCostReportedLive(t *testing.T) {
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&openrouter.OpenRouter{}))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(openrouter.ModelRef(chatModel, nil)),
		ai.WithPrompt("Name one primary color. Answer with the word alone."),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if cost := resp.Usage.Custom["cost"]; cost <= 0 {
		t.Errorf("Usage.Custom[\"cost\"] = %v, want the price OpenRouter charged (usage %+v)",
			cost, resp.Usage)
	}
}

// TestErrorStatusClassifiedLive pins that a request the gateway refuses reaches
// the caller as a classified status rather than an opaque error, on both
// transports. Two refusals are checked rather than one, so the assertion is
// that they are told apart rather than merely classified: middleware routes
// around a failure by status, and one that collapses every refusal into the
// same value is no better than none.
//
// Streaming is the half worth spending a live check on. NewStreaming returns
// before the response arrives, so a refusal surfaces at the stream rather than
// at the call, and only the real gateway says whether that assumption holds.
//
// A provider dying part-way through a generation is the other source of a
// stream error and is deliberately not here: it needs an upstream to fail
// mid-response, which no request can provoke. That path is pinned against the
// documented shape in TestGenerateStreamTopLevelFailureEndsStream.
func TestErrorStatusClassifiedLive(t *testing.T) {
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY is not set")
	}

	ctx := context.Background()
	keyed := genkit.Init(ctx, genkit.WithPlugins(&openrouter.OpenRouter{}))
	// Deliberately not shaped like a key. OpenRouter rejects any bearer token
	// it does not recognize, and a realistic-looking placeholder only trips
	// secret scanning on the way to the same 401.
	rejected := genkit.Init(ctx, genkit.WithPlugins(&openrouter.OpenRouter{APIKey: "invalid"}))

	for name, tc := range map[string]struct {
		g     *genkit.Genkit
		model ai.ModelRef
		want  status.Name
	}{
		"rejected key": {
			g:     rejected,
			model: openrouter.ModelRef(chatModel, nil),
			want:  status.Unauthenticated,
		},
		"no provider serves the model": {
			g: keyed,
			model: openrouter.ModelRef(chatModel, &openrouter.ChatConfig{
				Provider: &openrouter.ProviderRouting{Only: []string{"not-a-provider"}},
			}),
			want: status.NotFound,
		},
	} {
		t.Run(name, func(t *testing.T) {
			for _, streaming := range []string{"call", "stream"} {
				t.Run(streaming, func(t *testing.T) {
					opts := []ai.GenerateOption{
						ai.WithModel(tc.model),
						ai.WithPrompt("Name one primary color. Answer with the word alone."),
					}
					if streaming == "stream" {
						opts = append(opts, ai.WithStreaming(
							func(context.Context, *ai.ModelResponseChunk) error { return nil }))
					}

					resp, err := genkit.Generate(ctx, tc.g, opts...)
					if err == nil {
						t.Fatalf("Generate() error = nil, want the request refused (response %+v)", resp)
					}
					got, classified := status.Classified(err)
					if !classified || got != tc.want {
						t.Errorf("status = %q (classified %v), want %q: %v", got, classified, tc.want, err)
					}
				})
			}
		})
	}
}

// TestGatewayControlsAcceptedLive pins the fields the gateway exists for
// against the real API. OpenRouter answers a malformed provider object or
// models list with a 400 rather than ignoring it, so a request that comes back
// at all is the assertion.
//
// The assertion is deliberately not about the answer's content. Price sorting
// routes to whichever endpoint is cheapest at the time, which may be a heavily
// quantized one, so tying this to a correct arithmetic result would make it
// fail on the routing working exactly as asked.
func TestGatewayControlsAcceptedLive(t *testing.T) {
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&openrouter.OpenRouter{}))

	for name, config := range map[string]*openrouter.ChatConfig{
		"provider routing": {
			MaxOutputTokens: 512,
			Provider: &openrouter.ProviderRouting{
				Sort:              openrouter.ProviderSortPrice,
				DataCollection:    openrouter.DataCollectionDeny,
				RequireParameters: openai.Ptr(true),
			},
		},
		// The fallback list is for a model that fails at request time, not for
		// one that does not exist: OpenRouter validates the primary model ID up
		// front and answers an unknown one with a 400 rather than falling
		// through. So this pins that a well-formed list is accepted; which
		// entry serves the request is not deterministic enough to assert.
		"fallback chain": {
			MaxOutputTokens: 512,
			Models:          []string{visionModel},
		},
		"transforms and session": {
			MaxOutputTokens: 512,
			Transforms:      []string{"middle-out"},
			SessionID:       "genkit-livetest",
			Metadata:        map[string]string{"suite": "genkit-livetest"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(openrouter.ModelRef(chatModel, config)),
				ai.WithPrompt("Name one primary color. Answer with the word alone."),
			)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if strings.TrimSpace(resp.Text()) == "" {
				// The budget is the usual suspect: it reaches OpenRouter as
				// max_tokens, which a reasoning model spends on thinking
				// before any visible text, so too small a cap returns an
				// empty answer with a length finish reason.
				t.Errorf("Text() is empty, want the request served (finish reason %q, usage %+v)",
					resp.FinishReason, resp.Usage)
			}
		})
	}
}
