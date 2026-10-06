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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/oailive"
	"github.com/firebase/genkit/go/plugins/compat_oai/openrouter"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/openai/openai-go"
)

// The models the live checks spend on. They are ordinary catalog entries
// rather than anything the plugin knows about, so swap in whatever the key
// has credit for; the plugin resolves any ID the gateway serves.
const (
	chatModel      = "openai/gpt-5-mini"
	visionModel    = "anthropic/claude-haiku-4.5"
	reasoningModel = "anthropic/claude-haiku-4.5"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "OPENROUTER_API_KEY")
	g := livetest.Init(t, &openrouter.OpenRouter{})

	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: openrouter.ModelRef(chatModel, nil),
			// OpenRouter normalizes each vendor's thinking onto the
			// response's reasoning field, so the content reaches the caller.
			ReasoningModel: openrouter.ModelRef(reasoningModel, &openrouter.ChatConfig{
				MaxOutputTokens: 4096,
				Reasoning:       &openrouter.ReasoningConfig{MaxTokens: 1024},
			}),
			ReasoningContent: true,
			VisionModel:      openrouter.ModelRef(visionModel, nil),
			LimitConfig:      &openrouter.ChatConfig{MaxOutputTokens: 16},
			// Deliberately not shaped like a key. OpenRouter rejects any
			// bearer token it does not recognize, and a realistic-looking
			// placeholder only trips secret scanning on the way to the same
			// 401.
			BadKeyPlugin: &openrouter.OpenRouter{APIKey: "invalid"},
			Skip: map[string]string{
				"generate/unknown model": "OpenRouter answers an unknown model ID with 400, not 404",
			},
		},
		ExtraConfig: map[string]any{
			"extra": map[string]any{"user": "genkit-livetest"},
		},
	})

	// OpenRouter prices a request and reports what it charged, with no
	// request field asking for it. The field that used to turn this on is
	// deprecated and does nothing, so the only check that the accounting
	// still arrives is against the real API.
	t.Run("cost reported", func(t *testing.T) {
		resp, err := genkit.Generate(t.Context(), g,
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
	})

	// A routing constraint no provider satisfies is the gateway's own
	// refusal, and must stay apart from a bad key (UNAUTHENTICATED, checked by
	// the shared suite) so middleware can route around it by status.
	t.Run("no provider serves the model", func(t *testing.T) {
		model := openrouter.ModelRef(chatModel, &openrouter.ChatConfig{
			Provider: &openrouter.ProviderRouting{Only: []string{"not-a-provider"}},
		})
		for _, streaming := range []bool{false, true} {
			opts := []ai.GenerateOption{
				ai.WithModel(model),
				ai.WithPrompt("Name one primary color. Answer with the word alone."),
			}
			if streaming {
				opts = append(opts, ai.WithStreaming(
					func(context.Context, *ai.ModelResponseChunk) error { return nil }))
			}
			resp, err := genkit.Generate(t.Context(), g, opts...)
			if err == nil {
				t.Fatalf("Generate(streaming %v) error = nil, want the request refused (response %+v)", streaming, resp)
			}
			if got, ok := status.Classified(err); !ok || got != status.NotFound {
				t.Errorf("Generate(streaming %v) status = %q (classified %v), want %q: %v", streaming, got, ok, status.NotFound, err)
			}
		}
	})

	// The fields the gateway exists for. OpenRouter answers a malformed
	// provider object or models list with a 400 rather than ignoring it, so
	// a request that comes back at all is the assertion. It is deliberately
	// not about the answer's content: price sorting routes to whichever
	// endpoint is cheapest at the time, which may be a heavily quantized one.
	t.Run("gateway controls accepted", func(t *testing.T) {
		for name, config := range map[string]*openrouter.ChatConfig{
			"provider routing": {
				MaxOutputTokens: 512,
				Provider: &openrouter.ProviderRouting{
					Sort:              openrouter.ProviderSortPrice,
					DataCollection:    openrouter.DataCollectionDeny,
					RequireParameters: openai.Ptr(true),
				},
			},
			// The fallback list is for a model that fails at request time,
			// not for one that does not exist: OpenRouter validates the
			// primary model ID up front and answers an unknown one with a 400
			// rather than falling through. So this pins that a well-formed
			// list is accepted; which entry serves the request is not
			// deterministic enough to assert.
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
				resp, err := genkit.Generate(t.Context(), g,
					ai.WithModel(openrouter.ModelRef(chatModel, config)),
					ai.WithPrompt("Name one primary color. Answer with the word alone."),
				)
				if err != nil {
					t.Fatalf("Generate() error = %v", err)
				}
				if strings.TrimSpace(resp.Text()) == "" {
					// The budget is the usual suspect: it reaches OpenRouter
					// as max_tokens, which a reasoning model spends on
					// thinking before any visible text, so too small a cap
					// returns an empty answer with a length finish reason.
					t.Errorf("Text() is empty, want the request served (finish reason %q, usage %+v)",
						resp.FinishReason, resp.Usage)
				}
			})
		}
	})
}
