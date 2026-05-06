// Copyright 2025 Google LLC
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

package modelgarden_test

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func TestLlamaLive(t *testing.T) {
	if _, ok := requireEnv("GOOGLE_CLOUD_PROJECT"); !ok {
		t.Skip("GOOGLE_CLOUD_PROJECT not found in the environment")
	}
	if _, ok := requireEnv("GOOGLE_CLOUD_LOCATION"); !ok {
		t.Skip("GOOGLE_CLOUD_LOCATION not found in the environment")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&modelgarden.Llama{}))

	t.Run("invalid model", func(t *testing.T) {
		m := modelgarden.LlamaModel(g, "meta/llama-does-not-exist")
		if m != nil {
			t.Fatalf("model should have been empty, got: %#v", m)
		}
	})

	t.Run("basic generation", func(t *testing.T) {
		m := modelgarden.LlamaModel(g, "meta/llama-3.3-70b-instruct-maas")
		if m == nil {
			t.Fatal("meta/llama-3.3-70b-instruct-maas model was not registered")
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithSystem("You are a helpful assistant. Reply in one short sentence."),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Say hello."))),
		)
		if err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(resp.Text()) == "" {
			t.Fatal("expected a non-empty response")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		m := modelgarden.LlamaModel(g, "meta/llama-3.3-70b-instruct-maas")
		out := ""
		final, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Count from one to three."),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				for _, p := range c.Content {
					if p.IsText() {
						out += p.Text
					}
				}
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}
		if out == "" {
			t.Fatal("expected streamed content")
		}
		if final.Usage.InputTokens == 0 || final.Usage.OutputTokens == 0 {
			t.Fatalf("empty usage stats: %#v", *final.Usage)
		}
	})
}
