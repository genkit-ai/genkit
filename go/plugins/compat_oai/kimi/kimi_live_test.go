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

package kimi_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
)

func TestPluginLive(t *testing.T) {
	apiKey := os.Getenv("KIMI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("MOONSHOT_API_KEY")
	}
	if apiKey == "" {
		t.Skip("KIMI_API_KEY and MOONSHOT_API_KEY are not set")
	}

	ctx := context.Background()
	plugin := &kimi.Kimi{}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("kimi/kimi-k3"),
	)

	t.Run("kimi k3 complete", func(t *testing.T) {
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("What is the capital of France? Answer with the city only."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "paris") {
			t.Fatalf("Text() = %q, want Paris", resp.Text())
		}
	})

	t.Run("kimi k2.6 streaming reasoning", func(t *testing.T) {
		var streamedReasoning, streamedText strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithModelName("kimi/kimi-k2.6"),
			ai.WithPrompt("Explain briefly why the sky appears blue."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				streamedReasoning.WriteString(chunk.Reasoning())
				streamedText.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if streamedReasoning.String() == "" {
			t.Fatal("streamed reasoning is empty")
		}
		if streamedText.String() == "" {
			t.Fatal("streamed text is empty")
		}
		if resp.Reasoning() != streamedReasoning.String() {
			t.Fatalf("final reasoning = %q, want streamed %q", resp.Reasoning(), streamedReasoning.String())
		}
		if resp.Text() != streamedText.String() {
			t.Fatalf("final text = %q, want streamed %q", resp.Text(), streamedText.String())
		}
	})
}
