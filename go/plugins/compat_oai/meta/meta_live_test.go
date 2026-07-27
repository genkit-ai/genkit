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

package meta_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/meta"
)

func TestPluginLive(t *testing.T) {
	apiKey := os.Getenv("MODEL_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("META_API_KEY")
	}
	if apiKey == "" {
		t.Skip("META_API_KEY and MODEL_API_KEY are not set")
	}

	ctx := context.Background()
	plugin := &meta.Meta{}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("meta/"+meta.ModelMuseSpark11),
	)

	config := map[string]any{"reasoning_effort": "high"}
	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("What is the capital of France? Answer with the city only."),
			ai.WithConfig(config),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "paris") {
			t.Fatalf("Text() = %q, want Paris", resp.Text())
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var streamed strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Reply with exactly: Muse streaming works"),
			ai.WithConfig(config),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				streamed.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if streamed.String() == "" {
			t.Fatal("streamed response is empty")
		}
		if resp.Text() != streamed.String() {
			t.Fatalf("final text = %q, want streamed %q", resp.Text(), streamed.String())
		}
	})
}
