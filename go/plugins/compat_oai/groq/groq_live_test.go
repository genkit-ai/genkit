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

package groq_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/groq"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("GROQ_API_KEY") == "" {
		t.Skip("GROQ_API_KEY is not set")
	}

	ctx := context.Background()
	plugin := &groq.Groq{}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("groq/"+groq.ModelLlama3370bVersatile),
	)

	t.Run("complete", func(t *testing.T) {
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

	t.Run("streaming", func(t *testing.T) {
		var streamedText strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Say hello in one short sentence."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				streamedText.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if streamedText.String() == "" {
			t.Fatal("streamed text is empty")
		}
		if resp.Text() != streamedText.String() {
			t.Fatalf("final text = %q, want streamed %q", resp.Text(), streamedText.String())
		}
	})
}
