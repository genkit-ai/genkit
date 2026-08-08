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

package main

import (
	"context"
	"errors"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func main() {
	ctx := context.Background()

	g := genkit.Init(ctx, genkit.WithPlugins(&modelgarden.Anthropic{}))

	// Define a simple flow that generates jokes about a given topic
	genkit.DefineFlow(g, "jokesFlow", func(ctx context.Context, input string) (string, error) {
		m := modelgarden.AnthropicModel(g, "claude-3-5-sonnet-v2")
		if m == nil {
			return "", errors.New("jokesFlow: failed to find model")
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1.0),
			}),
			ai.WithPrompt(`Tell a short joke about %s`, input))
		if err != nil {
			return "", err
		}

		text := resp.Text()
		return text, nil
	})

	// Vertex Model Garden - Claude Sonnet 4.6.
	genkit.DefineFlow(g, "claudeSonnet46VertexModelGardenFlow", func(ctx context.Context, input string) (string, error) {
		if input == "" {
			input = "airplane food"
		}
		m := modelgarden.AnthropicModel(g, "claude-sonnet-4-6")
		if m == nil {
			return "", errors.New("claudeSonnet46VertexModelGardenFlow: failed to find model")
		}
		return genkit.GenerateText(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				MaxTokens: 1024,
			}),
			ai.WithPrompt("Share a joke about %s.", input),
		)
	})

	// Vertex Model Garden - Claude Opus 4.6.
	genkit.DefineFlow(g, "claudeOpus46VertexModelGardenFlow", func(ctx context.Context, input string) (string, error) {
		if input == "" {
			input = "airplane food"
		}
		m := modelgarden.AnthropicModel(g, "claude-opus-4-6")
		if m == nil {
			return "", errors.New("claudeOpus46VertexModelGardenFlow: failed to find model")
		}
		return genkit.GenerateText(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				MaxTokens: 1024,
			}),
			ai.WithPrompt("Share a joke about %s.", input),
		)
	})

	// Vertex Model Garden - Claude Opus 4.7.
	genkit.DefineFlow(g, "claudeOpus47VertexModelGardenFlow", func(ctx context.Context, input string) (string, error) {
		if input == "" {
			input = "airplane food"
		}
		m := modelgarden.AnthropicModel(g, "claude-opus-4-7")
		if m == nil {
			return "", errors.New("claudeOpus47VertexModelGardenFlow: failed to find model")
		}
		return genkit.GenerateText(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				MaxTokens: 1024,
			}),
			ai.WithPrompt("Share a joke about %s.", input),
		)
	})

	<-ctx.Done()
}
