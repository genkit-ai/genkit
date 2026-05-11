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
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func main() {
	ctx := context.Background()

	g := genkit.Init(ctx, genkit.WithPlugins(
		&modelgarden.Anthropic{},
		&modelgarden.Llama{},
		&modelgarden.Mistral{},
	))

	// Anthropic flows.
	defineFlow(g, "jokesFlow",
		modelgarden.AnthropicModel(g, "claude-3-5-sonnet-v2@20241022"),
		"Tell a short joke about %s",
		ai.WithConfig(&anthropic.MessageNewParams{
			MaxTokens:   256,
			Temperature: anthropic.Float(1.0),
		}),
	)
	defineFlow(g, "opus45Flow",
		modelgarden.AnthropicModel(g, "claude-opus-4-5@20251101"),
		"Write a haiku about %s",
		ai.WithConfig(&anthropic.MessageNewParams{MaxTokens: 256}),
	)

	// Llama flow.
	defineFlow(g, "llamaFlow",
		modelgarden.LlamaModel(g, "meta/llama-3.3-70b-instruct-maas"),
		"In one short sentence, describe %s",
	)

	// Mistral / Codestral flows.
	defineFlow(g, "mistralFlow",
		modelgarden.MistralModel(g, "mistralai/mistral-small-2503"),
		"List three interesting facts about %s",
	)
	defineFlow(g, "codestralFlow",
		modelgarden.MistralModel(g, "mistralai/codestral-2"),
		"Write a small Go function that does the following: %s",
	)

	<-ctx.Done()
}

// defineFlow registers a Dev UI flow that generates from the given model using
// a prompt template. Extra GenerateOption values (e.g. provider-specific
// config) are appended to the base options.
func defineFlow(
	g *genkit.Genkit,
	name string,
	m ai.Model,
	promptTemplate string,
	extra ...ai.GenerateOption,
) {
	genkit.DefineFlow(g, name, func(ctx context.Context, input string) (string, error) {
		if m == nil {
			return "", fmt.Errorf("%s: model not registered", name)
		}
		opts := append([]ai.GenerateOption{
			ai.WithModel(m),
			ai.WithPrompt(promptTemplate, input),
		}, extra...)
		resp, err := genkit.Generate(ctx, g, opts...)
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})
}
