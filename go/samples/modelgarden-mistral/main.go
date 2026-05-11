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

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func main() {
	ctx := context.Background()

	// Mistral and Codestral MaaS are served from us-central1. Override Location
	// if your project has Mistral enabled in a different region.
	g := genkit.Init(ctx, genkit.WithPlugins(
		&modelgarden.Mistral{Location: "us-central1"},
	))

	// Define a flow that uses Mistral Small to list three interesting facts.
	genkit.DefineFlow(g, "mistralFlow", func(ctx context.Context, input string) (string, error) {
		m := modelgarden.MistralModel(g, "mistralai/mistral-small-2503")
		if m == nil {
			return "", errors.New("mistralFlow: failed to find model")
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("List three interesting facts about %s", input))
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	// Define a flow that uses Codestral 2 to write a small code snippet.
	genkit.DefineFlow(g, "codestralFlow", func(ctx context.Context, input string) (string, error) {
		m := modelgarden.MistralModel(g, "mistralai/codestral-2")
		if m == nil {
			return "", errors.New("codestralFlow: failed to find model")
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Write a small Go function that does the following: %s", input))
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	<-ctx.Done()
}
