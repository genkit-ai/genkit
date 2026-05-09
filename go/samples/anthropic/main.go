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

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	ant "github.com/firebase/genkit/go/plugins/anthropic"
)

func main() {
	ctx := context.Background()

	g := genkit.Init(ctx, genkit.WithPlugins(&ant.Anthropic{}))

	// Define a simple flow that generates a short story about a given topic
	genkit.DefineStreamingFlow(g, "storyFlow", func(ctx context.Context, input string, cb ai.ModelStreamCallback) (string, error) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("anthropic/claude-sonnet-4-20250514"),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   *anthropic.IntPtr(2000),
				Thinking: anthropic.ThinkingConfigParamUnion{
					OfEnabled: &anthropic.ThinkingConfigEnabledParam{
						BudgetTokens: *anthropic.IntPtr(1024),
					},
				},
			}),
			ai.WithStreaming(cb),
			ai.WithPrompt(`Tell a short story about %s`, input))
		if err != nil {
			return "", err
		}

		return resp.Text(), nil
	})

	<-ctx.Done()
}
