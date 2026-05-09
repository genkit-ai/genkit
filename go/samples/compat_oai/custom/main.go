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
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	oai "github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
)

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENROUTER_API_KEY environment variable not set")
	}

	g := genkit.Init(ctx, genkit.WithPlugins(&oai.OpenAICompatible{
		Provider: "openrouter",
		APIKey:   apiKey,
		BaseURL:  "https://openrouter.ai/api/v1",
	}),
		genkit.WithDefaultModel("openrouter/tngtech/deepseek-r1t2-chimera:free"))

	prompt := "tell me a joke"
	config := &openai.ChatCompletionNewParams{
		Temperature: openai.Float(0.7),
		MaxTokens:   openai.Int(1000),
		TopP:        openai.Float(0.9),
	}

	resp, err := genkit.Generate(context.Background(), g,
		ai.WithConfig(config),
		ai.WithPrompt(prompt))
	if err != nil {
		log.Fatalf("failed to generate contents: %v", err)
	}
	log.Println("Joke:", resp.Text())
}
