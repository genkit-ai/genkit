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

// This sample demonstrates the DeepSeek plugin: a flow that generates a joke
// with a model pinned through deepseek.ModelRef and its typed config.
//
// To run:
//
//	export DEEPSEEK_API_KEY=...
//	go run .
//
// In another terminal:
//
//	curl -X POST http://localhost:8080/jokesFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "bananas"}'
package main

import (
	"context"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
	"github.com/firebase/genkit/go/plugins/server"
)

func main() {
	ctx := context.Background()

	// The plugin reads the API key from the DEEPSEEK_API_KEY environment variable.
	g := genkit.Init(ctx, genkit.WithPlugins(&deepseek.DeepSeek{}))

	// Define a flow that generates a joke about a given topic. Thinking is on
	// by default for DeepSeek's V4 models, so the config turns it off for a
	// quick conversational answer.
	genkit.DefineFlow(g, "jokesFlow", func(ctx context.Context, topic string) (string, error) {
		if topic == "" {
			topic = "airplane food"
		}

		return genkit.GenerateText(ctx, g,
			ai.WithModel(deepseek.ModelRef("deepseek-v4-flash", &deepseek.ChatConfig{
				Thinking: &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeDisabled},
			})),
			ai.WithPrompt("Share a joke about %s.", topic),
		)
	})

	// Optionally, start a web server to make the flow callable via HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
