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

// This sample demonstrates the DashScope plugin for Alibaba Cloud's Qwen
// models: a streaming flow that generates a joke with a model pinned through
// dashscope.ModelRef and its typed config.
//
// Run it:
//
//	export DASHSCOPE_API_KEY=...
//	go run .
//
// Or with the Dev UI, to call the flow from a browser and read a trace of
// every run at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Streaming needs ?stream=true:
//
//	curl -N -X POST 'http://localhost:8080/jokesFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"topic": "bananas"}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/openai/openai-go"
)

// JokeRequest is what the flow takes. A struct rather than a bare string lets
// the field carry a description and a default, which the Dev UI pre-fills its
// form from. The default is not applied in transit, and a field without
// omitempty is required.
type JokeRequest struct {
	Topic string `json:"topic" jsonschema:"default=airplane food" jsonschema_description:"What the joke should be about"`
}

// model pins the model and its config in one place, so switching either is a
// one-line change. The config turns on DashScope's thinking mode with a budget
// on its length.
var model = dashscope.ModelRef("qwen-plus", &dashscope.ChatConfig{
	EnableThinking: openai.Ptr(true),
	ThinkingBudget: openai.Ptr(2048),
})

func main() {
	ctx := context.Background()

	// The plugin reads the API key from the DASHSCOPE_API_KEY environment variable.
	g := genkit.Init(ctx, genkit.WithPlugins(&dashscope.DashScope{}))

	// Passing sendChunk straight to WithStreaming forwards the model's chunks
	// to the caller untouched.
	genkit.DefineStreamingFlow(g, "jokesFlow",
		func(ctx context.Context, input JokeRequest, sendChunk ai.ModelStreamCallback) (string, error) {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(model),
				ai.WithPrompt("Share a joke about %s.", input.Topic),
				ai.WithStreaming(sendChunk),
			)
			if err != nil {
				return "", fmt.Errorf("could not generate joke: %w", err)
			}

			return resp.Text(), nil
		},
	)

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
