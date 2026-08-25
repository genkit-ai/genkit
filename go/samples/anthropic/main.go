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

// This sample demonstrates the Anthropic plugin, which speaks Anthropic's
// native Messages API: a streaming flow that generates a joke with a model
// pinned through anthropic.ModelRef and the Anthropic SDK's own request type
// as its config.
//
// compat_oai/anthropic reaches the same Claude models through the
// OpenAI-compatible endpoint instead. What differs is the response: this
// plugin returns Claude's thinking as Genkit reasoning parts with their
// signatures preserved, while the compatible endpoint keeps the thinking
// server-side.
//
// Run it:
//
//	export ANTHROPIC_API_KEY=...
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

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/anthropic"
	"github.com/firebase/genkit/go/plugins/server"
)

// JokeRequest is what the flow takes. A struct rather than a bare string lets
// the field carry a description and a default, which the Dev UI pre-fills its
// form from. The default is not applied in transit, and a field without
// omitempty is required.
type JokeRequest struct {
	Topic string `json:"topic" jsonschema:"default=airplane food" jsonschema_description:"What the joke should be about"`
}

// model pins the model and its config in one place, so switching either is a
// one-line change. Sonnet 5 thinks adaptively, choosing its own budget per
// request, so a fixed Thinking.OfEnabled budget is rejected outright; Effort is
// the knob instead, and low keeps a quick joke quick.
//
// This package and the Anthropic SDK are both named anthropic, so one of them
// needs an import alias; the import above aliases the SDK to sdk.
var model = anthropic.ModelRef("claude-sonnet-5", &sdk.MessageNewParams{
	MaxTokens: 4000,
	Thinking: sdk.ThinkingConfigParamUnion{
		OfAdaptive: &sdk.ThinkingConfigAdaptiveParam{},
	},
	OutputConfig: sdk.OutputConfigParam{Effort: sdk.OutputConfigEffortLow},
})

func main() {
	ctx := context.Background()

	// The plugin reads the API key from the ANTHROPIC_API_KEY environment variable.
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{}))

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
