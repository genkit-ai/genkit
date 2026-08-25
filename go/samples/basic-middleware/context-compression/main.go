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

// This sample demonstrates the ContextCompression middleware on a research
// loop whose tools return deliberately verbose output, so the context grows
// past its budget within a few turns.
//
// The middleware compresses only the view each model call sends to the
// provider; the conversation history the caller gets back is never replaced.
// What happened is recorded as message metadata under
// middleware.CompressionMetadataKey:
//
//   - every model message carries {"inputTokens": N}, the provider-reported
//     context size of the call that produced it, and
//   - the last message covered by a compaction carries the summary and stats,
//     so a chat client can render "history was compacted here" in place while
//     still showing every original message.
//
// The flow below returns that evidence alongside the answer: the per-turn
// token counts and each compaction stamp, read straight from the history.
// In a Dev UI trace the outer generate spans show the full history while the
// model spans show the compacted view that was actually sent — including the
// synthetic "[Previous conversation summary ...]" user message.
//
// The MaxInputTokens budget here is unrealistically small (2000) so a
// compaction reliably fires within one request; real applications set it
// near their model's practical context budget.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to inspect the compacted model requests at
// http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP:
//
//	curl -X POST 'http://localhost:8080/researchFlow' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"topic": "quantum error correction"}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/middleware"
	"github.com/firebase/genkit/go/plugins/server"
)

// TopicRequest is what the flow takes. The field carries a description and a
// default, which the Dev UI pre-fills its form from.
type TopicRequest struct {
	Topic string `json:"topic" jsonschema:"default=quantum error correction" jsonschema_description:"What to research"`
}

// ResearchResult pairs the answer with the compression evidence read back
// from the history, so the Dev UI shows what the middleware did.
type ResearchResult struct {
	Answer string `json:"answer"`
	// Messages is the total history length; every message survives
	// compression.
	Messages int `json:"messages"`
	// InputTokensPerTurn is the provider-reported context size of each model
	// call, read from the per-message metadata stamps.
	InputTokensPerTurn []int `json:"inputTokensPerTurn"`
	// Compactions holds each compaction stamp's stats, in history order.
	Compactions []map[string]any `json:"compactions,omitempty"`
}

func main() {
	ctx := context.Background()

	// Registering the Middleware plugin exposes the built-in middleware
	// (ContextCompression, Retry, Fallback, ...) to the Dev UI.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &middleware.Middleware{}))

	// Both tools return far more text than any answer needs, standing in for
	// real search APIs and document fetches.
	search := genkit.DefineTool(g, "search", "Search for information on a topic",
		func(ctx *ai.ToolContext, input struct {
			Query string `json:"query" jsonschema_description:"The search query"`
		}) (string, error) {
			return fmt.Sprintf("Search results for %q:\n%s", input.Query,
				strings.Repeat("A promising direction discussed at length in the literature. ", 120)), nil
		})
	readDocument := genkit.DefineTool(g, "readDocument", "Read the contents of a document",
		func(ctx *ai.ToolContext, input struct {
			URL string `json:"url" jsonschema_description:"The document to read"`
		}) (string, error) {
			return fmt.Sprintf("Contents of %s:\n%s", input.URL,
				strings.Repeat("Detailed technical content with many specifics. ", 150)), nil
		})

	genkit.DefineFlow(g, "researchFlow",
		func(ctx context.Context, input TopicRequest) (*ResearchResult, error) {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModelName("googleai/gemini-3.6-flash"),
				ai.WithPrompt("Research %s: search for at least three angles, read the most "+
					"promising documents, then give a concise synthesis.", input.Topic),
				ai.WithTools(search, readDocument),
				ai.WithMaxTurns(20),
				ai.WithUse(&middleware.ContextCompression{
					// Unrealistically small, so a compaction fires in this demo.
					MaxInputTokens:        2000,
					DedupeToolResponses:   &middleware.CompressionDedupe{},
					TruncateToolResponses: &middleware.CompressionToolTruncation{MaxChars: 2000},
					Summarizer: &middleware.CompressionSummarizer{
						// Usually a cheaper model than the primary one; the
						// same model keeps this sample to one dependency.
						Model: googlegenai.ModelRef("googleai/gemini-3.6-flash", nil),
					},
				}),
			)
			if err != nil {
				return nil, fmt.Errorf("research failed: %w", err)
			}

			// The full history is intact; the compression record lives in
			// message metadata. A chat client reads the same stamps to place
			// "history was compacted here" markers.
			result := &ResearchResult{Answer: resp.Text()}
			for _, msg := range resp.History() {
				result.Messages++
				stamp, _ := msg.Metadata[middleware.CompressionMetadataKey].(map[string]any)
				if stamp == nil {
					continue
				}
				if tokens, ok := stamp["inputTokens"].(int); ok {
					result.InputTokensPerTurn = append(result.InputTokensPerTurn, tokens)
				}
				if stats, ok := stamp["stats"].(map[string]any); ok {
					result.Compactions = append(result.Compactions, stats)
				}
			}
			return result, nil
		})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
