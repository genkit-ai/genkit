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

// This sample meters model usage per tenant with an observer.
//
// A WrapModel hook sees only the model calls of the Generate it is attached
// to. Here the answering model calls a tool that asks a second, cheaper model,
// and that call happens in a Generate of its own. An aix.Observer added to the
// context with tracing.WithInstrumentation sees both, so the meter bills each
// model at its own rate:
//
//	ctx = tracing.WithInstrumentation(ctx, aix.Observer{ModelDone: ...})
//
// Run it:
//
//	go run .
//
// Or with the Dev UI at http://localhost:4000:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP, then ask for the tenant's running total:
//
//	curl -X POST http://localhost:8080/answerFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"tenant": "acme", "question": "Why is the sky blue?"}}'
//
//	curl -X POST http://localhost:8080/usageFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "acme"}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
)

// QuestionRequest is what answerFlow takes. The fields carry descriptions and
// defaults, which the Dev UI pre-fills its form from.
type QuestionRequest struct {
	Tenant   string `json:"tenant" jsonschema:"default=acme" jsonschema_description:"Who to bill"`
	Question string `json:"question" jsonschema:"default=Why is the sky blue?" jsonschema_description:"What to answer"`
}

// Meter holds each tenant's usage, by model. Observer hooks may run
// concurrently, so it is guarded.
type Meter struct {
	mu    sync.Mutex
	usage map[string]map[string]*ai.GenerationUsage
}

// Add records one model call's usage against tenant.
func (m *Meter) Add(tenant, model string, u *ai.GenerationUsage) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.usage[tenant] == nil {
		m.usage[tenant] = map[string]*ai.GenerationUsage{}
	}
	m.usage[tenant][model] = ai.SumUsage(m.usage[tenant][model], u)
}

// Tokens returns tenant's total tokens, by model.
func (m *Meter) Tokens(tenant string) map[string]int {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := map[string]int{}
	for model, u := range m.usage[tenant] {
		out[model] = u.TotalTokens
	}
	return out
}

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
	meter := &Meter{usage: map[string]map[string]*ai.GenerationUsage{}}

	// The tool's Generate is a separate call, which middleware on the outer
	// Generate does not see.
	factCheck := genkit.DefineTool(g, "factCheck", "Checks a claim and says whether it holds.",
		func(ctx *ai.ToolContext, in struct {
			Claim string `json:"claim"`
		}) (string, error) {
			return genkit.GenerateText(ctx, g,
				ai.WithModelName("googleai/gemini-flash-lite-latest"),
				ai.WithPrompt("In one sentence, is this true? %s", in.Claim),
			)
		})

	genkit.DefineFlow(g, "answerFlow", func(ctx context.Context, in QuestionRequest) (string, error) {
		// Every model call under ctx reaches the observer, the tool's
		// included, and failed calls too.
		ctx = tracing.WithInstrumentation(ctx, aix.Observer{
			ModelDone: func(_ context.Context, call *aix.ModelCall) {
				if u := call.Response.Usage; u != nil {
					meter.Add(in.Tenant, call.Model, u)
				}
			},
		})
		answer, err := genkit.GenerateText(ctx, g,
			ai.WithModelName("googleai/gemini-flash-latest"),
			ai.WithPrompt("Answer in two sentences, fact-checking your main claim first: %s", in.Question),
			ai.WithTools(factCheck),
		)
		if err != nil {
			return "", fmt.Errorf("could not answer: %w", err)
		}
		return answer, nil
	})

	genkit.DefineFlow(g, "usageFlow", func(ctx context.Context, tenant string) (map[string]int, error) {
		return meter.Tokens(tenant), nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	if err := server.Start(ctx, "127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}
