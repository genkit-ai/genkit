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

// This sample demonstrates constrained output with a *recursive* Go type: an
// org chart, where each employee has direct reports who are themselves
// employees. The Gemini plugin sends the output schema via
// GenerateContentConfig.ResponseJsonSchema, which expresses recursion as JSON
// Schema $ref/$defs (unrolled server-side), so self-referential structs
// round-trip correctly instead of collapsing to an "any" schema.
//
// To run:
//
//	go run .
//
// In another terminal, generate an org chart for a company:
//
//	curl -N -X POST http://localhost:8080/orgChartFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "a 20-person coffee roasting startup"}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// Employee is a self-referential type: each employee's DirectReports are
// themselves employees, forming an org chart of arbitrary depth. This recursion
// is the whole point of the sample — it is what the ResponseJsonSchema
// migration enables.
type Employee struct {
	Name          string      `json:"name" jsonschema:"description=The employee's full name"`
	Title         string      `json:"title" jsonschema:"description=The employee's job title"`
	DirectReports []*Employee `json:"directReports,omitempty" jsonschema:"description=Employees who report directly to this person"`
}

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. The API key is read from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
	//
	// To fall back to the older ResponseSchema field (no recursion support),
	// set the LegacyResponseSchema flag:
	//
	//	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{LegacyResponseSchema: true}))
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	DefineOrgChart(g)

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineOrgChart defines a flow that generates a company org chart from a short
// company description. The output type *Employee is recursive, so its inferred
// JSON schema uses $ref/$defs.
func DefineOrgChart(g *genkit.Genkit) {
	genkit.DefineFlow(g, "orgChartFlow",
		func(ctx context.Context, company string) (*Employee, error) {
			ceo, _, err := genkit.GenerateData[Employee](ctx, g,
				ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
					ThinkingConfig: &genai.ThinkingConfig{
						ThinkingBudget: genai.Ptr[int32](0),
					},
				})),
				ai.WithSystem("You design plausible company org charts. Start at the CEO and "+
					"nest direct reports two or three levels deep."),
				ai.WithPrompt("Build an org chart for: %s", company),
			)
			if err != nil {
				return nil, fmt.Errorf("could not generate org chart: %w", err)
			}
			return ceo, nil
		})
}
