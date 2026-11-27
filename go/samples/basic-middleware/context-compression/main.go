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

// This sample demonstrates the experimental ContextCompression middleware on
// a research loop that fills the context with verbose tool output.
//
// The model searches for reports and fetches each one in its own turn. The
// token budget is small on purpose, so compression starts partway through
// the loop: older report bodies are truncated, and once that is not enough,
// a smaller model summarizes the earlier turns. The model works from the
// compressed view, while the history the flow returns keeps every message in
// full, with the compression recorded in message metadata.
//
// The flow reports what was compressed next to the answer: the stats the
// middleware puts on the response, how many messages the history holds, and
// how many of them the model received on its last call.
//
// Run it (needs a Gemini API key in the environment):
//
//	go run .
//
// Or with the Dev UI, to watch the summarizer call nest inside the loop at
// http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP:
//
//	curl -X POST 'http://localhost:8080/researchFlow' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"project": "Project Alpha"}}'
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
	middlewarex "github.com/firebase/genkit/go/plugins/middleware/exp"
	"github.com/firebase/genkit/go/plugins/server"
)

// ResearchRequest is what the flow takes. The field carries a description and
// a default, which the Dev UI pre-fills its form from.
type ResearchRequest struct {
	Project string `json:"project" jsonschema:"default=Project Alpha" jsonschema_description:"The project to investigate"`
}

// ResearchResult is the answer and a record of what was compressed.
type ResearchResult struct {
	Answer string `json:"answer"`
	// Compression is the stats the middleware put on the response.
	Compression any `json:"compression,omitempty"`
	// HistoryMessages is how many messages the returned history holds.
	HistoryMessages int `json:"historyMessages"`
	// ModelMessages is how many messages the model received on its last call.
	ModelMessages int `json:"modelMessages"`
}

// Report is what fetchReport returns.
type Report struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	Details string `json:"details"`
}

// reports is the database the tools read.
var reports = map[string]Report{
	"report-101": {Title: "Performance Audit Q1", Details: "Audit across cluster us-central. Average latency was 240ms under 50k QPS. " +
		"Query optimization resolved 4 of 5 slow queries found during cache warmup. CPU stayed under 65% on every replica."},
	"report-102": {Title: "Security and Access Log", Details: "Quarterly access review. 14 service accounts audited and 2 stale credentials revoked. " +
		"Every API endpoint now enforces mTLS, and tokens rotate every hour. No intrusion attempts were detected."},
	"report-103": {Title: "Deployment Incident Post-Mortem", Details: "Root cause of the Feb 12 interruption: the canary failed to halt the rollout " +
		"because of an unhandled rejection in the health-check probe. Rollback took 14 minutes. The probe timeout dropped from 30s to 5s."},
}

func main() {
	ctx := context.Background()

	// Registering the experimental middleware plugin exposes ContextCompression
	// to the Dev UI.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &middlewarex.Middleware{}))

	searchReports := genkit.DefineTool(g, "searchReports", "Searches for the IDs of reports about a project.",
		func(ctx *ai.ToolContext, input struct {
			Project string `json:"project"`
		}) ([]string, error) {
			return []string{"report-101", "report-102", "report-103"}, nil
		})

	fetchReport := genkit.DefineTool(g, "fetchReport", "Fetches the full text of one report by ID.",
		func(ctx *ai.ToolContext, input struct {
			ReportID string `json:"reportId"`
		}) (Report, error) {
			r, ok := reports[input.ReportID]
			if !ok {
				return Report{}, fmt.Errorf("no report %q", input.ReportID)
			}
			r.ID = input.ReportID
			// Real reports come with appendices that bury the findings.
			r.Details += strings.Repeat(" Appendix: raw metrics, dashboards and on-call notes.", 30)
			return r, nil
		})

	genkit.DefineFlow(g, "researchFlow", func(ctx context.Context, input ResearchRequest) (*ResearchResult, error) {
		// lastView records what the model received on its latest call. Listed
		// after ContextCompression, it sees the compressed view.
		var lastView []*ai.Message
		recordView := ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
			return &ai.Hooks{WrapModel: func(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
				lastView = params.Request.Messages
				return next(ctx, params)
			}}, nil
		})

		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-flash-latest"),
			ai.WithSystem("You are an investigative research assistant. Search for reports first, "+
				"then fetch every report, one fetchReport call per turn, and finish with a short summary of the findings."),
			ai.WithPrompt("Investigate %s.", input.Project),
			ai.WithTools(searchReports, fetchReport),
			ai.WithMaxTurns(10),
			ai.WithUse(
				&middlewarex.ContextCompression{
					// Small on purpose, so compression starts partway through the
					// loop. A real budget sits well below the model's window.
					MaxInputTokens:      1000,
					DedupeToolResponses: &middlewarex.CompressionDedupe{},
					TruncateToolResponses: &middlewarex.CompressionToolTruncation{
						MaxChars:       300,
						PreserveRecent: 1,
					},
					Summarize: &middlewarex.CompressionSummarizer{
						Model:          googlegenai.ModelRef("googleai/gemini-flash-lite-latest", nil),
						PreserveRecent: 2,
					},
					SkipSummarizationThreshold: 0.3,
				},
				recordView,
				// Listed after ContextCompression, Retry resends the compressed
				// view when the model is briefly unavailable.
				&middleware.Retry{},
			),
		)
		if err != nil {
			return nil, fmt.Errorf("research failed: %w", err)
		}

		result := &ResearchResult{
			Answer:          resp.Text(),
			HistoryMessages: len(resp.History()),
			ModelMessages:   len(lastView),
		}
		if custom, ok := resp.Custom.(map[string]any); ok {
			result.Compression = custom["contextCompression"]
		}
		return result, nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	if err := server.Start(ctx, "127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}
