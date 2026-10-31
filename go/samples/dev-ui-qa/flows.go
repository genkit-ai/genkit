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

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

// ticketInput is the structured input of the flows that exercise the schema
// form of the Dev UI runner.
type ticketInput struct {
	Subject string   `json:"subject" jsonschema:"description=short summary of the request"`
	Urgency int      `json:"urgency" jsonschema:"description=1 (low) to 5 (page someone),minimum=1,maximum=5"`
	Tags    []string `json:"tags,omitempty" jsonschema:"description=optional labels"`
}

// ticketOutput is the structured output rendered in the runner result pane.
type ticketOutput struct {
	Queue    string   `json:"queue"`
	Priority string   `json:"priority"`
	Tags     []string `json:"tags"`
}

// progress is the chunk type of the streaming flow.
type progress struct {
	Step  int    `json:"step"`
	Total int    `json:"total"`
	Note  string `json:"note"`
}

// weatherInput is the input of the tool the model calls in the tool-loop flow.
type weatherInput struct {
	City string `json:"city" jsonschema:"description=city to look up"`
}

// approvalInterrupt is the typed interrupt metadata surfaced to the caller
// when the spend tool needs a human decision.
type approvalInterrupt struct {
	Reason string  `json:"reason"`
	Amount float64 `json:"amount"`
}

type spendInput struct {
	Amount float64 `json:"amount" jsonschema:"description=amount in dollars"`
}

// resumeInput carries the human decision back into a second run of the flow.
// The Dev UI has no native "resume" control, so the decision has to travel as
// flow input; leaving it empty reproduces the paused state.
type resumeInput struct {
	Amount   float64 `json:"amount" jsonschema:"description=amount in dollars"`
	Decision string  `json:"decision" jsonschema:"description=empty to pause; approve to restart the tool; deny to answer for it"`
}

// registerFlowAndToolCases covers the "Flows and tools" audit section:
// streaming chunk rendering, classified provider errors on an error flow
// (ccfe1093d overhaul), interrupt/resume from the UI, multi-turn code
// execution custom parts (GGA-4), and a tool name containing "/" (GGA-25).
// Also home for the trace cases of "Traces and logs": tool-loop spans,
// thinking parts, raw request/response visibility (GGA-49).
func registerFlowAndToolCases(g *genkit.Genkit) {
	// Structured in/out: does the runner render a schema form and a typed
	// result, or fall back to a raw JSON textarea?
	genkit.DefineFlow(g, "triageTicket", func(ctx context.Context, in ticketInput) (ticketOutput, error) {
		queue := "support"
		if in.Urgency >= 4 {
			queue = "oncall"
		}
		return ticketOutput{
			Queue:    queue,
			Priority: fmt.Sprintf("P%d", 6-in.Urgency),
			Tags:     append([]string{"triaged"}, in.Tags...),
		}, nil
	})

	// Streaming chunks: the UI should render each chunk as it arrives and
	// then replace them with the final output.
	genkit.DefineStreamingFlow(g, "streamProgress", func(ctx context.Context, steps int, stream core.StreamCallback[progress]) (string, error) {
		if steps <= 0 {
			steps = 5
		}
		for i := 1; i <= steps; i++ {
			if stream != nil {
				if err := stream(ctx, progress{Step: i, Total: steps, Note: fmt.Sprintf("working on step %d", i)}); err != nil {
					return "", err
				}
			}
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(300 * time.Millisecond):
			}
		}
		return fmt.Sprintf("completed %d steps", steps), nil
	})

	// Error classification: a flow that fails with a typed status the UI is
	// expected to surface as more than "500 Internal Server Error".
	genkit.DefineFlow(g, "failingFlow", func(ctx context.Context, mode string) (string, error) {
		switch mode {
		case "invalid":
			return "", core.NewError(core.INVALID_ARGUMENT, "mode %q is rejected on purpose", mode)
		case "unauthenticated":
			return "", core.NewPublicError(core.UNAUTHENTICATED, "no credentials for the downstream provider", map[string]any{"provider": "fake"})
		case "plain":
			return "", fmt.Errorf("plain go error with no status")
		default:
			return "", core.NewError(core.FAILED_PRECONDITION, "unknown mode %q; try invalid|unauthenticated|plain", mode)
		}
	})

	// Standalone tools. They are runnable on their own in the Dev UI action
	// list and are also reachable through the model in toolLoop below.
	lookupWeather := genkit.DefineTool(g, "lookupWeather", "Looks up the current weather for a city.",
		func(ctx *ai.ToolContext, in weatherInput) (string, error) {
			return fmt.Sprintf("It is 22C and sunny in %s.", in.City), nil
		})

	// GGA-25: a tool whose name contains "/" - the same separator the action
	// keys use, so a naive parse of "/tool/<name>" splits in the wrong place.
	// It is deliberately not attached to toolLoop: googlegenai rejects "/" in
	// a tool name before the request leaves the process, which would mask the
	// tool-loop case behind that failure.
	mathAdd := genkit.DefineTool(g, "math/add", "Adds two numbers.",
		func(ctx *ai.ToolContext, in struct {
			A float64 `json:"a"`
			B float64 `json:"b"`
		}) (float64, error) {
			return in.A + in.B, nil
		})

	// Interrupt/resume: the tool pauses for approval above a threshold.
	spend := genkit.DefineTool(g, "spendBudget", "Spends part of the team budget. Requires approval over $100.",
		func(ctx *ai.ToolContext, in spendInput) (string, error) {
			if !ctx.IsResumed() && in.Amount > 100 {
				return "", ai.InterruptWith(ctx, approvalInterrupt{Reason: "approval_required", Amount: in.Amount})
			}
			return fmt.Sprintf("spent $%.2f", in.Amount), nil
		})

	if !modelBackendAvailable() {
		return
	}

	// Tool loop: one model turn that calls a tool and then answers. The trace
	// should show nested generate + tool spans.
	genkit.DefineFlow(g, "toolLoop", func(ctx context.Context, city string) (string, error) {
		if city == "" {
			city = "Lagos"
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName(qaModelName),
			ai.WithSystem("Answer using the tools you are given. Keep answers to one sentence."),
			ai.WithPrompt("What is the weather in %s, and in Nairobi?", city),
			ai.WithTools(lookupWeather),
		)
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	// A real provider failure (not a locally constructed one): a model ID the
	// backend resolves dynamically and the provider then rejects. This is the
	// surface the ccfe1093d error overhaul classifies.
	genkit.DefineFlow(g, "providerError", func(ctx context.Context, model string) (string, error) {
		if model == "" {
			model = "googleai/gemini-does-not-exist"
		}
		resp, err := genkit.Generate(ctx, g, ai.WithModelName(model), ai.WithPrompt("hello"))
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	// Interrupt then resume. First run with an empty decision to see the
	// paused state, then re-run with approve/deny to drive the resume.
	genkit.DefineFlow(g, "resumeSpend", func(ctx context.Context, in resumeInput) (string, error) {
		if in.Amount == 0 {
			in.Amount = 250
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName(qaModelName),
			ai.WithSystem("Call the spendBudget tool for the requested amount. Do not ask the user to confirm first; the tool itself handles approval."),
			ai.WithPrompt("Please spend $%.2f on laptops.", in.Amount),
			ai.WithTools(spend),
			ai.WithToolChoice(ai.ToolChoiceRequired),
		)
		if err != nil {
			return "", err
		}

		for resp.FinishReason == ai.FinishReasonInterrupted {
			if in.Decision == "" {
				return fmt.Sprintf("paused: %d interrupt(s) awaiting a decision; re-run with decision=approve or decision=deny", len(resp.Interrupts())), nil
			}
			var restarts, responses []*ai.Part
			for _, interrupt := range resp.Interrupts() {
				switch in.Decision {
				case "approve":
					part, err := spend.RestartWith(interrupt)
					if err != nil {
						return "", fmt.Errorf("RestartWith: %w", err)
					}
					restarts = append(restarts, part)
				default:
					part, err := spend.RespondWith(interrupt, "denied by reviewer")
					if err != nil {
						return "", fmt.Errorf("RespondWith: %w", err)
					}
					responses = append(responses, part)
				}
			}
			resp, err = genkit.Generate(ctx, g,
				ai.WithModelName(qaModelName),
				ai.WithMessages(resp.History()...),
				ai.WithTools(spend),
				ai.WithToolRestarts(restarts...),
				ai.WithToolResponses(responses...),
			)
			if err != nil {
				return "", err
			}
		}
		return resp.Text(), nil
	})

	// GGA-25 as a model call: attach the slash-named tool to a real generate
	// and see which side rejects it. Input picks the backend.
	genkit.DefineFlow(g, "slashToolLoop", func(ctx context.Context, backend string) (string, error) {
		model := qaModelName
		if backend == "googleai" {
			model = "googleai/gemini-flash-latest"
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName(model),
			ai.WithPrompt("Add 2 and 40 using the math/add tool."),
			ai.WithTools(mathAdd),
		)
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	// Interrupted generation: returns while the tool call is pending so the
	// UI can show an interrupted finish reason and the interrupt metadata.
	genkit.DefineFlow(g, "interruptedSpend", func(ctx context.Context, amount float64) (string, error) {
		if amount == 0 {
			amount = 250
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName(qaModelName),
			ai.WithSystem("Use the spendBudget tool for any spending request."),
			ai.WithPrompt("Please spend $%.2f on laptops.", amount),
			ai.WithTools(spend),
		)
		if err != nil {
			return "", err
		}
		if resp.FinishReason != ai.FinishReasonInterrupted {
			return fmt.Sprintf("not interrupted: %s", resp.Text()), nil
		}
		for _, p := range resp.Interrupts() {
			if meta, ok := ai.InterruptAs[approvalInterrupt](p); ok {
				return fmt.Sprintf("interrupted: %s for $%.2f", meta.Reason, meta.Amount), nil
			}
		}
		return "interrupted with no typed metadata", nil
	})

	// GGA-4: multi-turn code execution. Turn one runs code; turn two replays
	// the history, which now carries the custom executableCode and
	// codeExecutionResult parts.
	if os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "" {
		genkit.DefineFlow(g, "codeExecTwoTurn", func(ctx context.Context, _ struct{}) (string, error) {
			cfg := &genai.GenerateContentConfig{
				Tools: []*genai.Tool{{CodeExecution: &genai.ToolCodeExecution{}}},
			}
			first, err := genkit.Generate(ctx, g,
				ai.WithModelName("googleai/gemini-flash-latest"),
				ai.WithConfig(cfg),
				ai.WithPrompt("Find the sum of the first 5 prime numbers by running code."),
			)
			if err != nil {
				return "", fmt.Errorf("turn 1: %w", err)
			}
			code := googlegenai.GetExecutableCode(first.Message)
			result := googlegenai.GetCodeExecutionResult(first.Message)

			second, err := genkit.Generate(ctx, g,
				ai.WithModelName("googleai/gemini-flash-latest"),
				ai.WithConfig(cfg),
				ai.WithMessages(first.History()...),
				ai.WithPrompt("Now do the same for the first 10 prime numbers."),
			)
			if err != nil {
				return "", fmt.Errorf("turn 2 (turn 1 ran %d chars of %s code, outcome %s): %w",
					len(code.Code), code.Language, result.Outcome, err)
			}
			return fmt.Sprintf("turn 1: %s\n\nturn 2: %s", first.Text(), second.Text()), nil
		})
	}
}

// qaModelName is the model the tool-loop cases generate with. Anthropic is the
// default because the googleai catalog retires model IDs out from under a key
// (gemini-2.5-flash now 404s), and a QA app should fail on the case under test,
// not on model availability.
const qaModelName = "anthropic/claude-opus-5"

// modelBackendAvailable reports whether the plugin behind qaModelName was
// initialized in main; the tool-loop flows are skipped without it rather than
// registering flows that fail on every run.
func modelBackendAvailable() bool {
	return os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("ANTHROPIC_AUTH_TOKEN") != ""
}
