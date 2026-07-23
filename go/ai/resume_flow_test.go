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
//
// SPDX-License-Identifier: Apache-2.0

// Interrupt/resume flow tests. These drive the generate loop with restart and
// respond parts built by the real constructors (the typed
// [ai.InterruptibleTool.Restart]/[ai.InterruptibleTool.Respond] methods and
// the type-erased [tool.Restart]/[tool.Respond]), covering
// the loop's consumption of the interrupt/resume wire format end to end. They
// live in an external test package because ai/tool imports ai, so in-package
// ai tests cannot use it.
package ai_test

import (
	"context"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

func newFlowRegistry(t *testing.T) api.Registry {
	t.Helper()
	r := registry.New()
	ai.ConfigureFormats(r)
	return r
}

// defineRoundTripModel defines a model that emits reqs (typically tool
// requests) on the first turn and the final text "done" once a tool response
// is in history, driving a single tool round per Generate.
func defineRoundTripModel(r api.Registry, name string, reqs ...*ai.Part) *ai.Model {
	return ai.DefineModel(r, name,
		&ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			for _, m := range req.Messages {
				if m.Role == ai.RoleTool {
					return &ai.ModelResponse{
						Request:      req,
						Message:      ai.NewModelTextMessage("done"),
						FinishReason: ai.FinishReasonStop,
					}, nil
				}
			}
			return &ai.ModelResponse{
				Request:      req,
				Message:      &ai.Message{Role: ai.RoleModel, Content: reqs},
				FinishReason: ai.FinishReasonStop,
			}, nil
		})
}

type transferIn struct {
	Amount float64 `json:"amount"`
}
type transferInterrupt struct {
	Reason string  `json:"reason"`
	Amount float64 `json:"amount"`
}
type confirmation struct {
	Approved bool `json:"approved"`
}

// TestInterruptResume_TypedRoundTrip pins the core interrupt/resume contract:
// the tool interrupts with typed data on the first pass, the caller reads it
// with tool.InterruptData and restarts with typed data via the Restart method,
// and the
// resumed value reaches the function's *Res parameter on re-execution.
func TestInterruptResume_TypedRoundTrip(t *testing.T) {
	r := newFlowRegistry(t)
	model := defineRoundTripModel(r, "test/model", ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "transfer", Input: map[string]any{"amount": 200},
	}))

	var gotResume *confirmation
	transfer := ai.DefineInterruptibleTool(r, "transfer", "transfers money",
		func(ctx context.Context, in transferIn, res *confirmation) (string, error) {
			if res == nil {
				return "", tool.Interrupt(transferInterrupt{Reason: "large_amount", Amount: in.Amount})
			}
			gotResume = res
			if !res.Approved {
				return "cancelled", nil
			}
			return "completed", nil
		})

	resp, err := ai.Generate(context.Background(), r,
		ai.WithModel(model),
		ai.WithPrompt("transfer 200"),
		ai.WithTools(transfer))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %d (finish=%s)", len(interrupts), resp.FinishReason)
	}

	meta, ok := tool.InterruptData[transferInterrupt](interrupts[0])
	if !ok {
		t.Fatal("failed to decode the typed interrupt data")
	}
	if meta.Reason != "large_amount" || meta.Amount != 200 {
		t.Errorf("interrupt data = %+v, want {large_amount 200}", meta)
	}

	restart, err := transfer.Restart(interrupts[0], transfer.WithResume(confirmation{Approved: true}))
	if err != nil {
		t.Fatalf("Restart: %v", err)
	}

	resp2, err := ai.Generate(context.Background(), r,
		ai.WithModel(model),
		ai.WithMessages(resp.History()...),
		ai.WithTools(transfer),
		ai.WithToolRestarts(restart))
	if err != nil {
		t.Fatalf("resume Generate: %v", err)
	}
	if gotResume == nil || !gotResume.Approved {
		t.Errorf("resumed tool saw %+v, want Approved=true", gotResume)
	}
	if got := resp2.Text(); got != "done" {
		t.Errorf("final text after resume = %q, want %q", got, "done")
	}
}

// TestInterruptResume_ReplaceInput covers input replacement on resume: the
// tool re-executes with the new input and can read the original input via
// tool.OriginalInput.
func TestInterruptResume_ReplaceInput(t *testing.T) {
	r := newFlowRegistry(t)
	model := defineRoundTripModel(r, "test/model", ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "transfer", Input: map[string]any{"amount": 200},
	}))

	var gotAmount, gotOriginal float64
	transfer := ai.DefineInterruptibleTool(r, "transfer", "transfers money",
		func(ctx context.Context, in transferIn, res *confirmation) (string, error) {
			if res == nil {
				return "", tool.Interrupt(transferInterrupt{Reason: "too_much", Amount: in.Amount})
			}
			gotAmount = in.Amount
			if orig, ok := tool.OriginalInput[transferIn](ctx); ok {
				gotOriginal = orig.Amount
			}
			return "completed", nil
		})

	resp, err := ai.Generate(context.Background(), r,
		ai.WithModel(model),
		ai.WithPrompt("transfer 200"),
		ai.WithTools(transfer))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %d", len(interrupts))
	}

	restart, err := transfer.Restart(interrupts[0],
		transfer.WithResume(confirmation{Approved: true}),
		transfer.WithNewInput(transferIn{Amount: 50}))
	if err != nil {
		t.Fatalf("Restart: %v", err)
	}

	if _, err = ai.Generate(context.Background(), r,
		ai.WithModel(model),
		ai.WithMessages(resp.History()...),
		ai.WithTools(transfer),
		ai.WithToolRestarts(restart)); err != nil {
		t.Fatalf("resume Generate: %v", err)
	}
	if gotAmount != 50 {
		t.Errorf("resumed tool saw amount %v, want 50 (replaced)", gotAmount)
	}
	if gotOriginal != 200 {
		t.Errorf("original input amount = %v, want 200", gotOriginal)
	}
}

// TestInterruptRespond_ResolvesWithoutReexecution covers the respond path: a
// pre-computed output resolves the interrupt and the tool does not run again.
func TestInterruptRespond_ResolvesWithoutReexecution(t *testing.T) {
	r := newFlowRegistry(t)
	model := defineRoundTripModel(r, "test/model", ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "transfer", Input: map[string]any{"amount": 200},
	}))

	calls := 0
	transfer := ai.DefineInterruptibleTool(r, "transfer", "transfers money",
		func(ctx context.Context, in transferIn, res *confirmation) (string, error) {
			calls++
			return "", tool.Interrupt(transferInterrupt{Reason: "confirm", Amount: in.Amount})
		})

	resp, err := ai.Generate(context.Background(), r,
		ai.WithModel(model),
		ai.WithPrompt("transfer 200"),
		ai.WithTools(transfer))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %d", len(interrupts))
	}

	respond, err := transfer.Respond(interrupts[0], "cancelled")
	if err != nil {
		t.Fatalf("Respond: %v", err)
	}

	resp2, err := ai.Generate(context.Background(), r,
		ai.WithModel(model),
		ai.WithMessages(resp.History()...),
		ai.WithTools(transfer),
		ai.WithToolResponses(respond))
	if err != nil {
		t.Fatalf("respond Generate: %v", err)
	}
	if calls != 1 {
		t.Errorf("tool ran %d times, want 1 (Respond must not re-execute)", calls)
	}
	if got := resp2.Text(); got != "done" {
		t.Errorf("final text = %q, want %q", got, "done")
	}
}

// TestInterruptResume_MixedToolRequests covers a turn with two tool requests
// where only one interrupts: the loop must pair the caller's respond/restart
// directive with the interrupted request and settle the other on resume.
func TestInterruptResume_MixedToolRequests(t *testing.T) {
	type conditionalIn struct {
		Value     string
		Interrupt bool
	}
	type resumableIn struct {
		Action string
		Data   string
	}
	type restartMeta struct {
		Data   string `json:"data"`
		Source string `json:"source,omitempty"`
	}
	type interruptMeta struct {
		Reason string `json:"reason"`
		Value  string `json:"value"`
	}

	r := newFlowRegistry(t)

	conditionalTool := ai.DefineInterruptibleTool(r, "conditional", "tool that may interrupt based on input",
		func(ctx context.Context, input conditionalIn, res *restartMeta) (string, error) {
			if input.Interrupt {
				return "", tool.Interrupt(interruptMeta{Reason: "user_intervention_required", Value: input.Value})
			}
			return "processed: " + input.Value, nil
		})

	resumableTool := ai.DefineInterruptibleTool(r, "resumable", "tool that can be resumed",
		func(ctx context.Context, input resumableIn, res *restartMeta) (string, error) {
			if res != nil {
				return "resumed with: " + res.Data + ", original: " + input.Data, nil
			}
			return "first run: " + input.Data, nil
		})

	// Always emits both tool requests; the resume turns use finishModel below.
	toolModel := ai.DefineModel(r, "test/toolmodel",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Request: req,
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewTextPart("I need to use some tools."),
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "conditional",
							Ref:   "tool1",
							Input: map[string]any{"Value": "test_data", "Interrupt": true},
						}),
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "resumable",
							Ref:   "tool2",
							Input: map[string]any{"Action": "process", "Data": "initial_data"},
						}),
					},
				},
				FinishReason: ai.FinishReasonStop,
			}, nil
		})

	finishModel := ai.DefineModel(r, "test/finish",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Request:      req,
				Message:      ai.NewModelTextMessage("done"),
				FinishReason: ai.FinishReasonStop,
			}, nil
		})

	interruptedTurn := func(t *testing.T) *ai.ModelResponse {
		t.Helper()
		res, err := ai.Generate(context.Background(), r,
			ai.WithModel(toolModel),
			ai.WithPrompt("use tools"),
			ai.WithTools(conditionalTool, resumableTool))
		if err != nil {
			t.Fatal(err)
		}
		if res.FinishReason != ai.FinishReasonInterrupted {
			t.Fatalf("expected finish reason 'interrupted', got %q", res.FinishReason)
		}
		if len(res.Message.Content) != 3 {
			t.Fatalf("expected 3 content parts, got %d", len(res.Message.Content))
		}
		return res
	}

	t.Run("interrupt surfaces with typed data", func(t *testing.T) {
		res := interruptedTurn(t)
		interrupted := res.Message.Content[1]
		if !interrupted.IsInterrupt() {
			t.Fatal("expected second part to be an interrupted tool request")
		}
		meta, ok := tool.InterruptData[interruptMeta](interrupted)
		if !ok {
			t.Fatal("failed to decode interrupt metadata")
		}
		if meta.Reason != "user_intervention_required" {
			t.Errorf("interrupt reason = %q, want 'user_intervention_required'", meta.Reason)
		}
	})

	t.Run("respond directive resolves the interrupted request", func(t *testing.T) {
		res := interruptedTurn(t)
		respond, err := tool.Respond(res.Message.Content[1], "user_provided_response")
		if err != nil {
			t.Fatalf("Respond: %v", err)
		}

		resumeRes, err := ai.Generate(context.Background(), r,
			ai.WithModel(finishModel),
			ai.WithMessages(res.History()...),
			ai.WithTools(conditionalTool, resumableTool),
			ai.WithToolResponses(respond))
		if err != nil {
			t.Fatal(err)
		}
		if resumeRes.FinishReason == ai.FinishReasonInterrupted {
			t.Error("expected generation to not be interrupted after responding")
		}
	})

	t.Run("restart directive re-executes the interrupted request", func(t *testing.T) {
		res := interruptedTurn(t)
		restart, err := tool.Restart(res.Message.Content[1],
			ai.WithResume(restartMeta{Data: "restart_context"}),
			ai.WithNewInput(conditionalIn{Value: "restarted_data", Interrupt: false}))
		if err != nil {
			t.Fatalf("Restart: %v", err)
		}

		resumeRes, err := ai.Generate(context.Background(), r,
			ai.WithModel(finishModel),
			ai.WithMessages(res.History()...),
			ai.WithTools(conditionalTool, resumableTool),
			ai.WithToolRestarts(restart))
		if err != nil {
			t.Fatal(err)
		}
		if resumeRes.FinishReason == ai.FinishReasonInterrupted {
			t.Error("expected generation to not be interrupted after restarting")
		}
	})
}

// TestMiddlewareHookOrderOnToolRestart pins hook ordering on the restart path:
// resume handling lives inside the outer generate span, so the restarted tool
// fires before the recursive follow-up iteration's generate+model.
func TestMiddlewareHookOrderOnToolRestart(t *testing.T) {
	r := newFlowRegistry(t)

	type restartInput struct {
		Interrupt bool `json:"interrupt"`
	}

	restartable := ai.DefineTool(r, "restartable", "interrupts, then runs on resume",
		func(ctx context.Context, in restartInput) (string, error) {
			if in.Interrupt {
				return "", tool.Interrupt(nil)
			}
			return "ok", nil
		})

	model := defineRoundTripModel(r, "test/restartModel", ai.NewToolRequestPart(&ai.ToolRequest{
		Name:  "restartable",
		Ref:   "t1",
		Input: map[string]any{"interrupt": true},
	}))

	first, err := ai.Generate(context.Background(), r,
		ai.WithModel(model), ai.WithPrompt("go"), ai.WithTools(restartable))
	if err != nil {
		t.Fatal(err)
	}
	if first.FinishReason != ai.FinishReasonInterrupted {
		t.Fatalf("expected FinishReason=interrupted, got %q", first.FinishReason)
	}
	interruptedPart := first.Message.Content[0]

	var mu sync.Mutex
	var order []string
	record := func(s string) {
		mu.Lock()
		defer mu.Unlock()
		order = append(order, s)
	}

	tracker := ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapGenerate: func(ctx context.Context, p *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
				record("generate")
				return next(ctx, p)
			},
			WrapModel: func(ctx context.Context, p *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
				record("model")
				return next(ctx, p)
			},
			WrapTool: func(ctx context.Context, p *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
				record("tool")
				return next(ctx, p)
			},
		}, nil
	})

	restart, err := tool.Restart(interruptedPart,
		ai.WithNewInput(restartInput{Interrupt: false}))
	if err != nil {
		t.Fatalf("Restart: %v", err)
	}

	resumed, err := ai.Generate(context.Background(), r,
		ai.WithModel(model),
		ai.WithMessages(first.History()...),
		ai.WithTools(restartable),
		ai.WithToolRestarts(restart),
		ai.WithUse(tracker),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.FinishReason == ai.FinishReasonInterrupted {
		t.Fatal("expected completion after restart, got interrupted")
	}

	want := []string{"generate", "tool", "generate", "model"}
	if len(order) != len(want) {
		t.Fatalf("hook order: got %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Errorf("order[%d] = %q, want %q", i, order[i], want[i])
		}
	}
}

// TestInterruptibleToolMethods_ValidateOwnership checks that the typed Restart
// and Respond methods reject an interrupt part belonging to a different tool,
// and that a Restart with no options produces a bare restart (omitting
// WithResume means restarting without data).
func TestInterruptibleToolMethods_ValidateOwnership(t *testing.T) {
	mine := ai.NewInterruptibleTool("mine", "d",
		func(ctx context.Context, _ struct{}, _ *confirmation) (string, error) { return "", nil })

	foreign := ai.NewToolRequestPart(&ai.ToolRequest{Name: "other"})
	foreign.Interrupt = &ai.ToolInterrupt{}

	if _, err := mine.Restart(foreign, mine.WithResume(confirmation{Approved: true})); err == nil {
		t.Error("Restart must reject a part for a different tool")
	}
	if _, err := mine.Respond(foreign, "out"); err == nil {
		t.Error("Respond must reject a part for a different tool")
	}

	own := ai.NewToolRequestPart(&ai.ToolRequest{Name: "mine"})
	own.Interrupt = &ai.ToolInterrupt{}
	bare, err := mine.Restart(own)
	if err != nil {
		t.Fatalf("bare Restart: %v", err)
	}
	if bare.Restart == nil || bare.Restart.Resume != nil {
		t.Errorf("bare restart state = %+v, want bare restart (nil Resume)", bare.Restart)
	}
}
