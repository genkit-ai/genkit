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

package exp

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/middleware"
)

// defineStepModel defines a model that asks for the "step" tool until the
// conversation holds `steps` tool responses, then answers "done". Each call
// reports usage (10 input tokens when nil) and is counted in calls. It
// returns the model and the tool.
func defineStepModel(t *testing.T, g *genkit.Genkit, steps int, calls *atomic.Int32, usage *ai.GenerationUsage) (ai.Model, ai.Tool) {
	t.Helper()
	if usage == nil {
		usage = &ai.GenerationUsage{InputTokens: 10}
	}
	step := genkit.DefineTool(g, "step", "takes a step",
		func(ctx *ai.ToolContext, in struct {
			V string `json:"v"`
		}) (string, error) {
			return "stepped", nil
		})
	m := toolModel(t, g, "test/steps", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls.Add(1)
		responses := 0
		for _, msg := range req.Messages {
			for _, p := range msg.Content {
				if p.IsToolResponse() {
					responses++
				}
			}
		}
		resp := textResp(req, "done")
		if responses < steps {
			resp = toolReqResp(req, &ai.ToolRequest{Name: "step", Input: map[string]any{"v": "x"}})
		}
		resp.FinishReason = ai.FinishReasonStop
		resp.Usage = usage
		return resp, nil
	})
	return m, step
}

func TestBudgetStopsTheRunBeforeTheTurnPastItsLimit(t *testing.T) {
	g := newTestGenkit(t)
	var calls atomic.Int32
	m, step := defineStepModel(t, g, 100, &calls, nil)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(step),
		ai.WithUse(&Budget{Limit: ai.GenerationUsage{InputTokens: 25}}),
	)
	if !errors.Is(err, ai.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	// 10 and 20 are under 25, so the second and third turns run; 30 is not,
	// so the fourth never does.
	if got := calls.Load(); got != 3 {
		t.Errorf("model calls = %d, want 3", got)
	}
	if resp.FinishReason != ai.FinishReasonAborted {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, ai.FinishReasonAborted)
	}
}

// A hook can call the model more than once per turn. Each of those calls is
// billed, so each one counts.
func TestBudgetCountsCallsAHookMakes(t *testing.T) {
	g := newTestGenkit(t)
	var calls atomic.Int32
	m, step := defineStepModel(t, g, 100, &calls, nil)
	twice := ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{WrapModel: func(ctx context.Context, p *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
			if _, err := next(ctx, p); err != nil {
				return nil, err
			}
			return next(ctx, p)
		}}, nil
	})

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(step),
		ai.WithUse(&Budget{Limit: ai.GenerationUsage{InputTokens: 20}}, twice),
	)
	if !errors.Is(err, ai.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("model calls = %d, want 2: the first turn's two calls spend the budget", got)
	}
}

// Any field the limit sets is a cap, a provider's custom cost included.
func TestBudgetCapsCustomUsage(t *testing.T) {
	g := newTestGenkit(t)
	var calls atomic.Int32
	m, step := defineStepModel(t, g, 100, &calls, &ai.GenerationUsage{Custom: map[string]float64{"cost": 0.5}})

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(step),
		ai.WithUse(&Budget{Limit: ai.GenerationUsage{Custom: map[string]float64{"cost": 1}}}),
	)
	if !errors.Is(err, ai.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("model calls = %d, want 2", got)
	}
}

// A limit that is not a finite number can never be reached, so New rejects
// it and names the field.
func TestBudgetRejectsANonFiniteLimit(t *testing.T) {
	for _, v := range []float64{math.NaN(), math.Inf(1)} {
		_, err := Budget{Limit: ai.GenerationUsage{InputTokens: 10, Custom: map[string]float64{"cost": v}}}.New(ctx)
		if !errors.Is(err, status.ErrInvalidArgument) || !strings.Contains(err.Error(), "custom.cost") {
			t.Errorf("New with cost %v = %v, want INVALID_ARGUMENT naming custom.cost", v, err)
		}
	}
}

// A subagent's calls stay out of its parent's reported usage, but a budget
// on the parent's run counts them, so delegation cannot spend around it.
func TestBudgetRunScopeCountsSubagents(t *testing.T) {
	g := newTestGenkit(t)
	child := genkitx.DefineAgent[any](g, "child", aix.InlinePrompt{
		ai.WithModel(toolModel(t, g, "test/child", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			resp := textResp(req, "answered")
			resp.Usage = &ai.GenerationUsage{InputTokens: 10}
			return resp, nil
		})),
	})
	ask := genkit.DefineTool(g, "ask", "asks the child agent",
		func(ctx *ai.ToolContext, in struct {
			V string `json:"v"`
		}) (string, error) {
			out, err := child.RunText(ctx, "question")
			if err != nil {
				return "", err
			}
			return out.Message.Text(), nil
		})
	var calls atomic.Int32
	parent := toolModel(t, g, "test/parent", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls.Add(1)
		resp := toolReqResp(req, &ai.ToolRequest{Name: "ask", Input: map[string]any{"v": "x"}})
		resp.Usage = &ai.GenerationUsage{InputTokens: 10}
		return resp, nil
	})

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(parent),
		ai.WithPrompt("go"),
		ai.WithTools(ask),
		ai.WithUse(&Budget{Limit: ai.GenerationUsage{InputTokens: 15}}),
	)
	if !errors.Is(err, ai.ErrBudgetExceeded) {
		t.Fatalf("err = %v, want ErrBudgetExceeded", err)
	}
	// The parent's 10 plus the child's 10 reach 15 before the second turn.
	if got := calls.Load(); got != 1 {
		t.Errorf("parent model calls = %d, want 1", got)
	}
}

// A session budget carries across invocations: the next invocation of a
// session that is over its limit stops before it calls the model.
func TestBudgetSessionScope(t *testing.T) {
	g := newTestGenkit(t)
	var calls atomic.Int32
	m, step := defineStepModel(t, g, 2, &calls, nil)
	agent := genkitx.DefineAgent[any](g, "budgeted", aix.InlinePrompt{
		ai.WithModel(m),
		ai.WithTools(step),
		ai.WithUse(&Budget{Limit: ai.GenerationUsage{InputTokens: 15}, Scope: BudgetScopeSession}),
	}, aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()))

	// 10, then 20: the third turn would start over the limit.
	out, err := agent.RunText(ctx, "go")
	if err != nil {
		t.Fatal(err)
	}
	if out.FinishReason != aix.AgentFinishReasonAborted {
		t.Fatalf("FinishReason = %q, want %q", out.FinishReason, aix.AgentFinishReasonAborted)
	}

	before := calls.Load()
	out, err = agent.RunText(ctx, "again", aix.WithSessionID[any](out.SessionID))
	if err != nil {
		t.Fatal(err)
	}
	if out.FinishReason != aix.AgentFinishReasonAborted {
		t.Errorf("FinishReason = %q, want %q", out.FinishReason, aix.AgentFinishReasonAborted)
	}
	if calls.Load() != before {
		t.Errorf("model calls = %d after the invocation, want %d", calls.Load(), before)
	}
}

func TestBudgetRejectsAnUnenforceableSessionBudget(t *testing.T) {
	t.Run("client-managed agent", func(t *testing.T) {
		// The caller sends the state back, usage included, so it could
		// reset the count; only a store makes it hold.
		g := newTestGenkit(t)
		var calls atomic.Int32
		m, _ := defineStepModel(t, g, 0, &calls, nil)
		agent := genkitx.DefineAgent[any](g, "clientManaged", aix.InlinePrompt{
			ai.WithModel(m),
			ai.WithUse(&Budget{Limit: ai.GenerationUsage{InputTokens: 15}, Scope: BudgetScopeSession}),
		})
		out, err := agent.RunText(ctx, "go")
		if err != nil {
			t.Fatal(err)
		}
		if out.Error == nil || out.Error.Status != status.FailedPrecondition {
			t.Errorf("Error = %+v, want FAILED_PRECONDITION", out.Error)
		}
	})
	t.Run("interrupt", func(t *testing.T) {
		// A later turn would start over the limit with no tool call to
		// pause on.
		_, err := Budget{Limit: ai.GenerationUsage{InputTokens: 15}, Scope: BudgetScopeSession, Interrupt: true}.New(ctx)
		if !errors.Is(err, status.ErrInvalidArgument) {
			t.Errorf("New err = %v, want INVALID_ARGUMENT", err)
		}
	})
}

// With Interrupt, a run that reaches its limit pauses at its tool calls and
// says what it spent against which limit; restarting them continues with a
// fresh count.
func TestBudgetInterruptPausesTheRun(t *testing.T) {
	g := newTestGenkit(t)
	var calls atomic.Int32
	m, step := defineStepModel(t, g, 2, &calls, nil)
	budget := &Budget{Limit: ai.GenerationUsage{InputTokens: 15}, Interrupt: true}

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(step),
		ai.WithUse(budget),
	)
	if err != nil {
		t.Fatal(err)
	}
	interrupts := resp.Interrupts()
	if resp.FinishReason != ai.FinishReasonInterrupted || len(interrupts) != 1 {
		t.Fatalf("FinishReason = %q with %d interrupts, want one interrupt", resp.FinishReason, len(interrupts))
	}
	call, ok := BudgetInterrupted(interrupts[0])
	if !ok {
		t.Fatalf("BudgetInterrupted(%+v) = false, want the budget's hold", interrupts[0])
	}
	data, _ := ai.InterruptAs[BudgetExceeded](call.Part)
	if data.Spent["inputTokens"] != 20 || data.Limit["inputTokens"] != 15 {
		t.Errorf("hold = %v spent of %v, want 20 of 15", data.Spent, data.Limit)
	}

	// Answering the budget's own hold renews the limit.
	resp, err = genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(resp.History()...),
		ai.WithTools(step),
		ai.WithResume(call.Restart(nil)),
		ai.WithUse(budget),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Errorf("Text = %q, want %q", resp.Text(), "done")
	}
}

// A resume continues the run's count, so answering another middleware's hold
// cannot renew the budget: only answering the budget's own hold does, and a
// new turn of the conversation starts from zero.
func TestBudgetCountsARunAcrossResumes(t *testing.T) {
	g := newTestGenkit(t)
	var calls atomic.Int32
	m, step := defineStepModel(t, g, 1, &calls, nil) // 10 input tokens a call
	use := ai.WithUse(&middleware.ToolApproval{}, &Budget{Limit: ai.GenerationUsage{InputTokens: 10}, Interrupt: true})
	approve := middleware.ToolCallDecision{Approved: true}
	generate := func(opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
		return genkit.Generate(ctx, g, append([]ai.GenerateOption{ai.WithModel(m), ai.WithTools(step), use}, opts...)...)
	}
	// The conversation crosses the wire between calls, as a client's would.
	overWire := func(msgs []*ai.Message) []*ai.Message {
		b, err := json.Marshal(msgs)
		if err != nil {
			t.Fatal(err)
		}
		var out []*ai.Message
		if err := json.Unmarshal(b, &out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	only := func(resp *ai.ModelResponse, claim func(*ai.Part) bool) *ai.Part {
		t.Helper()
		if resp == nil || resp.FinishReason != ai.FinishReasonInterrupted || len(resp.Interrupts()) != 1 || !claim(resp.Interrupts()[0]) {
			t.Fatalf("response = %+v, want one interrupt of the expected kind", resp)
		}
		return resp.Interrupts()[0]
	}

	// The first turn spends the whole budget, and the approval gate holds
	// its call before the budget sees it.
	resp, err := generate(ai.WithPrompt("go"))
	if err != nil {
		t.Fatal(err)
	}
	held := only(resp, func(p *ai.Part) bool { _, ok := middleware.ToolApprovalInterrupted(p); return ok })
	approval, _ := middleware.ToolApprovalInterrupted(held)

	// Approving it resumes the same run, still over its limit: the budget
	// holds the call instead of letting it run on a fresh count.
	resp, _ = generate(ai.WithMessages(overWire(resp.History())...), ai.WithResume(approval.Restart(approve)))
	held = only(resp, func(p *ai.Part) bool { _, ok := BudgetInterrupted(p); return ok })
	if n := calls.Load(); n != 1 {
		t.Fatalf("model called %d times, want 1: the approval must not buy another turn", n)
	}

	// Answering the budget renews it, and the run finishes.
	cont, _ := BudgetInterrupted(held)
	resp, err = generate(ai.WithMessages(overWire(resp.History())...), ai.WithResume(cont.Restart(nil)))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Fatalf("Text = %q, want done", resp.Text())
	}

	// A new turn is a new run, whatever the last one spent.
	if _, err := generate(ai.WithMessages(overWire(resp.History())...), ai.WithPrompt("again")); err != nil {
		t.Errorf("new turn = %v, want a fresh budget", err)
	}

	// Without Interrupt, the approved call runs and the run stops before
	// the model call it would lead to.
	calls.Store(0)
	use = ai.WithUse(&middleware.ToolApproval{}, &Budget{Limit: ai.GenerationUsage{InputTokens: 10}})
	resp, err = generate(ai.WithPrompt("go"))
	if err != nil {
		t.Fatal(err)
	}
	approval, _ = middleware.ToolApprovalInterrupted(only(resp, func(p *ai.Part) bool { _, ok := middleware.ToolApprovalInterrupted(p); return ok }))
	_, err = generate(ai.WithMessages(overWire(resp.History())...), ai.WithResume(approval.Restart(approve)))
	if !errors.Is(err, ai.ErrBudgetExceeded) || calls.Load() != 1 {
		t.Errorf("approved resume = %v after %d model calls, want ErrBudgetExceeded after 1", err, calls.Load())
	}
}
