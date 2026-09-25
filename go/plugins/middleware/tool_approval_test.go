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

package middleware

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/registry"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func defineToolModel(t *testing.T, r *registry.Registry, name string, fn ai.ModelFunc) ai.Model {
	t.Helper()
	return registerTestModel(r, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, fn)
}

func defineTool(t *testing.T, r api.Registry, name string) ai.Tool {
	t.Helper()
	return registerTestTool(r, name, "test tool",
		func(ctx *ai.ToolContext, input struct {
			V string `json:"v"`
		}) (string, error) {
			return "result:" + input.V, nil
		})
}

// spanCollector is a minimal sdktrace.SpanExporter that records finished
// spans so a test can assert on them.
type spanCollector struct {
	mu    sync.Mutex
	spans []sdktrace.ReadOnlySpan
}

func (c *spanCollector) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.spans = append(c.spans, spans...)
	return nil
}

func (c *spanCollector) Shutdown(context.Context) error { return nil }

// byName returns every recorded span with the given name.
func (c *spanCollector) byName(name string) []sdktrace.ReadOnlySpan {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []sdktrace.ReadOnlySpan
	for _, s := range c.spans {
		if s.Name() == name {
			out = append(out, s)
		}
	}
	return out
}

// spanAttr returns the string value of the named span attribute, if present.
func spanAttr(span sdktrace.ReadOnlySpan, key string) (string, bool) {
	for _, kv := range span.Attributes() {
		if string(kv.Key) == key {
			return kv.Value.AsString(), true
		}
	}
	return "", false
}

// twoToolModelHandler returns a model handler that requests two tools on the first call,
// then returns a final text response when it sees tool responses.
func twoToolModelHandler(tool1, tool2 string) ai.ModelFunc {
	return func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					return &ai.ModelResponse{
						Request: req,
						Message: ai.NewModelTextMessage("done"),
					}, nil
				}
			}
		}
		return &ai.ModelResponse{
			Request: req,
			Message: &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{Name: tool1, Input: map[string]any{"v": "1"}}),
					ai.NewToolRequestPart(&ai.ToolRequest{Name: tool2, Input: map[string]any{"v": "2"}}),
				},
			},
		}, nil
	}
}

func TestToolApprovalAllowsApprovedTools(t *testing.T) {
	r := newTestRegistry(t)

	m := defineToolModel(t, r, "test/twotools", twoToolModelHandler("allowed", "alsoAllowed"))
	allowed := defineTool(t, r, "allowed")
	alsoAllowed := defineTool(t, r, "alsoAllowed")

	ta := &ToolApproval{AllowedTools: []string{"allowed", "alsoAllowed"}}
	registerTestMiddleware(r, "toolApproval", ta)

	resp, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(allowed, alsoAllowed),
		ai.WithUse(ta),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Errorf("got %q, want %q", resp.Text(), "done")
	}
	if resp.FinishReason == "interrupted" {
		t.Error("did not expect interrupted finish reason")
	}
}

func TestToolApprovalInterruptsUnapprovedTools(t *testing.T) {
	r := newTestRegistry(t)

	m := defineToolModel(t, r, "test/twotools", twoToolModelHandler("safe", "dangerous"))
	safe := defineTool(t, r, "safe")
	dangerous := defineTool(t, r, "dangerous")

	ta := &ToolApproval{AllowedTools: []string{"safe"}}
	registerTestMiddleware(r, "toolApproval", ta)

	resp, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(safe, dangerous),
		ai.WithUse(ta),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "interrupted" {
		t.Errorf("got finish reason %q, want %q", resp.FinishReason, "interrupted")
	}

	interrupts := resp.Interrupts()
	if len(interrupts) == 0 {
		t.Fatal("expected at least one interrupt")
	}

	found := false
	for _, p := range interrupts {
		if p.ToolRequest != nil && p.ToolRequest.Name == "dangerous" {
			found = true
		}
		if p.ToolRequest != nil && p.ToolRequest.Name == "safe" {
			t.Error("did not expect interrupt for 'safe' tool")
		}
	}
	if !found {
		t.Error("expected interrupt for 'dangerous' tool")
	}
}

// TestToolApprovalInterruptIsTracedOnce verifies the interrupt lands in the
// trace as exactly one span named for the blocked tool. ToolApproval emits no
// span of its own: the generate engine attributes a short-circuited tool call
// to the tool, so hand-rolling one here would double-count it.
func TestToolApprovalInterruptIsTracedOnce(t *testing.T) {
	r := newTestRegistry(t)

	m := defineToolModel(t, r, "test/twotools", twoToolModelHandler("safe", "dangerous"))
	safe := defineTool(t, r, "safe")
	dangerous := defineTool(t, r, "dangerous")

	collector := &spanCollector{}
	sp := sdktrace.NewSimpleSpanProcessor(collector)
	tp := tracing.TracerProvider()
	tp.RegisterSpanProcessor(sp)
	t.Cleanup(func() { tp.UnregisterSpanProcessor(sp) })

	ta := &ToolApproval{AllowedTools: []string{"safe"}}
	if _, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(safe, dangerous),
		ai.WithUse(ta),
	); err != nil {
		t.Fatal(err)
	}

	spans := collector.byName("dangerous")
	if len(spans) != 1 {
		t.Fatalf("got %d spans named %q, want exactly 1", len(spans), "dangerous")
	}
	// An approval interrupt resolves the call without running the tool, and
	// must still look exactly like a tool call in the trace.
	const subtypeKey = "genkit:metadata:subtype"
	subtype, ok := spanAttr(spans[0], subtypeKey)
	if !ok {
		t.Fatalf("span %q: missing attribute %q", "dangerous", subtypeKey)
	}
	if want := string(api.ActionTypeToolV2); subtype != want {
		t.Errorf("span %q: %s = %q, want %q", "dangerous", subtypeKey, subtype, want)
	}
}

func TestToolApprovalEmptyListInterruptsAll(t *testing.T) {
	r := newTestRegistry(t)

	singleToolHandler := func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("done")}, nil
				}
			}
		}
		return &ai.ModelResponse{
			Request: req,
			Message: &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{Name: "myTool", Input: map[string]any{"v": "1"}}),
				},
			},
		}, nil
	}

	m := defineToolModel(t, r, "test/singletool", singleToolHandler)
	myTool := defineTool(t, r, "myTool")

	ta := &ToolApproval{}
	registerTestMiddleware(r, "toolApproval", ta)

	resp, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(myTool),
		ai.WithUse(ta),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "interrupted" {
		t.Errorf("got finish reason %q, want %q", resp.FinishReason, "interrupted")
	}
}

func TestToolApprovalResumedCallRuns(t *testing.T) {
	r := newTestRegistry(t)

	m := defineToolModel(t, r, "test/singletool", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("done")}, nil
				}
			}
		}
		return &ai.ModelResponse{
			Request: req,
			Message: &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{Name: "needsApproval", Input: map[string]any{"v": "1"}}),
				},
			},
		}, nil
	})
	needsApproval := defineTool(t, r, "needsApproval")

	ta := &ToolApproval{} // deny all
	registerTestMiddleware(r, "toolApproval", ta)

	resp, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(needsApproval),
		ai.WithUse(ta),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "interrupted" {
		t.Fatalf("got finish reason %q, want %q", resp.FinishReason, "interrupted")
	}

	// Build a restart part for each interrupt with explicit approval metadata.
	var restarts []*ai.Part
	for _, p := range resp.Interrupts() {
		restart := ai.NewToolRequestPart(p.ToolRequest)
		restart.Metadata = map[string]any{"resumed": map[string]any{"toolApproved": true}}
		restarts = append(restarts, restart)
	}

	resp, err = ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithMessages(resp.History()...),
		ai.WithTools(needsApproval),
		ai.WithToolRestarts(restarts...),
		ai.WithUse(ta),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Errorf("got %q, want %q", resp.Text(), "done")
	}
}

// Resuming without the explicit toolApproved flag must still interrupt, so
// unrelated resume flows cannot bypass approval.
func TestToolApprovalResumedWithoutApprovalInterrupts(t *testing.T) {
	r := newTestRegistry(t)

	m := defineToolModel(t, r, "test/singletool", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("done")}, nil
				}
			}
		}
		return &ai.ModelResponse{
			Request: req,
			Message: &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{Name: "needsApproval", Input: map[string]any{"v": "1"}}),
				},
			},
		}, nil
	})
	needsApproval := defineTool(t, r, "needsApproval")

	ta := &ToolApproval{}
	registerTestMiddleware(r, "toolApproval", ta)

	resp, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(needsApproval),
		ai.WithUse(ta),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "interrupted" {
		t.Fatalf("got finish reason %q, want %q", resp.FinishReason, "interrupted")
	}

	// Bare `resumed: true` without `toolApproved: true` must NOT bypass approval;
	// the tool re-interrupts, which surfaces as a FAILED_PRECONDITION error from
	// ai.Generate for the restarted turn.
	var restarts []*ai.Part
	for _, p := range resp.Interrupts() {
		restart := ai.NewToolRequestPart(p.ToolRequest)
		restart.Metadata = map[string]any{"resumed": true}
		restarts = append(restarts, restart)
	}

	_, err = ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithMessages(resp.History()...),
		ai.WithTools(needsApproval),
		ai.WithToolRestarts(restarts...),
		ai.WithUse(ta),
	)
	if err == nil {
		t.Fatal("expected error from re-interrupted restart, got nil")
	}
}

// TestToolApprovalReleasesTheToolsOwnInterrupt pins the two-step flow with a
// tool that asks its own question: the hold is the middleware's interrupt;
// the approval answers it, after which the tool runs afresh, asks its own
// question with no resume data, and the answer to that question passes the
// gate, since the call was approved.
func TestToolApprovalReleasesTheToolsOwnInterrupt(t *testing.T) {
	r := newTestRegistry(t)
	m := defineToolModel(t, r, "test/transfer", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, msg := range req.Messages {
			if msg.Role == ai.RoleTool {
				return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("done")}, nil
			}
		}
		return &ai.ModelResponse{
			Request: req,
			Message: &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{Name: "transfer", Input: map[string]any{"amount": 200}}),
				},
			},
		}, nil
	})
	type transferInput struct {
		Amount float64 `json:"amount"`
	}
	var resumes []map[string]any
	transfer := ai.NewTool("transfer", "moves money",
		func(tc *ai.ToolContext, _ transferInput) (string, error) {
			resumes = append(resumes, tc.Resumed)
			if !tc.IsResumed() {
				return "", tool.Interrupt(tc, nil)
			}
			if tc.Resumed["approved"] != true {
				return "cancelled", nil
			}
			return "completed", nil
		})
	transfer.Register(r)
	ta := &ToolApproval{} // deny all
	registerTestMiddleware(r, "toolApproval", ta)
	generate := func(opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
		return ai.Generate(ctx, r, append([]ai.GenerateOption{ai.WithModel(m), ai.WithTools(transfer), ai.WithUse(ta)}, opts...)...)
	}
	restart := func(p *ai.Part, resume map[string]any) ai.GenerateOption {
		part, err := transfer.RestartWith(p, ai.WithResumedMetadata[transferInput](resume))
		if err != nil {
			t.Fatal(err)
		}
		return ai.WithToolRestarts(part)
	}

	resp, err := generate(ai.WithPrompt("go"))
	if err != nil {
		t.Fatal(err)
	}
	held := resp.Interrupts()
	if len(held) != 1 || held[0].Interrupt == nil || held[0].Interrupt.RaisedBy != ta.Name() {
		t.Fatalf("interrupts = %+v, want one hold raised by %s", held, ta.Name())
	}

	resp2, err := generate(ai.WithMessages(resp.History()...), restart(held[0], map[string]any{"toolApproved": true}))
	if !errors.Is(err, status.ErrFailedPrecondition) || resp2 == nil {
		t.Fatalf("approval = (%v, %v), want the tool's own interrupt under FAILED_PRECONDITION", resp2, err)
	}
	if len(resumes) != 1 || resumes[0] != nil {
		t.Fatalf("tool saw resumes %v, want one fresh call: the approval must not reach it", resumes)
	}
	asked := resp2.Interrupts()
	if len(asked) != 1 || asked[0].Interrupt == nil || asked[0].Interrupt.RaisedBy != "" {
		t.Fatalf("interrupts = %+v, want the tool's own question", asked)
	}

	resp3, err := generate(ai.WithMessages(resp2.History()...), restart(asked[0], map[string]any{"approved": true}))
	if err != nil {
		t.Fatal(err)
	}
	if resp3.Text() != "done" {
		t.Errorf("got %q, want %q", resp3.Text(), "done")
	}
	if len(resumes) != 2 || resumes[1]["approved"] != true {
		t.Errorf("tool saw resumes %v, want the approval on its second call", resumes)
	}
}

// TestToolApprovalHoldWithoutRecordedStageNeedsApproval pins that a hold
// which does not record the stage that raised it, as one stored before
// stages were recorded, still needs an explicit approval: the restart's
// answer reaches the gate, a denial holds the call again, and only
// "toolApproved": true runs the tool.
func TestToolApprovalHoldWithoutRecordedStageNeedsApproval(t *testing.T) {
	r := newTestRegistry(t)
	m := defineToolModel(t, r, "test/singletool", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("done")}, nil
				}
			}
		}
		return &ai.ModelResponse{Request: req, Message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "needsApproval", Input: map[string]any{"v": "1"}}),
		}}}, nil
	})
	needsApproval := defineTool(t, r, "needsApproval")
	ta := &ToolApproval{}
	registerTestMiddleware(r, "toolApproval", ta)

	resp, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("go"), ai.WithTools(needsApproval), ai.WithUse(ta))
	if err != nil {
		t.Fatal(err)
	}
	// Drop the recorded stage, as a hold stored before it was recorded lacks it.
	raw, err := json.Marshal(resp.History())
	if err != nil {
		t.Fatal(err)
	}
	var wire []map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	for _, msg := range wire {
		for _, c := range msg["content"].([]any) {
			if md, ok := c.(map[string]any)["metadata"].(map[string]any); ok {
				delete(md, "interruptedBy")
			}
		}
	}
	if raw, err = json.Marshal(wire); err != nil {
		t.Fatal(err)
	}
	var history []*ai.Message
	if err := json.Unmarshal(raw, &history); err != nil {
		t.Fatal(err)
	}
	var held *ai.Part
	for _, msg := range history {
		for _, p := range msg.Content {
			if p.IsInterrupt() {
				held = p
			}
		}
	}
	if held == nil || held.Interrupt.RaisedBy != "" {
		t.Fatalf("held part = %+v, want an interrupt with no recorded stage", held)
	}

	resume := func(answer map[string]any) (*ai.ModelResponse, error) {
		restart := needsApproval.Restart(held, &ai.RestartOptions{ResumedMetadata: answer})
		return ai.Generate(ctx, r, ai.WithModel(m), ai.WithMessages(history...),
			ai.WithTools(needsApproval), ai.WithToolRestarts(restart), ai.WithUse(ta))
	}
	if _, err := resume(map[string]any{"toolApproved": false}); !errors.Is(err, status.ErrFailedPrecondition) {
		t.Fatalf("denied restart = %v, want the call held again", err)
	}
	if _, err := resume(nil); !errors.Is(err, status.ErrFailedPrecondition) {
		t.Fatalf("bare restart = %v, want the call held again", err)
	}
	resp, err = resume(map[string]any{"toolApproved": true})
	if err != nil {
		t.Fatalf("approved restart: %v", err)
	}
	if resp.Text() != "done" {
		t.Errorf("Text() = %q, want done", resp.Text())
	}
}
