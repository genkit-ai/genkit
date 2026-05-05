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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

func defineToolModel(t *testing.T, r *registry.Registry, name string, fn ai.ModelFunc) ai.Model {
	t.Helper()
	return ai.DefineModel(r, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, fn)
}

func defineTool(t *testing.T, r api.Registry, name string) ai.Tool {
	t.Helper()
	return ai.DefineTool(r, name, "test tool",
		func(ctx *ai.ToolContext, input struct {
			V string `json:"v"`
		}) (string, error) {
			return "result:" + input.V, nil
		})
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
	ai.DefineMiddleware(r, "toolApproval", ta)

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
	ai.DefineMiddleware(r, "toolApproval", ta)

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
	ai.DefineMiddleware(r, "toolApproval", ta)

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
	ai.DefineMiddleware(r, "toolApproval", ta)

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
	ai.DefineMiddleware(r, "toolApproval", ta)

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
