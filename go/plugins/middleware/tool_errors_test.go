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
	"errors"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp/tool"
	"github.com/firebase/genkit/go/core/api"
)

// failingTool registers a tool that always fails with err.
func failingTool(r api.Registry, name string, err error) ai.Tool {
	return registerTestTool(r, name, "always fails",
		func(ctx *ai.ToolContext, input struct{}) (string, error) {
			return "", err
		})
}

// toolLoopModel requests the named tools, one per turn, and then answers
// "done". It records every tool response it receives in *got.
func toolLoopModel(got *[]*ai.Part, names ...string) ai.ModelFunc {
	turn := 0
	return func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if last := req.Messages[len(req.Messages)-1]; last.Role == ai.RoleTool {
			*got = append(*got, last.Content...)
		}
		if turn == len(names) {
			return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("done")}, nil
		}
		turn++
		return &ai.ModelResponse{Request: req, Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{Name: names[turn-1], Input: map[string]any{}})},
		}}, nil
	}
}

// errorMessage returns the error a tool error response carries, or fails t.
func errorMessage(t *testing.T, p *ai.Part) string {
	t.Helper()
	if !p.IsToolError() {
		t.Fatalf("part %+v is not a tool error response", p)
	}
	msg, _ := p.ToolResponse.Output.(map[string]any)["error"].(string)
	return msg
}

func TestToolErrorsReturnsErrorsOnEveryTurn(t *testing.T) {
	r := newTestRegistry(t)
	var got []*ai.Part
	m := defineToolModel(t, r, "test/loop", toolLoopModel(&got, "flaky", "missing", "flaky"))
	flaky := failingTool(r, "flaky", errors.New("upstream timeout"))

	resp, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(flaky),
		ai.WithUse(&ToolErrors{}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Errorf("Text() = %q, want %q", resp.Text(), "done")
	}
	if len(got) != 3 {
		t.Fatalf("model received %d tool responses, want 3", len(got))
	}
	for i, want := range []string{"upstream timeout", `tool "missing" not found`, "upstream timeout"} {
		if msg := errorMessage(t, got[i]); !strings.Contains(msg, want) {
			t.Errorf("response %d error = %q, want it to contain %q", i, msg, want)
		}
	}
}

func TestToolErrorsLimitedToListedTools(t *testing.T) {
	r := newTestRegistry(t)
	var got []*ai.Part
	m := defineToolModel(t, r, "test/loop", toolLoopModel(&got, "covered", "uncovered"))
	covered := failingTool(r, "covered", errors.New("boom"))
	uncovered := failingTool(r, "uncovered", errors.New("boom"))

	_, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(covered, uncovered),
		ai.WithUse(&ToolErrors{Tools: []string{"covered"}}),
	)
	if !errors.Is(err, ai.ErrToolFailed) {
		t.Fatalf("err = %v, want ErrToolFailed from the uncovered tool", err)
	}
	if len(got) != 1 || errorMessage(t, got[0]) == "" {
		t.Errorf("model received %v, want the covered tool's error alone", got)
	}
}

func TestToolErrorsCombineAcrossInstances(t *testing.T) {
	r := newTestRegistry(t)
	var got []*ai.Part
	m := defineToolModel(t, r, "test/loop", toolLoopModel(&got, "a", "b"))
	a := failingTool(r, "a", errors.New("boom"))
	b := failingTool(r, "b", errors.New("boom"))

	_, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(a, b),
		ai.WithUse(&ToolErrors{Tools: []string{"a"}}, &ToolErrors{Tools: []string{"b"}}),
	)
	if err != nil {
		t.Fatalf("err = %v, want each tool covered by one of the two instances", err)
	}
	if len(got) != 2 {
		t.Fatalf("model received %d tool responses, want 2", len(got))
	}
}

func TestToolErrorsKeepsApprovalInterrupts(t *testing.T) {
	r := newTestRegistry(t)
	var got []*ai.Part
	m := defineToolModel(t, r, "test/loop", toolLoopModel(&got, "dangerous"))
	dangerous := defineTool(t, r, "dangerous")

	resp, err := ai.Generate(ctx, r,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(dangerous),
		ai.WithUse(&ToolErrors{}, &ToolApproval{}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "interrupted" {
		t.Errorf("FinishReason = %q, want interrupted", resp.FinishReason)
	}
}

// TestToolFailWithoutMiddleware checks the author's side of the feature: an
// error made with tool.Fail answers the call with no middleware installed,
// carrying the tool's own message.
func TestToolFailWithoutMiddleware(t *testing.T) {
	r := newTestRegistry(t)
	var got []*ai.Part
	m := defineToolModel(t, r, "test/loop", toolLoopModel(&got, "lookup"))
	lookup := failingTool(r, "lookup", tool.Fail(errors.New("no such city")))

	if _, err := ai.Generate(ctx, r, ai.WithModel(m), ai.WithPrompt("go"), ai.WithTools(lookup)); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("model received %d tool responses, want 1", len(got))
	}
	if msg := errorMessage(t, got[0]); msg != "no such city" {
		t.Errorf("error = %q, want %q", msg, "no such city")
	}
}
