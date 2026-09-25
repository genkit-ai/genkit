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

// Tests for tools.go that drive tools the way callers do, through the ai/tool
// runtime verbs. They live in package
// ai_test rather than in tools_test.go because ai/tool imports ai: an internal
// test file importing it would close an import cycle, so the external test
// package is the escape hatch.
package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/google/go-cmp/cmp"
)

// newToolTestRegistry returns a registry with the formats and generate action
// configured, ready for ai.Generate / ai.GenerateStream.
func newToolTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	reg := registry.New()
	ai.ConfigureFormats(reg)
	ai.DefineGenerateAction(context.Background(), reg)
	return reg
}

// defineTestModel builds and registers a model, the two steps
// genkit.DefineModel fuses for an application.
func defineTestModel(reg api.Registry, name string, opts *ai.ModelOptions, fn ai.ModelFunc) ai.Model {
	m := ai.NewModel(name, opts, fn)
	m.Register(reg)
	return m
}

// defineTestTool builds and registers a tool written against ai.ToolContext
// from a function that only wants a context.Context. The adapter is what
// makes these tests double as coverage that the ai/tool verbs work from a
// ToolContext tool: the context they receive is the ToolContext itself.
func defineTestTool[In, Out any](reg api.Registry, name, description string, fn func(context.Context, In) (Out, error)) *ai.ToolAction[In, Out] {
	tl := ai.NewTool(name, description, func(tc *ai.ToolContext, in In) (Out, error) {
		return fn(tc, in)
	})
	tl.Register(reg)
	return tl
}

// defineToolThenFinishModel defines "test/model": on the first turn it returns
// reqs (typically tool requests), and once a tool response is in history it
// returns the final text "done". This drives a single tool round per Generate.
func defineToolThenFinishModel(reg *registry.Registry, reqs ...*ai.Part) {
	defineTestModel(reg, "test/model",
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

type weatherIn struct {
	City string `json:"city"`
}

// TestTool_AttachParts verifies AttachParts folds extra content into the tool's
// multipart response without changing the function signature, for a tool
// written against ToolContext.
func TestTool_AttachParts(t *testing.T) {
	reg := newToolTestRegistry(t)
	shot := defineTestTool(reg, "screenshot", "takes a screenshot",
		func(ctx context.Context, _ struct{}) (string, error) {
			// A nil part is ignored, so a failed constructor result can be
			// passed without a check.
			tool.AttachParts(ctx, nil, ai.NewMediaPart("image/png", "pngbytes"))
			return "captured", nil
		})

	resp, err := shot.RunRawMultipart(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRawMultipart: %v", err)
	}
	if resp.Output != "captured" {
		t.Errorf("output = %v, want %q", resp.Output, "captured")
	}
	if len(resp.Content) != 1 || !resp.Content[0].IsMedia() {
		t.Fatalf("expected one attached media part, got %+v", resp.Content)
	}
}

// TestMultipartTool_AttachParts verifies attached parts are appended to the
// content a multipart tool returns itself, rather than replacing it.
func TestMultipartTool_AttachParts(t *testing.T) {
	tl := ai.NewMultipartTool("chart", "charts and annotates",
		func(tc *ai.ToolContext, _ struct{}) (*ai.MultipartToolResponse, error) {
			tool.AttachParts(tc, ai.NewMediaPart("image/png", "annotation"))
			return &ai.MultipartToolResponse{
				Output:  "charted",
				Content: []*ai.Part{ai.NewMediaPart("image/png", "chart")},
			}, nil
		})

	resp, err := tl.RunRawMultipart(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRawMultipart: %v", err)
	}
	if len(resp.Content) != 2 || resp.Content[0].Text != "chart" || resp.Content[1].Text != "annotation" {
		t.Fatalf("content = %+v, want the returned part followed by the attached one", resp.Content)
	}

	// A multipart function may return no response at all; the attached parts
	// still need somewhere to land.
	silent := ai.NewMultipartTool("silent", "attaches, returns nothing",
		func(tc *ai.ToolContext, _ struct{}) (*ai.MultipartToolResponse, error) {
			tool.AttachParts(tc, ai.NewMediaPart("image/png", "only"))
			return nil, nil
		})
	resp, err = silent.RunRawMultipart(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRawMultipart: %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "only" {
		t.Fatalf("content = %+v, want the attached part on an empty response", resp.Content)
	}
}

// TestTool_SendPartialNoOpWithoutStreaming confirms SendPartial is a safe no-op
// when no streaming callback is wired (here, a direct RunRaw).
func TestTool_SendPartialNoOpWithoutStreaming(t *testing.T) {
	reg := newToolTestRegistry(t)
	tl := defineTestTool(reg, "noop", "streams when it can",
		func(ctx context.Context, _ struct{}) (string, error) {
			tool.SendPartial(ctx, map[string]any{"progress": 50})
			return "ok", nil
		})

	out, err := tl.RunRaw(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRaw: %v", err)
	}
	if out != "ok" {
		t.Errorf("output = %v, want %q", out, "ok")
	}
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

// approve builds the restart part that re-executes the interrupted call to
// tl with {"approved": true} as its resume data.
func approve[In, Out any](t *testing.T, tl *ai.ToolAction[In, Out], interrupt *ai.Part) *ai.Part {
	t.Helper()
	restart, err := tl.RestartWith(interrupt, ai.WithResumedMetadata[In](map[string]any{"approved": true}))
	if err != nil {
		t.Fatalf("RestartWith: %v", err)
	}
	return restart
}

// singleInterrupt returns the one interrupt part of resp.
func singleInterrupt(t *testing.T, resp *ai.ModelResponse) *ai.Part {
	t.Helper()
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %d (finish=%s)", len(interrupts), resp.FinishReason)
	}
	return interrupts[0]
}

// TestToolContextTool_InterruptAndResumeData covers the verbs from a tool
// written against ToolContext: it interrupts with tool.Interrupt and, when
// restarted, reads the typed answer with tool.ResumeData off the context it
// was given (the ToolContext itself).
func TestToolContextTool_InterruptAndResumeData(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "gate", Input: map[string]any{"amount": 200},
	}))

	var (
		gotResume confirmation
		gotOK     bool
	)
	gate := defineTestTool(reg, "gate", "interrupts once, then reads its resume data",
		func(ctx context.Context, in transferIn) (string, error) {
			res, ok := tool.ResumeData[confirmation](ctx)
			if !ok {
				return "", tool.Interrupt(ctx, transferInterrupt{Reason: "confirm", Amount: in.Amount})
			}
			gotResume, gotOK = res, ok
			return "ok", nil
		})

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(gate))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupt := singleInterrupt(t, resp)
	meta, ok := ai.InterruptAs[transferInterrupt](interrupt)
	if !ok || meta.Reason != "confirm" || meta.Amount != 200 {
		t.Fatalf("InterruptAs = %+v, %v; want the typed interrupt data", meta, ok)
	}
	resp, err = ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithMessages(resp.History()...),
		ai.WithTools(gate), ai.WithToolRestarts(approve(t, gate, interrupt)))
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if got := resp.Text(); got != "done" {
		t.Errorf("final text = %q, want %q", got, "done")
	}
	if !gotOK || !gotResume.Approved {
		t.Errorf("ResumeData = %+v, %v; want {true}, true", gotResume, gotOK)
	}
}

// TestInterruptAs_DecodesIntoAnyMatchingType pins that the interrupt data a
// tool sent as a struct reads back into any type with the same JSON shape:
// the loop records the data as the JSON object it serializes to, so a handler
// in another package with its own view of the payload decodes it.
func TestInterruptAs_DecodesIntoAnyMatchingType(t *testing.T) {
	type transferInterruptView struct {
		Reason string  `json:"reason"`
		Amount float64 `json:"amount"`
	}
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "transfer", Input: map[string]any{"amount": 200},
	}))
	transfer := defineTestTool(reg, "transfer", "transfers money",
		func(ctx context.Context, in transferIn) (string, error) {
			return "", tool.Interrupt(ctx, transferInterrupt{Reason: "large_amount", Amount: in.Amount})
		})
	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(transfer))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupt := singleInterrupt(t, resp)

	if _, ok := interrupt.Interrupt.Data.(map[string]any); !ok {
		t.Errorf("Interrupt.Data = %T, want the JSON object the tool's struct serializes to", interrupt.Interrupt.Data)
	}
	view, ok := ai.InterruptAs[transferInterruptView](interrupt)
	if !ok || view.Reason != "large_amount" || view.Amount != 200 {
		t.Errorf("InterruptAs[view] = (%+v, %v), want the payload decoded", view, ok)
	}
	same, ok := ai.InterruptAs[transferInterrupt](interrupt)
	if !ok || same.Reason != "large_amount" {
		t.Errorf("InterruptAs[same type] = (%+v, %v), want the payload decoded", same, ok)
	}
}

// TestInterrupt_NonObjectData_ReturnsClearError pins that interrupting with a
// scalar fails the call with a clear error when the loop records the
// interrupt.
func TestInterrupt_NonObjectData_ReturnsClearError(t *testing.T) {
	for _, tc := range []struct {
		name   string
		define func(reg *registry.Registry) ai.Tool
	}{
		{"plain", func(reg *registry.Registry) ai.Tool {
			return defineTestTool(reg, "bad", "interrupts with a scalar",
				func(ctx context.Context, _ struct{}) (string, error) {
					return "", tool.Interrupt(ctx, "not an object")
				})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reg := newToolTestRegistry(t)
			tl := tc.define(reg)
			defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "bad", Input: map[string]any{}}))

			_, err := ai.Generate(context.Background(), reg,
				ai.WithModelName("test/model"),
				ai.WithPrompt("go"),
				ai.WithTools(tl))
			if err == nil {
				t.Fatal("expected an error interrupting with non-object data")
			}
			if !strings.Contains(err.Error(), "JSON object") {
				t.Errorf("error = %q, want it to mention the JSON object constraint", err)
			}
		})
	}
}

// TestSendPartial_StreamsPartialToolResponse asserts a tool's SendPartial calls
// arrive on the stream as partial tool responses, distinguishable via
// IsPartial / ToolResponses.
func TestSendPartial_StreamsPartialToolResponse(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "progressTool", Input: map[string]any{}}))

	defineTestTool(reg, "progressTool", "streams progress",
		func(ctx context.Context, _ struct{}) (string, error) {
			tool.SendPartial(ctx, map[string]any{"progress": 50})
			return "complete", nil
		})

	var partials []*ai.Part
	for val, err := range ai.GenerateStream(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("go"),
		ai.WithTools(ai.ToolName("progressTool"))) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
		if val.Done {
			continue
		}
		for _, p := range val.Chunk.ToolResponses() {
			if p.IsPartial() {
				partials = append(partials, p)
			}
		}
	}

	if len(partials) == 0 {
		t.Fatal("expected at least one partial tool response on the stream")
	}
	if partials[0].ToolResponse.Name != "progressTool" {
		t.Errorf("partial tool name = %q, want %q", partials[0].ToolResponse.Name, "progressTool")
	}
}

// TestSendPartial_StreamsFromARestartedTool pins that a restarted tool
// streams as a tool of a model turn does: its SendPartial calls arrive on the
// resumed generation's stream, tagged with the request they belong to.
func TestSendPartial_StreamsFromARestartedTool(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "progressTool", Ref: "r1", Input: map[string]any{}}))
	progress := defineTestTool(reg, "progressTool", "streams progress after approval",
		func(ctx context.Context, _ struct{}) (string, error) {
			if _, ok := tool.ResumeData[confirmation](ctx); !ok {
				return "", tool.Interrupt(ctx, nil)
			}
			tool.SendPartial(ctx, map[string]any{"progress": 50})
			return "complete", nil
		})

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(progress))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	restart := approve(t, progress, singleInterrupt(t, resp))

	var partials []*ai.Part
	for val, err := range ai.GenerateStream(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithMessages(resp.History()...),
		ai.WithTools(progress),
		ai.WithToolRestarts(restart)) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
		if val.Done {
			continue
		}
		for _, p := range val.Chunk.ToolResponses() {
			if p.IsPartial() {
				partials = append(partials, p)
			}
		}
	}
	if len(partials) != 1 {
		t.Fatalf("got %d partial tool responses on the stream, want 1", len(partials))
	}
	if got := partials[0].ToolResponse; got.Name != "progressTool" || got.Ref != "r1" {
		t.Errorf("partial = %s#%s, want progressTool#r1", got.Name, got.Ref)
	}
}

// TestConcurrentStreamingTools_NoDataRace is the regression for the streaming
// race: when a model emits multiple tool calls in one turn and more than one
// streams via SendPartial, the per-tool senders run on concurrent goroutines.
// They must be serialized so they don't race on the shared stream callback.
// Run under `go test -race` to detect a regression.
func TestConcurrentStreamingTools_NoDataRace(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg,
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "toolA", Input: map[string]any{}}),
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "toolB", Input: map[string]any{}}))

	// A rendezvous so both tools enter their SendPartial loops at the same
	// time, maximizing the chance of overlapping callback invocations.
	var ready sync.WaitGroup
	ready.Add(2)
	start := make(chan struct{})
	go func() { ready.Wait(); close(start) }()

	streamer := func(ctx context.Context, _ struct{}) (string, error) {
		ready.Done()
		<-start
		for i := 0; i < 200; i++ {
			tool.SendPartial(ctx, map[string]any{"n": i})
		}
		return "ok", nil
	}
	defineTestTool(reg, "toolA", "streams", streamer)
	defineTestTool(reg, "toolB", "streams", streamer)

	for _, err := range ai.GenerateStream(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("go"),
		ai.WithTools(ai.ToolName("toolA"), ai.ToolName("toolB"))) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
	}
}

// TestAttachParts_FromWrapToolHook pins that the part sink spans the whole
// tool call: a WrapTool hook attaches parts before and after running the
// tool, and they land on the tool response next to the tool's own, in call
// order, instead of vanishing because the sink was installed inside the call.
func TestAttachParts_FromWrapToolHook(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "shot", Input: map[string]any{}}))
	shot := defineTestTool(reg, "shot", "takes a screenshot",
		func(ctx context.Context, _ struct{}) (string, error) {
			tool.AttachParts(ctx, ai.NewTextPart("tool"))
			return "captured", nil
		})
	attach := ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapTool: func(ctx context.Context, p *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
				tool.AttachParts(ctx, ai.NewTextPart("before"))
				resp, err := next(ctx, p)
				tool.AttachParts(ctx, ai.NewTextPart("after"))
				return resp, err
			},
		}, nil
	})

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("go"),
		ai.WithTools(shot),
		ai.WithUse(attach))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var got []string
	for _, m := range resp.History() {
		if m.Role != ai.RoleTool {
			continue
		}
		for _, p := range m.Content[0].ToolResponse.Content {
			got = append(got, p.Text)
		}
	}
	if diff := cmp.Diff([]string{"before", "tool", "after"}, got); diff != "" {
		t.Errorf("attached parts mismatch (-want +got):\n%s", diff)
	}
}

// restartThen defines "outer", a tool that interrupts on its first call and,
// once restarted, runs then, and generates until outer is restarted with an
// approval. It returns outer's output.
func restartThen(t *testing.T, reg *registry.Registry, then func(ctx context.Context) (string, error)) string {
	t.Helper()
	outer := defineTestTool(reg, "outer", "outer",
		func(ctx context.Context, _ struct{}) (string, error) {
			if _, ok := tool.ResumeData[confirmation](ctx); !ok {
				return "", tool.Interrupt(ctx, nil)
			}
			return then(ctx)
		})
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "outer", Input: map[string]any{}}))
	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(outer))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	resp, err = ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithMessages(resp.History()...),
		ai.WithTools(outer), ai.WithToolRestarts(approve(t, outer, singleInterrupt(t, resp))))
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	for _, m := range resp.History() {
		for _, p := range m.Content {
			if p.IsToolResponse() && p.ToolResponse.Name == "outer" {
				out, _ := p.ToolResponse.Output.(string)
				return out
			}
		}
	}
	t.Fatal("no response from outer in history")
	return ""
}

// TestToolCall_NestedGenerateAnswersNoRestart pins that a Generate a restarted
// tool runs is a fresh generation: its tools see no resume, so a tool that
// asks for approval asks rather than taking the enclosing call's answer as
// its own.
func TestToolCall_NestedGenerateAnswersNoRestart(t *testing.T) {
	reg := newToolTestRegistry(t)
	innerResumed := false
	inner := defineTestTool(reg, "inner", "inner",
		func(ctx context.Context, _ struct{}) (string, error) {
			if _, ok := tool.ResumeData[map[string]any](ctx); ok {
				innerResumed = true
				return "ran", nil
			}
			return "", tool.Interrupt(ctx, nil)
		})
	defineTestModel(reg, "test/inner",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Request: req,
				Message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{Name: "inner", Input: map[string]any{}}),
				}},
			}, nil
		})

	out := restartThen(t, reg, func(ctx context.Context) (string, error) {
		resp, err := ai.Generate(ctx, reg,
			ai.WithModelName("test/inner"), ai.WithPrompt("go"), ai.WithTools(inner))
		if err != nil {
			return "", err
		}
		return string(resp.FinishReason), nil
	})
	if innerResumed {
		t.Error("the nested tool took the enclosing restart's answer as its own")
	}
	if out != string(ai.FinishReasonInterrupted) {
		t.Errorf("nested generation finished %q, want %q", out, ai.FinishReasonInterrupted)
	}
}

// TestToolCall_DirectRunIsItsOwnCall pins that a tool a restarted tool runs
// directly is a call of its own: the parts it attaches land on its own
// response rather than the enclosing call's, and it sees no resume.
func TestToolCall_DirectRunIsItsOwnCall(t *testing.T) {
	reg := newToolTestRegistry(t)
	chartResumed := false
	chart := defineTestTool(reg, "chart", "chart",
		func(ctx context.Context, _ struct{}) (string, error) {
			_, chartResumed = tool.ResumeData[map[string]any](ctx)
			tool.AttachParts(ctx, ai.NewMediaPart("image/png", "chart"))
			return "chart", nil
		})

	out := restartThen(t, reg, func(ctx context.Context) (string, error) {
		resp, err := chart.RunRawMultipart(ctx, struct{}{})
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("chart parts: %d", len(resp.Content)), nil
	})
	if chartResumed {
		t.Error("the directly run tool took the enclosing restart's answer as its own")
	}
	if out != "chart parts: 1" {
		t.Errorf("outer saw %q, want the attached part on the directly run tool's response", out)
	}
}

// TestToolCall_RetriedToolStageKeepsTheRestart pins that a hook running the
// tool stage more than once, as a retry does, delivers the restart to every
// run: each run of the tool stage is the call's own.
func TestToolCall_RetriedToolStageKeepsTheRestart(t *testing.T) {
	reg := newToolTestRegistry(t)
	retry := ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapTool: func(ctx context.Context, p *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
				if _, err := next(ctx, p); err != nil {
					return nil, err
				}
				return next(ctx, p)
			},
		}, nil
	})
	var runs []bool
	tl := defineTestTool(reg, "outer", "outer",
		func(ctx context.Context, _ struct{}) (string, error) {
			res, ok := tool.ResumeData[confirmation](ctx)
			if !ok {
				return "", tool.Interrupt(ctx, nil)
			}
			runs = append(runs, res.Approved)
			return "ok", nil
		})
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "outer", Input: map[string]any{}}))
	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(tl), ai.WithUse(retry))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if _, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithMessages(resp.History()...),
		ai.WithTools(tl), ai.WithUse(retry), ai.WithToolRestarts(approve(t, tl, singleInterrupt(t, resp)))); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if diff := cmp.Diff([]bool{true, true}, runs); diff != "" {
		t.Errorf("resume per run (-want +got):\n%s", diff)
	}
}

// pauseOnce defines "transfer", a tool that pauses on its first pass and,
// when restarted, records the resume data it read and completes.
func pauseOnce(t *testing.T, reg *registry.Registry) (*ai.ToolAction[transferIn, string], func() *confirmation) {
	t.Helper()
	var got *confirmation
	tl := defineTestTool(reg, "transfer", "transfers money",
		func(ctx context.Context, in transferIn) (string, error) {
			res, ok := tool.ResumeData[confirmation](ctx)
			if !ok {
				return "", tool.Interrupt(ctx, transferInterrupt{Reason: "large_amount", Amount: in.Amount})
			}
			got = &res
			return "completed", nil
		})
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "transfer", Input: map[string]any{"amount": 200},
	}))
	return tl, func() *confirmation { return got }
}

// restartWith builds the restart part that re-executes the interrupted call
// to tl with resume as its resume data.
func restartWith[In, Out any](t *testing.T, tl *ai.ToolAction[In, Out], interrupt *ai.Part, resume map[string]any, opts ...ai.RestartWithOption[In]) *ai.Part {
	t.Helper()
	restart, err := tl.RestartWith(interrupt, append(opts, ai.WithResumedMetadata[In](resume))...)
	if err != nil {
		t.Fatalf("RestartWith: %v", err)
	}
	return restart
}

// gate is an inline WrapTool middleware that holds every call until a restart
// answers it with {"ok": true}, logging what each invocation saw: "held",
// "answered", or "released". Two gates in one chain share the inline name,
// which the chain tells apart.
func gate(log *[]string) ai.MiddlewareFunc {
	return func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapTool: func(ctx context.Context, p *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
				if tool.Released(ctx) {
					*log = append(*log, "released")
					return next(ctx, p)
				}
				if answer, ok := tool.ResumeData[map[string]any](ctx); ok {
					*log = append(*log, "answered")
					if answer["ok"] == true {
						return next(ctx, p)
					}
				}
				*log = append(*log, "held")
				return nil, tool.Interrupt(ctx, map[string]any{"gate": "held"})
			},
		}, nil
	}
}

// viaJSON round-trips messages through JSON, as a client that stores or
// forwards a conversation does, so a test can pin what survives the wire.
func viaJSON(t *testing.T, msgs []*ai.Message) []*ai.Message {
	t.Helper()
	raw, err := json.Marshal(msgs)
	if err != nil {
		t.Fatalf("marshal history: %v", err)
	}
	var out []*ai.Message
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal history: %v", err)
	}
	return out
}

// TestRestart_AnswersTheStageThatInterrupted pins that a restart answers
// whoever interrupted. A WrapTool hook that holds a call raises its own
// interrupt: the restart that answers it is read by the hook alone, and the
// tool then runs as a fresh call, asks its own question with no resume data,
// and gets that answer while the hook, which released the call before, lets
// the restart through. The stage rides on the interrupted request in
// history, so the flow survives a wire hop.
func TestRestart_AnswersTheStageThatInterrupted(t *testing.T) {
	for _, tc := range []struct {
		name string
		hop  func(t *testing.T, msgs []*ai.Message) []*ai.Message
	}{
		{"in process", func(_ *testing.T, msgs []*ai.Message) []*ai.Message { return msgs }},
		{"after a wire hop", viaJSON},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reg := newToolTestRegistry(t)
			transfer, saw := pauseOnce(t, reg)
			var log []string
			hold := gate(&log)
			resume := func(history []*ai.Message, part *ai.Part) (*ai.ModelResponse, error) {
				return ai.Generate(context.Background(), reg,
					ai.WithModelName("test/model"),
					ai.WithMessages(tc.hop(t, history)...),
					ai.WithTools(transfer),
					ai.WithUse(hold),
					ai.WithToolRestarts(part))
			}

			resp, err := ai.Generate(context.Background(), reg,
				ai.WithModelName("test/model"),
				ai.WithPrompt("transfer 200"),
				ai.WithTools(transfer),
				ai.WithUse(hold))
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			held := singleInterrupt(t, resp)
			if held.Interrupt == nil || held.Interrupt.RaisedBy != "inline" {
				t.Fatalf("held part interrupt = %+v, want one raised by the inline hook", held.Interrupt)
			}

			// Answering the hook releases the call: the tool runs afresh,
			// with no resume, and asks its own question, which the loop
			// reports as a re-interrupt next to the partial response.
			resp2, err := resume(resp.History(), restartWith(t, transfer, held, map[string]any{"ok": true}))
			if !errors.Is(err, status.ErrFailedPrecondition) || resp2 == nil {
				t.Fatalf("resume = (%v, %v), want the tool's own interrupt under FAILED_PRECONDITION", resp2, err)
			}
			if res := saw(); res != nil {
				t.Fatalf("tool saw resume = %+v, want none: the answer to the hook must not reach it", *res)
			}
			asked := singleInterrupt(t, resp2)
			if asked.Interrupt == nil || asked.Interrupt.RaisedBy != "" {
				t.Fatalf("re-interrupt = %+v, want one the tool raised", asked.Interrupt)
			}

			// Answering the tool passes the hook, which released the call.
			resp3, err := resume(resp2.History(), restartWith(t, transfer, asked, map[string]any{"approved": true}))
			if err != nil {
				t.Fatalf("second resume: %v", err)
			}
			if resp3.Text() != "done" {
				t.Errorf("Text() = %q, want done", resp3.Text())
			}
			if res := saw(); res == nil || !res.Approved {
				t.Errorf("tool saw resume = %+v, want approved", res)
			}
			if diff := cmp.Diff([]string{"held", "answered", "released"}, log); diff != "" {
				t.Errorf("hook saw (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRestart_TwoGatesAnswerInTurn pins the chain positions a restart is
// delivered by: with two holding hooks, answering the first reaches it alone
// and the second then holds; answering the second reaches it while the first,
// which released the call, lets it through; and answering the tool's own
// question passes both. Both hooks are inline, so they share a name and the
// chain tells them apart.
func TestRestart_TwoGatesAnswerInTurn(t *testing.T) {
	reg := newToolTestRegistry(t)
	transfer, saw := pauseOnce(t, reg)
	var logA, logB []string
	a, b := gate(&logA), gate(&logB)
	generate := func(opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
		return ai.Generate(context.Background(), reg, append([]ai.GenerateOption{
			ai.WithModelName("test/model"), ai.WithTools(transfer), ai.WithUse(a, b),
		}, opts...)...)
	}
	answer := func(t *testing.T, part *ai.Part) ai.GenerateOption {
		t.Helper()
		return ai.WithToolRestarts(restartWith(t, transfer, part, map[string]any{"ok": true}))
	}

	resp, err := generate(ai.WithPrompt("transfer 200"))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	heldA := singleInterrupt(t, resp)
	if heldA.Interrupt.RaisedBy != "inline" {
		t.Fatalf("first hold raised by %q, want the first inline hook", heldA.Interrupt.RaisedBy)
	}

	resp2, err := generate(ai.WithMessages(resp.History()...), answer(t, heldA))
	if !errors.Is(err, status.ErrFailedPrecondition) || resp2 == nil {
		t.Fatalf("answering the first hook = (%v, %v), want the second hook's hold", resp2, err)
	}
	heldB := singleInterrupt(t, resp2)
	if heldB.Interrupt.RaisedBy != "inline#2" {
		t.Fatalf("second hold raised by %q, want the second inline hook", heldB.Interrupt.RaisedBy)
	}

	resp3, err := generate(ai.WithMessages(resp2.History()...), answer(t, heldB))
	if !errors.Is(err, status.ErrFailedPrecondition) || resp3 == nil {
		t.Fatalf("answering the second hook = (%v, %v), want the tool's own interrupt", resp3, err)
	}
	asked := singleInterrupt(t, resp3)
	if asked.Interrupt.RaisedBy != "" {
		t.Fatalf("re-interrupt raised by %q, want the tool", asked.Interrupt.RaisedBy)
	}
	if res := saw(); res != nil {
		t.Fatalf("tool saw resume = %+v before it was answered", *res)
	}

	resp4, err := generate(ai.WithMessages(resp3.History()...),
		ai.WithToolRestarts(restartWith(t, transfer, asked, map[string]any{"approved": true})))
	if err != nil {
		t.Fatalf("answering the tool: %v", err)
	}
	if resp4.Text() != "done" {
		t.Errorf("Text() = %q, want done", resp4.Text())
	}
	if res := saw(); res == nil || !res.Approved {
		t.Errorf("tool saw resume = %+v, want approved", res)
	}
	if diff := cmp.Diff([]string{"held", "answered", "released", "released"}, logA); diff != "" {
		t.Errorf("first hook saw (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"held", "answered", "released"}, logB); diff != "" {
		t.Errorf("second hook saw (-want +got):\n%s", diff)
	}
}

// TestRestart_ReleasesOnlyTheHooksThatLetTheCallThrough pins that a restart
// answering the tool releases only the hooks the interrupted call records
// letting it through, and only while the input stands: a gate added to the
// chain since holds the call, and so does the recorded gate once the restart
// replaces the input.
func TestRestart_ReleasesOnlyTheHooksThatLetTheCallThrough(t *testing.T) {
	setup := func(t *testing.T) (*registry.Registry, *ai.ToolAction[transferIn, string], ai.MiddlewareFunc, *[]string, *ai.ModelResponse, *ai.Part) {
		t.Helper()
		reg := newToolTestRegistry(t)
		transfer, _ := pauseOnce(t, reg)
		var log []string
		a := gate(&log)
		generate := func(opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
			return ai.Generate(context.Background(), reg, append([]ai.GenerateOption{
				ai.WithModelName("test/model"), ai.WithTools(transfer), ai.WithUse(a),
			}, opts...)...)
		}
		resp, err := generate(ai.WithPrompt("transfer 200"))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		restart := restartWith(t, transfer, singleInterrupt(t, resp), map[string]any{"ok": true})
		resp, err = generate(ai.WithMessages(resp.History()...), ai.WithToolRestarts(restart))
		if !errors.Is(err, status.ErrFailedPrecondition) || resp == nil {
			t.Fatalf("answering the gate = (%v, %v), want the tool's own interrupt", resp, err)
		}
		asked := singleInterrupt(t, resp)
		if diff := cmp.Diff([]string{"inline"}, asked.Interrupt.ReleasedBy); diff != "" {
			t.Fatalf("ReleasedBy (-want +got):\n%s", diff)
		}
		return reg, transfer, a, &log, resp, asked
	}

	t.Run("a gate added since holds", func(t *testing.T) {
		reg, transfer, a, _, resp, asked := setup(t)
		var logB []string
		b := gate(&logB)
		_, err := ai.Generate(context.Background(), reg,
			ai.WithModelName("test/model"), ai.WithMessages(viaJSON(t, resp.History())...),
			ai.WithTools(transfer), ai.WithUse(a, b),
			ai.WithToolRestarts(restartWith(t, transfer, asked, map[string]any{"approved": true})))
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Fatalf("resume = %v, want the new gate's hold", err)
		}
		if diff := cmp.Diff([]string{"held"}, logB); diff != "" {
			t.Errorf("new gate saw (-want +got):\n%s", diff)
		}
	})

	t.Run("a replaced input faces the gate again", func(t *testing.T) {
		reg, transfer, a, log, resp, asked := setup(t)
		_, err := ai.Generate(context.Background(), reg,
			ai.WithModelName("test/model"), ai.WithMessages(resp.History()...),
			ai.WithTools(transfer), ai.WithUse(a),
			ai.WithToolRestarts(restartWith(t, transfer, asked, map[string]any{"approved": true},
				ai.WithNewInput(transferIn{Amount: 5000}))))
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Fatalf("resume = %v, want the gate's hold on the new input", err)
		}
		if got := (*log)[len(*log)-1]; got != "held" {
			t.Errorf("gate's last decision = %q, want held; log %v", got, *log)
		}
	})
}
