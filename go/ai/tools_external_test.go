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
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/core/api"
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
