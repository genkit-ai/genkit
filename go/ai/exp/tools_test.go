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
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp/tool"
	"github.com/firebase/genkit/go/internal/registry"
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

// defineToolThenFinishModel defines "test/model": on the first turn it returns
// reqs (typically tool requests), and once a tool response is in history it
// returns the final text "done". This drives a single tool round per Generate.
func defineToolThenFinishModel(reg *registry.Registry, reqs ...*ai.Part) {
	ai.DefineModel(reg, "test/model",
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

// TestDefineTool_PlainContext exercises the headline ergonomics: a tool whose
// function takes a plain context.Context (not ai.ToolContext), driven end to
// end through Generate.
func TestDefineTool_PlainContext(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "getWeather", Input: map[string]any{"city": "Paris"},
	}))

	var gotCity string
	weather := DefineTool(reg, "getWeather", "fetches the weather",
		func(ctx context.Context, in weatherIn) (string, error) {
			gotCity = in.City
			return "Sunny", nil
		})

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("weather in Paris?"),
		ai.WithTools(weather))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if gotCity != "Paris" {
		t.Errorf("tool saw city %q, want %q", gotCity, "Paris")
	}
	if got := resp.Text(); got != "done" {
		t.Errorf("final text = %q, want %q", got, "done")
	}
}

// TestTool_AttachParts verifies AttachParts folds extra content into the tool's
// multipart response without changing the function signature.
func TestTool_AttachParts(t *testing.T) {
	reg := newToolTestRegistry(t)
	shot := DefineTool(reg, "screenshot", "takes a screenshot",
		func(ctx context.Context, _ struct{}) (string, error) {
			tool.AttachParts(ctx, ai.NewMediaPart("image/png", "pngbytes"))
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

// TestTool_SendPartialNoOpWithoutStreaming confirms SendPartial is a safe no-op
// when no streaming callback is wired (here, a direct RunRaw).
func TestTool_SendPartialNoOpWithoutStreaming(t *testing.T) {
	reg := newToolTestRegistry(t)
	tl := DefineTool(reg, "noop", "streams when it can",
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

// TestDefineInterruptibleTool_TypedRoundTrip pins the core interrupt/resume
// contract: the tool interrupts with typed data on the first pass, the caller
// reads it with InterruptAs, resumes with typed data via the tool's Resume, and
// the resumed value reaches the function's *Res parameter on re-execution.
func TestDefineInterruptibleTool_TypedRoundTrip(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "transfer", Input: map[string]any{"amount": 200},
	}))

	var gotResume *confirmation
	transfer := DefineInterruptibleTool(reg, "transfer", "transfers money",
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

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("transfer 200"),
		ai.WithTools(transfer))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %d (finish=%s)", len(interrupts), resp.FinishReason)
	}

	meta, ok := tool.InterruptAs[transferInterrupt](interrupts[0])
	if !ok {
		t.Fatal("InterruptAs failed to decode the typed interrupt data")
	}
	if meta.Reason != "large_amount" || meta.Amount != 200 {
		t.Errorf("interrupt data = %+v, want {large_amount 200}", meta)
	}

	restart, err := transfer.Resume(interrupts[0], confirmation{Approved: true})
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}

	resp2, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
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

// TestInterruptibleTool_ValidatesOwnership checks the typed Resume/Respond
// reject an interrupt part that belongs to a different tool.
func TestInterruptibleTool_ValidatesOwnership(t *testing.T) {
	mine := NewInterruptibleTool("mine", "d",
		func(ctx context.Context, _ struct{}, _ *confirmation) (string, error) { return "", nil })

	foreign := ai.NewToolRequestPart(&ai.ToolRequest{Name: "other"})
	foreign.Metadata = map[string]any{"interrupt": true}

	if _, err := mine.Resume(foreign, confirmation{Approved: true}); err == nil {
		t.Error("Resume must reject a part for a different tool")
	}
	if _, err := mine.Respond(foreign, "out"); err == nil {
		t.Error("Respond must reject a part for a different tool")
	}
}

// TestResume_NonObjectData_ReturnsClearError covers the documented constraint:
// resume data must serialize to a JSON object, and a scalar yields an
// actionable error rather than an opaque json failure.
func TestResume_NonObjectData_ReturnsClearError(t *testing.T) {
	part := ai.NewToolRequestPart(&ai.ToolRequest{Name: "x"})
	part.Metadata = map[string]any{"interrupt": true}

	_, err := tool.Resume(part, "just a string")
	if err == nil {
		t.Fatal("expected an error resuming with non-object data")
	}
	if !strings.Contains(err.Error(), "JSON object") {
		t.Errorf("error = %q, want it to mention the JSON object constraint", err)
	}
}

// TestInterrupt_NonObjectData_ReturnsClearError covers the same constraint on
// the interrupt side: interrupting with a scalar surfaces a clear error when
// the tool runs.
func TestInterrupt_NonObjectData_ReturnsClearError(t *testing.T) {
	reg := newToolTestRegistry(t)
	tl := DefineInterruptibleTool(reg, "bad", "interrupts with a scalar",
		func(ctx context.Context, _ struct{}, _ *struct{}) (string, error) {
			return "", tool.Interrupt("not an object")
		})

	_, err := tl.RunRaw(context.Background(), struct{}{})
	if err == nil {
		t.Fatal("expected an error interrupting with non-object data")
	}
	if !strings.Contains(err.Error(), "JSON object") {
		t.Errorf("error = %q, want it to mention the JSON object constraint", err)
	}
}

// TestSendPartial_StreamsPartialToolResponse asserts a tool's SendPartial calls
// arrive on the stream as partial tool responses, distinguishable via
// IsPartial / ToolResponses.
func TestSendPartial_StreamsPartialToolResponse(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "progressTool", Input: map[string]any{}}))

	DefineTool(reg, "progressTool", "streams progress",
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
	DefineTool(reg, "toolA", "streams", streamer)
	DefineTool(reg, "toolB", "streams", streamer)

	for _, err := range ai.GenerateStream(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("go"),
		ai.WithTools(ai.ToolName("toolA"), ai.ToolName("toolB"))) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
	}
}
