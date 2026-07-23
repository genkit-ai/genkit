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

package tool

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

// interruptPart builds a minimal interrupted tool-request part for the helpers
// under test.
func interruptPart() *ai.Part {
	p := ai.NewToolRequestPart(&ai.ToolRequest{
		Name:  "transfer",
		Ref:   "call-1",
		Input: map[string]any{"amount": float64(200)},
	})
	p.Interrupt = &ai.ToolInterrupt{Data: map[string]any{"reason": "large_amount"}}
	p.Metadata = map[string]any{"keep": "me"}
	return p
}

type resumeData struct {
	Approved bool `json:"approved"`
}

func TestRestart_BuildsRestartPart(t *testing.T) {
	part := interruptPart()
	got, err := Restart(part, ai.WithResume(resumeData{Approved: true}))
	if err != nil {
		t.Fatalf("Restart: %v", err)
	}
	if !got.IsToolRequest() {
		t.Fatal("Restart must produce a tool request part")
	}
	if got.IsInterrupt() {
		t.Error("the restart part must not still be marked as an interrupt")
	}
	// Identity (name/ref/input) is preserved so the loop can match the restart.
	if got.ToolRequest.Name != "transfer" || got.ToolRequest.Ref != "call-1" {
		t.Errorf("identity = %q/%q, want transfer/call-1", got.ToolRequest.Name, got.ToolRequest.Ref)
	}
	// Resume data travels on the typed Restart state.
	if got.Restart == nil {
		t.Fatal("restart part must carry Restart state")
	}
	resume, ok := got.Restart.Resume.(resumeData)
	if !ok {
		t.Fatalf("Restart.Resume = %T, want resumeData", got.Restart.Resume)
	}
	if !resume.Approved {
		t.Errorf("Restart.Resume = %+v, want Approved=true", resume)
	}
	// Unrelated metadata survives the clone.
	if got.Metadata["keep"] != "me" {
		t.Errorf("unrelated metadata was dropped: %v", got.Metadata)
	}
	// The original part is not mutated.
	if !part.IsInterrupt() {
		t.Error("Restart must not mutate the source part's interrupt state")
	}
}

func TestRestart_WithNewInput(t *testing.T) {
	part := interruptPart()
	newInput := map[string]any{"amount": float64(50)}
	got, err := Restart(part, ai.WithResume(resumeData{Approved: true}), ai.WithNewInput(newInput))
	if err != nil {
		t.Fatalf("Restart: %v", err)
	}
	gotInput, ok := got.ToolRequest.Input.(map[string]any)
	if !ok || gotInput["amount"] != float64(50) {
		t.Errorf("restart input = %v, want replaced input %v", got.ToolRequest.Input, newInput)
	}
	// The original input travels on the Restart state for tool.OriginalInput.
	if got.Restart == nil {
		t.Fatal("restart part must carry Restart state")
	}
	orig, ok := got.Restart.OriginalInput.(map[string]any)
	if !ok || orig["amount"] != float64(200) {
		t.Errorf("Restart.OriginalInput = %v, want the original input", got.Restart.OriginalInput)
	}
}

func TestRestart_Bare(t *testing.T) {
	part := interruptPart()
	got, err := Restart(part, ai.WithNewInput(map[string]any{"amount": float64(50)}))
	if err != nil {
		t.Fatalf("Restart: %v", err)
	}
	// A nil resume payload marks a bare restart.
	if got.Restart == nil || got.Restart.Resume != nil {
		t.Errorf("Restart = %+v, want bare restart state (nil Resume)", got.Restart)
	}
	if got.Interrupt != nil {
		t.Error("interrupt state must be cleared")
	}
	if got.Metadata["keep"] != "me" {
		t.Error("unrelated metadata must be preserved")
	}
}

func TestRestart_RejectsDuplicateNewInput(t *testing.T) {
	_, err := Restart(interruptPart(), ai.WithResume(resumeData{Approved: true}),
		ai.WithNewInput(map[string]any{"a": 1}), ai.WithNewInput(map[string]any{"a": 2}))
	if err == nil {
		t.Error("Restart with ai.WithNewInput twice must error")
	}
}

func TestRestart_RejectsNonInterrupt(t *testing.T) {
	if _, err := Restart(nil); err == nil {
		t.Error("Restart(nil) must error")
	}
	plain := ai.NewToolRequestPart(&ai.ToolRequest{Name: "x"}) // no interrupt state
	if _, err := Restart(plain); err == nil {
		t.Error("Restart of a non-interrupt part must error")
	}
}

func TestRestart_NonObjectDataIsClearError(t *testing.T) {
	_, err := Restart(interruptPart(), ai.WithResume("scalar"))
	if err == nil {
		t.Fatal("expected an error for non-object resume data")
	}
	if !strings.Contains(err.Error(), "JSON object") {
		t.Errorf("error = %q, want it to mention the JSON object constraint", err)
	}
}

func TestRespond_BuildsResponsePart(t *testing.T) {
	got, err := Respond(interruptPart(), map[string]any{"status": "cancelled"})
	if err != nil {
		t.Fatalf("Respond: %v", err)
	}
	if !got.IsToolResponse() {
		t.Fatal("Respond must produce a tool response part")
	}
	if got.ToolResponse.Name != "transfer" || got.ToolResponse.Ref != "call-1" {
		t.Errorf("identity = %q/%q, want transfer/call-1", got.ToolResponse.Name, got.ToolResponse.Ref)
	}
	if got.Metadata["interruptResponse"] != true {
		t.Errorf("interruptResponse metadata = %v, want true", got.Metadata["interruptResponse"])
	}

	if _, err := Respond(ai.NewToolRequestPart(&ai.ToolRequest{Name: "x"}), "out"); err == nil {
		t.Error("Respond of a non-interrupt part must error")
	}
}

func TestInterruptData_RoundTrip(t *testing.T) {
	type meta struct {
		Reason string `json:"reason"`
	}
	got, ok := InterruptData[meta](interruptPart())
	if !ok {
		t.Fatal("InterruptData failed to decode interrupt metadata")
	}
	if got.Reason != "large_amount" {
		t.Errorf("reason = %q, want %q", got.Reason, "large_amount")
	}
	if _, ok := InterruptData[meta](ai.NewTextPart("hi")); ok {
		t.Error("InterruptData on a non-interrupt part must report ok=false")
	}
	if _, ok := InterruptData[meta](nil); ok {
		t.Error("InterruptData on a nil part must report ok=false")
	}
	bare := ai.NewToolRequestPart(&ai.ToolRequest{Name: "x"})
	bare.Interrupt = &ai.ToolInterrupt{}
	if _, ok := InterruptData[meta](bare); ok {
		t.Error("InterruptData on a bare (no data) interrupt must report ok=false")
	}
}

func TestAttachParts_CollectsViaContext(t *testing.T) {
	var got []*ai.Part
	ctx := base.ToolPartSinkKey.NewContext(context.Background(), func(part any) {
		if p, ok := part.(*ai.Part); ok {
			got = append(got, p)
		}
	})
	AttachParts(ctx, ai.NewTextPart("a"), ai.NewTextPart("b"))
	AttachParts(ctx, ai.NewMediaPart("image/png", "bytes"))
	if len(got) != 3 {
		t.Fatalf("collected %d parts, want 3", len(got))
	}
	// Without a sink in context, AttachParts is a safe no-op.
	AttachParts(context.Background(), ai.NewTextPart("ignored"))
}

func TestOriginalInput_RoundTrip(t *testing.T) {
	type in struct {
		City string `json:"city"`
	}
	ctx := base.ToolOriginalInputKey.NewContext(context.Background(), in{City: "Paris"})
	got, ok := OriginalInput[in](ctx)
	if !ok || got.City != "Paris" {
		t.Errorf("OriginalInput = %+v, %v; want {Paris}, true", got, ok)
	}
	if _, ok := OriginalInput[in](context.Background()); ok {
		t.Error("OriginalInput without a stored value must report ok=false")
	}
}

func TestResumedAs_RoundTrip(t *testing.T) {
	ctx := base.ToolResumeKey.NewContext(context.Background(),
		map[string]any{"approved": true})
	got, ok := ResumeData[resumeData](ctx)
	if !ok || !got.Approved {
		t.Errorf("ResumedAs = %+v, %v; want {Approved:true}, true", got, ok)
	}
	if _, ok := ResumeData[resumeData](context.Background()); ok {
		t.Error("ResumedAs without resume metadata must report ok=false")
	}
}

func TestSendPartial_InvokesSenderWhenPresent(t *testing.T) {
	var got any
	ctx := base.ToolPartialSenderKey.NewContext(context.Background(),
		func(_ context.Context, output any) { got = output })
	SendPartial(ctx, map[string]any{"progress": 50})
	m, ok := got.(map[string]any)
	if !ok || m["progress"] != 50 {
		t.Errorf("sender received %v, want {progress:50}", got)
	}
	// No sender wired: no-op, no panic.
	SendPartial(context.Background(), "ignored")
}

func TestSendChunk_InvokesSenderWhenPresent(t *testing.T) {
	var got *ai.ModelResponseChunk
	ctx := base.ToolChunkSenderKey.NewContext(context.Background(),
		func(_ context.Context, chunk any) { got, _ = chunk.(*ai.ModelResponseChunk) })
	want := &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart("hi")}}
	SendChunk(ctx, want)
	if got != want {
		t.Errorf("sender received %v, want %v", got, want)
	}
	// No sender wired: no-op, no panic.
	SendChunk(context.Background(), want)
}
