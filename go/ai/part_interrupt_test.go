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

package ai

import (
	"strings"
	"testing"
)

// interruptPart builds a minimal interrupted tool-request part for the
// [Part.InterruptAs], [Part.ToRestart], and [Part.ToResponse] tests.
func interruptPart() *Part {
	p := NewToolRequestPart(&ToolRequest{
		Name:  "transfer",
		Ref:   "call-1",
		Input: map[string]any{"amount": float64(200)},
	})
	p.Interrupt = &ToolInterrupt{Data: map[string]any{"reason": "large_amount"}}
	p.Metadata = map[string]any{"keep": "me"}
	return p
}

type resumeData struct {
	Approved bool `json:"approved"`
}

func TestToRestart_BuildsRestartPart(t *testing.T) {
	part := interruptPart()
	got, err := part.ToRestart(WithResume(resumeData{Approved: true}))
	if err != nil {
		t.Fatalf("ToRestart: %v", err)
	}
	if !got.IsToolRequest() {
		t.Fatal("ToRestart must produce a tool request part")
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
		t.Error("ToRestart must not mutate the source part's interrupt state")
	}
}

func TestToRestart_WithNewInput(t *testing.T) {
	part := interruptPart()
	newInput := map[string]any{"amount": float64(50)}
	got, err := part.ToRestart(WithResume(resumeData{Approved: true}), WithNewInput(newInput))
	if err != nil {
		t.Fatalf("ToRestart: %v", err)
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

func TestToRestart_Bare(t *testing.T) {
	part := interruptPart()
	got, err := part.ToRestart(WithNewInput(map[string]any{"amount": float64(50)}))
	if err != nil {
		t.Fatalf("ToRestart: %v", err)
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

func TestToRestart_RejectsDuplicateNewInput(t *testing.T) {
	_, err := interruptPart().ToRestart(WithResume(resumeData{Approved: true}),
		WithNewInput(map[string]any{"a": 1}), WithNewInput(map[string]any{"a": 2}))
	if err == nil {
		t.Error("ToRestart with WithNewInput twice must error")
	}
}

func TestToRestart_RejectsNonInterrupt(t *testing.T) {
	var nilPart *Part
	if _, err := nilPart.ToRestart(); err == nil {
		t.Error("ToRestart on a nil part must error")
	}
	plain := NewToolRequestPart(&ToolRequest{Name: "x"}) // no interrupt state
	if _, err := plain.ToRestart(); err == nil {
		t.Error("ToRestart of a non-interrupt part must error")
	}
}

func TestToRestart_NonObjectDataIsClearError(t *testing.T) {
	_, err := interruptPart().ToRestart(WithResume("scalar"))
	if err == nil {
		t.Fatal("expected an error for non-object resume data")
	}
	if !strings.Contains(err.Error(), "JSON object") {
		t.Errorf("error = %q, want it to mention the JSON object constraint", err)
	}
}

func TestToResponse_BuildsResponsePart(t *testing.T) {
	got, err := interruptPart().ToResponse(map[string]any{"status": "cancelled"})
	if err != nil {
		t.Fatalf("ToResponse: %v", err)
	}
	if !got.IsToolResponse() {
		t.Fatal("ToResponse must produce a tool response part")
	}
	if got.ToolResponse.Name != "transfer" || got.ToolResponse.Ref != "call-1" {
		t.Errorf("identity = %q/%q, want transfer/call-1", got.ToolResponse.Name, got.ToolResponse.Ref)
	}
	if got.Metadata["interruptResponse"] != true {
		t.Errorf("interruptResponse metadata = %v, want true", got.Metadata["interruptResponse"])
	}

	plain := NewToolRequestPart(&ToolRequest{Name: "x"})
	if _, err := plain.ToResponse("out"); err == nil {
		t.Error("ToResponse of a non-interrupt part must error")
	}
}

func TestInterruptRestarts_BlanketRestartsEveryInterrupt(t *testing.T) {
	first := interruptPart()
	second := interruptPart()
	second.ToolRequest.Ref = "call-2"

	resp := &ModelResponse{
		Message: &Message{
			Role: RoleModel,
			Content: []*Part{
				NewTextPart("some prose"), // ignored
				first,                     // restarted
				NewToolRequestPart(&ToolRequest{Name: "plain"}), // not an interrupt
				second, // restarted
			},
		},
	}

	got := resp.InterruptRestarts()
	if len(got) != 2 {
		t.Fatalf("InterruptRestarts() returned %d parts, want 2", len(got))
	}
	for i, p := range got {
		if !p.IsToolRequest() {
			t.Errorf("part %d is not a tool request", i)
		}
		if p.IsInterrupt() {
			t.Errorf("part %d must not still be marked as an interrupt", i)
		}
		// A bare restart carries restart state with no resume payload.
		if p.Restart == nil || p.Restart.Resume != nil {
			t.Errorf("part %d = %+v, want bare restart state (nil Resume)", i, p.Restart)
		}
	}
	if got[0].ToolRequest.Ref != "call-1" || got[1].ToolRequest.Ref != "call-2" {
		t.Errorf("refs = %q/%q, want call-1/call-2", got[0].ToolRequest.Ref, got[1].ToolRequest.Ref)
	}
	// Sources are not mutated.
	if !first.IsInterrupt() || !second.IsInterrupt() {
		t.Error("InterruptRestarts must not mutate the source parts")
	}
}

func TestInterruptRestarts_EmptyWhenNoInterrupts(t *testing.T) {
	var nilResp *ModelResponse
	if got := nilResp.InterruptRestarts(); len(got) != 0 {
		t.Errorf("InterruptRestarts() on a nil response = %v, want empty", got)
	}
	resp := &ModelResponse{Message: NewModelTextMessage("no interrupts here")}
	if got := resp.InterruptRestarts(); len(got) != 0 {
		t.Errorf("InterruptRestarts() = %v, want empty", got)
	}
}

func TestInterruptAs_RoundTrip(t *testing.T) {
	type meta struct {
		Reason string `json:"reason"`
	}
	got, ok := interruptPart().InterruptAs[meta]()
	if !ok {
		t.Fatal("InterruptAs failed to decode interrupt metadata")
	}
	if got.Reason != "large_amount" {
		t.Errorf("reason = %q, want %q", got.Reason, "large_amount")
	}
	if _, ok := NewTextPart("hi").InterruptAs[meta](); ok {
		t.Error("InterruptAs on a non-interrupt part must report ok=false")
	}
	var nilPart *Part
	if _, ok := nilPart.InterruptAs[meta](); ok {
		t.Error("InterruptAs on a nil part must report ok=false")
	}
	bare := NewToolRequestPart(&ToolRequest{Name: "x"})
	bare.Interrupt = &ToolInterrupt{}
	if _, ok := bare.InterruptAs[meta](); ok {
		t.Error("InterruptAs on a bare (no data) interrupt must report ok=false")
	}
}
