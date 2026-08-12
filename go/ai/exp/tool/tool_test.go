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
	p.Metadata = map[string]any{
		"interrupt": map[string]any{"reason": "large_amount"},
		"keep":      "me",
	}
	return p
}

type resumeData struct {
	Approved bool `json:"approved"`
}

func TestResume_BuildsRestartPart(t *testing.T) {
	part := interruptPart()
	got, err := Resume(part, resumeData{Approved: true})
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if !got.IsToolRequest() {
		t.Fatal("Resume must produce a tool request part")
	}
	if got.IsInterrupt() {
		t.Error("the restart part must not still be marked as an interrupt")
	}
	// Identity (name/ref/input) is preserved so the loop can match the restart.
	if got.ToolRequest.Name != "transfer" || got.ToolRequest.Ref != "call-1" {
		t.Errorf("identity = %q/%q, want transfer/call-1", got.ToolRequest.Name, got.ToolRequest.Ref)
	}
	// Resume data is carried as a JSON object under the "resumed" key.
	resumed, ok := got.Metadata["resumed"].(map[string]any)
	if !ok {
		t.Fatalf("resumed metadata = %T, want map[string]any", got.Metadata["resumed"])
	}
	if resumed["approved"] != true {
		t.Errorf("resumed[approved] = %v, want true", resumed["approved"])
	}
	// Unrelated metadata survives the clone.
	if got.Metadata["keep"] != "me" {
		t.Errorf("unrelated metadata was dropped: %v", got.Metadata)
	}
	// The original part is not mutated.
	if part.Metadata["interrupt"] == nil {
		t.Error("Resume must not mutate the source part's metadata")
	}
}

func TestResume_RejectsNonInterrupt(t *testing.T) {
	if _, err := Resume(nil, resumeData{}); err == nil {
		t.Error("Resume(nil) must error")
	}
	plain := ai.NewToolRequestPart(&ai.ToolRequest{Name: "x"}) // no interrupt metadata
	if _, err := Resume(plain, resumeData{}); err == nil {
		t.Error("Resume of a non-interrupt part must error")
	}
}

func TestResume_NonObjectDataIsClearError(t *testing.T) {
	_, err := Resume(interruptPart(), "scalar")
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

func TestInterruptAs_RoundTrip(t *testing.T) {
	type meta struct {
		Reason string `json:"reason"`
	}
	got, ok := InterruptAs[meta](interruptPart())
	if !ok {
		t.Fatal("InterruptAs failed to decode interrupt metadata")
	}
	if got.Reason != "large_amount" {
		t.Errorf("reason = %q, want %q", got.Reason, "large_amount")
	}
	if _, ok := InterruptAs[meta](ai.NewTextPart("hi")); ok {
		t.Error("InterruptAs on a non-interrupt part must report ok=false")
	}
}

func TestAttachParts_CollectsViaContext(t *testing.T) {
	ctx, collect := NewPartsContext(context.Background())
	AttachParts(ctx, ai.NewTextPart("a"), ai.NewTextPart("b"))
	AttachParts(ctx, ai.NewMediaPart("image/png", "bytes"))
	if got := collect(); len(got) != 3 {
		t.Fatalf("collected %d parts, want 3", len(got))
	}
	// Without a collector in context, AttachParts is a safe no-op.
	AttachParts(context.Background(), ai.NewTextPart("ignored"))
}

func TestOriginalInput_RoundTrip(t *testing.T) {
	type in struct {
		City string `json:"city"`
	}
	ctx := SetOriginalInput(context.Background(), in{City: "Paris"})
	got, ok := OriginalInput[in](ctx)
	if !ok || got.City != "Paris" {
		t.Errorf("OriginalInput = %+v, %v; want {Paris}, true", got, ok)
	}
	if _, ok := OriginalInput[in](context.Background()); ok {
		t.Error("OriginalInput without a stored value must report ok=false")
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
