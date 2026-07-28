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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

type resumeData struct {
	Approved bool `json:"approved"`
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
