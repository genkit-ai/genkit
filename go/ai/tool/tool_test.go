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
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

func TestInterrupt_CarriesData(t *testing.T) {
	type payload struct {
		Reason string `json:"reason"`
	}
	err := Interrupt(context.Background(), payload{Reason: "large_amount"})

	var ie *base.ToolInterruptError
	if !errors.As(err, &ie) {
		t.Fatalf("Interrupt returned %T, want *base.ToolInterruptError", err)
	}
	// The payload is normalized to the JSON object it serializes to when the
	// interrupt is raised, the shape it has after a wire hop.
	if got, ok := ie.Data.(map[string]any); !ok || got["reason"] != "large_amount" {
		t.Errorf("interrupt data = %#v, want map{reason: large_amount}", ie.Data)
	}
	if ok, m := ai.IsToolInterruptError(err); !ok || m["reason"] != "large_amount" {
		t.Errorf("ai.IsToolInterruptError = (%v, %v), want (true, {reason: large_amount})", ok, m)
	}
}

// TestInterrupt_BareAndNonObject pins the two edges of the payload contract: a
// nil (or nil map) is a bare interrupt with no data, and a value that is not
// a JSON object is refused at the raise site with a plain error, so it never
// reaches a reader as an interrupt with its payload silently dropped.
func TestInterrupt_BareAndNonObject(t *testing.T) {
	for name, data := range map[string]any{"nil": nil, "nil map": map[string]any(nil)} {
		t.Run(name, func(t *testing.T) {
			err := Interrupt(context.Background(), data)
			var ie *base.ToolInterruptError
			if !errors.As(err, &ie) || ie.Data != nil {
				t.Fatalf("Interrupt(%v) = %#v, want a bare interrupt", data, err)
			}
			if ok, m := ai.IsToolInterruptError(err); !ok || m != nil {
				t.Errorf("ai.IsToolInterruptError = (%v, %v), want (true, nil)", ok, m)
			}
		})
	}
	for name, data := range map[string]any{"string": "approve?", "slice": []string{"a"}, "number": 3} {
		t.Run(name, func(t *testing.T) {
			err := Interrupt(context.Background(), data)
			if ok, _ := ai.IsToolInterruptError(err); ok {
				t.Fatalf("Interrupt(%v) raised an interrupt, want a plain error", data)
			}
			if err == nil || !strings.Contains(err.Error(), "JSON object") {
				t.Errorf("Interrupt(%v) = %v, want an error naming the JSON object constraint", data, err)
			}
		})
	}
}

// partSinkContext mirrors what ai installs around a tool call.
func partSinkContext() (context.Context, func() []*ai.Part) {
	sink := &base.PartSink{}
	ctx := base.ToolPartSinkKey.NewContext(context.Background(), sink)
	return ctx, func() []*ai.Part {
		var parts []*ai.Part
		for _, p := range sink.Close() {
			parts = append(parts, p.(*ai.Part))
		}
		return parts
	}
}

func TestAttachParts_CollectsViaContext(t *testing.T) {
	ctx, collect := partSinkContext()
	AttachParts(ctx, ai.NewTextPart("a"), ai.NewTextPart("b"))
	AttachParts(ctx, ai.NewMediaPart("image/png", "bytes"))
	if got := collect(); len(got) != 3 {
		t.Fatalf("collected %d parts, want 3", len(got))
	}
	// Without a sink in context, AttachParts is a safe no-op.
	AttachParts(context.Background(), ai.NewTextPart("ignored"))
}

// TestAttachParts_Concurrent exercises the documented promise that a tool
// function may attach parts from goroutines it spawns. Meaningful under -race.
func TestAttachParts_Concurrent(t *testing.T) {
	ctx, collect := partSinkContext()
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			AttachParts(ctx, ai.NewTextPart(string(rune('a'+i%26))))
		}()
	}
	wg.Wait()
	if got := collect(); len(got) != 20 {
		t.Errorf("collected %d parts, want 20", len(got))
	}
}

func TestOriginalInput_RoundTrip(t *testing.T) {
	type in struct {
		City string `json:"city"`
	}
	ctx := base.ToolOriginalInputKey.NewContext(context.Background(), map[string]any{"city": "Paris"})
	got, ok := OriginalInput[in](ctx)
	if !ok || got.City != "Paris" {
		t.Errorf("OriginalInput = %+v, %v; want {Paris}, true", got, ok)
	}
	if _, ok := OriginalInput[in](context.Background()); ok {
		t.Error("OriginalInput without a stored value must report ok=false")
	}
}

func TestResumeData_RoundTrip(t *testing.T) {
	type confirmation struct {
		Approved bool `json:"approved"`
	}
	ctx := base.ToolResumeKey.NewContext(context.Background(), map[string]any{"approved": true})
	got, ok := ResumeData[confirmation](ctx)
	if !ok || !got.Approved {
		t.Errorf("ResumeData = %+v, %v; want {true}, true", got, ok)
	}
	// A bare restart carries an empty payload: the call is still a resumption,
	// and the data decodes to the zero value.
	bare := base.ToolResumeKey.NewContext(context.Background(), map[string]any{})
	if got, ok := ResumeData[confirmation](bare); !ok || got.Approved {
		t.Errorf("ResumeData on a bare restart = %+v, %v; want {false}, true", got, ok)
	}
	if _, ok := ResumeData[confirmation](context.Background()); ok {
		t.Error("ResumeData outside a resumed call must report ok=false")
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
