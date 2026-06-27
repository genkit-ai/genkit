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

import "testing"

func TestNewPartialToolResponsePart(t *testing.T) {
	p := NewPartialToolResponsePart(&ToolResponse{Name: "t", Output: map[string]any{"progress": 50}})
	if !p.IsToolResponse() {
		t.Fatal("partial part must be a tool response")
	}
	if !p.IsPartial() {
		t.Error("NewPartialToolResponsePart must be marked partial")
	}
	if NewToolResponsePart(&ToolResponse{Name: "t"}).IsPartial() {
		t.Error("a plain tool response must not be partial")
	}
	if NewToolRequestPart(&ToolRequest{Name: "t"}).IsPartial() {
		t.Error("a tool request must not be partial")
	}
	if (*Part)(nil).IsPartial() {
		t.Error("nil part must not be partial")
	}
}

func TestModelResponseChunk_ToolResponses(t *testing.T) {
	partial := NewPartialToolResponsePart(&ToolResponse{Name: "t", Output: "progress"})
	final := NewToolResponsePart(&ToolResponse{Name: "t", Output: "done"})
	chunk := &ModelResponseChunk{Content: []*Part{NewTextPart("hi"), partial, final}}

	got := chunk.ToolResponses()
	if len(got) != 2 {
		t.Fatalf("ToolResponses() len = %d, want 2", len(got))
	}
	if !got[0].IsPartial() {
		t.Error("first tool response should be the streamed partial")
	}
	if got[1].IsPartial() {
		t.Error("second tool response should be the final (non-partial) result")
	}

	if got := (*ModelResponseChunk)(nil).ToolResponses(); len(got) != 0 {
		t.Errorf("nil chunk ToolResponses() = %v, want empty", got)
	}
}

func TestModelResponseChunk_TextIgnoresNonTextParts(t *testing.T) {
	// A lone non-text part (here reasoning) must not leak into Text(); the old
	// single-part fast path returned Content[0].Text regardless of kind.
	if got := (&ModelResponseChunk{Content: []*Part{NewReasoningPart("thinking", nil)}}).Text(); got != "" {
		t.Errorf("Text() with a single reasoning part = %q, want empty", got)
	}
	// Text and data parts concatenate; other kinds are skipped.
	mixed := &ModelResponseChunk{Content: []*Part{
		NewTextPart("a"),
		NewReasoningPart("ignore", nil),
		NewTextPart("b"),
	}}
	if got := mixed.Text(); got != "ab" {
		t.Errorf("Text() = %q, want %q", got, "ab")
	}
	if got := (*ModelResponseChunk)(nil).Text(); got != "" {
		t.Errorf("nil chunk Text() = %q, want empty", got)
	}
}
