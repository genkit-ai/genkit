// Copyright 2025 Google LLC
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

package genai

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/firebase/genkit/go/ai"
)

func TestMapRole(t *testing.T) {
	if got := MapRole(ai.RoleModel); got != "assistant" {
		t.Errorf("MapRole(model) = %q, want assistant", got)
	}
	if got := MapRole(ai.RoleUser); got != "user" {
		t.Errorf("MapRole(user) = %q, want user", got)
	}
}

func TestMapPart(t *testing.T) {
	text := MapPart(ai.NewTextPart("hi"))
	if text["type"] != "text" || text["content"] != "hi" {
		t.Errorf("text part = %+v", text)
	}

	tr := MapPart(ai.NewToolRequestPart(&ai.ToolRequest{Name: "getWeather", Ref: "call1", Input: map[string]any{"city": "SF"}}))
	if tr["type"] != "tool_call" || tr["name"] != "getWeather" || tr["id"] != "call1" {
		t.Errorf("tool request part = %+v", tr)
	}

	resp := MapPart(ai.NewToolResponsePart(&ai.ToolResponse{Ref: "call1", Output: map[string]any{"temp": 20}}))
	if resp["type"] != "tool_call_response" || resp["id"] != "call1" {
		t.Errorf("tool response part = %+v", resp)
	}

}

func TestMapMediaPart(t *testing.T) {
	tests := []struct {
		name string
		part *ai.Part
		want map[string]any
	}{
		{
			name: "inline data uri becomes blob without payload",
			part: ai.NewMediaPart("image/png", "data:image/png;base64,AAAAAAAA"),
			want: map[string]any{"type": "blob", "modality": "image", "mime_type": "image/png", "size_bytes": 6},
		},
		{
			name: "base64 padding is excluded from size",
			part: ai.NewMediaPart("audio/wav", "data:audio/wav;base64,AAAAAA=="),
			want: map[string]any{"type": "blob", "modality": "audio", "mime_type": "audio/wav", "size_bytes": 4},
		},
		{
			name: "mime type falls back to the data uri header",
			part: ai.NewMediaPart("", "data:video/mp4;base64,AAAA"),
			want: map[string]any{"type": "blob", "modality": "video", "mime_type": "video/mp4", "size_bytes": 3},
		},
		{
			name: "non-base64 data uri reports raw length",
			part: ai.NewMediaPart("", "data:text/plain,hello"),
			want: map[string]any{"type": "blob", "modality": "document", "mime_type": "text/plain", "size_bytes": 5},
		},
		{
			name: "remote url becomes uri",
			part: ai.NewMediaPart("application/pdf", "gs://bucket/doc.pdf"),
			want: map[string]any{"type": "uri", "modality": "document", "mime_type": "application/pdf", "uri": "gs://bucket/doc.pdf"},
		},
		{
			name: "unknown mime type omits mime_type",
			part: ai.NewMediaPart("", "https://example.com/x"),
			want: map[string]any{"type": "uri", "modality": "document", "uri": "https://example.com/x"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, MapPart(tt.part)); diff != "" {
				t.Errorf("MapPart() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMapPartNilPointers(t *testing.T) {
	// A part whose kind says tool request/response but whose pointer is nil
	// (malformed payload) must not panic.
	tr := MapPart(&ai.Part{Kind: ai.PartToolRequest})
	if tr["type"] != "tool_call" || tr["name"] != "" || tr["arguments"] != nil {
		t.Errorf("nil tool request part = %+v", tr)
	}
	resp := MapPart(&ai.Part{Kind: ai.PartToolResponse})
	if resp["type"] != "tool_call_response" || resp["response"] != nil {
		t.Errorf("nil tool response part = %+v", resp)
	}
}

func TestNormalizeMessages(t *testing.T) {
	msgs := []*ai.Message{
		ai.NewSystemTextMessage("be nice"),
		ai.NewUserTextMessage("hello"),
		ai.NewModelTextMessage("hi there"),
	}
	got := NormalizeMessages(msgs)
	if len(got.SystemInstructions) != 1 {
		t.Fatalf("system instructions = %+v", got.SystemInstructions)
	}
	if got.SystemInstructions[0]["content"] != "be nice" {
		t.Errorf("system instruction content = %+v", got.SystemInstructions[0])
	}
	if len(got.Messages) != 2 {
		t.Fatalf("messages = %+v", got.Messages)
	}
	if got.Messages[0]["role"] != "user" {
		t.Errorf("first message role = %v", got.Messages[0]["role"])
	}
	if got.Messages[1]["role"] != "assistant" {
		t.Errorf("second message role = %v", got.Messages[1]["role"])
	}
}

func TestMapOutputMessage(t *testing.T) {
	got := MapOutputMessage(ai.NewModelTextMessage("done"), "stop")
	if got["role"] != "assistant" || got["finish_reason"] != "stop" {
		t.Errorf("output message = %+v", got)
	}
}

func TestHasToolRequestPart(t *testing.T) {
	with := []*ai.Part{ai.NewTextPart("x"), ai.NewToolRequestPart(&ai.ToolRequest{Name: "t"})}
	if !HasToolRequestPart(with) {
		t.Error("expected true with a tool request part")
	}
	without := []*ai.Part{ai.NewTextPart("x")}
	if HasToolRequestPart(without) {
		t.Error("expected false without a tool request part")
	}
}
