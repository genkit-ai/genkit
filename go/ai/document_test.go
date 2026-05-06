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

package ai

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestDocumentFromText(t *testing.T) {
	const data = "robot overlord"
	d := DocumentFromText(data, nil)
	if len(d.Content) != 1 {
		t.Fatalf("got %d parts, want 1", len(d.Content))
	}
	p := d.Content[0]
	if !p.IsText() {
		t.Errorf("IsText() == %t, want %t", p.IsText(), true)
	}
	if got := p.Text; got != data {
		t.Errorf("Data() == %q, want %q", got, data)
	}
}

// TODO: verify that this works with the data that genkit passes.
func TestDocumentJSON(t *testing.T) {
	d := Document{
		Content: []*Part{
			&Part{
				Kind: PartText,
				Text: "hi",
			},
			&Part{
				Kind:        PartMedia,
				ContentType: "text/plain",
				Text:        "data:,bye",
			},
			&Part{
				Kind: PartData,
				Text: "somedata\x00string",
			},
			&Part{
				Kind: PartToolRequest,
				ToolRequest: &ToolRequest{
					Name:  "tool1",
					Input: map[string]any{"arg1": 3.3, "arg2": "foo"},
				},
			},
			&Part{
				Kind: PartToolResponse,
				ToolResponse: &ToolResponse{
					Name:   "tool1",
					Output: map[string]any{"res1": 4.4, "res2": "bar"},
				},
			},
		},
	}

	b, err := json.Marshal(&d)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("marshaled:%s\n", string(b))

	var d2 Document
	if err := json.Unmarshal(b, &d2); err != nil {
		t.Fatal(err)
	}

	cmpPart := func(a, b *Part) bool {
		if a.Kind != b.Kind {
			return false
		}
		switch a.Kind {
		case PartText:
			return a.Text == b.Text
		case PartMedia:
			return a.ContentType == b.ContentType && a.Text == b.Text
		case PartData:
			return a.Text == b.Text
		case PartToolRequest:
			return reflect.DeepEqual(a.ToolRequest, b.ToolRequest)
		case PartToolResponse:
			return reflect.DeepEqual(a.ToolResponse, b.ToolResponse)
		default:
			t.Fatalf("bad part kind %v", a.Kind)
			return false
		}
	}

	diff := cmp.Diff(d, d2, cmp.Comparer(cmpPart))
	if diff != "" {
		t.Errorf("mismatch (-want, +got)\n%s", diff)
	}
}

func TestReasoningPartJSON(t *testing.T) {
	reasoningText := "This is my reasoning process"
	signature := []byte("sig123")

	originalPart := NewReasoningPart(reasoningText, signature)

	b, err := json.Marshal(originalPart)
	if err != nil {
		t.Fatalf("failed to marshal reasoning part: %v", err)
	}

	t.Logf("marshaled reasoning part: %s\n", string(b))

	var unmarshaledPart Part
	if err := json.Unmarshal(b, &unmarshaledPart); err != nil {
		t.Fatalf("failed to unmarshal reasoning part: %v", err)
	}

	if !unmarshaledPart.IsReasoning() {
		t.Errorf("unmarshaled part is not reasoning, got kind: %v", unmarshaledPart.Kind)
	}

	if unmarshaledPart.Text != reasoningText {
		t.Errorf("unmarshaled reasoning text = %q, want %q", unmarshaledPart.Text, reasoningText)
	}

	if unmarshaledPart.ContentType != "plain/text" {
		t.Errorf("unmarshaled reasoning content type = %q, want %q", unmarshaledPart.ContentType, "plain/text")
	}
}

func TestNewDataPart(t *testing.T) {
	t.Run("creates data part with content", func(t *testing.T) {
		p := NewDataPart("some binary data")

		if p.Kind != PartData {
			t.Errorf("Kind = %v, want %v", p.Kind, PartData)
		}
		if p.Text != "some binary data" {
			t.Errorf("Text = %q, want %q", p.Text, "some binary data")
		}
	})

	t.Run("creates data part with empty content", func(t *testing.T) {
		p := NewDataPart("")

		if p.Kind != PartData {
			t.Errorf("Kind = %v, want %v", p.Kind, PartData)
		}
		if p.Text != "" {
			t.Errorf("Text = %q, want empty string", p.Text)
		}
	})
}

func TestNewCustomPart(t *testing.T) {
	t.Run("creates custom part with value", func(t *testing.T) {
		custom := map[string]any{"key": "value", "count": 42}
		p := NewCustomPart(custom)

		if p.Kind != PartCustom {
			t.Errorf("Kind = %v, want %v", p.Kind, PartCustom)
		}
		if p.Custom == nil {
			t.Fatal("Custom is nil")
		}
		if p.Custom["key"] != "value" {
			t.Errorf("Custom[key] = %v, want %q", p.Custom["key"], "value")
		}
	})

	t.Run("creates custom part with nil value", func(t *testing.T) {
		p := NewCustomPart(nil)

		if p.Kind != PartCustom {
			t.Errorf("Kind = %v, want %v", p.Kind, PartCustom)
		}
		if p.Custom != nil {
			t.Errorf("Custom = %v, want nil", p.Custom)
		}
	})
}

func TestPartIsData(t *testing.T) {
	tests := []struct {
		name string
		part *Part
		want bool
	}{
		{"data part", NewDataPart("{}"), true},
		{"text part", NewTextPart("hello"), false},
		{"media part", NewMediaPart("image/png", "data:..."), false},
		{"nil part", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.part.IsData()
			if got != tt.want {
				t.Errorf("IsData() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPartIsInterrupt(t *testing.T) {
	t.Run("interrupt tool request returns true", func(t *testing.T) {
		p := &Part{
			Kind: PartToolRequest,
			ToolRequest: &ToolRequest{
				Name:  "test",
				Input: map[string]any{},
			},
			Metadata: map[string]any{
				"interrupt": true,
			},
		}

		if !p.IsInterrupt() {
			t.Error("IsInterrupt() = false, want true")
		}
	})

	t.Run("non-interrupt tool request returns false", func(t *testing.T) {
		p := &Part{
			Kind: PartToolRequest,
			ToolRequest: &ToolRequest{
				Name:  "test",
				Input: map[string]any{},
			},
		}

		if p.IsInterrupt() {
			t.Error("IsInterrupt() = true, want false")
		}
	})

	t.Run("non-tool-request part returns false", func(t *testing.T) {
		p := NewTextPart("hello")

		if p.IsInterrupt() {
			t.Error("IsInterrupt() = true, want false")
		}
	})

	t.Run("nil part returns false", func(t *testing.T) {
		var p *Part
		if p.IsInterrupt() {
			t.Error("IsInterrupt() = true, want false")
		}
	})
}

func TestPartIsCustom(t *testing.T) {
	tests := []struct {
		name string
		part *Part
		want bool
	}{
		{"custom part", NewCustomPart(map[string]any{"key": "value"}), true},
		{"text part", NewTextPart("hello"), false},
		{"data part", NewDataPart("data"), false},
		{"nil part", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.part.IsCustom()
			if got != tt.want {
				t.Errorf("IsCustom() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsImageContentType(t *testing.T) {
	tests := []struct {
		contentType string
		want        bool
	}{
		{"image/png", true},
		{"image/jpeg", true},
		{"image/gif", true},
		{"image/webp", true},
		{"data:image/png;base64,...", true},
		{"video/mp4", false},
		{"audio/mp3", false},
		{"text/plain", false},
		{"application/json", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.contentType, func(t *testing.T) {
			got := IsImageContentType(tt.contentType)
			if got != tt.want {
				t.Errorf("IsImageContentType(%q) = %v, want %v", tt.contentType, got, tt.want)
			}
		})
	}
}

func TestIsVideoContentType(t *testing.T) {
	tests := []struct {
		contentType string
		want        bool
	}{
		{"video/mp4", true},
		{"video/webm", true},
		{"video/mpeg", true},
		{"data:video/mp4;base64,...", true},
		{"image/png", false},
		{"audio/mp3", false},
		{"text/plain", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.contentType, func(t *testing.T) {
			got := IsVideoContentType(tt.contentType)
			if got != tt.want {
				t.Errorf("IsVideoContentType(%q) = %v, want %v", tt.contentType, got, tt.want)
			}
		})
	}
}

func TestIsAudioContentType(t *testing.T) {
	tests := []struct {
		contentType string
		want        bool
	}{
		{"audio/mp3", true},
		{"audio/wav", true},
		{"audio/ogg", true},
		{"audio/mpeg", true},
		{"data:audio/mp3;base64,...", true},
		{"image/png", false},
		{"video/mp4", false},
		{"text/plain", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.contentType, func(t *testing.T) {
			got := IsAudioContentType(tt.contentType)
			if got != tt.want {
				t.Errorf("IsAudioContentType(%q) = %v, want %v", tt.contentType, got, tt.want)
			}
		})
	}
}

func TestNewResponseForToolRequest(t *testing.T) {
	t.Run("creates tool response for tool request part", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name:  "calculator",
			Input: map[string]any{"a": 1, "b": 2},
		})
		output := map[string]any{"result": 3}

		resp := NewResponseForToolRequest(reqPart, output)

		if resp.Kind != PartToolResponse {
			t.Errorf("Kind = %v, want %v", resp.Kind, PartToolResponse)
		}
		if resp.ToolResponse == nil {
			t.Fatal("ToolResponse is nil")
		}
		if resp.ToolResponse.Name != "calculator" {
			t.Errorf("Name = %q, want %q", resp.ToolResponse.Name, "calculator")
		}
		if resp.ToolResponse.Output.(map[string]any)["result"] != 3 {
			t.Errorf("Output mismatch")
		}
	})

	t.Run("preserves ref from original request", func(t *testing.T) {
		reqPart := NewToolRequestPart(&ToolRequest{
			Name: "tool",
			Ref:  "request-123",
		})

		resp := NewResponseForToolRequest(reqPart, "output")

		if resp.ToolResponse.Ref != "request-123" {
			t.Errorf("Ref = %q, want %q", resp.ToolResponse.Ref, "request-123")
		}
	})

	t.Run("returns nil for non-tool-request part", func(t *testing.T) {
		textPart := NewTextPart("not a tool request")

		resp := NewResponseForToolRequest(textPart, "output")

		if resp != nil {
			t.Error("expected nil for non-tool-request part")
		}
	})
}

// TestPartClone verifies that Part.Clone produces an independent copy.
// Every Part field is populated so that adding a new field without updating
// this test (and Clone) causes a failure.
func TestPartClone(t *testing.T) {
	orig := &Part{
		Kind:        PartToolRequest,
		ContentType: "application/json",
		Text:        "body",
		ToolRequest: &ToolRequest{Name: "tool", Input: map[string]any{"a": 1}},
		// Normally a Part wouldn't have both ToolRequest and ToolResponse,
		// but we populate everything to catch missing fields.
		ToolResponse: &ToolResponse{Name: "tool", Output: "ok"},
		Resource:     &ResourcePart{Uri: "res://x"},
		Custom:       map[string]any{"ck": "cv"},
		Metadata:     map[string]any{"sig": []byte{1, 2, 3}, "key": "val"},
	}

	// Guard: every field in the fixture must be non-zero.
	// If someone adds a new field to Part this will fail, forcing them to
	// add it here and verify Clone handles it.
	rv := reflect.ValueOf(orig).Elem()
	for i := range rv.NumField() {
		if rv.Field(i).IsZero() {
			t.Fatalf("Part field %q is zero in test fixture — populate it and verify Clone handles it", rv.Type().Field(i).Name)
		}
	}

	cp := orig.Clone()

	// Values must match.
	if !reflect.DeepEqual(orig, cp) {
		t.Fatal("Clone() values differ from original")
	}

	// Mutating clone's maps must not affect the original.
	cp.Metadata["extra"] = true
	if _, ok := orig.Metadata["extra"]; ok {
		t.Error("mutating clone Metadata affected original")
	}

	cp.Custom["extra"] = true
	if _, ok := orig.Custom["extra"]; ok {
		t.Error("mutating clone Custom affected original")
	}

	// Go types in metadata (e.g. []byte) must be preserved, not string-ified.
	sig, ok := cp.Metadata["sig"].([]byte)
	if !ok {
		t.Fatalf("Metadata[sig] type = %T, want []byte", cp.Metadata["sig"])
	}
	if !bytes.Equal(sig, []byte{1, 2, 3}) {
		t.Errorf("Metadata[sig] = %v, want [1 2 3]", sig)
	}

	// nil Part.Clone() should return nil.
	var nilPart *Part
	if nilPart.Clone() != nil {
		t.Error("nil Part.Clone() should return nil")
	}
}

// TestMessageClone verifies that Message.Clone produces an independent copy.
// Every Message field is populated so that adding a new field without updating
// this test (and Clone) causes a failure.
func TestMessageClone(t *testing.T) {
	orig := &Message{
		Role:     RoleModel,
		Content:  []*Part{NewTextPart("hello"), NewTextPart("world")},
		Metadata: map[string]any{"k": "v"},
	}

	// Guard: every field must be non-zero.
	rv := reflect.ValueOf(orig).Elem()
	for i := range rv.NumField() {
		if rv.Field(i).IsZero() {
			t.Fatalf("Message field %q is zero in test fixture — populate it and verify Clone handles it", rv.Type().Field(i).Name)
		}
	}

	cp := orig.Clone()

	// Values must match.
	if !reflect.DeepEqual(orig, cp) {
		t.Fatal("Clone() values differ from original")
	}

	// Mutating clone's Content slice must not affect the original.
	cp.Content[0] = NewTextPart("replaced")
	if orig.Content[0].Text != "hello" {
		t.Error("mutating clone Content affected original")
	}

	// Mutating clone's Metadata must not affect the original.
	cp.Metadata["extra"] = true
	if _, ok := orig.Metadata["extra"]; ok {
		t.Error("mutating clone Metadata affected original")
	}

	// nil Message.Clone() should return nil.
	var nilMsg *Message
	if nilMsg.Clone() != nil {
		t.Error("nil Message.Clone() should return nil")
	}
}
