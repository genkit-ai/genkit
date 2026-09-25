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
				Data: map[string]any{"some": "data", "n": 3.3},
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
			return reflect.DeepEqual(a.Data, b.Data)
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

	if got := unmarshaledPart.Metadata["signature"]; got == nil {
		t.Errorf("unmarshaled reasoning part lost its signature, metadata = %v", unmarshaledPart.Metadata)
	}
}

func TestReasoningPartWithoutSignature(t *testing.T) {
	// A part with no signature carries no metadata at all. A metadata map
	// holding only a nil signature reads as "this part has metadata" to
	// consumers, which stops adjacent reasoning parts from being merged.
	p := NewReasoningPart("thinking", nil)
	if p.Metadata != nil {
		t.Errorf("Metadata = %v, want nil", p.Metadata)
	}

	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("failed to marshal reasoning part: %v", err)
	}
	if got, want := string(b), `{"reasoning":"thinking"}`; got != want {
		t.Errorf("marshaled = %s, want %s", got, want)
	}
}

func TestReasoningPartClonesSignature(t *testing.T) {
	// A caller reusing a buffer across streamed chunks must not be able to
	// rewrite a signature it has already handed off.
	buf := []byte("sig123")
	p := NewReasoningPart("thinking", buf)
	copy(buf, "XXXXXX")

	got, ok := p.Metadata["signature"].([]byte)
	if !ok {
		t.Fatalf("signature = %#v, want []byte", p.Metadata["signature"])
	}
	if string(got) != "sig123" {
		t.Errorf("signature = %q, want %q: the caller's buffer is aliased", got, "sig123")
	}
}

func TestEmptyReasoningPartRoundTrip(t *testing.T) {
	// The reasoning key marks the kind, so it has to survive an empty text:
	// dropping it turns the part into an empty text part on the way back, and
	// the wire schema lists reasoning as required.
	p := NewReasoningPart("", []byte("sig123"))

	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("failed to marshal reasoning part: %v", err)
	}

	var got Part
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("failed to unmarshal reasoning part: %v", err)
	}

	if !got.IsReasoning() {
		t.Errorf("empty reasoning part became kind %v, want %v (marshaled as %s)", got.Kind, PartReasoning, b)
	}
	if got.Text != "" {
		t.Errorf("Text = %q, want empty", got.Text)
	}
}

func TestNewDataPart(t *testing.T) {
	t.Run("creates data part with string content", func(t *testing.T) {
		p := NewDataPart("some binary data")

		if p.Kind != PartData {
			t.Errorf("Kind = %v, want %v", p.Kind, PartData)
		}
		if p.Data != "some binary data" {
			t.Errorf("Data = %v, want %q", p.Data, "some binary data")
		}
	})

	t.Run("creates data part with structured content", func(t *testing.T) {
		data := map[string]any{"name": "Alice", "age": 30}
		p := NewDataPart(data)

		if p.Kind != PartData {
			t.Errorf("Kind = %v, want %v", p.Kind, PartData)
		}
		if !reflect.DeepEqual(p.Data, data) {
			t.Errorf("Data = %v, want %v", p.Data, data)
		}
	})

	t.Run("round-trips structured data through JSON", func(t *testing.T) {
		p := NewDataPart(map[string]any{"envelopes": []any{map[string]any{"x": 1.0}}})
		p.Metadata = map[string]any{"mimeType": "application/a2ui+json"}

		b, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		want := `{"data":{"envelopes":[{"x":1}]},"metadata":{"mimeType":"application/a2ui+json"}}`
		if string(b) != want {
			t.Errorf("marshaled = %s, want %s", string(b), want)
		}

		var p2 Part
		if err := json.Unmarshal(b, &p2); err != nil {
			t.Fatal(err)
		}
		if p2.Kind != PartData {
			t.Errorf("Kind = %v, want %v", p2.Kind, PartData)
		}
		if !reflect.DeepEqual(p2.Data, p.Data) {
			t.Errorf("Data = %v, want %v", p2.Data, p.Data)
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
			Interrupt: &ToolInterrupt{},
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
		Data:        map[string]any{"dk": "dv"},
		ToolRequest: &ToolRequest{Name: "tool", Input: map[string]any{"a": 1}},
		// Normally a Part wouldn't have both ToolRequest and ToolResponse,
		// but we populate everything to catch missing fields.
		ToolResponse: &ToolResponse{Name: "tool", Output: "ok"},
		Resource:     &ResourcePart{Uri: "res://x"},
		Custom:       map[string]any{"ck": "cv"},
		Interrupt:    &ToolInterrupt{Data: map[string]any{"reason": "confirm"}},
		Restart:      &ToolRestart{Resume: map[string]any{"approved": true}, OriginalInput: map[string]any{"a": 0}},
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

	// A map-shaped Data value is cloned too, so mutating the clone's top-level
	// keys must not affect the original (data parts, e.g. A2UI envelopes, are
	// commonly map[string]any and shared by reference before this).
	cpData, _ := cp.Data.(map[string]any)
	if cpData == nil {
		t.Fatalf("clone Data type = %T, want map[string]any", cp.Data)
	}
	cpData["extra"] = true
	if _, ok := orig.Data.(map[string]any)["extra"]; ok {
		t.Error("mutating clone Data affected original")
	}

	// Interrupt and Restart are pointers; the clone must own its own, so
	// resolving an interrupt on a copy doesn't reach back into the original.
	cp.Interrupt.Resolved = true
	if orig.Interrupt.Resolved {
		t.Error("mutating clone Interrupt affected original")
	}
	cp.Restart.Resume = "other"
	if orig.Restart.Resume == "other" {
		t.Error("mutating clone Restart affected original")
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

// A []any Data value gets the same top-level isolation as a map, so switching a
// payload from an object to an array doesn't silently lose Clone's guarantee.
func TestPartCloneSliceData(t *testing.T) {
	orig := NewDataPart([]any{"a", "b"})
	cp := orig.Clone()
	cpData, ok := cp.Data.([]any)
	if !ok {
		t.Fatalf("clone Data type = %T, want []any", cp.Data)
	}
	cpData[0] = "mutated"
	if orig.Data.([]any)[0] != "a" {
		t.Error("mutating clone's slice Data affected the original")
	}
}

func TestPartDataString(t *testing.T) {
	tests := []struct {
		name string
		part *Part
		want string
	}{
		{"string payload as-is", NewDataPart("data:image/png;base64,aGVsbG8="), "data:image/png;base64,aGVsbG8="},
		{"map payload as JSON", NewDataPart(map[string]any{"k": "v"}), `{"k":"v"}`},
		{"slice payload as JSON", NewDataPart([]any{1.0, 2.0}), `[1,2]`},
		{"nil data", NewDataPart(nil), ""},
		{"nil part", (*Part)(nil), ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.part.DataString(); got != tt.want {
				t.Errorf("DataString() = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestPartInterruptWireRoundTrip pins the wire contract for the typed interrupt
// and restart state: it must fold into the metadata keys the JS runtime reads,
// and lift back out on the way in, so a message can cross runtimes unchanged.
func TestPartInterruptWireRoundTrip(t *testing.T) {
	tests := []struct {
		name      string
		part      func() *Part
		wantWire  map[string]any
		wantLifts func(t *testing.T, p *Part)
	}{
		{
			name: "interrupt with data",
			part: func() *Part {
				p := NewToolRequestPart(&ToolRequest{Name: "transfer", Ref: "r1"})
				p.Interrupt = &ToolInterrupt{Data: map[string]any{"reason": "large"}}
				p.Metadata = map[string]any{"keep": "me"}
				return p
			},
			wantWire: map[string]any{
				"interrupt": map[string]any{"reason": "large"},
				"keep":      "me",
			},
			wantLifts: func(t *testing.T, p *Part) {
				if !p.IsInterrupt() {
					t.Error("lifted part is not an interrupt")
				}
				data, _ := p.Interrupt.Data.(map[string]any)
				if data["reason"] != "large" {
					t.Errorf("lifted interrupt data = %v, want reason=large", p.Interrupt.Data)
				}
				if p.Metadata["keep"] != "me" {
					t.Errorf("unrelated metadata lost: %v", p.Metadata)
				}
			},
		},
		{
			name: "bare interrupt",
			part: func() *Part {
				p := NewToolRequestPart(&ToolRequest{Name: "transfer"})
				p.Interrupt = &ToolInterrupt{}
				return p
			},
			wantWire: map[string]any{"interrupt": true},
			wantLifts: func(t *testing.T, p *Part) {
				if p.Interrupt == nil || p.Interrupt.Data != nil {
					t.Errorf("lifted interrupt = %+v, want no data", p.Interrupt)
				}
			},
		},
		{
			name: "resolved interrupt",
			part: func() *Part {
				p := NewToolRequestPart(&ToolRequest{Name: "transfer"})
				p.Interrupt = &ToolInterrupt{Data: map[string]any{"reason": "large"}, Resolved: true}
				return p
			},
			wantWire: map[string]any{
				"resolvedInterrupt": map[string]any{"reason": "large"},
			},
			wantLifts: func(t *testing.T, p *Part) {
				if p.Interrupt == nil || !p.Interrupt.Resolved {
					t.Errorf("lifted interrupt = %+v, want resolved", p.Interrupt)
				}
				if p.IsInterrupt() {
					t.Error("a resolved interrupt must not still await resolution")
				}
			},
		},
		{
			name: "restart with resume and replaced input",
			part: func() *Part {
				p := NewToolRequestPart(&ToolRequest{Name: "transfer", Input: map[string]any{"amount": 50.0}})
				p.Restart = &ToolRestart{
					Resume:        map[string]any{"approved": true},
					OriginalInput: map[string]any{"amount": 200.0},
				}
				return p
			},
			wantWire: map[string]any{
				"resumed":       map[string]any{"approved": true},
				"replacedInput": map[string]any{"amount": 200.0},
			},
			wantLifts: func(t *testing.T, p *Part) {
				if p.Restart == nil {
					t.Fatal("restart state was not lifted")
				}
				resume, _ := p.Restart.Resume.(map[string]any)
				if resume["approved"] != true {
					t.Errorf("lifted resume = %v, want approved=true", p.Restart.Resume)
				}
				orig, _ := p.Restart.OriginalInput.(map[string]any)
				if orig["amount"] != 200.0 {
					t.Errorf("lifted original input = %v, want amount=200", p.Restart.OriginalInput)
				}
			},
		},
		{
			name: "bare restart",
			part: func() *Part {
				p := NewToolRequestPart(&ToolRequest{Name: "transfer"})
				p.Restart = &ToolRestart{}
				return p
			},
			wantWire: map[string]any{"resumed": true},
			wantLifts: func(t *testing.T, p *Part) {
				if p.Restart == nil || p.Restart.Resume != nil {
					t.Errorf("lifted restart = %+v, want no resume payload", p.Restart)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			orig := tt.part()
			b, err := json.Marshal(orig)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}

			if diff := cmp.Diff(tt.wantWire, wireMetadataOf(t, orig)); diff != "" {
				t.Errorf("wire metadata mismatch (-want +got):\n%s", diff)
			}

			// The source part keeps its own metadata: folding is not mutation.
			if _, ok := orig.Metadata["interrupt"]; ok {
				t.Error("marshaling must not write wire keys back onto the part")
			}

			var lifted Part
			if err := json.Unmarshal(b, &lifted); err != nil {
				t.Fatalf("unmarshal part: %v", err)
			}
			tt.wantLifts(t, &lifted)
			if _, ok := lifted.Metadata["resumed"]; ok {
				t.Error("lifted wire keys must be removed from the metadata map")
			}
		})
	}
}

// TestPartWireMetadata_TypedStateWinsOverStaleKey pins that folding the typed
// interrupt state into the wire keys drops the key of the other resolution
// state: a part assembled with the raw "interrupt" key and then resolved on
// the field marshals as resolved only, so it does not read back as pending
// after a wire hop.
func TestPartWireMetadata_TypedStateWinsOverStaleKey(t *testing.T) {
	p := NewToolRequestPart(&ToolRequest{Name: "transfer"})
	p.Metadata = map[string]any{"interrupt": map[string]any{"reason": "old"}}
	p.Interrupt = &ToolInterrupt{Data: map[string]any{"reason": "old"}, Resolved: true}

	wire := wireMetadataOf(t, p)
	if _, ok := wire["interrupt"]; ok {
		t.Errorf("wire metadata = %v, want the stale interrupt key dropped", wire)
	}
	if _, ok := wire["resolvedInterrupt"]; !ok {
		t.Errorf("wire metadata = %v, want resolvedInterrupt", wire)
	}

	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var lifted Part
	if err := json.Unmarshal(b, &lifted); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if lifted.IsInterrupt() || lifted.Interrupt == nil || !lifted.Interrupt.Resolved {
		t.Errorf("after a wire hop the part reads as %+v, want a resolved interrupt", lifted.Interrupt)
	}
}

// TestPartWireMarkersReadByTruthiness pins how the wire keys are read when a
// part arrives from a peer: the JS runtime tests them by truthiness, so null
// false, "" and 0 mean no state, true means state with no payload, and any
// other value is the payload as sent, even one Go cannot deliver to a tool.
func TestPartWireMarkersReadByTruthiness(t *testing.T) {
	tests := []struct {
		name          string
		wire          string
		wantInterrupt bool
		wantRestart   bool
		wantResume    any
	}{
		{"interrupt null", `{"toolRequest":{"name":"t"},"metadata":{"interrupt":null}}`, false, false, nil},
		{"interrupt false", `{"toolRequest":{"name":"t"},"metadata":{"interrupt":false}}`, false, false, nil},
		{"interrupt true", `{"toolRequest":{"name":"t"},"metadata":{"interrupt":true}}`, true, false, nil},
		{"interrupt empty string", `{"toolRequest":{"name":"t"},"metadata":{"interrupt":""}}`, false, false, nil},
		{"interrupt zero", `{"toolRequest":{"name":"t"},"metadata":{"interrupt":0}}`, false, false, nil},
		{"resumed null", `{"toolRequest":{"name":"t"},"metadata":{"resumed":null}}`, false, false, nil},
		{"resumed false", `{"toolRequest":{"name":"t"},"metadata":{"resumed":false}}`, false, false, nil},
		{"resumed true", `{"toolRequest":{"name":"t"},"metadata":{"resumed":true}}`, false, true, nil},
		{"resumed zero", `{"toolRequest":{"name":"t"},"metadata":{"resumed":0}}`, false, false, nil},
		{"resumed scalar", `{"toolRequest":{"name":"t"},"metadata":{"resumed":"approved"}}`, false, true, "approved"},
		{"interrupt and resumed", `{"toolRequest":{"name":"t"},"metadata":{"interrupt":{"why":"x"},"resumed":{"ok":true}}}`, true, true, map[string]any{"ok": true}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var p Part
			if err := json.Unmarshal([]byte(tt.wire), &p); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got := p.IsInterrupt(); got != tt.wantInterrupt {
				t.Errorf("IsInterrupt() = %v, want %v", got, tt.wantInterrupt)
			}
			if got := p.IsRestart(); got != tt.wantRestart {
				t.Errorf("IsRestart() = %v, want %v", got, tt.wantRestart)
			}
			if tt.wantRestart {
				if diff := cmp.Diff(tt.wantResume, p.Restart.Resume); diff != "" {
					t.Errorf("Restart.Resume mismatch (-want +got):\n%s", diff)
				}
			}
			if err := p.Validate(); err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
		})
	}
}

// TestPartWireState_Hygiene pins the cases where wire state could go stale
// or vanish: decoding into a Part that already holds state replaces it, a
// restart that keeps the input drops a stale "replacedInput" key, and a nil
// map payload on a typed field marshals as the bare marker rather than null.
func TestPartWireState_Hygiene(t *testing.T) {
	t.Run("decoding into a used part replaces its state", func(t *testing.T) {
		var p Part
		if err := json.Unmarshal([]byte(`{"toolRequest":{"name":"t"},"metadata":{"interrupt":true}}`), &p); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if err := json.Unmarshal([]byte(`{"toolRequest":{"name":"t"},"metadata":{"resolvedInterrupt":true}}`), &p); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if p.IsInterrupt() || p.Interrupt == nil || !p.Interrupt.Resolved {
			t.Errorf("interrupt = %+v, want the resolved state of the second decode", p.Interrupt)
		}
	})
	t.Run("a stale replacedInput key is dropped", func(t *testing.T) {
		p := NewToolRequestPart(&ToolRequest{Name: "t", Input: map[string]any{"a": 1}})
		p.Metadata = map[string]any{"replacedInput": map[string]any{"a": 0}}
		p.Restart = &ToolRestart{}
		if wire := wireMetadataOf(t, p); wire["replacedInput"] != nil {
			t.Errorf("wire metadata = %v, want no replacedInput", wire)
		}
	})
	t.Run("a nil map payload is the bare marker", func(t *testing.T) {
		p := NewToolRequestPart(&ToolRequest{Name: "t"})
		p.Interrupt = &ToolInterrupt{Data: map[string]any(nil)}
		p.Restart = &ToolRestart{Resume: map[string]any(nil)}
		wire := wireMetadataOf(t, p)
		if wire["interrupt"] != true || wire["resumed"] != true {
			t.Errorf("wire metadata = %v, want the bare markers", wire)
		}
	})
}

func TestPartKindString(t *testing.T) {
	for kind, want := range map[PartKind]string{
		PartText:         "text",
		PartMedia:        "media",
		PartData:         "data",
		PartToolRequest:  "toolRequest",
		PartToolResponse: "toolResponse",
		PartCustom:       "custom",
		PartReasoning:    "reasoning",
		PartResource:     "resource",
		PartKind(99):     "unknown",
	} {
		if got := kind.String(); got != want {
			t.Errorf("PartKind(%d).String() = %q, want %q", int8(kind), got, want)
		}
	}
}

func TestPartValidate(t *testing.T) {
	toolReq := func() *Part { return NewToolRequestPart(&ToolRequest{Name: "t"}) }

	valid := []struct {
		name string
		part *Part
	}{
		{"text", NewTextPart("hi")},
		{"media", NewMediaPart("image/png", "data:...")},
		{"tool request", toolReq()},
		{"tool response", NewToolResponsePart(&ToolResponse{Name: "t"})},
		{"interrupted tool request", func() *Part { p := toolReq(); p.Interrupt = &ToolInterrupt{}; return p }()},
		{"restarted tool request", func() *Part { p := toolReq(); p.Restart = &ToolRestart{}; return p }()},
		{"resolved interrupt restarted", func() *Part {
			p := toolReq()
			p.Interrupt = &ToolInterrupt{Resolved: true}
			p.Restart = &ToolRestart{}
			return p
		}()},
		// The JS restartTool spreads the interrupted part's metadata onto the
		// restart, so a restart still carrying the interrupt it resolves is
		// the normal cross-runtime shape.
		{"restart of an unresolved interrupt", func() *Part {
			p := toolReq()
			p.Interrupt = &ToolInterrupt{}
			p.Restart = &ToolRestart{}
			return p
		}()},
		{"custom", NewCustomPart(map[string]any{"k": "v"})},
		{"resource", NewResourcePart("res://x")},
	}
	for _, tt := range valid {
		t.Run("valid/"+tt.name, func(t *testing.T) {
			if err := tt.part.Validate(); err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
		})
	}

	invalid := []struct {
		name string
		part *Part
	}{
		{"nil part", nil},
		{"unknown kind", &Part{Kind: PartKind(99)}},
		{"tool request without request", &Part{Kind: PartToolRequest}},
		{"tool response without response", &Part{Kind: PartToolResponse}},
		{"custom without custom data", &Part{Kind: PartCustom}},
		{"resource without resource", &Part{Kind: PartResource}},
		{"interrupt on a text part", func() *Part { p := NewTextPart("hi"); p.Interrupt = &ToolInterrupt{}; return p }()},
		{"restart on a tool response", func() *Part {
			p := NewToolResponsePart(&ToolResponse{Name: "t"})
			p.Restart = &ToolRestart{}
			return p
		}()},
		{"tool response on a tool request", func() *Part {
			p := toolReq()
			p.ToolResponse = &ToolResponse{Name: "t"}
			return p
		}()},
	}
	for _, tt := range invalid {
		t.Run("invalid/"+tt.name, func(t *testing.T) {
			if err := tt.part.Validate(); err == nil {
				t.Error("Validate() = nil, want an error")
			}
		})
	}
}
