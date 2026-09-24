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
	"encoding/json"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestNewPartAndIsPart(t *testing.T) {
	env := Envelope{"createSurface": map[string]any{"surfaceId": "s1", "catalogId": "c"}}
	p := newPart([]Envelope{env})

	if !IsPart(p) {
		t.Fatal("newPart should produce a part recognized by IsPart")
	}
	if mt, _ := p.Metadata["mimeType"].(string); mt != A2UIMimeType {
		t.Errorf("mimeType = %v, want %v", p.Metadata["mimeType"], A2UIMimeType)
	}
	if !IsPart(p) {
		t.Error("IsPart(newPart(...)) = false")
	}
	if IsPart(ai.NewTextPart("hello")) {
		t.Error("IsPart(text part) = true, want false")
	}
}

func TestNewPartWireShape(t *testing.T) {
	env := Envelope{"createSurface": map[string]any{"surfaceId": "s1"}}
	p := newPart([]Envelope{env})
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"data":{"envelopes":[{"createSurface":{"surfaceId":"s1"}}]},"metadata":{"mimeType":"application/a2ui+json"}}`
	if string(b) != want {
		t.Errorf("wire shape = %s\nwant       = %s", string(b), want)
	}
}

func TestEnvelopesFromParts(t *testing.T) {
	p := newPart([]Envelope{
		{"createSurface": map[string]any{"surfaceId": "s1"}},
		{"deleteSurface": map[string]any{"surfaceId": "s1"}},
	})
	parts := []*ai.Part{ai.NewTextPart("prose"), p}

	envs := EnvelopesFromParts(parts)
	if len(envs) != 2 {
		t.Fatalf("got %d envelopes, want 2", len(envs))
	}
	if _, ok := envs[0]["createSurface"]; !ok {
		t.Errorf("first envelope missing createSurface: %v", envs[0])
	}
	if _, ok := envs[1]["deleteSurface"]; !ok {
		t.Errorf("second envelope missing deleteSurface: %v", envs[1])
	}
}

func TestEnvelopesFromPartsRoundTrip(t *testing.T) {
	// A part that has gone through JSON (as it would arriving from a client)
	// must still be detected and read.
	p := newPart([]Envelope{{"action": map[string]any{"name": "submit", "surfaceId": "s1"}}})
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ai.Part
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	if !IsPart(&decoded) {
		t.Fatal("round-tripped part not recognized by IsPart")
	}
	envs := EnvelopesFromParts([]*ai.Part{&decoded})
	if len(envs) != 1 {
		t.Fatalf("got %d envelopes, want 1", len(envs))
	}
	action, _ := envs[0]["action"].(map[string]any)
	if action["name"] != "submit" {
		t.Errorf("action name = %v, want submit", action["name"])
	}
}
