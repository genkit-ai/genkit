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

import "github.com/firebase/genkit/go/ai"

// newPart builds the canonical "a2ui part": a Genkit data part whose Data is the
// A2UI spec's {"messages": [...]} payload wrapper around the array of A2UI
// envelopes, tagged with A2UIMimeType. The spec puts the list directly in data,
// but a data part's data must be an object on every runtime (Dart expects a
// map), so the list sits under the spec's own wrapper key.
func newPart(envelopes []Envelope) *ai.Part {
	// Store as []any so it round-trips through JSON identically to a decoded
	// payload (encoding/json decodes arrays into []any).
	arr := make([]any, len(envelopes))
	for i, e := range envelopes {
		arr[i] = e
	}
	p := ai.NewDataPart(map[string]any{"messages": arr})
	p.Metadata = map[string]any{"mimeType": A2UIMimeType}
	return p
}

// IsPart reports whether p is an a2ui data part (mime application/a2ui+json
// carrying a "messages" array, or "envelopes" for a part written before the
// rename).
func IsPart(p *ai.Part) bool {
	if p == nil || !p.IsData() || p.Metadata == nil {
		return false
	}
	if mt, _ := p.Metadata["mimeType"].(string); mt != A2UIMimeType {
		return false
	}
	_, ok := messagesOf(p)
	return ok
}

// messagesOf returns the message list in an a2ui part's data: "messages", or
// "envelopes" for a part written before the rename, still read so persisted
// sessions replay.
func messagesOf(p *ai.Part) ([]any, bool) {
	data, ok := p.Data.(map[string]any)
	if !ok {
		return nil, false
	}
	for _, key := range []string{"messages", "envelopes"} {
		if v, present := data[key]; present {
			list, ok := v.([]any)
			return list, ok
		}
	}
	return nil, false
}

// EnvelopesFromParts extracts all A2UI envelopes carried by the given parts.
// Pass a message's, chunk's, or response's content. Returns nil for content
// that carries no a2ui parts (e.g. plain prose).
func EnvelopesFromParts(parts []*ai.Part) []Envelope {
	var out []Envelope
	for _, p := range parts {
		if !IsPart(p) {
			continue
		}
		raw, _ := messagesOf(p)
		for _, e := range raw {
			if env, ok := e.(map[string]any); ok {
				out = append(out, env)
			}
		}
	}
	return out
}
