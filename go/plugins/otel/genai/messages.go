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
	"encoding/json"
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// NormalizedMessages is a message list split into system instructions and the
// remaining conversation messages, both in the GenAI content schema.
type NormalizedMessages struct {
	Messages           []map[string]any
	SystemInstructions []map[string]any
}

// MapRole maps a Genkit role to the GenAI role name: Genkit "model" becomes
// "assistant"; other roles pass through.
func MapRole(role ai.Role) string {
	if role == ai.RoleModel {
		return "assistant"
	}
	return string(role)
}

// MapPart converts a single Genkit part to a GenAI content part.
//
// It discriminates on the part kind. Unknown/opaque kinds fall back to a
// generic text part carrying the JSON encoding, so nothing is silently dropped
// from captured content.
func MapPart(part *ai.Part) map[string]any {
	if part == nil {
		return map[string]any{"type": "text", "content": ""}
	}
	switch {
	case part.IsText():
		return map[string]any{"type": "text", "content": part.Text}
	case part.IsReasoning():
		return map[string]any{"type": "reasoning", "content": part.Text}
	case part.IsToolRequest():
		tr := part.ToolRequest
		if tr == nil {
			return map[string]any{"type": "tool_call", "name": "", "arguments": nil}
		}
		m := map[string]any{"type": "tool_call", "name": tr.Name, "arguments": tr.Input}
		if tr.Ref != "" {
			m["id"] = tr.Ref
		}
		return m
	case part.IsToolResponse():
		tr := part.ToolResponse
		if tr == nil {
			return map[string]any{"type": "tool_call_response", "response": nil}
		}
		m := map[string]any{"type": "tool_call_response", "response": tr.Output}
		if tr.Ref != "" {
			m["id"] = tr.Ref
		}
		return m
	case part.IsMedia():
		return mapMediaPart(part)
	default:
		// Unknown/opaque part: represent it structurally without losing the
		// fact that it existed.
		b, err := json.Marshal(part)
		if err != nil {
			b = []byte("")
		}
		return map[string]any{"type": "text", "content": string(b)}
	}
}

// mapMediaPart maps a media part (whose Text holds the URL) to a spec "uri"
// part, or to a "blob" part for an inline data: URI.
//
// Blob payloads are deliberately not captured: a base64 image can be
// megabytes, which gets truncated mid-JSON by attribute length limits and can
// push an OTLP export past the receiver's message size cap, failing the whole
// batch. The blob part carries the decoded size instead of "content", which
// keeps it a valid spec GenericPart.
func mapMediaPart(part *ai.Part) map[string]any {
	url := part.Text
	mimeType := part.ContentType
	if rest, ok := strings.CutPrefix(url, "data:"); ok {
		header, payload, _ := strings.Cut(rest, ",")
		if mimeType == "" {
			mimeType, _, _ = strings.Cut(header, ";")
		}
		m := map[string]any{
			"type":       "blob",
			"modality":   mediaModality(mimeType),
			"size_bytes": dataURIPayloadSize(header, payload),
		}
		if mimeType != "" {
			m["mime_type"] = mimeType
		}
		return m
	}
	m := map[string]any{"type": "uri", "modality": mediaModality(mimeType), "uri": url}
	if mimeType != "" {
		m["mime_type"] = mimeType
	}
	return m
}

// mediaModality maps a MIME type to the spec's modality vocabulary. The spec
// requires a modality, so anything that is not image/video/audio (including
// an unknown type) is reported as a document.
func mediaModality(mimeType string) string {
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return "image"
	case strings.HasPrefix(mimeType, "video/"):
		return "video"
	case strings.HasPrefix(mimeType, "audio/"):
		return "audio"
	default:
		return "document"
	}
}

// dataURIPayloadSize estimates the decoded size of a data: URI payload without
// decoding it. Non-base64 payloads are reported by their raw length.
func dataURIPayloadSize(header, payload string) int {
	if !strings.HasSuffix(header, ";base64") {
		return len(payload)
	}
	return len(payload)*3/4 - strings.Count(payload[max(0, len(payload)-2):], "=")
}

// mapParts maps a slice of Genkit parts to GenAI content parts.
func mapParts(parts []*ai.Part) []map[string]any {
	out := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		out = append(out, MapPart(p))
	}
	return out
}

// MapMessage converts a single Genkit message to a GenAI message.
func MapMessage(message *ai.Message) map[string]any {
	return map[string]any{
		"role":  MapRole(message.Role),
		"parts": mapParts(message.Content),
	}
}

// NormalizeMessages splits messages into system_instructions (role system) and
// the remaining conversation messages, each normalized to the GenAI content
// schema.
func NormalizeMessages(messages []*ai.Message) NormalizedMessages {
	var system, rest []map[string]any
	for _, message := range messages {
		if message == nil {
			continue
		}
		if message.Role == ai.RoleSystem {
			// System instructions are represented by their parts directly.
			system = append(system, mapParts(message.Content)...)
		} else {
			rest = append(rest, MapMessage(message))
		}
	}
	return NormalizedMessages{Messages: rest, SystemInstructions: system}
}

// MapOutputMessage maps a response message to a GenAI output message, attaching
// the mapped finishReason.
func MapOutputMessage(message *ai.Message, finishReason string) map[string]any {
	m := MapMessage(message)
	m["finish_reason"] = finishReason
	return m
}

// HasToolRequestPart reports whether any part in content is a tool request.
func HasToolRequestPart(content []*ai.Part) bool {
	for _, p := range content {
		if p.IsToolRequest() {
			return true
		}
	}
	return false
}
