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

import "testing"

func TestSplitModelName(t *testing.T) {
	tests := []struct {
		in         string
		wantPrefix string
		wantModel  string
	}{
		{"googleai/gemini-flash-latest", "googleai", "gemini-flash-latest"},
		{"vertexai/gemini-flash-latest", "vertexai", "gemini-flash-latest"},
		{"bare-model", "", "bare-model"},
		{"a/b/c", "a", "b/c"},
	}
	for _, tt := range tests {
		prefix, model := SplitModelName(tt.in)
		if prefix != tt.wantPrefix || model != tt.wantModel {
			t.Errorf("SplitModelName(%q) = (%q, %q), want (%q, %q)", tt.in, prefix, model, tt.wantPrefix, tt.wantModel)
		}
	}
}

func TestDeriveProviderName(t *testing.T) {
	tests := map[string]string{
		"":             "",
		"googleai":     "gcp.gemini",
		"google-genai": "gcp.gemini",
		"vertexai":     "gcp.vertex_ai",
		"vertex_ai":    "gcp.vertex_ai",
		"openai":       "openai",
		"anthropic":    "anthropic",
		"CustomPlugin": "customplugin",
	}
	for prefix, want := range tests {
		if got := DeriveProviderName(prefix); got != want {
			t.Errorf("DeriveProviderName(%q) = %q, want %q", prefix, got, want)
		}
	}
}

func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		reason string
		failed bool
		want   string
	}{
		{"stop", false, "stop"},
		{"length", false, "length"},
		{"blocked", false, "content_filter"},
		{"interrupted", false, "stop"},
		{"other", false, "stop"},
		{"other", true, "error"},
		{"unknown", true, "error"},
		{"", false, "stop"},
		{"weird", true, "error"},
	}
	for _, tt := range tests {
		if got := MapFinishReason(tt.reason, tt.failed); got != tt.want {
			t.Errorf("MapFinishReason(%q, %v) = %q, want %q", tt.reason, tt.failed, got, tt.want)
		}
	}
}

func TestDeriveOutputType(t *testing.T) {
	tests := []struct {
		format      string
		contentType string
		want        string
	}{
		{"json", "", "json"},
		{"", "application/json", "json"},
		{"text", "", "text"},
		{"", "text/plain", "text"},
		{"", "", ""},
		{"media", "image/png", ""},
	}
	for _, tt := range tests {
		if got := DeriveOutputType(tt.format, tt.contentType); got != tt.want {
			t.Errorf("DeriveOutputType(%q, %q) = %q, want %q", tt.format, tt.contentType, got, tt.want)
		}
	}
}

func TestParseContentCapturingMode(t *testing.T) {
	tests := []struct {
		in     string
		want   ContentCapturingMode
		wantOK bool
	}{
		{"", NoContent, true},
		{"no_content", NoContent, true},
		{"SPAN_ONLY", SpanOnly, true},
		{"span_and_event", SpanAndEvent, true},
		{"  event_only  ", EventOnly, true},
		{"bogus", NoContent, false},
	}
	for _, tt := range tests {
		got, ok := ParseContentCapturingMode(tt.in)
		if got != tt.want || ok != tt.wantOK {
			t.Errorf("ParseContentCapturingMode(%q) = (%q, %v), want (%q, %v)", tt.in, got, ok, tt.want, tt.wantOK)
		}
	}
}
