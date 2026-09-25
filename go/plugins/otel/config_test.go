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

package otel

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// geminiLikeConfig mimics googlegenai's GenerateContentConfig: pointer fields
// and a fractional TopK, which a typed decode into GenerationCommonConfig
// (TopK int) rejects outright.
type geminiLikeConfig struct {
	Temperature     *float32 `json:"temperature,omitempty"`
	TopK            *float32 `json:"topK,omitempty"`
	MaxOutputTokens int32    `json:"maxOutputTokens,omitempty"`
}

// anthropicLikeConfig mimics the anthropic SDK's MessageNewParams wire names.
type anthropicLikeConfig struct {
	MaxTokens     int64    `json:"max_tokens"`
	TopP          float64  `json:"top_p,omitempty"`
	TopK          int64    `json:"top_k,omitempty"`
	StopSequences []string `json:"stop_sequences,omitempty"`
}

func ptr[T any](v T) *T { return &v }

func TestRequestConfigAttributes(t *testing.T) {
	tests := []struct {
		name   string
		config any
		want   map[string]any
	}{
		{
			name:   "explicit zero temperature and fractional topK",
			config: &geminiLikeConfig{Temperature: ptr(float32(0)), TopK: ptr(float32(2.5)), MaxOutputTokens: 64},
			want: map[string]any{
				genai.AttrRequestTemperature: 0.0,
				genai.AttrRequestTopK:        2.5,
				genai.AttrRequestMaxTokens:   int64(64),
			},
		},
		{
			name:   "snake_case sdk config",
			config: anthropicLikeConfig{MaxTokens: 1024, TopP: 0.5, TopK: 40, StopSequences: []string{"END"}},
			want: map[string]any{
				genai.AttrRequestMaxTokens:     int64(1024),
				genai.AttrRequestTopP:          0.5,
				genai.AttrRequestTopK:          40.0,
				genai.AttrRequestStopSequences: []string{"END"},
			},
		},
		{
			name:   "one mistyped field does not drop the rest",
			config: map[string]any{"topK": "high", "temperature": 0.75, "seed": 42.0, "candidateCount": 1.0},
			want: map[string]any{
				genai.AttrRequestTemperature: 0.75,
				genai.AttrRequestSeed:        int64(42),
			},
		},
		{
			name:   "choice count other than 1 is recorded",
			config: map[string]any{"n": 3.0, "stop": "STOP"},
			want: map[string]any{
				genai.AttrRequestChoiceCount:   int64(3),
				genai.AttrRequestStopSequences: []string{"STOP"},
			},
		},
		{
			name:   "common config omits unset fields",
			config: &ai.GenerationCommonConfig{TopP: 0.5},
			want:   map[string]any{genai.AttrRequestTopP: 0.5},
		},
		{
			name:   "nil config",
			config: nil,
			want:   map[string]any{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := map[string]any{}
			for _, kv := range requestConfigAttributes(&ai.ModelRequest{Config: tt.config}) {
				got[string(kv.Key)] = attrValue(kv.Value)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("requestConfigAttributes() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// attrValue unwraps an attribute value into a plain Go value for comparison.
func attrValue(v attribute.Value) any {
	switch v.Type() {
	case attribute.FLOAT64:
		return v.AsFloat64()
	case attribute.INT64:
		return v.AsInt64()
	case attribute.STRINGSLICE:
		return v.AsStringSlice()
	default:
		return v.AsInterface()
	}
}
