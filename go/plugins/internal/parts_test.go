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

package internal

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/google/go-cmp/cmp"
)

func TestMergeAdjacentText(t *testing.T) {
	signed := func(p *ai.Part) *ai.Part {
		p.Metadata = map[string]any{"signature": []byte("sig")}
		return p
	}
	tool := ai.NewToolRequestPart(&ai.ToolRequest{Name: "lookup"})

	tests := []struct {
		name string
		in   []*ai.Part
		want []*ai.Part
	}{
		{
			name: "streamed text becomes one part",
			in:   []*ai.Part{ai.NewTextPart("In Go"), ai.NewTextPart(", type parameters use []."), ai.NewTextPart("")},
			want: []*ai.Part{ai.NewTextPart("In Go, type parameters use [].")},
		},
		{
			name: "reasoning and text merge separately, in order",
			in: []*ai.Part{
				ai.NewReasoningPart("think", nil), ai.NewReasoningPart("ing", nil),
				ai.NewTextPart("ans"), ai.NewTextPart("wer"),
			},
			want: []*ai.Part{ai.NewReasoningPart("thinking", nil), ai.NewTextPart("answer")},
		},
		{
			name: "a signed part keeps its signature and joins nothing",
			in: []*ai.Part{
				ai.NewTextPart("In Go"), ai.NewTextPart(", type parameters use []."),
				signed(ai.NewTextPart("")),
				ai.NewTextPart("d"), ai.NewTextPart("e"),
				ai.NewReasoningPart("r", []byte("sig")), ai.NewReasoningPart("s", nil),
			},
			want: []*ai.Part{
				ai.NewTextPart("In Go, type parameters use []."),
				signed(ai.NewTextPart("")),
				ai.NewTextPart("de"),
				ai.NewReasoningPart("r", []byte("sig")), ai.NewReasoningPart("s", nil),
			},
		},
		{
			name: "a tool request is a boundary",
			in:   []*ai.Part{ai.NewTextPart("a"), tool, ai.NewTextPart("b"), ai.NewTextPart("c")},
			want: []*ai.Part{ai.NewTextPart("a"), tool, ai.NewTextPart("bc")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := cloneParts(tt.in)
			got := MergeAdjacentText(tt.in)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("MergeAdjacentText() mismatch (-want +got):\n%s", diff)
			}
			// The stream callback has already handed the input parts out.
			if diff := cmp.Diff(before, tt.in); diff != "" {
				t.Errorf("MergeAdjacentText() modified its input (-before +after):\n%s", diff)
			}
		})
	}
}

func cloneParts(parts []*ai.Part) []*ai.Part {
	out := make([]*ai.Part, len(parts))
	for i, p := range parts {
		c := *p
		out[i] = &c
	}
	return out
}
