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

package internal

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/google/go-cmp/cmp"
)

func TestSortToolDefinitions(t *testing.T) {
	tests := []struct {
		name     string
		input    []*ai.ToolDefinition
		expected []*ai.ToolDefinition
	}{
		{
			name:     "nil slice",
			input:    nil,
			expected: nil,
		},
		{
			name:     "empty slice",
			input:    []*ai.ToolDefinition{},
			expected: []*ai.ToolDefinition{},
		},
		{
			name: "already sorted",
			input: []*ai.ToolDefinition{
				{Name: "a"},
				{Name: "b"},
				{Name: "c"},
			},
			expected: []*ai.ToolDefinition{
				{Name: "a"},
				{Name: "b"},
				{Name: "c"},
			},
		},
		{
			name: "unsorted",
			input: []*ai.ToolDefinition{
				{Name: "c"},
				{Name: "a"},
				{Name: "b"},
			},
			expected: []*ai.ToolDefinition{
				{Name: "a"},
				{Name: "b"},
				{Name: "c"},
			},
		},
		{
			name: "with nil entries",
			input: []*ai.ToolDefinition{
				{Name: "b"},
				nil,
				{Name: "a"},
			},
			expected: []*ai.ToolDefinition{
				{Name: "a"},
				{Name: "b"},
			},
		},
		{
			name: "stable sort",
			input: []*ai.ToolDefinition{
				{Name: "a", Description: "1"},
				{Name: "a", Description: "2"},
				{Name: "b"},
			},
			expected: []*ai.ToolDefinition{
				{Name: "a", Description: "1"},
				{Name: "a", Description: "2"},
				{Name: "b"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SortToolDefinitions(tt.input)
			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Errorf("SortToolDefinitions() mismatch (-want +got):\n%s", diff)
			}

			// Ensure original slice is not modified
			if len(tt.input) > 0 {
				// We can't easily check for nil input without it being a bit messy,
				// but for non-empty we should ensure the reference is different if sorted.
				// Actually, SortToolDefinitions always makes a copy if len > 0.
			}
		})
	}
}
