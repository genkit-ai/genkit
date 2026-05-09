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

package anthropic

import (
	"testing"
)

func TestResolveModelID(t *testing.T) {
	availableModels := []string{
		"claude-opus-4-6",
		"claude-opus-4-5-20251101",
		"claude-opus-4-1-20250805",
		"claude-opus-4-20250514",
		"claude-sonnet-4-5-20250929",
		"claude-sonnet-4-20250514",
		"claude-haiku-4-5-20251001",
	}

	tests := []struct {
		input    string
		expected string
		found    bool
	}{
		// Exact matches
		{"claude-opus-4-6", "claude-opus-4-6", true},
		{"claude-opus-4-1-20250805", "claude-opus-4-1-20250805", true},
		{"claude-opus-4-20250514", "claude-opus-4-20250514", true},

		// Aliases
		{"claude-opus-4-5", "claude-opus-4-5-20251101", true},
		{"claude-sonnet-4-5", "claude-sonnet-4-5-20250929", true},
		{"claude-sonnet-4", "claude-sonnet-4-20250514", true},
		{"claude-opus-4", "claude-opus-4-20250514", true},
		{"claude-haiku-4-5", "claude-haiku-4-5-20251001", true},

		// Non-existent
		{"claude-2", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, found := resolveModelID(tt.input, availableModels)
			if found != tt.found {
				t.Errorf("found = %v, want %v", found, tt.found)
			}
			if got != tt.expected {
				t.Errorf("got = %q, want %q", got, tt.expected)
			}
		})
	}
}
