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

package googlegenai

import "testing"

func TestDeprecatedGenAIModels(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"gemini-3-pro-preview", true},
		{"gemini-2.5-pro", false},
		{"gemini-2.5-flash", false},
		{"", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, got := deprecatedGenAIModels[tc.name]
			if got != tc.want {
				t.Errorf("deprecatedGenAIModels[%q] presence = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}
