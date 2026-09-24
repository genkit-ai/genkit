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

package base

import "testing"

func TestToolInterruptError_Error(t *testing.T) {
	tests := []struct {
		name string
		data any
		want string
	}{
		{"object data", map[string]any{"key": "value"}, "tool execution interrupted: \n\n{\n  \"key\": \"value\"\n}"},
		{"bare interrupt", nil, "tool execution interrupted"},
		{"unmarshalable data", func() {}, "tool execution interrupted"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &ToolInterruptError{Data: tt.data}
			if got := err.Error(); got != tt.want {
				t.Errorf("Error() = %q, want %q", got, tt.want)
			}
		})
	}
}
