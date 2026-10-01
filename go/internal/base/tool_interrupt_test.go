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

import (
	"encoding/json"
	"testing"
	"time"
)

type objectMarshaler struct{}

func (objectMarshaler) MarshalJSON() ([]byte, error) { return []byte(`{"ok": true}`), nil }

// TestCheckObjectPayload pins which payloads pass as a JSON object. A type
// with its own encoding is judged by what it encodes to, not by its kind:
// time.Time is a struct that encodes to a string.
func TestCheckObjectPayload(t *testing.T) {
	tests := []struct {
		name string
		data any
		ok   bool
	}{
		{"nil", nil, true},
		{"map", map[string]any{"a": 1}, true},
		{"struct", struct{ A int }{1}, true},
		{"struct pointer", &struct{ A int }{1}, true},
		{"object marshaler", objectMarshaler{}, true},
		{"string", "approved", false},
		{"number", 1, false},
		{"slice", []any{1}, false},
		{"time", time.Now(), false},
		{"time pointer", new(time.Time), false},
		{"raw message array", json.RawMessage(`[1]`), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckObjectPayload(tt.data, "payload")
			if (err == nil) != tt.ok {
				t.Errorf("CheckObjectPayload(%T) = %v, want ok=%v", tt.data, err, tt.ok)
			}
		})
	}
}
