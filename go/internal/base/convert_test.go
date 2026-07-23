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
	"errors"
	"testing"
)

type exactTarget struct {
	Name  string `json:"name,omitempty"`
	Count int    `json:"count,omitempty"`
}

type exactOther struct {
	Name string `json:"name,omitempty"`
}

func TestConvertToExact(t *testing.T) {
	tests := []struct {
		name         string
		in           any
		want         exactTarget
		wantMismatch bool
	}{
		{name: "nil yields zero value", in: nil, want: exactTarget{}},
		{name: "exact type", in: exactTarget{Name: "a", Count: 1}, want: exactTarget{Name: "a", Count: 1}},
		{name: "pointer is dereferenced", in: &exactTarget{Name: "b"}, want: exactTarget{Name: "b"}},
		{name: "typed nil pointer yields zero value", in: (*exactTarget)(nil), want: exactTarget{}},
		{name: "map is decoded", in: map[string]any{"name": "c", "count": 2}, want: exactTarget{Name: "c", Count: 2}},
		{name: "mismatched struct is rejected", in: exactOther{Name: "d"}, wantMismatch: true},
		{name: "mismatched pointer is rejected", in: &exactOther{Name: "d"}, wantMismatch: true},
		{name: "mismatched primitive is rejected", in: 42, wantMismatch: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ConvertToExact[exactTarget](tt.in)
			if tt.wantMismatch {
				if !errors.Is(err, ErrTypeMismatch) {
					t.Fatalf("ConvertToExact(%v) error = %v, want ErrTypeMismatch", tt.in, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ConvertToExact(%v) unexpected error: %v", tt.in, err)
			}
			if got != tt.want {
				t.Errorf("ConvertToExact(%v) = %+v, want %+v", tt.in, got, tt.want)
			}
		})
	}

	t.Run("map decode error is not a type mismatch", func(t *testing.T) {
		_, err := ConvertToExact[exactTarget](map[string]any{"count": "not a number"})
		if err == nil || errors.Is(err, ErrTypeMismatch) {
			t.Fatalf("ConvertToExact() error = %v, want decode error distinct from ErrTypeMismatch", err)
		}
	})

	t.Run("any accepts everything", func(t *testing.T) {
		got, err := ConvertToExact[any](exactOther{Name: "d"})
		if err != nil {
			t.Fatalf("ConvertToExact[any]() unexpected error: %v", err)
		}
		if got != (exactOther{Name: "d"}) {
			t.Errorf("ConvertToExact[any]() = %v, want passthrough", got)
		}
	})
}

func TestSchemaMapFor(t *testing.T) {
	t.Run("struct type", func(t *testing.T) {
		schema := SchemaMapFor[exactTarget]()
		props, ok := schema["properties"].(map[string]any)
		if !ok {
			t.Fatalf("SchemaMapFor() = %v, want schema with properties", schema)
		}
		if _, ok := props["name"]; !ok {
			t.Errorf("schema missing name property: %v", props)
		}
	})

	t.Run("pointer type", func(t *testing.T) {
		schema := SchemaMapFor[*exactTarget]()
		if _, ok := schema["properties"].(map[string]any); !ok {
			t.Errorf("SchemaMapFor[*T]() = %v, want schema with properties", schema)
		}
	})

	t.Run("interface type yields nil", func(t *testing.T) {
		if schema := SchemaMapFor[any](); schema != nil {
			t.Errorf("SchemaMapFor[any]() = %v, want nil", schema)
		}
	})
}
