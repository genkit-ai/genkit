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

package base

import "testing"

type sample struct {
	Name string
	N    int
}

func TestIsNil(t *testing.T) {
	tests := []struct {
		name string
		got  bool
		want bool
	}{
		{"nil interface", IsNil[any](nil), true},
		{"nil pointer", IsNil[*sample](nil), true},
		{"nil slice", IsNil[[]int](nil), true},
		{"nil map", IsNil[map[string]int](nil), true},

		{"empty slice", IsNil[[]int]([]int{}), false},
		{"empty map", IsNil[map[string]int](map[string]int{}), false},
		{"zero struct", IsNil(sample{}), false},
		{"populated struct", IsNil(sample{Name: "x", N: 1}), false},
		{"empty string", IsNil(""), false},
		{"zero int", IsNil(0), false},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s: got %v, want %v", tt.name, tt.got, tt.want)
		}
	}
}

func TestIsZero(t *testing.T) {
	tests := []struct {
		name string
		got  bool
		want bool
	}{
		// Nil-like values: zero matches the IsNil behavior.
		{"nil interface", IsZero[any](nil), true},
		{"nil pointer", IsZero[*sample](nil), true},
		{"nil slice", IsZero[[]int](nil), true},
		{"nil map", IsZero[map[string]int](nil), true},

		// Empty-but-non-nil collections are not zero.
		{"empty slice", IsZero[[]int]([]int{}), false},
		{"empty map", IsZero[map[string]int](map[string]int{}), false},

		// Structs: zero-value vs populated. This is the case IsNil misses.
		{"zero struct", IsZero(sample{}), true},
		{"struct with name", IsZero(sample{Name: "x"}), false},
		{"struct with n", IsZero(sample{N: 1}), false},
		{"populated struct", IsZero(sample{Name: "x", N: 1}), false},

		// Pointers: nil and non-nil pointing to a zero-value pointee are both
		// considered zero (transitive unwrap).
		{"non-nil pointer to zero struct", IsZero(&sample{}), true},
		{"non-nil pointer to populated struct", IsZero(&sample{Name: "x"}), false},
		{"pointer to zero int", IsZero(new(int)), true},
		{"pointer to nonzero int", func() bool { n := 7; return IsZero(&n) }(), false},

		// Primitives.
		{"empty string", IsZero(""), true},
		{"non-empty string", IsZero("x"), false},
		{"zero int", IsZero(0), true},
		{"nonzero int", IsZero(7), false},
		{"false bool", IsZero(false), true},
		{"true bool", IsZero(true), false},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s: got %v, want %v", tt.name, tt.got, tt.want)
		}
	}
}
