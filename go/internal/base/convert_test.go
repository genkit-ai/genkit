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

type convSrc struct {
	Count int64  `json:"count"`
	Name  string `json:"name"`
}

type convOther struct {
	Title string `json:"title"`
}

func TestConvertToExactShapes(t *testing.T) {
	t.Run("nil yields the zero value", func(t *testing.T) {
		got, err := ConvertToExact[convSrc](nil)
		if err != nil {
			t.Fatal(err)
		}
		if got != (convSrc{}) {
			t.Errorf("got %+v, want zero", got)
		}
	})

	t.Run("exact type passes through untouched", func(t *testing.T) {
		src := convSrc{Count: 1, Name: "a"}
		got, err := ConvertToExact[convSrc](src)
		if err != nil {
			t.Fatal(err)
		}
		if got != src {
			t.Errorf("got %+v, want %+v", got, src)
		}
	})

	t.Run("pointer is dereferenced", func(t *testing.T) {
		got, err := ConvertToExact[convSrc](&convSrc{Name: "a"})
		if err != nil {
			t.Fatal(err)
		}
		if got.Name != "a" {
			t.Errorf("got %+v, want Name=a", got)
		}
	})

	t.Run("nil pointer yields the zero value", func(t *testing.T) {
		got, err := ConvertToExact[convSrc]((*convSrc)(nil))
		if err != nil {
			t.Fatal(err)
		}
		if got != (convSrc{}) {
			t.Errorf("got %+v, want zero", got)
		}
	})

	t.Run("value converts to a pointer target", func(t *testing.T) {
		got, err := ConvertToExact[*convSrc](convSrc{Name: "a"})
		if err != nil {
			t.Fatal(err)
		}
		if got == nil || got.Name != "a" {
			t.Errorf("got %+v, want Name=a", got)
		}
	})

	// The wire forms the reflection API produces. A prompt with no declared
	// input type passes arrays and scalars through unchanged, so a helper that
	// only accepted map[string]any would reject them.
	t.Run("map wire form decodes into a struct", func(t *testing.T) {
		got, err := ConvertToExact[convSrc](map[string]any{"count": 2, "name": "a"})
		if err != nil {
			t.Fatal(err)
		}
		if got.Count != 2 || got.Name != "a" {
			t.Errorf("got %+v", got)
		}
	})

	t.Run("array wire form decodes into a slice", func(t *testing.T) {
		got, err := ConvertToExact[[]string]([]any{"a", "b"})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 2 || got[0] != "a" {
			t.Errorf("got %v", got)
		}
	})

	t.Run("scalar wire form decodes", func(t *testing.T) {
		got, err := ConvertToExact[string]("hello")
		if err != nil {
			t.Fatal(err)
		}
		if got != "hello" {
			t.Errorf("got %q", got)
		}
	})
}

// TestConvertToExactRefusesStructReinterpretation pins the one case leniency
// got wrong: a JSON round-trip between unrelated structs succeeds and leaves
// every field zero, so the caller silently gets a blank value.
func TestConvertToExactRefusesStructReinterpretation(t *testing.T) {
	_, err := ConvertToExact[convOther](convSrc{Count: 1, Name: "a"})
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	if !errors.Is(err, ErrTypeMismatch) {
		t.Errorf("err = %v, want it to wrap ErrTypeMismatch", err)
	}

	if _, ok := ConvertTo[convOther](convSrc{Name: "a"}); ok {
		t.Error("ConvertTo reported success for an unrelated struct")
	}
}

// TestConvertToExactKeepsIntegers covers the widening a plain JSON round-trip
// causes: a loosely typed target cannot restore integer-ness on its own, so the
// decoded value is normalized against the source's inferred schema.
func TestConvertToExactKeepsIntegers(t *testing.T) {
	t.Run("struct to a loose map", func(t *testing.T) {
		got, err := ConvertToExact[map[string]any](convSrc{Count: 1230000000, Name: "a"})
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := got["count"].(int64); !ok {
			t.Errorf("count = %T(%v), want int64", got["count"], got["count"])
		}
	})

	t.Run("struct slice to a loose slice", func(t *testing.T) {
		got, err := ConvertToExact[[]any]([]convSrc{{Count: 7}})
		if err != nil {
			t.Fatal(err)
		}
		first, ok := got[0].(map[string]any)
		if !ok {
			t.Fatalf("got[0] = %T, want map", got[0])
		}
		if _, ok := first["count"].(int64); !ok {
			t.Errorf("count = %T(%v), want int64", first["count"], first["count"])
		}
	})

	// A value that is already in wire form was normalized by the transport, so
	// it is returned untouched rather than re-normalized.
	t.Run("wire form is passed through", func(t *testing.T) {
		src := map[string]any{"count": int64(5)}
		got, err := ConvertToExact[map[string]any](src)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := got["count"].(int64); !ok {
			t.Errorf("count = %T, want int64", got["count"])
		}
	})

	// A concrete target restores the types itself, so no normalization runs.
	t.Run("concrete target is untouched", func(t *testing.T) {
		got, err := ConvertToExact[convSrc](map[string]any{"count": 1230000000})
		if err != nil {
			t.Fatal(err)
		}
		if got.Count != 1230000000 {
			t.Errorf("count = %d", got.Count)
		}
	})

	// A pointer restores no more than what it points at, so it normalizes too.
	// The value has to be built rather than asserted, since an assertion only
	// matches the value form.
	t.Run("pointer to a loose map", func(t *testing.T) {
		got, err := ConvertToExact[*map[string]any](convSrc{Count: 1230000000})
		if err != nil {
			t.Fatal(err)
		}
		if got == nil {
			t.Fatal("got nil")
		}
		if _, ok := (*got)["count"].(int64); !ok {
			t.Errorf("count = %T(%v), want int64", (*got)["count"], (*got)["count"])
		}
	})

	t.Run("pointer to any", func(t *testing.T) {
		got, err := ConvertToExact[*any](convSrc{Count: 1230000000})
		if err != nil {
			t.Fatal(err)
		}
		if got == nil {
			t.Fatal("got nil")
		}
		m, ok := (*got).(map[string]any)
		if !ok {
			t.Fatalf("*got = %T, want map[string]any", *got)
		}
		if _, ok := m["count"].(int64); !ok {
			t.Errorf("count = %T(%v), want int64", m["count"], m["count"])
		}
	})
}
