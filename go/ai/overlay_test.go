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

package ai

import (
	"fmt"
	"reflect"
	"testing"
)

// TestOverlayCoversEveryField is the guard that keeps the hand-written Overlay
// methods honest. A field added to either options struct and forgotten in
// Overlay would silently drop a caller's value; here it fails the build's tests
// instead, naming the field.
//
// It also holds both structs to the invariant Overlay is built on: every field
// must distinguish its zero value from a meaningful one, so a bool field (whose
// false is both "unset" and "off") is rejected outright rather than merged
// wrongly.
func TestOverlayCoversEveryField(t *testing.T) {
	t.Run("ModelOptions", func(t *testing.T) {
		checkOverlay(t, func(base, override ModelOptions) ModelOptions { return base.Overlay(override) })
	})
	t.Run("EmbedderOptions", func(t *testing.T) {
		checkOverlay(t, func(base, override EmbedderOptions) EmbedderOptions { return base.Overlay(override) })
	})
}

// checkOverlay exercises overlay once per field of T: the override's value must
// win when set, and the base's must survive when the override leaves it zero.
func checkOverlay[T any](t *testing.T, overlay func(base, override T) T) {
	t.Helper()
	typ := reflect.TypeFor[T]()
	for i := range typ.NumField() {
		field := typ.Field(i)
		if !field.IsExported() {
			continue
		}
		if field.Type.Kind() == reflect.Bool {
			t.Errorf("%s.%s is a bool, whose zero value cannot mean 'not specified'; Overlay cannot merge it",
				typ.Name(), field.Name)
			continue
		}
		baseVal, err := nonZeroValue(field.Type, 1)
		if err != nil {
			t.Errorf("%s.%s: %v", typ.Name(), field.Name, err)
			continue
		}
		overrideVal, _ := nonZeroValue(field.Type, 2)

		var base, override T
		reflect.ValueOf(&base).Elem().Field(i).Set(baseVal)
		reflect.ValueOf(&override).Elem().Field(i).Set(overrideVal)

		if got := reflect.ValueOf(overlay(base, override)).Field(i); !reflect.DeepEqual(got.Interface(), overrideVal.Interface()) {
			t.Errorf("%s.%s: Overlay kept %v, want the override's %v; the field is missing from Overlay",
				typ.Name(), field.Name, got.Interface(), overrideVal.Interface())
		}
		var zero T
		if got := reflect.ValueOf(overlay(base, zero)).Field(i); !reflect.DeepEqual(got.Interface(), baseVal.Interface()) {
			t.Errorf("%s.%s: an unset override replaced the base's %v with %v",
				typ.Name(), field.Name, baseVal.Interface(), got.Interface())
		}
	}
}

// nonZeroValue builds a distinguishable non-zero value of type t. seed makes
// two calls for the same type differ, so a field Overlay ignores shows up as
// the base's value surviving where the override's was expected.
func nonZeroValue(t reflect.Type, seed int) (reflect.Value, error) {
	v := reflect.New(t).Elem()
	switch t.Kind() {
	case reflect.String:
		v.SetString(fmt.Sprintf("value-%d", seed))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(int64(seed))
	case reflect.Float32, reflect.Float64:
		v.SetFloat(float64(seed))
	case reflect.Pointer:
		v.Set(reflect.New(t.Elem()))
	case reflect.Slice:
		elem, err := nonZeroValue(t.Elem(), seed)
		if err != nil {
			return v, err
		}
		v.Set(reflect.Append(v, elem))
	case reflect.Map:
		key, err := nonZeroValue(t.Key(), seed)
		if err != nil {
			return v, err
		}
		v.Set(reflect.MakeMap(t))
		v.SetMapIndex(key, reflect.ValueOf(any(seed)))
	default:
		return v, fmt.Errorf("no non-zero value for kind %s; teach nonZeroValue about it", t.Kind())
	}
	return v, nil
}

// TestCloneSupportsDetachesEveryField is the same guard for the clone helpers:
// a slice or map field added to a supports struct and not cloned would leave
// two models built from one shared value aliasing each other's memory.
func TestCloneSupportsDetachesEveryField(t *testing.T) {
	t.Run("ModelSupports", func(t *testing.T) {
		checkCloneDetaches(t, cloneModelSupports)
	})
	t.Run("EmbedderSupports", func(t *testing.T) {
		checkCloneDetaches(t, cloneEmbedderSupports)
	})
	t.Run("RetrieverSupports", func(t *testing.T) {
		checkCloneDetaches(t, cloneRetrieverSupports)
	})
}

// checkCloneDetaches fills every field of T with a non-zero value, clones, and
// checks the clone is equal but shares no reference-typed field with the
// original.
func checkCloneDetaches[T any](t *testing.T, clone func(*T) *T) {
	t.Helper()
	typ := reflect.TypeFor[T]()
	orig := reflect.New(typ)
	for i := range typ.NumField() {
		field := typ.Field(i)
		if !field.IsExported() || field.Type.Kind() == reflect.Bool {
			continue
		}
		val, err := nonZeroValue(field.Type, 1)
		if err != nil {
			t.Fatalf("%s.%s: %v", typ.Name(), field.Name, err)
		}
		orig.Elem().Field(i).Set(val)
	}
	got := reflect.ValueOf(clone(orig.Interface().(*T))).Elem()
	for i := range typ.NumField() {
		field := typ.Field(i)
		want := orig.Elem().Field(i)
		if !reflect.DeepEqual(got.Field(i).Interface(), want.Interface()) {
			t.Errorf("%s.%s: clone changed the value: got %v, want %v",
				typ.Name(), field.Name, got.Field(i).Interface(), want.Interface())
		}
		switch field.Type.Kind() {
		case reflect.Slice, reflect.Map, reflect.Pointer:
			if got.Field(i).Pointer() == want.Pointer() {
				t.Errorf("%s.%s: clone aliases the original's %s; the field is missing from the clone helper",
					typ.Name(), field.Name, field.Type.Kind())
			}
		}
	}
}
