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

import (
	"net/url"
	"reflect"
)

// An Environment is the execution context in which the program is running.
type Environment string

const (
	EnvironmentDev  Environment = "dev"  // development: testing, debugging, etc.
	EnvironmentProd Environment = "prod" // production: user data, SLOs, etc.
)

// Zero returns the Zero value for T.
func Zero[T any]() T {
	var z T
	return z
}

// Clean returns a valid filename for id.
func Clean(id string) string {
	return url.PathEscape(id)
}

// IsNil returns true if v is nil or a nil pointer/interface/map/slice/channel/func.
func IsNil[T any](v T) bool {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Invalid:
		return true
	case reflect.Ptr, reflect.Interface, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func:
		return rv.IsNil()
	default:
		return false
	}
}

// IsZero returns true if v is the zero value for its type.
//
// Unlike [IsNil], it returns true for zero-value structs and primitives (e.g.
// Recipe{}, "", 0) in addition to nil pointers/interfaces/maps/slices/channels/funcs.
// Pointers and interfaces are followed transitively, so a non-nil pointer that
// points to a zero-value struct (&Recipe{}) is also considered zero. This is
// useful for "is there meaningful content here?" checks regardless of whether
// the value is held by reference or by value.
//
// Empty-but-non-nil slices and maps (e.g. []int{}) are still considered
// non-zero, matching reflect.Value.IsZero semantics.
func IsZero[T any](v T) bool {
	rv := reflect.ValueOf(v)
	for rv.IsValid() && (rv.Kind() == reflect.Ptr || rv.Kind() == reflect.Interface) {
		if rv.IsNil() {
			return true
		}
		rv = rv.Elem()
	}
	if !rv.IsValid() {
		return true
	}
	return rv.IsZero()
}
