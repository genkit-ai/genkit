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
	"bytes"
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
)

// ObjectPayload converts an interrupt or resume payload to the JSON object the
// wire contract requires: nil, a nil map and a nil pointer stay nil (a bare
// interrupt or restart), a map[string]any is returned as is, and any other
// value that is a JSON object by construction (see [IsJSONObject]) is
// converted through JSON. A scalar, slice or array is rejected; what names the
// payload in the error.
func ObjectPayload(data any, what string) (map[string]any, error) {
	if IsNil(data) {
		return nil, nil
	}
	if m, ok := data.(map[string]any); ok {
		return m, nil
	}
	if err := CheckObjectPayload(data, what); err != nil {
		return nil, err
	}
	m, err := StructToMap(data)
	if err != nil {
		return nil, fmt.Errorf("%s must serialize to a JSON object (a struct or map), got %T: %w", what, data, err)
	}
	return m, nil
}

// CheckObjectPayload is the check half of [ObjectPayload], for a caller that
// leaves the conversion to the reader: it costs a type inspection, not a JSON
// round trip, except for a type with its own encoding, which is encoded to
// see what it produces. nil passes, as a bare interrupt or restart.
func CheckObjectPayload(data any, what string) error {
	if IsNil(data) || IsJSONObject(data) {
		return nil
	}
	if t := reflect.TypeOf(data); encodesItself(t) {
		// json.Marshal compacts a MarshalJSON result, so an object starts
		// with its brace.
		if b, err := json.Marshal(data); err == nil && bytes.HasPrefix(b, []byte("{")) {
			return nil
		}
	}
	return fmt.Errorf("%s must serialize to a JSON object (a struct or map), got %T", what, data)
}

// IsJSONObject reports whether v serializes to a JSON object by construction:
// a struct, possibly behind pointers, or a map with string keys. Nil, scalars,
// slices and arrays do not, and neither does a type with its own JSON or text
// encoding, such as [time.Time], which can serialize to anything.
func IsJSONObject(v any) bool {
	t := reflect.TypeOf(v)
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil || encodesItself(t) {
		return false
	}
	switch t.Kind() {
	case reflect.Struct:
		return true
	case reflect.Map:
		return t.Key().Kind() == reflect.String
	}
	return false
}

var textMarshalerType = reflect.TypeFor[encoding.TextMarshaler]()

// encodesItself reports whether t, a pointer type it points through, or a
// pointer to it has its own JSON or text encoding, which overrides the shape
// its kind suggests.
func encodesItself(t reflect.Type) bool {
	for t.Kind() == reflect.Pointer {
		if t.Implements(jsonMarshalerType) || t.Implements(textMarshalerType) {
			return true
		}
		t = t.Elem()
	}
	pt := reflect.PointerTo(t)
	return t.Implements(jsonMarshalerType) || pt.Implements(jsonMarshalerType) ||
		t.Implements(textMarshalerType) || pt.Implements(textMarshalerType)
}
