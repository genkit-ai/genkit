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
	"fmt"
	"reflect"
)

// ToolInterruptError is the error a tool call returns to pause generation and
// hand control back to the caller. ai/tool.Interrupt is the one public way to
// raise it; ai recognizes it with errors.As when the tool returns, records Data
// on the interrupted tool request part, and reports it through
// ai.IsToolInterruptError. The type is internal so that raising an interrupt
// has a single entry point and the in-process representation can change with
// the wire contract.
//
// Data is the interrupt payload: nil for a bare interrupt, otherwise the JSON
// object it serializes to, as a map[string]any. ai/tool.Interrupt normalizes
// it with [ObjectPayload] when the interrupt is raised, so a payload has one
// shape in process and after a wire hop; ai normalizes again when the tool
// returns, for an error built with a struct directly.
//
// RaisedBy names the stage that raised the interrupt: the id of a WrapTool
// hook, read from the context under [ToolHookKey] by ai/tool.Interrupt, or
// empty for the tool function itself. ai records it on the interrupted part,
// and on restart delivers the answer to that stage alone.
//
// Released lists, in chain order, the ids of the WrapTool hooks that let the
// call through before it interrupted. The hook chain in ai/generate.go
// records each as the interrupt passes back out through it; nil when no hook
// let the call through.
type ToolInterruptError struct {
	Data     any
	RaisedBy string
	Released []string
}

func (e *ToolInterruptError) Error() string {
	if e.Data != nil {
		if data, err := json.MarshalIndent(e.Data, "", "  "); err == nil {
			return fmt.Sprintf("tool execution interrupted: \n\n%s", string(data))
		}
	}
	return "tool execution interrupted"
}

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
// round trip. nil passes, as a bare interrupt or restart.
func CheckObjectPayload(data any, what string) error {
	if IsNil(data) || IsJSONObject(data) {
		return nil
	}
	return fmt.Errorf("%s must serialize to a JSON object (a struct or map), got %T", what, data)
}

// IsJSONObject reports whether v serializes to a JSON object by construction:
// a struct, possibly behind pointers, or a map with string keys. Nil, scalars,
// slices and arrays do not.
func IsJSONObject(v any) bool {
	t := reflect.TypeOf(v)
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil {
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
