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

package otelgenai

import (
	"encoding/json"
	"sort"

	"go.opentelemetry.io/otel/attribute"
)

// structuredValue converts a JSON-shaped value to a structured attribute value
// (MAP / SLICE / scalars), as the spec's operation.details event expects.
//
// Message parts carry arbitrary Go values (tool arguments and results), so the
// value is first normalized through JSON: that gives the same shape the span
// attribute's JSON string has, and only leaves maps, slices, and scalars.
func structuredValue(v any) attribute.Value {
	b, err := json.Marshal(v)
	if err != nil {
		return attribute.StringValue(jsonString(v))
	}
	var norm any
	if err := json.Unmarshal(b, &norm); err != nil {
		return attribute.StringValue(string(b))
	}
	return toAttributeValue(norm)
}

// toAttributeValue converts a value decoded by encoding/json into any.
func toAttributeValue(v any) attribute.Value {
	switch v := v.(type) {
	case nil:
		return attribute.Value{} // EMPTY
	case bool:
		return attribute.BoolValue(v)
	case float64:
		// encoding/json decodes every number as float64; keep whole numbers
		// (token counts, sizes) as ints.
		if i := int64(v); float64(i) == v {
			return attribute.Int64Value(i)
		}
		return attribute.Float64Value(v)
	case string:
		return attribute.StringValue(v)
	case []any:
		vals := make([]attribute.Value, len(v))
		for i, e := range v {
			vals[i] = toAttributeValue(e)
		}
		return attribute.SliceValue(vals...)
	case map[string]any:
		// Sorted for deterministic output.
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		kvs := make([]attribute.KeyValue, len(keys))
		for i, k := range keys {
			kvs[i] = attribute.KeyValue{Key: attribute.Key(k), Value: toAttributeValue(v[k])}
		}
		return attribute.MapValue(kvs...)
	default:
		return attribute.StringValue(jsonString(v))
	}
}
