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

package jsonschema

import (
	"reflect"

	"github.com/firebase/genkit/go/internal/base"
	"github.com/invopop/jsonschema"
)

// ReflectConfigSchema reflects an SDK request-params struct into the JSON
// schema map a model advertises as its config. The provider SDKs wrap
// optional primitives in generic Opt types that marshal as bare values but
// reflect as objects, so those are mapped to their wire shapes; overrides
// adds SDK-specific mappings by type name (e.g. inline unions) that take
// precedence.
func ReflectConfigSchema(config any, overrides map[string]*jsonschema.Schema) map[string]any {
	r := jsonschema.Reflector{
		DoNotReference:             true, // Prevent $ref usage
		AllowAdditionalProperties:  false,
		ExpandedStruct:             true,
		RequiredFromJSONSchemaTags: true,
	}
	r.Mapper = func(t reflect.Type) *jsonschema.Schema {
		if s, ok := overrides[t.Name()]; ok {
			return s
		}
		switch t.Name() {
		case "Opt[float64]":
			return &jsonschema.Schema{Type: "number"}
		case "Opt[int64]":
			return &jsonschema.Schema{Type: "integer"}
		case "Opt[string]":
			return &jsonschema.Schema{Type: "string"}
		case "Opt[bool]":
			return &jsonschema.Schema{Type: "boolean"}
		}
		return nil
	}
	return base.SchemaAsMap(r.Reflect(config))
}
