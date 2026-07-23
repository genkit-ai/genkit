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

package ai

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
)

// Named is the argument contract for options that accept an action by value
// or by reference (e.g. [WithModel], [WithTools], [WithEmbedder],
// [WithEvaluator]). Concrete action values ([*Model], [*Tool], [*Embedder],
// [*Evaluator]) satisfy it directly, and [ActionRef] (or [ToolName] for
// tools) satisfies it for lazy, by-name references.
type Named interface {
	// Name returns the registry name of the action.
	Name() string
}

// ActionRef is a lazy, by-name reference to a registered action, optionally
// carrying a default configuration. Use it where a [Named] is expected but
// the action value is not in scope, e.g. referencing a plugin-provided model
// by its registry name:
//
//	ai.Generate(ctx, r, ai.WithModel(ai.NewActionRef("googleai/gemini-flash-latest", nil)))
//
// ActionRef supports JSON marshaling: it serializes as {"name": "...",
// "config": ...} and unmarshals from either that object form or a plain
// string (interpreted as the name).
type ActionRef struct {
	name   string
	config any
}

var _ Named = ActionRef{}

// NewActionRef creates a new [ActionRef] with the given name and default
// configuration. The config is used by [Generate] (and prompt execution) when
// no explicit [WithConfig] is provided.
func NewActionRef(name string, config any) ActionRef {
	return ActionRef{name: name, config: config}
}

// Name returns the name of the referenced action.
func (r ActionRef) Name() string {
	return r.name
}

// Config returns the configuration to use by default for the referenced action.
func (r ActionRef) Config() any {
	return r.config
}

// MarshalJSON implements [json.Marshaler]. ActionRef always marshals as a
// JSON object with "name" and optional "config" fields.
func (r ActionRef) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Name   string `json:"name"`
		Config any    `json:"config,omitempty"`
	}{
		Name:   r.name,
		Config: r.config,
	})
}

// UnmarshalJSON implements [json.Unmarshaler]. It accepts either a JSON
// object with "name" and optional "config" fields, or a plain string
// (interpreted as the action name).
func (r *ActionRef) UnmarshalJSON(data []byte) error {
	// Try string shorthand first.
	var name string
	if err := json.Unmarshal(data, &name); err == nil {
		r.name = name
		r.config = nil
		return nil
	}
	var obj struct {
		Name   string          `json:"name"`
		Config json.RawMessage `json:"config,omitempty"`
	}
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	r.name = obj.Name
	r.config = nil
	if len(obj.Config) > 0 {
		var config any
		if err := json.Unmarshal(obj.Config, &config); err != nil {
			return err
		}
		r.config = config
	}
	return nil
}

// JSONSchema implements the invopop/jsonschema customSchemaImpl interface
// so that schema reflection produces the correct object schema instead of
// an empty object (ActionRef has only unexported fields).
func (ActionRef) JSONSchema() *jsonschema.Schema {
	props := jsonschema.NewProperties()
	props.Set("name", &jsonschema.Schema{
		Type:        "string",
		Description: "Action name (e.g. \"googleai/gemini-flash-latest\")",
	})
	props.Set("config", &jsonschema.Schema{
		Description: "Optional action configuration",
	})
	return &jsonschema.Schema{
		Type:       "object",
		Properties: props,
		Required:   []string{"name"},
	}
}
