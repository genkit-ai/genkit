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

// Named is the capability shared by everything that carries a registry name:
// the action values ([*Model], [*BackgroundModel], [*Embedder], [*Evaluator],
// [*Tool]) and the by-name references ([ModelRef], [EmbedderRef],
// [EvaluatorRef], [ToolName]).
//
// Named says a value has a name, not which kind of action it names, so it is
// deliberately not what the options accept. Each option takes its own argument
// contract instead: [ModelArg], [EmbedderArg], [EvaluatorArg], [ToolArg].
type Named interface {
	// Name returns the registry name of the action.
	Name() string
}

// ModelArg is the argument contract for [WithModel]. It is satisfied by
// [*Model], [*BackgroundModel], and [ModelRef].
//
// The unexported method seals the interface: only this package can produce a
// value that fits, which is what stops an embedder or a tool from being passed
// where a model is expected. The primitives are all the same underlying action
// type and the by-name refs have no behavior at all, so a nominal marker is
// the only way to tell the kinds apart.
type ModelArg interface {
	Named
	modelArg()
}

// EmbedderArg is the argument contract for [WithEmbedder]. It is satisfied by
// [*Embedder] and [EmbedderRef]. See [ModelArg] for why it is sealed.
type EmbedderArg interface {
	Named
	embedderArg()
}

// EvaluatorArg is the argument contract for [WithEvaluator]. It is satisfied
// by [*Evaluator] and [EvaluatorRef]. See [ModelArg] for why it is sealed.
type EvaluatorArg interface {
	Named
	evaluatorArg()
}

// ToolArg is the argument contract for [WithTools]. It is satisfied by any
// [AnyTool] (including [*Tool] and [*InterruptibleTool]) and by [ToolName].
// See [ModelArg] for why it is sealed.
type ToolArg interface {
	Named
	toolArg()
}

// actionRef is the shared implementation behind the by-name reference types.
// It is embedded rather than exported so the name, the default config, and the
// JSON handling are written once, while each kind stays a distinct type at the
// call site.
type actionRef struct {
	name   string
	config any
}

// Name returns the name of the referenced action.
func (r actionRef) Name() string {
	return r.name
}

// Config returns the configuration to use by default for the referenced action.
func (r actionRef) Config() any {
	return r.config
}

// MarshalJSON implements [json.Marshaler]. A ref always marshals as a JSON
// object with "name" and optional "config" fields.
func (r actionRef) MarshalJSON() ([]byte, error) {
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
func (r *actionRef) UnmarshalJSON(data []byte) error {
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

// actionRefSchema builds the object schema shared by the ref types. The refs
// implement the invopop/jsonschema customSchemaImpl interface so reflection
// produces this instead of an empty object (they have only unexported fields).
// example names the kind in the "name" description.
func actionRefSchema(example string) *jsonschema.Schema {
	props := jsonschema.NewProperties()
	props.Set("name", &jsonschema.Schema{
		Type:        "string",
		Description: "Action name (e.g. \"" + example + "\")",
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

// ModelRef is a lazy, by-name reference to a registered model, optionally
// carrying a default configuration. Use it where a [ModelArg] is expected but
// the [*Model] value is not in scope, e.g. referencing a plugin-provided model
// by its registry name:
//
//	ai.Generate(ctx, r, ai.WithModel(ai.NewModelRef("googleai/gemini-flash-latest", nil)))
//
// ModelRef supports JSON marshaling: it serializes as {"name": "...",
// "config": ...} and unmarshals from either that object form or a plain
// string (interpreted as the name).
type ModelRef struct {
	actionRef
}

// NewModelRef creates a [ModelRef] with the given name and default
// configuration. The config is used by [Generate] (and prompt execution) when
// no explicit [WithConfig] is provided.
func NewModelRef(name string, config any) ModelRef {
	return ModelRef{actionRef{name: name, config: config}}
}

func (ModelRef) modelArg() {}

// JSONSchema implements the invopop/jsonschema customSchemaImpl interface.
func (ModelRef) JSONSchema() *jsonschema.Schema {
	return actionRefSchema("googleai/gemini-flash-latest")
}

// EmbedderRef is a lazy, by-name reference to a registered embedder,
// optionally carrying a default configuration. Use it where an [EmbedderArg]
// is expected but the [*Embedder] value is not in scope. It marshals like
// [ModelRef].
type EmbedderRef struct {
	actionRef
}

// NewEmbedderRef creates an [EmbedderRef] with the given name and default
// configuration. The config is used by [Embed] when no explicit [WithConfig]
// is provided.
func NewEmbedderRef(name string, config any) EmbedderRef {
	return EmbedderRef{actionRef{name: name, config: config}}
}

func (EmbedderRef) embedderArg() {}

// JSONSchema implements the invopop/jsonschema customSchemaImpl interface.
func (EmbedderRef) JSONSchema() *jsonschema.Schema {
	return actionRefSchema("googleai/text-embedding-004")
}

// EvaluatorRef is a lazy, by-name reference to a registered evaluator,
// optionally carrying a default configuration. Use it where an [EvaluatorArg]
// is expected but the [*Evaluator] value is not in scope. It marshals like
// [ModelRef].
type EvaluatorRef struct {
	actionRef
}

// NewEvaluatorRef creates an [EvaluatorRef] with the given name and default
// configuration. The config is used by [Evaluate] when no explicit
// [WithConfig] is provided.
func NewEvaluatorRef(name string, config any) EvaluatorRef {
	return EvaluatorRef{actionRef{name: name, config: config}}
}

func (EvaluatorRef) evaluatorArg() {}

// JSONSchema implements the invopop/jsonschema customSchemaImpl interface.
func (EvaluatorRef) JSONSchema() *jsonschema.Schema {
	return actionRefSchema("genkitEval/faithfulness")
}

var (
	_ ModelArg     = ModelRef{}
	_ EmbedderArg  = EmbedderRef{}
	_ EvaluatorArg = EvaluatorRef{}
)
