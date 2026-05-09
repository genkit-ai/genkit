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

package api

import (
	"context"
	"encoding/json"
)

type ActionRunResult[T any] struct {
	Result  T
	TraceId string
	SpanId  string
}

// Action is the interface that all Genkit primitives (e.g. flows, models, tools) have in common.
type Action interface {
	Registerable
	// Name returns the name of the action.
	Name() string
	// RunJSON runs the action with the given JSON input and streaming callback and returns the output as JSON.
	RunJSON(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error) (json.RawMessage, error)
	// RunJSONWithTelemetry runs the action with the given JSON input and streaming callback and returns the output as JSON along with telemetry info.
	RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error) (*ActionRunResult[json.RawMessage], error)
	// Desc returns a descriptor of the action.
	Desc() ActionDesc
}

// Registerable allows a primitive to be registered with a registry.
type Registerable interface {
	Register(r Registry)
}

// An ActionType is the kind of an action.
type ActionType string

const (
	ActionTypeRetriever        ActionType = "retriever"
	ActionTypeIndexer          ActionType = "indexer"
	ActionTypeEmbedder         ActionType = "embedder"
	ActionTypeEvaluator        ActionType = "evaluator"
	ActionTypeFlow             ActionType = "flow"
	ActionTypeModel            ActionType = "model"
	ActionTypeBackgroundModel  ActionType = "background-model"
	ActionTypeExecutablePrompt ActionType = "executable-prompt"
	ActionTypeResource         ActionType = "resource"
	ActionTypeTool             ActionType = "tool"
	ActionTypeToolV2           ActionType = "tool.v2"
	ActionTypeUtil             ActionType = "util"
	ActionTypeCustom           ActionType = "custom"
	ActionTypeCheckOperation   ActionType = "check-operation"
	ActionTypeCancelOperation  ActionType = "cancel-operation"
)

// ActionDesc is a descriptor of an action.
type ActionDesc struct {
	Type         ActionType     `json:"type"`         // Type of the action.
	Key          string         `json:"key"`          // Key of the action.
	Name         string         `json:"name"`         // Name of the action.
	Description  string         `json:"description"`  // Description of the action.
	InputSchema  map[string]any `json:"inputSchema"`  // JSON schema to validate against the action's input.
	OutputSchema map[string]any `json:"outputSchema"` // JSON schema to validate against the action's output.
	Metadata     map[string]any `json:"metadata"`     // Metadata for the action.
}
