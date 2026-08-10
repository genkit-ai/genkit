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
	"iter"
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

// BidiJSONOptions carries the options for a JSON-encoded call to a bidi
// action, used by both the one-shot [BidiAction.RunBidiJSON] and the streaming
// [BidiAction.ConnectJSON]. It is the JSON counterpart of the typed init
// argument. A nil value is equivalent to zero options. The struct may gain
// fields over time; construct it by field name.
//
// Experimental: bidirectional streaming is experimental and subject to change.
type BidiJSONOptions struct {
	// Init is the JSON-encoded initial configuration for the call,
	// decoded into the action's Init type and validated against its
	// InitSchema. Empty or JSON-null means no init (the zero Init value).
	Init json.RawMessage
}

// BidiAction is implemented by actions that support bidirectional streaming.
// Non-bidi actions do not implement this interface; callers may detect bidi
// support with a type assertion. The descriptor's "bidi" metadata carries the
// same signal for tooling that only sees serialized descriptors.
//
// Experimental: bidirectional streaming is experimental and subject to change.
type BidiAction interface {
	Action
	// RunBidiJSON runs the bidi action as a single one-shot call: input is
	// delivered as the only chunk on the input stream, outgoing chunks are
	// forwarded to cb, and opts carries the session init. Input is required;
	// only a streaming session can defer it past startup.
	RunBidiJSON(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error, opts *BidiJSONOptions) (*ActionRunResult[json.RawMessage], error)
	// ConnectJSON starts a bidirectional streaming session using
	// JSON-encoded messages.
	ConnectJSON(ctx context.Context, opts *BidiJSONOptions) (BidiJSONConnection, error)
}

// BidiJSONConnection is a JSON-encoded view of an active bidirectional
// streaming session. It mirrors the typed BidiConnection API but works in
// terms of json.RawMessage payloads, allowing generic transports (e.g. the
// reflection API) to wire bidi actions without knowing their concrete types.
//
// Experimental: bidirectional streaming is experimental and subject to change.
type BidiJSONConnection interface {
	// Send encodes chunk as the action's In type and sends it to the action.
	// A chunk that fails to decode or validate against the action's input
	// schema fails the session: the error is returned and also becomes the
	// session's terminal error, reported by Output. Transports that need
	// per-chunk tolerance must validate before calling Send.
	Send(chunk json.RawMessage) error
	// Close signals that no more inputs will be sent.
	Close() error
	// Receive yields outgoing stream chunks encoded as JSON. The iterator
	// completes when the action finishes.
	Receive() iter.Seq2[json.RawMessage, error]
	// Output returns the final output encoded as JSON, blocking until the
	// action completes or the context is cancelled.
	Output() (json.RawMessage, error)
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
	ActionTypeAgentSnapshot    ActionType = "agent-snapshot"
	ActionTypeAgentAbort       ActionType = "agent-abort"
	ActionTypeCheckOperation   ActionType = "check-operation"
	ActionTypeCancelOperation  ActionType = "cancel-operation"
	ActionTypeAgent            ActionType = "agent"
)

// ActionDesc is a descriptor of an action.
type ActionDesc struct {
	Type         ActionType     `json:"type"`                   // Type of the action.
	Key          string         `json:"key"`                    // Key of the action.
	Name         string         `json:"name"`                   // Name of the action.
	Description  string         `json:"description"`            // Description of the action.
	InputSchema  map[string]any `json:"inputSchema"`            // JSON schema to validate against the action's input.
	OutputSchema map[string]any `json:"outputSchema"`           // JSON schema to validate against the action's output.
	StreamSchema map[string]any `json:"streamSchema,omitempty"` // JSON schema to validate against the action's outgoing streamed chunks.
	InitSchema   map[string]any `json:"initSchema,omitempty"`   // JSON schema to validate against the action's initial configuration (bidi only).
	Metadata     map[string]any `json:"metadata"`               // Metadata for the action.
}
