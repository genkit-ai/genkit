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

// Package wire holds the HTTP wire format that the action handler (package
// genkit) serves and the remote agent client (package ai/exp) speaks, so the
// two sides share one definition and cannot drift apart.
//
// A request is a POST whose body is a [Request]. A non-streaming response is a
// [ResultResponse] on success and an [Error] on failure, with the HTTP status
// mapped from the error's status. A streaming response is server-sent events,
// one "data: " line per event: a [MessageResponse] per chunk, then one
// [ResultResponse] or [ErrorResponse].
package wire

import (
	"context"
	"encoding/json"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
)

// Request is the body of a POST to an action.
type Request struct {
	// Data is the action's input.
	Data json.RawMessage `json:"data"`
	// Init is the session init of a bidi action, and is rejected for any
	// other action.
	Init json.RawMessage `json:"init,omitempty"`
}

// ResultResponse carries an action's final result.
type ResultResponse struct {
	Result json.RawMessage `json:"result"`
}

// MessageResponse carries one streamed chunk.
type MessageResponse struct {
	Message json.RawMessage `json:"message"`
}

// ErrorResponse carries the error that ended a stream.
type ErrorResponse struct {
	Error *Error `json:"error"`
}

// Error is a failure on the wire: the body of a non-streaming error response,
// and the payload of a streaming one. It carries no details, so internal
// failure text stays off the wire.
type Error struct {
	Status  status.Name `json:"status"`
	Message string      `json:"message"`
}

// ClientMessage returns the message a client may see for err, and whether it
// is err's own text. The text leaves the process only when err was built with
// [status.PublicErrorf], or in the dev environment, where hiding a failure only
// hides it from the developer causing it. Otherwise it is a generic message
// derived from the status, so schema dumps, provider text, and internal
// identifiers stay server-side.
func ClientMessage(err error) (msg string, own bool) {
	msg, public := status.PublicMessage(err)
	switch {
	case public:
		return msg, true
	case api.CurrentEnvironment() == api.EnvironmentDev:
		return err.Error(), true
	}
	return msg, false
}

// SSEDataPrefix starts every server-sent event line the handler writes.
const SSEDataPrefix = "data: "

type servedActionKey struct{}

// WithServedAction marks ctx as the context of a request that a transport
// serves for the action with the given key. The action reads it with
// [ServesAction] to shape what it returns for a client, such as redacting
// internal error text. The mark names one action, so actions that the served
// action calls in turn do not read it as their own.
func WithServedAction(ctx context.Context, key string) context.Context {
	return context.WithValue(ctx, servedActionKey{}, key)
}

// ServesAction reports whether ctx is the context of a transport serving the
// action with the given key.
func ServesAction(ctx context.Context, key string) bool {
	k, _ := ctx.Value(servedActionKey{}).(string)
	return k != "" && k == key
}
