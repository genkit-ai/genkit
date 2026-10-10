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
	"encoding/json"

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
// and the payload of a streaming one. It carries no details: they used to hold
// the full err.Error() text, which put internal failure detail on the wire.
type Error struct {
	Status  status.Name `json:"status"`
	Message string      `json:"message"`
}

// SSEDataPrefix starts every server-sent event line the handler writes.
const SSEDataPrefix = "data: "
