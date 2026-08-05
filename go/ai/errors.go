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

package ai

import "github.com/firebase/genkit/go/core/status"

// Failure modes generation reports. Match them with errors.Is rather than by
// inspecting message text:
//
//	if errors.Is(err, ai.ErrMaxTurnsExceeded) { ... }
//
// Each also matches the base sentinel it derives from, so
// errors.Is(err, status.ErrNotFound) still catches a missing model or tool.
var (
	// ErrModelNotFound means the named model is not registered. Usually the
	// providing plugin is missing from genkit.Init.
	ErrModelNotFound = status.ErrNotFound.Subtype("model not found")

	// ErrToolNotFound means the named tool is not registered, either on the
	// request or in the registry the model's tool call resolved against.
	ErrToolNotFound = status.ErrNotFound.Subtype("tool not found")

	// ErrMaxTurnsExceeded means the tool-calling loop hit its turn limit before
	// the model produced a final response. Raise the limit with WithMaxTurns, or
	// look for a tool the model keeps retrying.
	ErrMaxTurnsExceeded = status.ErrAborted.Subtype("max turns exceeded")

	// ErrToolFailed means a tool returned an error or produced output that does
	// not match its declared schema. The tool's own error is wrapped, so
	// errors.Is and errors.As still reach it; the status is INTERNAL because a
	// tool's failure is not a failure of the caller's request.
	ErrToolFailed = status.ErrInternal.Subtype("tool failed")

	// ErrUnsupportedByModel means the request used a capability the model does
	// not advertise (media, tools, tool choice, a system role, ...).
	ErrUnsupportedByModel = status.ErrInvalidArgument.Subtype("unsupported by model")

	// ErrInvalidPart means a Part is malformed for the operation at hand: the
	// wrong kind, missing a required field, or carrying a field its kind does
	// not allow.
	ErrInvalidPart = status.ErrInvalidArgument.Subtype("invalid part")

	// ErrUnresolvedToolRequest means a resumed generation left an interrupted
	// tool request without a Respond or Restart directive.
	ErrUnresolvedToolRequest = status.ErrInvalidArgument.Subtype("unresolved tool request")
)
