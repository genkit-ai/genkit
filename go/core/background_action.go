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

package core

import (
	"context"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// StartOpFunc starts a background operation.
type StartOpFunc[In, Out any] = func(ctx context.Context, input In) (*Operation[Out], error)

// CheckOpFunc checks the status of a background operation.
type CheckOpFunc[Out any] = func(ctx context.Context, op *Operation[Out]) (*Operation[Out], error)

// CancelOpFunc cancels a background operation.
type CancelOpFunc[Out any] = func(ctx context.Context, op *Operation[Out]) (*Operation[Out], error)

// Operation represents a long-running operation started by a background action.
type Operation[Out any] struct {
	Action   string         `json:"action"`             // Key of the action that created this operation.
	ID       string         `json:"id"`                 // ID of the operation.
	Done     bool           `json:"done"`               // Whether the operation is complete.
	Output   Out            `json:"output,omitempty"`   // Result when done.
	Error    error          `json:"error,omitempty"`    // Error if the operation failed.
	Metadata map[string]any `json:"metadata,omitempty"` // Additional metadata.
}

// action is an unexported alias of [Action] used as the embedded field in
// [BackgroundAction]. Embedding through the alias promotes Action's methods
// while leaving the field itself unexported, so the start action stays an
// internal detail rather than public surface.
type action[In, Out, Stream any] = Action[In, Out, Stream]

// BackgroundAction is a background action that can be used to start, check, and cancel background operations.
// It promotes the start action's methods (Name, Desc, Run, RunJSON), so it can
// be used anywhere an [api.Action] is accepted.
type BackgroundAction[In, Out any] struct {
	*action[In, *Operation[Out], struct{}]

	check  *Action[*Operation[Out], *Operation[Out], struct{}] // Sub-action that checks the status of a background operation.
	cancel *Action[*Operation[Out], *Operation[Out], struct{}] // Sub-action that cancels a background operation.
}

// BackgroundActionDef is the previous name for [BackgroundAction].
//
// Deprecated: use [BackgroundAction].
type BackgroundActionDef[In, Out any] = BackgroundAction[In, Out]

// Start starts a background operation.
func (b *BackgroundAction[In, Out]) Start(ctx context.Context, input In) (*Operation[Out], error) {
	return b.Run(ctx, input, nil)
}

// Check checks the status of a background operation.
func (b *BackgroundAction[In, Out]) Check(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
	return b.check.Run(ctx, op, nil)
}

// Cancel attempts to cancel a background operation. It returns an error if the background action does not support cancellation.
func (b *BackgroundAction[In, Out]) Cancel(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
	if !b.SupportsCancel() {
		return nil, status.Errorf(status.ErrUnavailable, "model %q does not support canceling operations", b.Name())
	}

	return b.cancel.Run(ctx, op, nil)
}

// SupportsCancel returns whether the background action supports cancellation.
func (b *BackgroundAction[In, Out]) SupportsCancel() bool {
	return b.cancel != nil
}

// Register registers the model with the given registry.
func (b *BackgroundAction[In, Out]) Register(r api.Registry) {
	b.action.Register(r)
	b.check.Register(r)
	if b.cancel != nil {
		b.cancel.Register(r)
	}
}

// BackgroundActionOptions configures a background action created with
// [NewBackgroundActionOf]. The descriptor slots are [ActionOptions] minus the
// stream slot: the component actions are non-streaming, so a background
// action never advertises a stream schema. When [ActionOptions] gains a
// field, mirror it here (and copy it through in [NewBackgroundActionOf])
// unless it is stream-specific.
//
// The operation lifecycle functions beyond start also live here. They are
// fields rather than constructor arguments so that adding one, or relaxing
// which ones are required, is never a signature change; In is reserved for
// future lifecycle functions typed on the action's input.
type BackgroundActionOptions[In, Out any] struct {
	Description  string         // Human-readable description of the action. Metadata["description"] is used if empty; see [ActionOptions].
	Metadata     map[string]any // Arbitrary key-value data attached to the action descriptor.
	InputSchema  map[string]any // JSON schema for the start action's input. Inferred from In if nil.
	OutputSchema map[string]any // JSON schema for the start action's output. Inferred if nil.

	// Check checks the status of a background operation. It is required:
	// polling is currently the only way callers resolve a pending operation.
	Check CheckOpFunc[Out]
	// Cancel cancels a background operation. Optional: nil means the action
	// does not support cancellation and no cancel action is registered.
	Cancel CancelOpFunc[Out]
}

// checkOpName and cancelOpName return the registry names of a background
// action's lifecycle companions: the action's own name followed by the
// operation the companion performs, e.g. "googleai/veo-2/check". Both the
// constructor and [LookupBackgroundAction] build the names this way, and
// tooling that reaches a companion directly (the Dev UI polls and cancels
// background tasks over the reflection API) expects the same shape.
func checkOpName(name string) string  { return name + "/check" }
func cancelOpName(name string) string { return name + "/cancel" }

// NewBackgroundActionOf creates a new background action without
// registering it. Register it with [BackgroundAction.Register].
//
// startFn starts an operation; the rest of the operation lifecycle rides in
// opts: [BackgroundActionOptions.Check] is required and
// [BackgroundActionOptions.Cancel] is optional.
func NewBackgroundActionOf[In, Out any](
	atype api.ActionType,
	name string,
	opts *BackgroundActionOptions[In, Out],
	startFn StartOpFunc[In, Out],
) *BackgroundAction[In, Out] {
	if name == "" {
		panic("core.NewBackgroundActionOf: name is required")
	}
	if startFn == nil {
		panic("core.NewBackgroundActionOf: startFn is required")
	}
	if opts == nil || opts.Check == nil {
		panic("core.NewBackgroundActionOf: opts.Check is required")
	}
	// Bind the lifecycle functions now so mutating opts after construction
	// cannot change the action's behavior.
	checkFn, cancelFn := opts.Check, opts.Cancel

	key := api.KeyFromName(atype, name)

	// One inferred Operation schema serves the whole bundle: the start action
	// outputs an Operation, and the check and cancel actions consume and
	// produce one, so absent caller overrides every Operation-typed slot below
	// shares this map rather than re-running the inference.
	opSchema := base.SchemaMapFor[*Operation[Out]]()

	startOutputSchema := opts.OutputSchema
	if startOutputSchema == nil {
		startOutputSchema = opSchema
	}
	startAction := NewActionOf(atype, name, &ActionOptions{
		Description:  opts.Description,
		Metadata:     opts.Metadata,
		InputSchema:  opts.InputSchema,
		OutputSchema: startOutputSchema,
	},
		func(ctx context.Context, input In) (*Operation[Out], error) {
			op, err := startFn(ctx, input)
			if err != nil {
				return nil, err
			}
			op.Action = key
			return op, nil
		})

	// The schema slots in opts describe the start action; the check and
	// cancel actions keep the shared description and metadata but always use
	// the Operation schema.
	opOpts := &ActionOptions{
		Description:  opts.Description,
		Metadata:     opts.Metadata,
		InputSchema:  opSchema,
		OutputSchema: opSchema,
	}

	checkAction := NewActionOf(api.ActionTypeCheckOperation, checkOpName(name), opOpts,
		func(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
			updatedOp, err := checkFn(ctx, op)
			if err != nil {
				return nil, err
			}
			updatedOp.Action = key
			return updatedOp, nil
		})

	var cancelAction *Action[*Operation[Out], *Operation[Out], struct{}]
	if cancelFn != nil {
		cancelAction = NewActionOf(api.ActionTypeCancelOperation, cancelOpName(name), opOpts,
			func(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
				cancelledOp, err := cancelFn(ctx, op)
				if err != nil {
					return nil, err
				}
				cancelledOp.Action = key
				return cancelledOp, nil
			})
	}

	return &BackgroundAction[In, Out]{
		action: startAction,
		check:  checkAction,
		cancel: cancelAction,
	}
}

// NewBackgroundAction creates a new background action without registering it.
//
// Deprecated: Use [NewBackgroundActionOf], which takes the action type first
// and a [BackgroundActionOptions] struct covering the schema slots and the
// check and cancel functions.
func NewBackgroundAction[In, Out any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	startFn StartOpFunc[In, Out],
	checkFn CheckOpFunc[Out],
	cancelFn CancelOpFunc[Out],
) *BackgroundAction[In, Out] {
	if name == "" {
		panic("core.NewBackgroundAction: name is required")
	}
	if startFn == nil {
		panic("core.NewBackgroundAction: startFn is required")
	}
	if checkFn == nil {
		panic("core.NewBackgroundAction: checkFn is required")
	}
	return NewBackgroundActionOf(atype, name, &BackgroundActionOptions[In, Out]{
		Metadata: metadata,
		Check:    checkFn,
		Cancel:   cancelFn,
	}, startFn)
}

// LookupBackgroundAction looks up a background action by key (which includes the action type, provider, and name).
func LookupBackgroundAction[In, Out any](r api.Registry, key string) *BackgroundAction[In, Out] {
	atype, provider, id := api.ParseKey(key)
	name := api.NewName(provider, id)

	startAction := ResolveActionFor[In, *Operation[Out], struct{}](r, atype, name)
	if startAction == nil {
		return nil
	}

	checkAction := ResolveActionFor[*Operation[Out], *Operation[Out], struct{}](r, api.ActionTypeCheckOperation, checkOpName(name))
	if checkAction == nil {
		return nil
	}

	cancelAction := ResolveActionFor[*Operation[Out], *Operation[Out], struct{}](r, api.ActionTypeCancelOperation, cancelOpName(name))

	return &BackgroundAction[In, Out]{
		action: startAction,
		check:  checkAction,
		cancel: cancelAction,
	}
}

// CheckOperation checks the status of a background operation by looking up the action and calling its Check method.
func CheckOperation[In, Out any](ctx context.Context, r api.Registry, op *Operation[Out]) (*Operation[Out], error) {
	if op == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "core.CheckOperation: operation is nil")
	}

	if op.Action == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "core.CheckOperation: operation is missing original request information")
	}

	m := LookupBackgroundAction[In, Out](r, op.Action)
	if m == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "core.CheckOperation: failed to resolve background model %q from original request", op.Action)
	}

	return m.Check(ctx, op)
}
