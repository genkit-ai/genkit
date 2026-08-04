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

// BackgroundAction is a background action that can be used to start, check, and cancel background operations.
type BackgroundAction[In, Out any] struct {
	*Action[In, *Operation[Out], struct{}]

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
	b.Action.Register(r)
	b.check.Register(r)
	if b.cancel != nil {
		b.cancel.Register(r)
	}
}

// BackgroundActionOptions configures the optional attributes of a background
// action. It is [ActionOptions] minus the stream slot: the component actions
// are non-streaming, so a background action never advertises a stream schema.
// A nil options value is valid: schemas are inferred from the action's type
// parameters. When [ActionOptions] gains a field, mirror it here (and copy it
// through in [NewBackgroundActionOf]) unless it is stream-specific.
type BackgroundActionOptions struct {
	Description  string         // Human-readable description of the action. Metadata["description"] is used if empty.
	Metadata     map[string]any // Arbitrary key-value data attached to the action descriptor.
	InputSchema  map[string]any // JSON schema for the start action's input. Inferred from In if nil.
	OutputSchema map[string]any // JSON schema for the start action's output. Inferred if nil.
}

// NewBackgroundActionOf creates a new background action without
// registering it. Register it with [BackgroundAction.Register].
//
// cancelFn is optional; nil means the action does not support cancellation
// and no cancel action is registered.
func NewBackgroundActionOf[In, Out any](
	atype api.ActionType,
	name string,
	opts *BackgroundActionOptions,
	startFn StartOpFunc[In, Out],
	checkFn CheckOpFunc[Out],
	cancelFn CancelOpFunc[Out],
) *BackgroundAction[In, Out] {
	if name == "" {
		panic("core.NewBackgroundActionOf: name is required")
	}
	if startFn == nil {
		panic("core.NewBackgroundActionOf: startFn is required")
	}
	if checkFn == nil {
		panic("core.NewBackgroundActionOf: checkFn is required")
	}

	if opts == nil {
		opts = &BackgroundActionOptions{}
	}

	key := api.KeyFromName(atype, name)

	startAction := NewActionOf(atype, name, &ActionOptions{
		Description:  opts.Description,
		Metadata:     opts.Metadata,
		InputSchema:  opts.InputSchema,
		OutputSchema: opts.OutputSchema,
	},
		func(ctx context.Context, input In) (*Operation[Out], error) {
			op, err := startFn(ctx, input)
			if err != nil {
				return nil, err
			}
			op.Action = key
			return op, nil
		})

	// The schema slots in opts describe the start action's In/Out types; the
	// check and cancel actions operate on Operation values, so they keep the
	// shared description and metadata but get their own schema, inferred once
	// here rather than per slot per sub-action.
	opSchema := base.SchemaMapFor[*Operation[Out]]()
	opOpts := &ActionOptions{
		Description:  opts.Description,
		Metadata:     opts.Metadata,
		InputSchema:  opSchema,
		OutputSchema: opSchema,
	}

	checkAction := NewActionOf(api.ActionTypeCheckOperation, name, opOpts,
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
		cancelAction = NewActionOf(api.ActionTypeCancelOperation, name, opOpts,
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
		Action: startAction,
		check:  checkAction,
		cancel: cancelAction,
	}
}

// NewBackgroundAction creates a new background action without registering it.
//
// Deprecated: Use [NewBackgroundActionOf], which takes the action type first
// and a [BackgroundActionOptions] struct covering the schema slots.
func NewBackgroundAction[In, Out any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	startFn StartOpFunc[In, Out],
	checkFn CheckOpFunc[Out],
	cancelFn CancelOpFunc[Out],
) *BackgroundAction[In, Out] {
	return NewBackgroundActionOf(atype, name, &BackgroundActionOptions{
		Metadata: metadata,
	}, startFn, checkFn, cancelFn)
}

// LookupBackgroundAction looks up a background action by key (which includes the action type, provider, and name).
func LookupBackgroundAction[In, Out any](r api.Registry, key string) *BackgroundAction[In, Out] {
	atype, provider, id := api.ParseKey(key)
	name := api.NewName(provider, id)

	startAction := ResolveActionFor[In, *Operation[Out], struct{}](r, atype, name)
	if startAction == nil {
		return nil
	}

	checkAction := ResolveActionFor[*Operation[Out], *Operation[Out], struct{}](r, api.ActionTypeCheckOperation, name)
	if checkAction == nil {
		return nil
	}

	cancelAction := ResolveActionFor[*Operation[Out], *Operation[Out], struct{}](r, api.ActionTypeCancelOperation, name)

	return &BackgroundAction[In, Out]{
		Action: startAction,
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
