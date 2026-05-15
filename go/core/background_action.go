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

// BackgroundActionDef is a background action that can be used to start, check, and cancel background operations.
//
// For internal use only.
type BackgroundActionDef[In, Out any] struct {
	*ActionDef[In, *Operation[Out], struct{}]

	check  *ActionDef[*Operation[Out], *Operation[Out], struct{}] // Sub-action that checks the status of a background operation.
	cancel *ActionDef[*Operation[Out], *Operation[Out], struct{}] // Sub-action that cancels a background operation.
}

// Start starts a background operation.
func (b *BackgroundActionDef[In, Out]) Start(ctx context.Context, input In) (*Operation[Out], error) {
	return b.Run(ctx, input, nil)
}

// Check checks the status of a background operation.
func (b *BackgroundActionDef[In, Out]) Check(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
	return b.check.Run(ctx, op, nil)
}

// Cancel attempts to cancel a background operation. It returns an error if the background action does not support cancellation.
func (b *BackgroundActionDef[In, Out]) Cancel(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
	if !b.SupportsCancel() {
		return nil, NewError(UNAVAILABLE, "model %q does not support canceling operations", b.Name())
	}

	return b.cancel.Run(ctx, op, nil)
}

// SupportsCancel returns whether the background action supports cancellation.
func (b *BackgroundActionDef[In, Out]) SupportsCancel() bool {
	return b.cancel != nil
}

// Register registers the model with the given registry.
func (b *BackgroundActionDef[In, Out]) Register(r api.Registry) {
	b.ActionDef.Register(r)
	b.check.Register(r)
	if b.cancel != nil {
		b.cancel.Register(r)
	}
}

// DefineBackgroundAction creates and registers a background action with three component actions
func DefineBackgroundAction[In, Out any](
	r api.Registry,
	name string,
	atype api.ActionType,
	metadata map[string]any,
	startFn StartOpFunc[In, Out],
	checkFn CheckOpFunc[Out],
	cancelFn CancelOpFunc[Out],
) *BackgroundActionDef[In, Out] {
	a := NewBackgroundAction(name, atype, metadata, startFn, checkFn, cancelFn)
	a.Register(r)
	return a
}

// NewBackgroundAction creates a new background action without registering it.
func NewBackgroundAction[In, Out any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	startFn StartOpFunc[In, Out],
	checkFn CheckOpFunc[Out],
	cancelFn CancelOpFunc[Out],
) *BackgroundActionDef[In, Out] {
	if name == "" {
		panic("core.NewBackgroundAction: name is required")
	}
	if startFn == nil {
		panic("core.NewBackgroundAction: startFn is required")
	}
	if checkFn == nil {
		panic("core.NewBackgroundAction: checkFn is required")
	}

	key := api.KeyFromName(atype, name)

	startAction := NewAction(name, atype, metadata, nil,
		func(ctx context.Context, input In) (*Operation[Out], error) {
			op, err := startFn(ctx, input)
			if err != nil {
				return nil, err
			}
			op.Action = key
			return op, nil
		})

	checkAction := NewAction(name, api.ActionTypeCheckOperation, metadata, nil,
		func(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
			updatedOp, err := checkFn(ctx, op)
			if err != nil {
				return nil, err
			}
			updatedOp.Action = key
			return updatedOp, nil
		})

	var cancelAction *ActionDef[*Operation[Out], *Operation[Out], struct{}]
	if cancelFn != nil {
		cancelAction = NewAction(name, api.ActionTypeCancelOperation, metadata, nil,
			func(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
				cancelledOp, err := cancelFn(ctx, op)
				if err != nil {
					return nil, err
				}
				cancelledOp.Action = key
				return cancelledOp, nil
			})
	}

	return &BackgroundActionDef[In, Out]{
		ActionDef: startAction,
		check:     checkAction,
		cancel:    cancelAction,
	}
}

// LookupBackgroundAction looks up a background action by key (which includes the action type, provider, and name).
func LookupBackgroundAction[In, Out any](r api.Registry, key string) *BackgroundActionDef[In, Out] {
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

	return &BackgroundActionDef[In, Out]{
		ActionDef: startAction,
		check:     checkAction,
		cancel:    cancelAction,
	}
}

// CheckOperation checks the status of a background operation by looking up the action and calling its Check method.
func CheckOperation[In, Out any](ctx context.Context, r api.Registry, op *Operation[Out]) (*Operation[Out], error) {
	if op == nil {
		return nil, NewError(INVALID_ARGUMENT, "core.CheckOperation: operation is nil")
	}

	if op.Action == "" {
		return nil, NewError(INVALID_ARGUMENT, "core.CheckOperation: operation is missing original request information")
	}

	m := LookupBackgroundAction[In, Out](r, op.Action)
	if m == nil {
		return nil, NewError(INVALID_ARGUMENT, "core.CheckOperation: failed to resolve background model %q from original request", op.Action)
	}

	return m.Check(ctx, op)
}
