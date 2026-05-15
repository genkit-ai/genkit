// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
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
	"context"
	"errors"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

// BackgroundModel represents a model that can run operations in the background.
type BackgroundModel interface {
	// Name returns the registry name of the background model.
	Name() string
	// Register registers the model with the given registry.
	Register(r api.Registry)
	// Start starts a background operation.
	Start(ctx context.Context, req *ModelRequest) (*ModelOperation, error)
	// Check checks the status of a background operation.
	Check(ctx context.Context, op *ModelOperation) (*ModelOperation, error)
	// Cancel cancels a background operation.
	Cancel(ctx context.Context, op *ModelOperation) (*ModelOperation, error)
	// SupportsCancel returns whether the background action supports cancellation.
	SupportsCancel() bool
}

// backgroundModel is the concrete implementation of BackgroundModel interface.
type backgroundModel struct {
	core.BackgroundActionDef[*ModelRequest, *ModelResponse]
}

// ModelOperation is a background operation for a model.
type ModelOperation = core.Operation[*ModelResponse]

// StartModelOpFunc starts a background model operation.
type StartModelOpFunc = func(ctx context.Context, req *ModelRequest) (*ModelOperation, error)

// CheckModelOpFunc checks the status of a background model operation.
type CheckModelOpFunc = func(ctx context.Context, op *ModelOperation) (*ModelOperation, error)

// CancelModelOpFunc cancels a background model operation.
type CancelModelOpFunc = func(ctx context.Context, op *ModelOperation) (*ModelOperation, error)

// BackgroundModelOptions holds configuration for defining a background model
type BackgroundModelOptions struct {
	ModelOptions
	Cancel   CancelModelOpFunc // Function that cancels a background model operation.
	Metadata map[string]any    // Additional metadata.
}

// LookupBackgroundModel looks up a BackgroundAction registered by [DefineBackgroundModel].
// It returns nil if the background model was not found.
func LookupBackgroundModel(r api.Registry, name string) BackgroundModel {
	key := api.KeyFromName(api.ActionTypeBackgroundModel, name)
	action := core.LookupBackgroundAction[*ModelRequest, *ModelResponse](r, key)
	if action == nil {
		return nil
	}
	return &backgroundModel{*action}
}

// NewBackgroundModel defines a new model that runs in the background.
func NewBackgroundModel(name string, opts *BackgroundModelOptions, startFn StartModelOpFunc, checkFn CheckModelOpFunc) BackgroundModel {
	if name == "" {
		panic("ai.NewBackgroundModel: name is required")
	}
	if startFn == nil {
		panic("ai.NewBackgroundModel: startFn is required")
	}
	if checkFn == nil {
		panic("ai.NewBackgroundModel: checkFn is required")
	}

	if opts == nil {
		opts = &BackgroundModelOptions{}
	}
	if opts.Label == "" {
		opts.Label = name
	}
	if opts.Supports == nil {
		opts.Supports = &ModelSupports{}
	}

	metadata := map[string]any{
		"type": api.ActionTypeBackgroundModel,
		"model": map[string]any{
			"label": opts.Label,
			"supports": map[string]any{
				"media":       opts.Supports.Media,
				"context":     opts.Supports.Context,
				"multiturn":   opts.Supports.Multiturn,
				"systemRole":  opts.Supports.SystemRole,
				"tools":       opts.Supports.Tools,
				"toolChoice":  opts.Supports.ToolChoice,
				"constrained": opts.Supports.Constrained,
				"output":      opts.Supports.Output,
				"contentType": opts.Supports.ContentType,
				"longRunning": opts.Supports.LongRunning,
			},
			"versions":      opts.Versions,
			"stage":         opts.Stage,
			"customOptions": opts.ConfigSchema,
		},
	}

	inputSchema := core.InferSchemaMap(ModelRequest{})
	if inputSchema != nil && opts.ConfigSchema != nil {
		if props, ok := inputSchema["properties"].(map[string]any); ok {
			props["config"] = opts.ConfigSchema
		}
	}

	mws := []ModelMiddleware{
		simulateSystemPrompt(&opts.ModelOptions, nil),
		augmentWithContext(&opts.ModelOptions, nil),
		validateSupport(name, &opts.ModelOptions),
	}
	fn := core.ChainMiddleware(mws...)(backgroundModelToModelFn(startFn))

	wrappedFn := func(ctx context.Context, req *ModelRequest) (*ModelOperation, error) {
		resp, err := fn(ctx, req, nil)
		if err != nil {
			return nil, err
		}

		return modelOpFromResponse(resp)
	}

	return &backgroundModel{*core.NewBackgroundAction(name, api.ActionTypeBackgroundModel, metadata, wrappedFn, checkFn, opts.Cancel)}
}

// DefineBackgroundModel defines and registers a new model that runs in the background.
func DefineBackgroundModel(r *registry.Registry, name string, opts *BackgroundModelOptions, fn StartModelOpFunc, checkFn CheckModelOpFunc) BackgroundModel {
	m := NewBackgroundModel(name, opts, fn, checkFn)
	m.Register(r)
	return m
}

// GenerateOperation generates a model response as a long-running operation based on the provided options.
func GenerateOperation(ctx context.Context, r *registry.Registry, opts ...GenerateOption) (*ModelOperation, error) {
	resp, err := Generate(ctx, r, opts...)
	if err != nil {
		return nil, err
	}

	return modelOpFromResponse(resp)
}

// CheckModelOperation checks the status of a background model operation by looking up the model and calling its Check method.
func CheckModelOperation(ctx context.Context, r api.Registry, op *ModelOperation) (*ModelOperation, error) {
	return core.CheckOperation[*ModelRequest](ctx, r, op)
}

// backgroundModelToModelFn wraps a background model start function into a [ModelFunc] for middleware compatibility.
func backgroundModelToModelFn(startFn StartModelOpFunc) ModelFunc {
	return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		op, err := startFn(ctx, req)
		if err != nil {
			return nil, err
		}

		var opError *OperationError
		if op.Error != nil {
			opError = &OperationError{Message: op.Error.Error()}
		}

		metadata := op.Metadata
		if metadata == nil {
			metadata = make(map[string]any)
		}

		return &ModelResponse{
			Operation: &Operation{
				Action:   op.Action,
				Id:       op.ID,
				Done:     op.Done,
				Output:   op.Output,
				Error:    opError,
				Metadata: metadata,
			},
			Request: req,
		}, nil
	}
}

// modelOpFromResponse extracts a [ModelOperation] from a [ModelResponse].
func modelOpFromResponse(resp *ModelResponse) (*ModelOperation, error) {
	if resp.Operation == nil {
		return nil, core.NewError(core.FAILED_PRECONDITION, "background model did not return an operation")
	}

	op := &ModelOperation{
		Action:   resp.Operation.Action,
		ID:       resp.Operation.Id,
		Done:     resp.Operation.Done,
		Metadata: resp.Operation.Metadata,
	}

	if resp.Operation.Error != nil {
		op.Error = errors.New(resp.Operation.Error.Message)
	}

	if resp.Operation.Output != nil {
		if modelResp, ok := resp.Operation.Output.(*ModelResponse); ok {
			op.Output = modelResp
		} else {
			return nil, core.NewError(core.INTERNAL, "operation output is not a model response")
		}
	}

	return op, nil
}
