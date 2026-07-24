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
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/registry"
)

// backgroundAction is an unexported alias of [core.BackgroundAction] used as
// the embedded field in [BackgroundModel], mirroring the `action` alias.
type backgroundAction[In, Out any] = core.BackgroundAction[In, Out]

// BackgroundModel is a model that runs operations in the background, backed
// by a registry background action. Create one with [DefineBackgroundModel] or
// [NewBackgroundModel], or fetch a registered one with
// [LookupBackgroundModel]. Use Start to begin an operation and Check/Cancel
// to manage it.
type BackgroundModel struct {
	backgroundAction[*ModelRequest, *ModelResponse]
}

var (
	_ api.Action = (*BackgroundModel)(nil)
	_ Named      = (*BackgroundModel)(nil)
)

// ModelOperation is a background operation for a model.
type ModelOperation = core.Operation[*ModelResponse]

// StartModelOpFunc starts a background model operation.
// Config is the model's typed configuration: the framework deserializes the
// request's raw config into it before calling the function (see [DefineBackgroundModel]).
type StartModelOpFunc[Config any] = func(ctx context.Context, req *ModelRequest, config Config) (*ModelOperation, error)

// rawStartModelOpFunc is the untyped start function shape used internally:
// the request's config still rides along as `any` inside the [ModelRequest].
type rawStartModelOpFunc = func(ctx context.Context, req *ModelRequest) (*ModelOperation, error)

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
func LookupBackgroundModel(r api.Registry, name string) *BackgroundModel {
	key := api.KeyFromName(api.ActionTypeBackgroundModel, name)
	action := core.LookupBackgroundAction[*ModelRequest, *ModelResponse](r, key)
	if action == nil {
		return nil
	}
	return &BackgroundModel{*action}
}

// NewBackgroundModel creates a new unregistered [BackgroundModel]. Register
// it with [BackgroundModel.Register] or use [DefineBackgroundModel] to define
// and register in one step.
//
// Config is the model's typed configuration; it is usually inferred from
// startFn's signature. See [NewModel] for how the request's config is
// deserialized.
func NewBackgroundModel[Config any](name string, opts *BackgroundModelOptions, startFn StartModelOpFunc[Config], checkFn CheckModelOpFunc) *BackgroundModel {
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

	configSchema, inputSchema := actionConfigSchemas[Config](opts.ConfigSchema, ModelRequest{}, "config")

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
			"customOptions": configSchema,
		},
	}

	typedStartFn := func(ctx context.Context, req *ModelRequest) (*ModelOperation, error) {
		// req.Config was normalized to the exact Config type by
		// normalizeConfig below, so this hits the fast path.
		cfg, err := resolveConfig[Config](req.Config)
		if err != nil {
			return nil, err
		}
		return startFn(ctx, req, cfg)
	}

	// normalizeConfig runs outermost so that the built-in wrappers and the
	// start function all see the typed, converted config on the request.
	fn := core.ChainMiddleware(
		normalizeConfig[Config](name, opts.Versions),
		simulateSystemPrompt(&opts.ModelOptions, nil),
		augmentWithContext(&opts.ModelOptions, nil),
		validateSupport(name, &opts.ModelOptions),
	)(backgroundModelToModelFn(typedStartFn))

	wrappedFn := func(ctx context.Context, req *ModelRequest) (*ModelOperation, error) {
		resp, err := fn(ctx, req, nil)
		if err != nil {
			return nil, err
		}

		return modelOpFromResponse(resp)
	}

	return &BackgroundModel{*core.NewBackgroundAction(api.ActionTypeBackgroundModel, name, &core.ActionOptions{
		Metadata:    metadata,
		InputSchema: inputSchema,
	}, wrappedFn, checkFn, opts.Cancel)}
}

// DefineBackgroundModel defines and registers a new model that runs in the background.
//
// Config is the model's typed configuration; it is usually inferred from fn's
// signature. See [NewModel] for how the request's config is deserialized.
func DefineBackgroundModel[Config any](r *registry.Registry, name string, opts *BackgroundModelOptions, fn StartModelOpFunc[Config], checkFn CheckModelOpFunc) *BackgroundModel {
	m := NewBackgroundModel(name, opts, fn, checkFn)
	m.Register(r)
	return m
}

// Name returns the registry name of the background model, or the empty
// string if the model is nil (e.g. from a failed lookup).
func (m *BackgroundModel) Name() string {
	if m == nil {
		return ""
	}
	return m.backgroundAction.Name()
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

// backgroundModelToModelFn wraps a background model start function into a [rawModelFunc] for middleware compatibility.
func backgroundModelToModelFn(startFn rawStartModelOpFunc) rawModelFunc {
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
				ID:       op.ID,
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
		return nil, status.Errorf(status.ErrFailedPrecondition, "background model did not return an operation")
	}

	op := &ModelOperation{
		Action:   resp.Operation.Action,
		ID:       resp.Operation.ID,
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
			return nil, status.Errorf(status.ErrInternal, "operation output is not a model response")
		}
	}

	return op, nil
}
