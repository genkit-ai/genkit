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
	"encoding/json"
	"errors"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/registry"
)

// BackgroundModel represents a model that can run operations in the
// background. It is the type to accept as an argument and to look up by name;
// implementations are created with [NewBackgroundModelAction], or
// [genkit.DefineBackgroundModelAction] in an application.
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

// backgroundAction is an unexported alias of [core.BackgroundAction] used as
// the embedded field in [BackgroundModelAction]; see the action alias in
// generate.go for why, including why the promoted methods are redeclared
// below.
type backgroundAction[In, Out any] = core.BackgroundAction[In, Out]

// BackgroundModelAction is a background model backed by registry actions. It
// is the concrete type returned by [NewBackgroundModelAction]; return it from
// a plugin's Init for the framework to register.
//
// It implements [BackgroundModel] and [api.Action], so it can be passed
// anywhere either is accepted. The [api.Action] side is what lets a plugin
// resolver return the whole model as the resolved action (googlegenai's
// resolveAction does this), and the three component actions register
// together through [BackgroundModelAction.Register].
type BackgroundModelAction struct {
	backgroundAction[*ModelRequest, *ModelResponse]
}

// Pinned here so that breaking either interface fails the build at the type
// rather than during plugin resolution.
var (
	_ api.Action      = (*BackgroundModelAction)(nil)
	_ BackgroundModel = (*BackgroundModelAction)(nil)
)

// Name returns the registry name of the background model.
func (b *BackgroundModelAction) Name() string { return b.backgroundAction.Name() }

// Register registers the model's start, check, and cancel actions with r. The
// cancel action is registered only if the model supports cancellation. A
// plugin that returns the model from its Init does not need to call this.
func (b *BackgroundModelAction) Register(r api.Registry) { b.backgroundAction.Register(r) }

// Desc returns the start action's descriptor: its name, schemas, and metadata.
func (b *BackgroundModelAction) Desc() api.ActionDesc { return b.backgroundAction.Desc() }

// Start starts a background operation and returns it without waiting for
// completion. Poll it with [BackgroundModelAction.Check].
func (b *BackgroundModelAction) Start(ctx context.Context, req *ModelRequest) (*ModelOperation, error) {
	if b == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "BackgroundModel.Start: start called on a nil background model; check that all background models are defined")
	}
	return b.backgroundAction.Start(ctx, req)
}

// Check returns the current state of a background operation.
func (b *BackgroundModelAction) Check(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
	if b == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "BackgroundModel.Check: check called on a nil background model; check that all background models are defined")
	}
	return b.backgroundAction.Check(ctx, op)
}

// Cancel cancels a running background operation. It fails with UNAVAILABLE if
// the model does not support cancellation; see
// [BackgroundModelAction.SupportsCancel].
func (b *BackgroundModelAction) Cancel(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
	if b == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "BackgroundModel.Cancel: cancel called on a nil background model; check that all background models are defined")
	}
	return b.backgroundAction.Cancel(ctx, op)
}

// SupportsCancel reports whether the model was defined with a cancel
// function.
func (b *BackgroundModelAction) SupportsCancel() bool { return b.backgroundAction.SupportsCancel() }

// RunJSON starts an operation from a JSON-encoded [ModelRequest] and returns
// the JSON-encoded operation. The framework uses it to serve reflection and
// registry-driven calls; prefer [BackgroundModelAction.Start].
func (b *BackgroundModelAction) RunJSON(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	if b == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "BackgroundModel.RunJSON: model called on a nil background model; check that all background models are defined")
	}
	return b.backgroundAction.RunJSON(ctx, input, cb)
}

// RunJSONWithTelemetry is [BackgroundModelAction.RunJSON] with the run's
// telemetry returned alongside the output.
func (b *BackgroundModelAction) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	if b == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "BackgroundModel.RunJSONWithTelemetry: model called on a nil background model; check that all background models are defined")
	}
	return b.backgroundAction.RunJSONWithTelemetry(ctx, input, cb)
}

// ModelOperation is a background operation for a model.
type ModelOperation = core.Operation[*ModelResponse]

// StartModelOpFunc starts a background model operation.
type StartModelOpFunc = func(ctx context.Context, req *ModelRequest) (*ModelOperation, error)

// BackgroundModelActionFunc is a [StartModelOpFunc] that additionally
// receives the request's typed Config: the framework deserializes the
// request's raw config into it before calling the function (see
// [NewBackgroundModelAction]).
type BackgroundModelActionFunc[Config any] = func(ctx context.Context, req *ModelRequest, config Config) (*ModelOperation, error)

// CheckModelOpFunc checks the status of a background model operation.
type CheckModelOpFunc = func(ctx context.Context, op *ModelOperation) (*ModelOperation, error)

// CancelModelOpFunc cancels a background model operation.
type CancelModelOpFunc = func(ctx context.Context, op *ModelOperation) (*ModelOperation, error)

// BackgroundModelOptions configures a background model created with
// [NewBackgroundModelAction]. It extends [ModelOptions] with the operation
// lifecycle hooks a background model needs; the required start and check
// functions are constructor arguments.
type BackgroundModelOptions struct {
	ModelOptions

	// Cancel cancels a running operation. Optional: nil means the model does
	// not support canceling operations.
	Cancel CancelModelOpFunc

	// Metadata is arbitrary key-value data attached to the action descriptor.
	// It is merged over [ModelOptions.Metadata]; this field wins on key
	// conflicts.
	Metadata map[string]any
}

// LookupBackgroundModel looks up a registered [BackgroundModel] by name.
// It returns nil if the background model was not found.
func LookupBackgroundModel(r api.Registry, name string) BackgroundModel {
	key := api.KeyFromName(api.ActionTypeBackgroundModel, name)
	action := core.LookupBackgroundAction[*ModelRequest, *ModelResponse](r, key)
	if action == nil {
		return nil
	}
	return &BackgroundModelAction{*action}
}

// NewBackgroundModelAction creates an unregistered [BackgroundModelAction]:
// return it from a plugin's Init for the framework to register, or call
// [BackgroundModelAction.Register] directly. Applications should define
// background models with [genkit.DefineBackgroundModelAction].
//
// Config is the model's typed configuration; it is usually inferred from
// startFn's signature. See [NewModelAction] for how the request's config
// is deserialized.
func NewBackgroundModelAction[Config any](
	name string,
	opts *BackgroundModelOptions,
	startFn BackgroundModelActionFunc[Config],
	checkFn CheckModelOpFunc,
) *BackgroundModelAction {
	if name == "" {
		panic("ai.NewBackgroundModelAction: name is required")
	}
	if startFn == nil {
		panic("ai.NewBackgroundModelAction: startFn is required")
	}
	if checkFn == nil {
		panic("ai.NewBackgroundModelAction: checkFn is required")
	}

	o := BackgroundModelOptions{}
	if opts != nil {
		o = *opts
	}
	labelExplicit := o.Label != ""
	if !labelExplicit {
		o.Label = name
	}
	o.Supports = cloneModelSupports(o.Supports)

	configSchema, inputSchema := modelConfigSchemas[Config](o.ConfigSchema, o.Versions)

	// The top-level Metadata wins over the embedded ModelOptions.Metadata on
	// key conflicts.
	metadata := modelActionMetadata(api.ActionTypeBackgroundModel, &o.ModelOptions, configSchema, o.ModelOptions.Metadata, o.Metadata)

	typedStartFn := func(ctx context.Context, req *ModelRequest) (*ModelOperation, error) {
		// req.Config was normalized to the exact Config type by
		// normalizeConfig below, so this hits the fast path.
		cfg, err := resolveConfig[Config](req.Config)
		if err != nil {
			return nil, err
		}
		return startFn(ctx, req, cfg)
	}

	mopts := &o.ModelOptions

	// normalizeConfig runs outermost so that the built-in wrappers and the
	// start function all see the typed, converted config on the request.
	fn := core.ChainMiddleware(
		normalizeConfig[Config](name, o.Versions),
		simulateSystemPrompt(mopts, nil),
		augmentWithContext(mopts, nil),
		validateSupport(name, mopts),
	)(backgroundModelToModelFn(typedStartFn))

	wrappedFn := func(ctx context.Context, req *ModelRequest) (*ModelOperation, error) {
		resp, err := fn(ctx, req, nil)
		if err != nil {
			return nil, err
		}

		return modelOpFromResponse(resp)
	}

	// The label doubles as the description on all three component actions,
	// matching the JS background model surface. A label that was only
	// defaulted from the name yields to an explicit caller
	// Metadata["description"]: leaving Description empty lets core's
	// metadata fallback apply it.
	description := o.Label
	if !labelExplicit {
		if _, ok := metadata["description"].(string); ok {
			description = ""
		}
	}
	return &BackgroundModelAction{*core.NewBackgroundActionOf(api.ActionTypeBackgroundModel, name, &core.BackgroundActionOptions[*ModelRequest, *ModelResponse]{
		Description: description,
		Metadata:    metadata,
		InputSchema: inputSchema,
		Check:       checkFn,
		Cancel:      o.Cancel,
	}, wrappedFn)}
}

// NewBackgroundModel defines a new model that runs in the background.
//
// Deprecated: Use [NewBackgroundModelAction], which passes the request's
// config to startFn as a typed value instead of leaving it type-erased on the
// request.
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
	return NewBackgroundModelAction(name, opts, func(ctx context.Context, req *ModelRequest, _ any) (*ModelOperation, error) {
		return startFn(ctx, req)
	}, checkFn)
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
		return nil, status.Errorf(status.ErrFailedPrecondition, "background model did not return an operation")
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
			return nil, status.Errorf(status.ErrInternal, "operation output is not a model response")
		}
	}

	return op, nil
}
