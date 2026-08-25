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

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// RetrieverFunc is the function type for retriever implementations.
type RetrieverFunc = func(context.Context, *RetrieverRequest) (*RetrieverResponse, error)

// RetrieverActionFunc is a [RetrieverFunc] that additionally receives the
// request's typed Config: the framework deserializes the request's raw
// options into it before calling the function (see [NewRetrieverAction]).
type RetrieverActionFunc[Config any] = func(context.Context, *RetrieverRequest, Config) (*RetrieverResponse, error)

// Retriever represents a document retriever. It is the type to accept as an
// argument and to look up by name; implementations are created with
// [NewRetrieverAction], or [genkit.DefineRetrieverAction] in an application.
type Retriever interface {
	// Name returns the name of the retriever.
	Name() string
	// Retrieve retrieves the documents.
	Retrieve(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error)
	// Register registers the retriever with the given registry.
	Register(r api.Registry)
}

// RetrieverAction is a retriever backed by a registry action. It is the
// concrete type returned by [NewRetrieverAction]; pass it to [WithRetriever] to
// retrieve with it, or return it from a plugin's Init for the framework to
// register.
//
// It implements [Retriever] and [api.Action], so it can be passed anywhere
// either is accepted. It also promotes [core.Action.Run], the typed
// equivalent of [RetrieverAction.Retrieve].
type RetrieverAction struct {
	action[*RetrieverRequest, *RetrieverResponse, struct{}]
}

// Pinned here so that breaking either interface fails the build at the type
// rather than at a call site.
var (
	_ api.Action = (*RetrieverAction)(nil)
	_ Retriever  = (*RetrieverAction)(nil)
)

// Name returns the registry name of the retriever.
func (r *RetrieverAction) Name() string { return r.action.Name() }

// Register registers the retriever with reg, making it available to lookups
// and to the Dev UI. A plugin that returns the retriever from its Init does
// not need to call this.
func (r *RetrieverAction) Register(reg api.Registry) { r.action.Register(reg) }

// Desc returns the retriever's action descriptor: its name, schemas, and
// metadata.
func (r *RetrieverAction) Desc() api.ActionDesc { return r.action.Desc() }

// RunJSON runs the retriever on a JSON-encoded [RetrieverRequest] and returns
// a JSON-encoded [RetrieverResponse]. The framework uses it to serve
// reflection and registry-driven calls; prefer [RetrieverAction.Retrieve].
func (r *RetrieverAction) RunJSON(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	if r == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Retriever.RunJSON: retriever called on a nil retriever; check that all retrievers are defined")
	}
	return r.action.RunJSON(ctx, input, cb)
}

// RunJSONWithTelemetry is [RetrieverAction.RunJSON] with the run's telemetry
// returned alongside the output.
func (r *RetrieverAction) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	if r == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Retriever.RunJSONWithTelemetry: retriever called on a nil retriever; check that all retrievers are defined")
	}
	return r.action.RunJSONWithTelemetry(ctx, input, cb)
}

// RetrieverArg is the interface for retriever arguments. It can either be the retriever action itself or a reference to be looked up.
type RetrieverArg interface {
	Name() string
}

// RetrieverRef is a struct to hold retriever name and configuration.
type RetrieverRef struct {
	name   string
	config any
}

// NewRetrieverRef creates a new RetrieverRef with the given name and configuration.
func NewRetrieverRef(name string, config any) RetrieverRef {
	return RetrieverRef{name: name, config: config}
}

// Name returns the name of the retriever.
func (r RetrieverRef) Name() string {
	return r.name
}

// Config returns the configuration to use by default for this retriever.
func (r RetrieverRef) Config() any {
	return r.config
}

// RetrieverSupports defines the supported capabilities of the retriever.
type RetrieverSupports struct {
	// Media indicates whether the retriever supports media content.
	Media bool `json:"media,omitempty"`
}

// RetrieverOptions represents the configuration options for a retriever.
type RetrieverOptions struct {
	// ConfigSchema is the JSON schema for the retriever's config.
	ConfigSchema map[string]any `json:"configSchema,omitempty"`
	// Label is a user-friendly name for the retriever.
	Label string `json:"label,omitempty"`
	// Supports defines the capabilities of the retriever, such as media support.
	Supports *RetrieverSupports `json:"supports,omitempty"`
	// Metadata is arbitrary key-value data attached to the action descriptor.
	Metadata map[string]any `json:"-"`
}

// NewRetrieverAction creates an unregistered [RetrieverAction]: return it from
// a plugin's Init for the framework to register, or call
// [RetrieverAction.Register] directly. Applications should define retrievers
// with [genkit.DefineRetrieverAction].
//
// Config is the retriever's typed configuration; it is usually inferred from
// fn's signature. The framework deserializes the request's raw options into
// Config before calling fn: the exact Config type (or a pointer to it) and
// map[string]any (from the Dev UI and other JSON callers) are accepted, and
// mismatched types are rejected. The request's [RetrieverRequest.Options] is
// normalized to the converted value, so it always matches the typed
// parameter. The config's JSON schema is inferred from Config unless
// [RetrieverOptions.ConfigSchema] overrides it.
func NewRetrieverAction[Config any](
	name string,
	opts *RetrieverOptions,
	fn RetrieverActionFunc[Config],
) *RetrieverAction {
	if name == "" {
		panic("ai.NewRetrieverAction: name is required")
	}

	o := RetrieverOptions{}
	if opts != nil {
		o = *opts
	}
	if o.Label == "" {
		o.Label = name
	}
	o.Supports = cloneRetrieverSupports(o.Supports)

	configSchema, inputSchema := actionConfigSchemas[Config](o.ConfigSchema, RetrieverRequest{}, "options")
	metadata := actionMetadata(api.ActionTypeRetriever, map[string]any{
		"info": map[string]any{
			"label":    o.Label,
			"supports": map[string]any{"media": o.Supports.Media},
		},
		"retriever": map[string]any{"customOptions": configSchema},
	}, o.Metadata)

	return &RetrieverAction{
		action: *core.NewActionOf(api.ActionTypeRetriever, name, &core.ActionOptions{
			Metadata:    metadata,
			InputSchema: inputSchema,
		}, typedConfigFn(func(r *RetrieverRequest) *any { return &r.Options }, fn)),
	}
}

// NewRetriever creates a new [Retriever].
//
// Deprecated: Use [NewRetrieverAction], which passes the request's options to
// fn as a typed value instead of leaving them type-erased on the request.
func NewRetriever(name string, opts *RetrieverOptions, fn RetrieverFunc) Retriever {
	if name == "" {
		panic("ai.NewRetriever: name is required")
	}
	return NewRetrieverAction(name, opts, func(ctx context.Context, req *RetrieverRequest, _ any) (*RetrieverResponse, error) {
		return fn(ctx, req)
	})
}

// LookupRetriever looks up a registered [Retriever] by name.
// It will try to resolve the retriever dynamically if the retriever is not found.
// It returns nil if the retriever was not resolved.
func LookupRetriever(r api.Registry, name string) Retriever {
	action := core.ResolveActionFor[*RetrieverRequest, *RetrieverResponse, struct{}](r, api.ActionTypeRetriever, name)
	if action == nil {
		return nil
	}
	return &RetrieverAction{*action}
}

// Retrieve runs the given [Retriever].
func (r *RetrieverAction) Retrieve(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
	if r == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Retriever.Retrieve: retriever called on a nil retriever; check that all retrievers are defined")
	}

	return r.Run(ctx, req, nil)
}

// Retrieve calls the retriever with the provided options.
func Retrieve(ctx context.Context, r api.Registry, opts ...RetrieverOption) (*RetrieverResponse, error) {
	retOpts := &retrieverOptions{}
	for _, opt := range opts {
		opt.applyRetriever(retOpts)
	}

	if len(retOpts.Documents) == 0 {
		return nil, errors.New("ai.Retrieve: a query document is required (WithDocs or WithTextDocs)")
	}
	if len(retOpts.Documents) > 1 {
		return nil, errors.New("ai.Retrieve: only supports a single document as input")
	}

	if retOpts.Retriever == nil {
		return nil, fmt.Errorf("ai.Retrieve: retriever must be set")
	}
	ret, ok := retOpts.Retriever.(Retriever)
	if !ok {
		ret = LookupRetriever(r, retOpts.Retriever.Name())
	}

	if ret == nil {
		return nil, fmt.Errorf("ai.Retrieve: retriever not found: %s", retOpts.Retriever.Name())
	}

	if retRef, ok := retOpts.Retriever.(RetrieverRef); ok && retOpts.Config == nil {
		if cfg := retRef.Config(); !base.IsNil(cfg) {
			retOpts.Config = cfg
		}
	}

	req := &RetrieverRequest{
		Query:   retOpts.Documents[0],
		Options: retOpts.Config,
	}

	return ret.Retrieve(ctx, req)
}
