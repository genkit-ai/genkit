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
	"fmt"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// EmbedderFunc is the function type for embedding documents.
type EmbedderFunc = func(context.Context, *EmbedRequest) (*EmbedResponse, error)

// EmbedderActionFunc is an [EmbedderFunc] that additionally receives the
// request's typed Config: the framework deserializes the request's raw
// options into it before calling the function (see [NewEmbedderAction]).
type EmbedderActionFunc[Config any] = func(context.Context, *EmbedRequest, Config) (*EmbedResponse, error)

// Embedder represents an embedder that can perform content embedding. It is
// the type to accept as an argument and to look up by name; implementations
// are created with [NewEmbedderAction], or [genkit.DefineEmbedderAction] in an
// application.
type Embedder interface {
	// Name returns the registry name of the embedder.
	Name() string
	// Embed embeds to content as part of the [EmbedRequest].
	Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error)
	// Register registers the embedder with the given registry.
	Register(r api.Registry)
}

// EmbedderArg is the interface for embedder arguments. It can either be the embedder action itself or a reference to be looked up.
type EmbedderArg interface {
	Name() string
}

// EmbedderRef is a struct to hold embedder name and configuration.
type EmbedderRef struct {
	name   string
	config any
}

// NewEmbedderRef creates a new EmbedderRef with the given name and configuration.
func NewEmbedderRef(name string, config any) EmbedderRef {
	return EmbedderRef{name: name, config: config}
}

// Name returns the name of the embedder.
func (e EmbedderRef) Name() string {
	return e.name
}

// Config returns the configuration to use by default for this embedder.
func (e EmbedderRef) Config() any {
	return e.config
}

// EmbedderSupports represents the supported capabilities of the embedder model.
type EmbedderSupports struct {
	// Input lists the types of data the model can process (e.g., "text", "image", "video").
	Input []string `json:"input,omitempty"`
	// Multilingual indicates whether the model supports multiple languages.
	Multilingual bool `json:"multilingual,omitempty"`
}

// EmbedderOptions represents the configuration options for an embedder.
type EmbedderOptions struct {
	// ConfigSchema is the JSON schema for the embedder's config.
	ConfigSchema map[string]any `json:"configSchema,omitempty"`
	// Label is a user-friendly name for the embedder model (e.g., "Google AI - Gemini Pro").
	Label string `json:"label,omitempty"`
	// Supports defines the capabilities of the embedder, such as input types and multilingual support.
	Supports *EmbedderSupports `json:"supports,omitempty"`
	// Dimensions specifies the number of dimensions in the embedding vector.
	Dimensions int `json:"dimensions,omitempty"`
	// Metadata is arbitrary key-value data attached to the action descriptor.
	Metadata map[string]any `json:"-"`
}

// EmbedderAction is an embedder backed by a registry action. It is the
// concrete type returned by [NewEmbedderAction]; pass it to [WithEmbedder] to
// use it for embedding, or return it from a plugin's Init for the framework
// to register.
//
// It implements [Embedder] and [api.Action], so it can be passed anywhere
// either is accepted. It also promotes [core.Action.Run], the typed
// equivalent of [EmbedderAction.Embed].
type EmbedderAction struct {
	action[*EmbedRequest, *EmbedResponse, struct{}]
}

// Pinned here so that breaking either interface fails the build at the type
// rather than at a call site.
var (
	_ api.Action = (*EmbedderAction)(nil)
	_ Embedder   = (*EmbedderAction)(nil)
)

// Name returns the registry name of the embedder.
func (e *EmbedderAction) Name() string { return e.action.Name() }

// Register registers the embedder with r, making it available to lookups and
// to the Dev UI. A plugin that returns the embedder from its Init does not
// need to call this.
func (e *EmbedderAction) Register(r api.Registry) { e.action.Register(r) }

// Desc returns the embedder's action descriptor: its name, schemas, and
// metadata.
func (e *EmbedderAction) Desc() api.ActionDesc { return e.action.Desc() }

// RunJSON runs the embedder on a JSON-encoded [EmbedRequest] and returns a
// JSON-encoded [EmbedResponse]. The framework uses it to serve reflection and
// registry-driven calls; prefer [EmbedderAction.Embed].
func (e *EmbedderAction) RunJSON(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	if e == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Embedder.RunJSON: embedder called on a nil embedder; check that all embedders are defined")
	}
	return e.action.RunJSON(ctx, input, cb)
}

// RunJSONWithTelemetry is [EmbedderAction.RunJSON] with the run's telemetry
// returned alongside the output.
func (e *EmbedderAction) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	if e == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Embedder.RunJSONWithTelemetry: embedder called on a nil embedder; check that all embedders are defined")
	}
	return e.action.RunJSONWithTelemetry(ctx, input, cb)
}

// NewEmbedderAction creates an unregistered [EmbedderAction]: return it from a
// plugin's Init for the framework to register, or call
// [EmbedderAction.Register] directly. Applications should define embedders
// with [genkit.DefineEmbedderAction].
//
// Config is the embedder's typed configuration; it is usually inferred from
// fn's signature. The framework deserializes the request's raw options into
// Config before calling fn: the exact Config type (or a pointer to it) and
// map[string]any (from the Dev UI and other JSON callers) are accepted, and
// mismatched types are rejected. The request's [EmbedRequest.Options] is
// normalized to the converted value, so it always matches the typed
// parameter. The config's JSON schema is inferred from Config unless
// [EmbedderOptions.ConfigSchema] overrides it.
func NewEmbedderAction[Config any](
	name string,
	opts *EmbedderOptions,
	fn EmbedderActionFunc[Config],
) *EmbedderAction {
	if name == "" {
		panic("ai.NewEmbedderAction: name is required")
	}

	o := EmbedderOptions{}
	if opts != nil {
		o = *opts
	}
	if o.Label == "" {
		o.Label = name
	}
	o.Supports = cloneEmbedderSupports(o.Supports)

	configSchema, inputSchema := actionConfigSchemas[Config](o.ConfigSchema, EmbedRequest{}, "options")
	metadata := actionMetadata(api.ActionTypeEmbedder, map[string]any{
		// TODO: This should be under "embedder" but JS has it as "info".
		"info": map[string]any{
			"label":      o.Label,
			"dimensions": o.Dimensions,
			"supports": map[string]any{
				"input":        o.Supports.Input,
				"multilingual": o.Supports.Multilingual,
			},
		},
		"embedder": map[string]any{"customOptions": configSchema},
	}, o.Metadata)

	return &EmbedderAction{
		action: *core.NewActionOf(api.ActionTypeEmbedder, name, &core.ActionOptions{
			Metadata:    metadata,
			InputSchema: inputSchema,
		}, typedConfigFn(func(r *EmbedRequest) *any { return &r.Options }, fn)),
	}
}

// NewEmbedder creates a new [Embedder].
//
// Deprecated: Use [NewEmbedderAction], which passes the request's options
// to fn as a typed value instead of leaving them type-erased on the request.
func NewEmbedder(name string, opts *EmbedderOptions, fn EmbedderFunc) Embedder {
	if name == "" {
		panic("ai.NewEmbedder: name is required")
	}
	return NewEmbedderAction(name, opts, func(ctx context.Context, req *EmbedRequest, _ any) (*EmbedResponse, error) {
		return fn(ctx, req)
	})
}

// LookupEmbedder looks up a registered [Embedder] by name.
// It will try to resolve the embedder dynamically if the embedder is not found.
// It returns nil if the embedder was not resolved.
func LookupEmbedder(r api.Registry, name string) Embedder {
	action := core.ResolveActionFor[*EmbedRequest, *EmbedResponse, struct{}](r, api.ActionTypeEmbedder, name)
	if action == nil {
		return nil
	}
	return &EmbedderAction{*action}
}

// Embed runs the given [Embedder].
func (e *EmbedderAction) Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
	if e == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Embedder.Embed: embedder called on a nil embedder; check that all embedders are defined")
	}

	return e.Run(ctx, req, nil)
}

// Embed invokes the embedder with provided options.
func Embed(ctx context.Context, r api.Registry, opts ...EmbedderOption) (*EmbedResponse, error) {
	embedOpts := &embedderOptions{}
	for _, opt := range opts {
		opt.applyEmbedder(embedOpts)
	}

	if embedOpts.Embedder == nil {
		return nil, fmt.Errorf("ai.Embed: embedder must be set")
	}
	e, ok := embedOpts.Embedder.(Embedder)
	if !ok {
		e = LookupEmbedder(r, embedOpts.Embedder.Name())
	}
	if e == nil {
		return nil, fmt.Errorf("ai.Embed: embedder not found: %s", embedOpts.Embedder.Name())
	}

	if embedRef, ok := embedOpts.Embedder.(EmbedderRef); ok && embedOpts.Config == nil {
		if cfg := embedRef.Config(); !base.IsNil(cfg) {
			embedOpts.Config = cfg
		}
	}

	req := &EmbedRequest{
		Input:   embedOpts.Documents,
		Options: embedOpts.Config,
	}

	return e.Embed(ctx, req)
}
