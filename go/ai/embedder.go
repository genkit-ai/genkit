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
	"fmt"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// EmbedderFunc is the function type for embedding documents.
type EmbedderFunc = func(context.Context, *EmbedRequest) (*EmbedResponse, error)

// TypedEmbedderFunc is an [EmbedderFunc] that additionally receives the
// request's typed Config: the framework deserializes the request's raw
// options into it before calling the function (see [NewTypedEmbedder]).
type TypedEmbedderFunc[Config any] = func(context.Context, *EmbedRequest, Config) (*EmbedResponse, error)

// Embedder represents an embedder that can perform content embedding.
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
}

// embedder is an action with functions specific to converting documents to multidimensional vectors such as Embed().
type embedder struct {
	core.Action[*EmbedRequest, *EmbedResponse, struct{}]
}

// NewTypedEmbedder creates a new [Embedder]. Register it with
// [Embedder.Register] to make it resolvable by name.
//
// Config is the embedder's typed configuration; it is usually inferred from
// fn's signature. The framework deserializes the request's raw options into
// Config before calling fn: the exact Config type (or a pointer to it) and
// map[string]any (from the Dev UI and other JSON callers) are accepted, and
// mismatched types are rejected. The request's [EmbedRequest.Options] is
// normalized to the converted value, so it always matches the typed
// parameter. The config's JSON schema is inferred from Config unless
// [EmbedderOptions.ConfigSchema] overrides it.
func NewTypedEmbedder[Config any](name string, opts *EmbedderOptions, fn TypedEmbedderFunc[Config]) Embedder {
	if name == "" {
		panic("ai.NewTypedEmbedder: name is required")
	}

	if opts == nil {
		opts = &EmbedderOptions{
			Label: name,
		}
	}
	if opts.Supports == nil {
		opts.Supports = &EmbedderSupports{}
	}

	configSchema, inputSchema := actionConfigSchemas[Config](opts.ConfigSchema, EmbedRequest{}, "options")

	metadata := map[string]any{
		"type": api.ActionTypeEmbedder,
		// TODO: This should be under "embedder" but JS has it as "info".
		"info": map[string]any{
			"label":      opts.Label,
			"dimensions": opts.Dimensions,
			"supports": map[string]any{
				"input":        opts.Supports.Input,
				"multilingual": opts.Supports.Multilingual,
			},
		},
		"embedder": map[string]any{
			"customOptions": configSchema,
		},
	}

	rawFn := func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
		cfg, err := resolveConfig[Config](req.Options)
		if err != nil {
			return nil, err
		}
		// Normalize the request so its type-erased Options always carries the
		// same converted value the typed parameter does.
		req.Options = cfg
		return fn(ctx, req, cfg)
	}

	return &embedder{
		Action: *core.NewActionOf(api.ActionTypeEmbedder, name, &core.ActionOptions{
			Metadata:    metadata,
			InputSchema: inputSchema,
		}, rawFn),
	}
}

// NewEmbedder creates a new [Embedder].
//
// Deprecated: Use [NewTypedEmbedder], which passes the request's options
// to fn as a typed value instead of leaving them type-erased on the request.
func NewEmbedder(name string, opts *EmbedderOptions, fn EmbedderFunc) Embedder {
	if name == "" {
		panic("ai.NewEmbedder: name is required")
	}
	return NewTypedEmbedder(name, opts, func(ctx context.Context, req *EmbedRequest, _ any) (*EmbedResponse, error) {
		return fn(ctx, req)
	})
}

// DefineEmbedder registers the given embed function as an action, and returns an
// [Embedder] that runs it.
//
// Deprecated: Use [NewTypedEmbedder] and register the result with
// [Embedder.Register].
func DefineEmbedder(r api.Registry, name string, opts *EmbedderOptions, fn EmbedderFunc) Embedder {
	e := NewEmbedder(name, opts, fn)
	e.Register(r)
	return e
}

// LookupEmbedder looks up an [Embedder] registered by [DefineEmbedder].
// It will try to resolve the embedder dynamically if the embedder is not found.
// It returns nil if the embedder was not resolved.
func LookupEmbedder(r api.Registry, name string) Embedder {
	action := core.ResolveActionFor[*EmbedRequest, *EmbedResponse, struct{}](r, api.ActionTypeEmbedder, name)
	if action == nil {
		return nil
	}
	return &embedder{
		Action: *action,
	}
}

// Embed runs the given [Embedder].
func (e *embedder) Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
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
