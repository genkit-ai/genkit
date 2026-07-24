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
)

// EmbedderFunc is the function type for embedding documents.
// Config is the embedder's typed configuration: the framework deserializes the
// request's raw config into it before calling the function (see [DefineEmbedder]).
type EmbedderFunc[Config any] = func(context.Context, *EmbedRequest, Config) (*EmbedResponse, error)

// Embedder is a content embedder backed by a registry action. Create one
// with [DefineEmbedder] or [NewEmbedder], or fetch a registered one with
// [LookupEmbedder]. Pass it to [WithEmbedder] to use it with [Embed].
type Embedder struct {
	action[*EmbedRequest, *EmbedResponse, struct{}]
}

var (
	_ api.Action = (*Embedder)(nil)
	_ Named      = (*Embedder)(nil)
)

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

// NewEmbedder creates a new unregistered [Embedder]. Register it with
// [Embedder.Register] or use [DefineEmbedder] to define and register in one
// step.
//
// Config is the embedder's typed configuration; it is usually inferred from
// fn's signature. The framework deserializes the request's raw config into
// Config before calling fn: the exact Config type (or a pointer to it) and
// map[string]any (from the Dev UI and other JSON callers) are accepted, and
// mismatched types are rejected. The config's JSON schema is inferred from
// Config unless [EmbedderOptions.ConfigSchema] overrides it.
func NewEmbedder[Config any](name string, opts *EmbedderOptions, fn EmbedderFunc[Config]) *Embedder {
	if name == "" {
		panic("ai.NewEmbedder: name is required")
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
		cfg, err := resolveConfig[Config](req.Config)
		if err != nil {
			return nil, err
		}
		// Normalize the request so its type-erased Config always carries the
		// same converted value the typed parameter does.
		req.Config = cfg
		return fn(ctx, req, cfg)
	}

	return &Embedder{
		action: *core.NewAction(api.ActionTypeEmbedder, name, &core.ActionOptions{
			Metadata:    metadata,
			InputSchema: inputSchema,
		}, rawFn),
	}
}

// DefineEmbedder registers the given embed function as an action, and returns an
// [Embedder] that runs it.
//
// Config is the embedder's typed configuration; it is usually inferred from
// fn's signature. See [NewEmbedder] for how the request's config is deserialized.
func DefineEmbedder[Config any](r api.Registry, name string, opts *EmbedderOptions, fn EmbedderFunc[Config]) *Embedder {
	e := NewEmbedder(name, opts, fn)
	e.Register(r)
	return e
}

// LookupEmbedder looks up an [Embedder] registered by [DefineEmbedder].
// It will try to resolve the embedder dynamically if the embedder is not found.
// It returns nil if the embedder was not resolved.
func LookupEmbedder(r api.Registry, name string) *Embedder {
	action := core.ResolveActionFor[*EmbedRequest, *EmbedResponse, struct{}](r, api.ActionTypeEmbedder, name)
	if action == nil {
		return nil
	}
	return &Embedder{
		action: *action,
	}
}

// Name returns the registry name of the embedder, or the empty string if the
// embedder is nil (e.g. from a failed lookup).
func (e *Embedder) Name() string {
	if e == nil {
		return ""
	}
	return e.action.Name()
}

// Embed runs the given [Embedder].
func (e *Embedder) Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
	if e == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Embedder.Embed: embedder called on a nil embedder; check that all embedders are defined")
	}

	return e.Run(ctx, req, nil)
}

// Embed invokes the embedder with provided options.
func Embed(ctx context.Context, r api.Registry, opts ...EmbedderOption) (*EmbedResponse, error) {
	embedOpts := &embedderOptions{}
	for _, opt := range opts {
		if err := opt.applyEmbedder(embedOpts); err != nil {
			return nil, fmt.Errorf("ai.Embed: error applying options: %w", err)
		}
	}

	if embedOpts.Embedder == nil {
		return nil, fmt.Errorf("ai.Embed: embedder must be set")
	}
	e, ok := embedOpts.Embedder.(*Embedder)
	if !ok {
		e = LookupEmbedder(r, embedOpts.Embedder.Name())
	}
	if e == nil {
		return nil, fmt.Errorf("ai.Embed: embedder not found: %s", embedOpts.Embedder.Name())
	}

	if ref, ok := embedOpts.Embedder.(ActionRef); ok && embedOpts.Config == nil {
		embedOpts.Config = ref.Config()
	}

	req := &EmbedRequest{
		Input:  embedOpts.Documents,
		Config: embedOpts.Config,
	}

	return e.Embed(ctx, req)
}
