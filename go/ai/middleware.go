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
	"reflect"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
)

// Hooks is the per-call bundle of hook functions produced by a [Middleware]'s
// New method. Each field is optional; a nil hook is treated as a pass-through.
type Hooks struct {
	// Tools are additional tools to register during the generation this
	// middleware is attached to. They are available to the model alongside
	// any user-supplied tools.
	Tools []Tool
	// WrapGenerate wraps each iteration of the tool loop. It sees the
	// accumulated request, the iteration index, and the streaming callback.
	// A single Generate() with N tool-call turns invokes this hook N+1 times.
	WrapGenerate func(ctx context.Context, params *GenerateParams, next GenerateNext) (*ModelResponse, error)
	// WrapModel wraps each model API call. Retry, fallback, and caching
	// middleware typically hook here.
	WrapModel func(ctx context.Context, params *ModelParams, next ModelNext) (*ModelResponse, error)
	// WrapTool wraps each tool execution. It may be called concurrently when
	// multiple tools execute in parallel for the same Generate() call; any
	// state closed over from the enclosing scope that this hook mutates must
	// be guarded with sync primitives.
	WrapTool func(ctx context.Context, params *ToolParams, next ToolNext) (*MultipartToolResponse, error)
}

// GenerateParams holds params for the WrapGenerate hook.
type GenerateParams struct {
	// Options is a per-turn copy of the options [Generate] was called with,
	// for the settings Request does not carry: model name, turn limit, resume
	// directives. Its Messages are the call's original ones; read
	// Request.Messages for the conversation as of this turn. Treat it as
	// read-only: the loop keeps its own copy, so writes here reach nothing.
	// The copy is shallow, so values it points at are still shared.
	Options *GenerateActionOptions
	// Request is the model request for this turn, with the messages
	// accumulated so far. Replace it, or edit it in place, to change what the
	// model receives. Each turn gets its own value, so an edit stays with this
	// turn unless it lands on a message shared with the next.
	Request *ModelRequest
	// Iteration is the current tool-loop iteration (0-indexed).
	Iteration int
	// MessageIndex is the index of the next message in the streamed response sequence.
	// Middleware that streams extra messages (e.g. injected user content) should emit
	// chunks at this index and advance it so downstream middleware and the model
	// receive the shifted value.
	MessageIndex int
	// Callback is the streaming callback supplied to [Generate], or nil if not streaming.
	// Middleware may invoke it to emit chunks, setting [ModelResponseChunk.Role] and
	// [ModelResponseChunk.Index] explicitly.
	Callback ModelStreamCallback
}

// ModelParams holds params for the WrapModel hook.
type ModelParams struct {
	// Request is the model request about to be sent.
	Request *ModelRequest
	// Callback is the streaming callback, or nil if not streaming.
	Callback ModelStreamCallback
}

// ToolParams holds params for the WrapTool hook.
type ToolParams struct {
	// Request is the tool request about to be executed.
	Request *ToolRequest
	// Tool is the resolved tool being called.
	Tool Tool
}

// GenerateNext is the next function in the WrapGenerate hook chain.
type GenerateNext = func(ctx context.Context, params *GenerateParams) (*ModelResponse, error)

// ModelNext is the next function in the WrapModel hook chain.
type ModelNext = func(ctx context.Context, params *ModelParams) (*ModelResponse, error)

// ToolNext is the next function in the WrapTool hook chain.
type ToolNext = func(ctx context.Context, params *ToolParams) (*MultipartToolResponse, error)

// Middleware is the contract every value passed to [WithUse] satisfies. The
// config struct both identifies the middleware (via [Name]) and produces a
// per-call [Hooks] bundle (via [New]).
//
// Plugin-level state belongs on unexported fields of the config type. A
// plugin's [MiddlewarePlugin.Middlewares] sets those fields on a prototype,
// which every JSON-dispatched call copies before unmarshalling its own
// config over the exported fields. Unexported state therefore carries into
// each call, and state that must be shared across calls (a client, a cache)
// belongs behind a pointer, which survives the copy pointing at the same
// object.
//
// Exported fields are per-call user config, never plugin defaults: a call
// that omits one gets the zero value, so defaults belong in New. Leave them
// zero on the prototype, since a copy would otherwise share their slices and
// maps with it.
type Middleware interface {
	// Name returns the registered middleware's unique identifier. Must be a
	// stable constant, since it is read from a zero value of the config type
	// during descriptor creation.
	Name() string
	// New produces a fresh [Hooks] bundle for one Generate() call. It is
	// invoked per-Generate, so any state the bundle's hooks need to share
	// (counters, caches) may be allocated in this method and closed over by
	// the returned hooks.
	New(ctx context.Context) (*Hooks, error)
}

// middlewareFactoryFunc is the closure stored on [MiddlewareDesc] that
// materializes a [Hooks] bundle from JSON config. It is produced by
// [NewMiddleware] and copies the prototype per call so unexported
// plugin-level state carries into each JSON-dispatched invocation.
type middlewareFactoryFunc = func(ctx context.Context, configJSON []byte) (*Hooks, error)

// middlewareRegistryPrefix is the registry-key prefix under which middleware
// descriptors are stored. The reflection API lists values under this prefix.
const middlewareRegistryPrefix = "/middleware/"

func middlewareRegistryKey(name string) string {
	return middlewareRegistryPrefix + name
}

// Register records this descriptor in the registry under its name so it can
// be resolved during JSON dispatch and surfaced to the Dev UI.
func (d *MiddlewareDesc) Register(r api.Registry) {
	r.RegisterValue(middlewareRegistryKey(d.Name), d)
}

// NewMiddleware constructs a descriptor without registering it. Useful for
// [MiddlewarePlugin.Middlewares] implementations that defer registration
// to [genkit.Init]. The prototype argument supplies the registered name (via
// its [Middleware.Name] method), the config schema, and any plugin-level
// state on unexported fields; see [Middleware] for what belongs where.
func NewMiddleware[M Middleware](description string, prototype M) *MiddlewareDesc {
	name := prototype.Name()
	return &MiddlewareDesc{
		Name:         name,
		Description:  description,
		ConfigSchema: core.InferSchemaMap(prototype),
		buildFromJSON: func(ctx context.Context, configJSON []byte) (*Hooks, error) {
			cfg := isolate(prototype)
			if len(configJSON) > 0 {
				if err := json.Unmarshal(configJSON, &cfg); err != nil {
					return nil, status.Errorf(status.ErrInvalidArgument, "middleware %q: %w", name, err)
				}
			}
			return cfg.New(ctx)
		},
	}
}

// isolate returns a copy of prototype that a call's config can be
// unmarshalled into without writing through to the registered prototype.
// Assignment already copies a value prototype; a pointer one would share its
// pointee, so the struct behind it is copied into a fresh allocation.
func isolate[M Middleware](prototype M) M {
	v := reflect.ValueOf(prototype)
	if v.Kind() != reflect.Pointer {
		return prototype
	}
	fresh := reflect.New(v.Type().Elem())
	if !v.IsNil() {
		fresh.Elem().Set(v.Elem())
	}
	// A nil prototype has no state to copy, but New still needs a receiver:
	// a call that sends no config unmarshals nothing and would get the nil.
	return fresh.Interface().(M)
}

// MiddlewareFunc adapts a per-call factory closure to the [Middleware]
// interface for ad-hoc inline use, without a registered descriptor or plugin
// wiring. The adapted middleware does not appear in the Dev UI.
//
// Example:
//
//	ai.WithUse(ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
//	    return &ai.Hooks{WrapModel: ...}, nil
//	}))
type MiddlewareFunc func(ctx context.Context) (*Hooks, error)

// Name returns the placeholder name shared by all [MiddlewareFunc] values.
// Uniqueness is unnecessary: inline middleware is resolved via the fast path
// in [resolveRefs] and never goes through a name-keyed registry lookup.
func (MiddlewareFunc) Name() string { return "inline" }

// New implements [Middleware] by calling f.
func (f MiddlewareFunc) New(ctx context.Context) (*Hooks, error) { return f(ctx) }

// middlewareRefArg is a lazy [Middleware] that carries only a registered
// name and an opaque config payload. It exists so data-driven sources (most
// notably dotprompt `use:` entries) can populate the same `[]Middleware`
// channel that [WithUse] uses, while [configsToRefs] strips the adapter out
// and routes resolution through the registry rather than the local fast
// path. The type is unexported on purpose: users who want this shape should
// build a [GenerateActionOptions] with [MiddlewareRef]s directly.
type middlewareRefArg struct {
	name   string
	config any
}

func (r middlewareRefArg) Name() string { return r.name }

// New is unreachable in normal use because [configsToRefs] swaps the adapter
// for a name-only [MiddlewareRef] before [resolveRefs] sees it; the error
// here surfaces a routing bug instead of returning nil hooks.
func (middlewareRefArg) New(context.Context) (*Hooks, error) {
	return nil, status.Errorf(status.ErrInternal, "ai: middlewareRefArg must be resolved via the registry")
}

// LookupMiddleware returns the registered middleware descriptor with the
// given name, or nil if no such descriptor exists in the registry or any
// ancestor. Primarily useful for inspection and for the reflection API;
// callers dispatching middleware should do so through [WithUse].
func LookupMiddleware(r api.Registry, name string) *MiddlewareDesc {
	v := r.LookupValue(middlewareRegistryKey(name))
	if v == nil {
		return nil
	}
	d, _ := v.(*MiddlewareDesc)
	return d
}

// MiddlewarePlugin is implemented by plugins that provide middleware. The
// returned descriptors are registered in the registry during [genkit.Init],
// with any plugin-level state captured by the descriptor's build closure via
// the prototype passed to [NewMiddleware].
type MiddlewarePlugin interface {
	Middlewares(ctx context.Context) ([]*MiddlewareDesc, error)
}

// configsToRefs converts a user-supplied slice of [Middleware] values into
// the [MiddlewareRef] entries carried on [GenerateActionOptions.Use]. For
// regular middleware the Go value is stored on each ref so [resolveRefs] can
// build the hooks bundle directly without a registry round trip. For
// [middlewareRefArg] only the name and raw config travel; this routes
// [resolveRefs] through the registry-lookup path so dotprompt-supplied
// middleware resolves like any other JSON-dispatched call.
func configsToRefs(configs []Middleware) ([]*MiddlewareRef, error) {
	if len(configs) == 0 {
		return nil, nil
	}
	refs := make([]*MiddlewareRef, 0, len(configs))
	for _, c := range configs {
		if c == nil {
			return nil, status.Errorf(status.ErrInvalidArgument, "ai: nil middleware")
		}
		if lazy, ok := c.(middlewareRefArg); ok {
			refs = append(refs, &MiddlewareRef{Name: lazy.name, Config: lazy.config})
			continue
		}
		refs = append(refs, &MiddlewareRef{Name: c.Name(), Config: c})
	}
	return refs, nil
}

// namedHooks pairs a middleware's registered name with the per-call [Hooks]
// bundle it produced. Middleware gets no spans of its own (its hooks wrap
// spans other actions create), so the name travels with the hooks to let the
// chain builders attribute log records to the middleware that ran.
type namedHooks struct {
	name  string
	hooks *Hooks
}

// resolveRefs resolves [MiddlewareRef] entries to named [Hooks] bundles. If
// ref.Config is a [Middleware] value, its New method is invoked directly
// (local fast path). Otherwise the descriptor is looked up in the registry
// and its build closure is invoked with the marshaled config (JSON dispatch,
// used for cross-runtime / Dev UI calls).
func resolveRefs(ctx context.Context, r api.Registry, refs []*MiddlewareRef) ([]namedHooks, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	bundles := make([]namedHooks, 0, len(refs))
	for _, ref := range refs {
		if mw, ok := ref.Config.(Middleware); ok {
			h, err := mw.New(ctx)
			if err != nil {
				return nil, status.Errorf(status.ErrInvalidArgument, "ai: failed to build middleware %q: %w", ref.Name, err)
			}
			if h == nil {
				return nil, status.Errorf(status.ErrInternal, "ai: middleware %q returned nil hooks", ref.Name)
			}
			bundles = append(bundles, namedHooks{name: ref.Name, hooks: h})
			continue
		}
		d := LookupMiddleware(r, ref.Name)
		if d == nil {
			return nil, status.Errorf(status.ErrNotFound, "ai: middleware %q not registered (is the providing plugin installed?)", ref.Name)
		}
		var configJSON []byte
		if ref.Config != nil {
			b, err := json.Marshal(ref.Config)
			if err != nil {
				return nil, status.Errorf(status.ErrInternal, "ai: failed to marshal config for middleware %q: %w", ref.Name, err)
			}
			configJSON = b
		}
		h, err := d.buildFromJSON(ctx, configJSON)
		if err != nil {
			return nil, status.Errorf(status.ErrInvalidArgument, "ai: failed to build middleware %q: %w", ref.Name, err)
		}
		if h == nil {
			return nil, status.Errorf(status.ErrInternal, "ai: middleware %q factory returned nil", ref.Name)
		}
		bundles = append(bundles, namedHooks{name: ref.Name, hooks: h})
	}
	return bundles, nil
}
