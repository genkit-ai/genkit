// Copyright 2026 Google LLC
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
	"errors"
	"maps"
	"slices"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// This file holds the typed-config plumbing shared by models, embedders, and
// evaluators. Requests carry config as `any` on the wire; the
// New*Action constructors wrap the user's typed function so that
// the raw value is deserialized into the Config type parameter before the
// function runs, and the request's type-erased config slot is normalized to
// that same converted value so the two views never disagree.

// typedConfigFn wraps fn so it receives the typed config alongside a request
// whose type-erased config slot has been normalized to the same value, meaning
// the request's two views of the config never disagree. slot points at that
// slot on the shallow copy this makes: the incoming request is caller-owned
// memory that may be reused across actions or turns, so it keeps carrying the
// raw config.
//
// Models go through [normalizeConfig] instead, which does the same thing as
// middleware so the built-in wrappers see the typed value too.
func typedConfigFn[Req, Resp, Config any](
	slot func(*Req) *any,
	fn func(context.Context, *Req, Config) (*Resp, error),
) func(context.Context, *Req) (*Resp, error) {
	return func(ctx context.Context, req *Req) (*Resp, error) {
		reqCopy := *req
		cfg, err := resolveConfigInto[Config](slot(&reqCopy))
		if err != nil {
			return nil, err
		}
		return fn(ctx, &reqCopy, cfg)
	}
}

// actionMetadata builds an action descriptor's metadata: the caller's maps are
// merged in order, then the built-in entries and the action type are stamped
// over them so they cannot be corrupted. Registry discovery depends on those.
func actionMetadata(atype api.ActionType, builtin map[string]any, callerMetadata ...map[string]any) map[string]any {
	size := len(builtin) + 1
	for _, m := range callerMetadata {
		size += len(m)
	}
	metadata := make(map[string]any, size)
	for _, m := range callerMetadata {
		maps.Copy(metadata, m)
	}
	maps.Copy(metadata, builtin)
	metadata["type"] = atype
	return metadata
}

// nullableConfigSchema wraps a config schema for the request input-schema
// slot so that an explicit JSON null is accepted on the wire: a typed-nil Go
// config marshals to null and resolves to the zero Config value, so it must
// not be rejected by input validation. The advertised config schema (the
// customOptions metadata) stays unwrapped.
func nullableConfigSchema(schema map[string]any) map[string]any {
	if schema == nil {
		return nil
	}
	return map[string]any{"anyOf": []any{schema, map[string]any{"type": "null"}}}
}

// normalizeConfig returns a model middleware that resolves the request's raw
// config into Config and passes the request on with its config slot
// normalized to the converted value. It runs as the outermost step of a
// model's built-in chain so that every wrapper after it, and the model
// function itself, sees the typed value; by then the config has already been
// validated against the model's config schema at the action boundary. The
// normalization happens on a shallow copy: the incoming request is
// caller-owned memory that may be reused across actions or turns, so it must
// keep carrying the raw config.
//
// Version validation runs here, against the raw config, because conversion is
// lossy: a "version" key sent by a JSON caller would be silently dropped when
// deserializing into a Config type that has no such field.
func normalizeConfig[Config any](model string, versions []string) ModelMiddleware {
	return func(next ModelFunc) ModelFunc {
		return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if err := validateVersion(model, versions, req.Config); err != nil {
				return nil, err
			}
			reqCopy := *req
			req = &reqCopy
			if _, err := resolveConfigInto[Config](&req.Config); err != nil {
				return nil, err
			}
			return next(ctx, req, cb)
		}
	}
}

// actionConfigSchemas returns the action's effective config schema (see
// effectiveConfigSchema) and the request input schema with its config slot
// replaced by the null-tolerant wrapping of that schema. reqZero is the zero
// request value to infer the input schema from and key is the config slot's
// wire name ("options" for embedders, retrievers, and evaluators; models go
// through [modelConfigSchemas]).
func actionConfigSchemas[Config any](override map[string]any, reqZero any, key string) (configSchema, inputSchema map[string]any) {
	configSchema, enforced := configSchemas[Config](override)
	return configSchema, requestInputSchema(reqZero, key, nullableConfigSchema(enforced))
}

// configSchemas returns the config schema the action advertises and the one it
// enforces on the wire. The advertised one is the contract, verbatim from the
// plugin or inferred from Config; the enforced one is a copy that additionally
// tolerates the nulls a partial config marshals to, which stays out of the
// advertised copy so the dev UI is not littered with it.
//
// Both a declared and an inferred schema are widened, so a config behaves the
// same whichever way its schema was arrived at. The copy is what keeps that
// safe: a plugin's declared schema is usually a package-level value.
func configSchemas[Config any](override map[string]any) (advertised, enforced map[string]any) {
	advertised = effectiveConfigSchema[Config](override)
	enforced = base.CloneSchema(advertised)
	tolerateNulls(enforced)
	return advertised, enforced
}

// modelConfigSchemas is [actionConfigSchemas] for the model config slot. It
// additionally keeps the framework-level "version" key admissible: callers
// pin a model version through the config, validateVersion consumes the key on
// the raw value, and conversion drops it, so the schema must not reject it.
// The property is added to the advertised schema as well, which is honest:
// the key is accepted on the wire.
//
// versions is the model's supported version list. Only a model that declares
// one gets the property: validateVersion rejects every value when the list is
// empty, so advertising the key there offers a field that can only error.
func modelConfigSchemas[Config any](override map[string]any, versions []string) (configSchema, inputSchema map[string]any) {
	configSchema, enforced := configSchemas[Config](override)
	if len(versions) > 0 {
		configSchema = withVersionProperty(configSchema)
		enforced = withVersionProperty(enforced)
	}
	return configSchema, requestInputSchema(ModelRequest{}, "config", nullableConfigSchema(enforced))
}

// effectiveConfigSchema returns the explicit override when set, otherwise the
// schema inferred from Config. Inferred schemas have their "required" lists
// stripped: a config is partial by nature (callers set only the fields they
// want to override), so a struct field lacking omitempty must not become a
// mandatory config key. An explicit override is the caller's contract and
// passes through untouched, which is also what keeps the in-place strip safe:
// it only ever reaches the fresh map [base.SchemaMapFor] allocates per call.
func effectiveConfigSchema[Config any](override map[string]any) map[string]any {
	if override != nil {
		return override
	}
	schema := base.SchemaMapFor[Config]()
	stripRequired(schema)
	return schema
}

// stripRequired removes "required" from schema and every subschema: a config
// is partial by nature, so a field lacking omitempty must not become mandatory.
func stripRequired(schema map[string]any) {
	delete(schema, "required")
	base.WalkSubschemas(schema, func(sub map[string]any) map[string]any {
		delete(sub, "required")
		return sub
	})
}

// tolerateNulls widens every subschema of an inferred config schema to also
// accept an explicit JSON null.
//
// A pointer, slice, or map field that lacks omitempty marshals to null when it
// is unset, so enforcing the field's declared type would reject a partially
// filled value of the config type itself: sending Config{Temperature: 0.5}
// fails on `config.list: Invalid type. Expected: array, given: null`. That is
// the same "a config is partial by nature" reasoning that strips "required",
// applied to the fields that are present but empty.
//
// Every field is widened, not just the nilable ones, because the inferred
// schema cannot tell a *string from a string. Widening costs nothing: null
// decodes to the zero value for every Go kind, which is what omitting the
// field would have produced.
func tolerateNulls(schema map[string]any) {
	base.WalkSubschemas(schema, allowNull)
}

// allowNull returns schema widened to accept null, adding to its "type" when
// it has one and wrapping it in an anyOf otherwise.
func allowNull(schema map[string]any) map[string]any {
	switch t := schema["type"].(type) {
	case string:
		if t != "null" {
			schema["type"] = []any{t, "null"}
		}
		return schema
	case []any:
		if !slices.Contains(t, any("null")) {
			schema["type"] = append(t, "null")
		}
		return schema
	}
	if len(schema) == 0 {
		return schema // Already unconstrained, so null is in.
	}
	return nullableConfigSchema(schema)
}

// withVersionProperty returns schema with a string "version" property added
// unless one is already declared or the schema constrains no properties. The
// maps are cloned before modification since an override is caller-owned.
func withVersionProperty(schema map[string]any) map[string]any {
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		return schema
	}
	if _, ok := props["version"]; ok {
		return schema
	}
	schema = maps.Clone(schema)
	props = maps.Clone(props)
	props["version"] = map[string]any{"type": "string"}
	schema["properties"] = props
	return schema
}

// resolveConfig converts the raw config value carried by a request into the
// typed Config the action was defined with. It accepts the exact Config type
// (or a pointer to it, which is dereferenced), a map[string]any (as sent by
// the Dev UI and other JSON callers, deserialized via a JSON round-trip), or
// nil (yielding the zero value). Any other type is rejected so one provider's
// config cannot be silently passed to another provider's action.
func resolveConfig[Config any](raw any) (Config, error) {
	cfg, err := base.ConvertToExact[Config](raw)
	if err != nil {
		// Public: naming the config type is what tells a caller which
		// provider's config they reached for, and it leaks nothing.
		if errors.Is(err, base.ErrTypeMismatch) {
			return cfg, status.PublicErrorf(status.ErrInvalidArgument, "invalid config type %T, want %T or map[string]any", raw, cfg)
		}
		return cfg, status.PublicErrorf(status.ErrInvalidArgument, "invalid config for %T; check that field names and value types match: %w", cfg, err)
	}
	return cfg, nil
}

// resolveConfigInto resolves the type-erased config slot at *slot into the
// typed Config (see [resolveConfig]) and normalizes the slot to the converted
// value so the request's two views of the config never disagree. The slot is
// left untouched when the resolved value is a nil pointer, map, or slice:
// boxing a typed nil would make the slot compare non-nil for a request that
// carried no config, flipping every `== nil` default check downstream while
// dereferences still panic.
func resolveConfigInto[Config any](slot *any) (Config, error) {
	cfg, err := resolveConfig[Config](*slot)
	if err != nil {
		return cfg, err
	}
	if !base.IsNil(cfg) {
		*slot = cfg
	}
	return cfg, nil
}
