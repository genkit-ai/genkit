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
	"fmt"
	"maps"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/base"
)

// This file holds the typed-config plumbing shared by models, embedders, and
// evaluators. Requests carry config as `any` on the wire; the
// New*Action constructors wrap the user's typed function so that
// the raw value is deserialized into the Config type parameter before the
// function runs, and the request's type-erased config slot is normalized to
// that same converted value so the two views never disagree.

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
	configSchema = effectiveConfigSchema[Config](override)
	return configSchema, requestInputSchema(reqZero, key, nullableConfigSchema(configSchema))
}

// modelConfigSchemas is [actionConfigSchemas] for the model config slot. It
// additionally keeps the framework-level "version" key admissible: callers
// pin a model version through the config, validateVersion consumes the key on
// the raw value, and conversion drops it, so the schema must not reject it.
// The property is added to the advertised schema as well, which is honest:
// the key is accepted on the wire.
func modelConfigSchemas[Config any](override map[string]any) (configSchema, inputSchema map[string]any) {
	configSchema = withVersionProperty(effectiveConfigSchema[Config](override))
	return configSchema, requestInputSchema(ModelRequest{}, "config", nullableConfigSchema(configSchema))
}

// effectiveConfigSchema returns the explicit override when set, otherwise the
// schema inferred from Config. Inferred schemas have their "required" lists
// stripped: a config is partial by nature (callers set only the fields they
// want to override), so a struct field lacking omitempty must not become a
// mandatory config key. An explicit override is the caller's contract and
// passes through untouched.
func effectiveConfigSchema[Config any](override map[string]any) map[string]any {
	if override != nil {
		return override
	}
	schema := base.SchemaMapFor[Config]()
	stripRequired(schema)
	return schema
}

// stripRequired removes "required" lists from schema and its nested object
// schemas (properties, items, and schema-valued additionalProperties).
func stripRequired(schema map[string]any) {
	if schema == nil {
		return
	}
	delete(schema, "required")
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, sub := range props {
			if m, ok := sub.(map[string]any); ok {
				stripRequired(m)
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		stripRequired(items)
	}
	if extra, ok := schema["additionalProperties"].(map[string]any); ok {
		stripRequired(extra)
	}
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
		if errors.Is(err, base.ErrTypeMismatch) {
			return cfg, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("invalid config type %T, want %T or map[string]any", raw, cfg), nil)
		}
		return cfg, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("invalid config for %T; check that field names and value types match: %v", cfg, err), nil)
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
