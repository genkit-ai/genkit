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

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// This file holds the typed-config plumbing shared by models, embedders, and
// evaluators. Requests carry config as `any` on the wire; the Define/New
// constructors wrap the user's typed function so that the raw value is
// deserialized into the Config type parameter before the function runs, and
// the request's type-erased config slot is normalized to that same converted
// value so the two views never disagree.

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
// config into Config and writes the converted value back into the request.
// It runs as the outermost step of a model's built-in chain so that every
// wrapper after it, and the model function itself, sees the typed value; by
// then the config has already been validated against the model's config
// schema at the action boundary.
//
// Version validation runs here, against the raw config, because conversion is
// lossy: a "version" key sent by a JSON caller would be silently dropped when
// deserializing into a Config type that has no such field.
func normalizeConfig[Config any](model string, versions []string) func(next rawModelFunc) rawModelFunc {
	return func(next rawModelFunc) rawModelFunc {
		return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if err := validateVersion(model, versions, req.Config); err != nil {
				return nil, err
			}
			cfg, err := resolveConfig[Config](req.Config)
			if err != nil {
				return nil, err
			}
			req.Config = cfg
			return next(ctx, req, cb)
		}
	}
}

// actionConfigSchemas returns the action's effective config schema (the
// explicit override when set, otherwise the schema inferred from Config) and
// the request input schema with its config slot replaced by the null-tolerant
// wrapping of that schema. reqZero is the zero request value to infer the
// input schema from and key is the config slot's wire name ("config" for
// models, "options" for embedders and evaluators).
func actionConfigSchemas[Config any](override map[string]any, reqZero any, key string) (configSchema, inputSchema map[string]any) {
	configSchema = override
	if configSchema == nil {
		configSchema = base.SchemaMapFor[Config]()
	}

	inputSchema = core.InferSchemaMap(reqZero)
	if inputSchema != nil && configSchema != nil {
		if props, ok := inputSchema["properties"].(map[string]any); ok {
			props[key] = nullableConfigSchema(configSchema)
		}
	}
	return configSchema, inputSchema
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
			return cfg, status.PublicErrorf(status.ErrInvalidArgument, "invalid config type %T, want %T or map[string]any", raw, cfg)
		}
		return cfg, status.PublicErrorf(status.ErrInvalidArgument, "invalid config for %T; check that field names and value types match: %v", cfg, err)
	}
	return cfg, nil
}
