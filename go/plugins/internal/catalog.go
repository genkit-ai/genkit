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

package internal

import "github.com/firebase/genkit/go/ai"

// A plugin's model catalog answers two questions that must agree: what
// ListActions advertises, and what ResolveAction builds when a request names a
// model. A caller who knows better than the catalog supplies an override
// through plugin config, and both paths overlay it on what the plugin already
// knows. Overlaying rather than replacing is what lets a caller pin one
// capability without restating the label, the config schema and the rest,
// which a plugin needs set for the model to work at all.
//
// A zero-value field means "not specified" and leaves the base value in place.
// Every field of both options structs distinguishes its zero value from a
// meaningful one: the maps and slices are nil when unset, Supports is a
// pointer, and the string and int fields have no meaningful zero.

// OverlayModelOptions returns base with every field set in override replacing
// base's. Fields left at their zero value in override keep base's value.
func OverlayModelOptions(base, override ai.ModelOptions) ai.ModelOptions {
	if override.ConfigSchema != nil {
		base.ConfigSchema = override.ConfigSchema
	}
	if override.Label != "" {
		base.Label = override.Label
	}
	if override.Stage != "" {
		base.Stage = override.Stage
	}
	if override.Supports != nil {
		base.Supports = override.Supports
	}
	if override.Versions != nil {
		base.Versions = override.Versions
	}
	if override.Metadata != nil {
		base.Metadata = override.Metadata
	}
	return base
}

// OverlayEmbedderOptions returns base with every field set in override
// replacing base's. Fields left at their zero value in override keep base's
// value.
func OverlayEmbedderOptions(base, override ai.EmbedderOptions) ai.EmbedderOptions {
	if override.ConfigSchema != nil {
		base.ConfigSchema = override.ConfigSchema
	}
	if override.Label != "" {
		base.Label = override.Label
	}
	if override.Supports != nil {
		base.Supports = override.Supports
	}
	if override.Dimensions != 0 {
		base.Dimensions = override.Dimensions
	}
	if override.Metadata != nil {
		base.Metadata = override.Metadata
	}
	return base
}
