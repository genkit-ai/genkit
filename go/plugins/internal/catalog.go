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

package internal

import (
	"strings"

	"github.com/firebase/genkit/go/core/api"
)

// A plugin's model catalog answers two questions that must agree: what
// ListActions advertises, and what ResolveAction builds to serve a request. A
// caller who knows better than the catalog supplies an override through plugin
// config, and both paths lay it over what the plugin already knows with
// [ai.ModelOptions.Overlay].

// LookupOverride returns the caller's entry for id in a plugin's override map,
// accepting the key bare or provider-prefixed since both name the same model
// everywhere else. id may itself carry the prefix.
func LookupOverride[T any](overrides map[string]T, provider, id string) (T, bool) {
	id = TrimProvider(provider, id)
	if o, ok := overrides[id]; ok {
		return o, true
	}
	o, ok := overrides[api.NewName(provider, id)]
	return o, ok
}

// TrimProvider returns id without its leading "provider/" prefix, inverting
// [api.NewName] and leaving an id that does not carry the prefix unchanged.
// Prefer it to [api.ParseName] when the provider is already known: only the
// prefix is removed, so an ID with slashes of its own (a tuned Vertex
// endpoint, say) survives intact.
func TrimProvider(provider, id string) string {
	return strings.TrimPrefix(id, provider+"/")
}

// ProviderLabel joins a provider's display name and a model's into the label
// the dev UI lists a model under, e.g. "Anthropic - Claude Opus 5".
func ProviderLabel(provider, name string) string {
	if provider == "" {
		return name
	}
	return provider + " - " + name
}
