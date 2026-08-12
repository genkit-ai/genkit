// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal"
)

// catalog is what one plugin instance knows about the models and embedders it
// serves: the capabilities the plugin ships, with the caller's overrides laid
// over them.
//
// Both paths that describe an action consult it, [listActions] and
// [resolveAction], which is what makes an override authoritative regardless of
// which path reaches a given ID first.
type catalog struct {
	provider  string
	models    map[string]ai.ModelOptions
	embedders map[string]ai.EmbedderOptions
}

func (ga *GoogleAI) catalog() catalog {
	return catalog{provider: googleAIProvider, models: ga.Models, embedders: ga.Embedders}
}

func (v *VertexAI) catalog() catalog {
	return catalog{provider: vertexAIProvider, models: v.Models, embedders: v.Embedders}
}

// modelOptions returns the capabilities to describe a model ID with: what
// [GetModelOptions] resolves for it, overlaid with the caller's entry when
// there is one. Gemini, Imagen and Veo IDs all resolve through here, so an
// entry works the same for each.
func (c catalog) modelOptions(id string) ai.ModelOptions {
	base := GetModelOptions(id, c.provider)
	if override, ok := internal.LookupOverride(c.models, c.provider, id); ok {
		return base.Overlay(override)
	}
	return base
}

// modelOverridden reports whether the caller described this model ID, which
// makes an ID the plugin does not ship a known one.
func (c catalog) modelOverridden(id string) bool {
	_, ok := internal.LookupOverride(c.models, c.provider, id)
	return ok
}

// embedderOptions returns the capabilities to describe an embedder ID with:
// what [GetEmbedderOptions] resolves for it, overlaid with the caller's entry
// when there is one.
func (c catalog) embedderOptions(id string) ai.EmbedderOptions {
	base := GetEmbedderOptions(id, c.provider)
	if override, ok := internal.LookupOverride(c.embedders, c.provider, id); ok {
		return base.Overlay(override)
	}
	return base
}
