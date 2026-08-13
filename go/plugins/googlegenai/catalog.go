// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal"
)

// catalog is the per-instance context for describing and building one plugin's
// actions: the capabilities the plugin ships with the caller's overrides laid
// over them, alongside the instance-level settings an action is built from.
// The provider is here for the same reason, since resolving capabilities and
// naming an action both depend on which backend this instance speaks to.
//
// Both paths that describe an action consult it, [listActions] and
// [resolveAction], which is what makes an override authoritative regardless of
// which path reaches a given ID first.
type catalog struct {
	provider  string
	models    map[string]ai.ModelOptions
	embedders map[string]ai.EmbedderOptions

	// legacyResponseSchema is [GoogleAI.LegacyResponseSchema] /
	// [VertexAI.LegacyResponseSchema]. It is settled per plugin instance rather
	// than per model, so it travels with the rest of the instance's settings
	// instead of as an argument through call paths (Veo, embedders) that have
	// no use for it. The zero value is the current default, so a catalog
	// literal that omits it gets ResponseJsonSchema.
	legacyResponseSchema bool
}

func (ga *GoogleAI) catalog() catalog {
	return catalog{
		provider:             googleAIProvider,
		models:               ga.Models,
		embedders:            ga.Embedders,
		legacyResponseSchema: ga.LegacyResponseSchema,
	}
}

func (v *VertexAI) catalog() catalog {
	return catalog{
		provider:             vertexAIProvider,
		models:               v.Models,
		embedders:            v.Embedders,
		legacyResponseSchema: v.LegacyResponseSchema,
	}
}

// modelOptions returns the capabilities to describe a model ID with: what
// [GetModelOptions] resolves for it, overlaid with the caller's entry when
// there is one. Gemini, Imagen and Veo IDs all resolve through here, so an
// entry works the same for each.
func (c catalog) modelOptions(id string) ai.ModelOptions {
	base := GetModelOptions(id, c.provider)
	if override, ok := c.models[id]; ok {
		return internal.OverlayModelOptions(base, override)
	}
	return base
}

// modelOverridden reports whether the caller described this model ID, which
// makes an ID the plugin does not ship a known one.
func (c catalog) modelOverridden(id string) bool {
	_, ok := c.models[id]
	return ok
}

// embedderOptions returns the capabilities to describe an embedder ID with:
// what [GetEmbedderOptions] resolves for it, overlaid with the caller's entry
// when there is one.
func (c catalog) embedderOptions(id string) ai.EmbedderOptions {
	base := GetEmbedderOptions(id, c.provider)
	if override, ok := c.embedders[id]; ok {
		return internal.OverlayEmbedderOptions(base, override)
	}
	return base
}
