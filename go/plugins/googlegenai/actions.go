// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"

	"github.com/firebase/genkit/go/core/api"
	"google.golang.org/genai"
)

// ListActions lists all the actions supported by the Google AI plugin.
func (ga *GoogleAI) ListActions(ctx context.Context) []api.ActionDesc {
	return listActions(ctx, ga.gclient, googleAIProvider)
}

// ListActions lists all the actions supported by the Vertex AI plugin.
func (v *VertexAI) ListActions(ctx context.Context) []api.ActionDesc {
	return listActions(ctx, v.gclient, vertexAIProvider)
}

// listActions is the shared implementation for listing actions.
func listActions(ctx context.Context, client *genai.Client, provider string) []api.ActionDesc {
	models, err := listGenaiModels(ctx, client)
	if err != nil {
		return nil
	}

	actions := []api.ActionDesc{}

	// Gemini models
	for _, name := range models.gemini {
		opts := GetModelOptions(name, provider)
		model := newModel(client, name, opts)
		if action, ok := model.(api.Action); ok {
			actions = append(actions, action.Desc())
		}
	}

	// Imagen models
	for _, name := range models.imagen {
		opts := GetModelOptions(name, provider)
		model := newModel(client, name, opts)
		if action, ok := model.(api.Action); ok {
			actions = append(actions, action.Desc())
		}
	}

	// Veo models (background models)
	for _, name := range models.veo {
		actions = append(actions, newVeoModel(client, name, GetModelOptions(name, provider)).Desc())
	}

	// Embedders
	for _, name := range models.embedders {
		opts := GetEmbedderOptions(name, provider)
		actions = append(actions, newEmbedder(client, name, &opts).Desc())
	}

	return actions
}

// ResolveAction resolves an action with the given name.
func (ga *GoogleAI) ResolveAction(atype api.ActionType, name string) api.Action {
	return resolveAction(ga.gclient, googleAIProvider, atype, name)
}

// ResolveAction resolves an action with the given name.
func (v *VertexAI) ResolveAction(atype api.ActionType, name string) api.Action {
	return resolveAction(v.gclient, vertexAIProvider, atype, name)
}

// resolveAction is the shared implementation for resolving actions.
func resolveAction(client *genai.Client, provider string, atype api.ActionType, name string) api.Action {
	mt := ClassifyModel(name)

	switch atype {
	case api.ActionTypeEmbedder:
		opts := GetEmbedderOptions(name, provider)
		return newEmbedder(client, name, &opts)

	case api.ActionTypeModel:
		// Veo models should not be resolved as regular models
		if mt == ModelTypeVeo {
			return nil
		}
		opts := GetModelOptions(name, provider)
		return newModel(client, name, opts).(api.Action)

	// A background model is a bundle: registering it registers both its start
	// and check actions, so the same value resolves either key. The registry
	// registers what we return and then looks up the key it was asked for.
	case api.ActionTypeBackgroundModel, api.ActionTypeCheckOperation:
		if mt != ModelTypeVeo {
			return nil
		}
		return newVeoModel(client, name, GetModelOptions(name, provider))
	}

	return nil
}
