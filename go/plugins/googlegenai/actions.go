// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"slices"

	"github.com/firebase/genkit/go/core/api"
	"google.golang.org/genai"
)

// ListActions lists all the actions supported by the Google AI plugin.
func (ga *GoogleAI) ListActions(ctx context.Context) []api.ActionDesc {
	return listActions(ctx, ga.gclient, ga.catalog())
}

// ListActions lists all the actions supported by the Vertex AI plugin.
func (v *VertexAI) ListActions(ctx context.Context) []api.ActionDesc {
	return listActions(ctx, v.gclient, v.catalog())
}

// listActions is the shared implementation for listing actions.
func listActions(ctx context.Context, client *genai.Client, c catalog) []api.ActionDesc {
	models, err := listGenaiModels(ctx, client)
	if err != nil {
		return nil
	}

	actions := []api.ActionDesc{}

	// Gemini and Imagen models
	for _, name := range slices.Concat(models.gemini, models.imagen) {
		actions = append(actions, newModel(client, name, c.modelOptions(name)).Desc())
	}

	// Veo models (background models)
	for _, name := range models.veo {
		actions = append(actions, newVeoModel(client, name, c.modelOptions(name)).Desc())
	}

	// Embedders
	for _, name := range models.embedders {
		opts := c.embedderOptions(name)
		actions = append(actions, newEmbedder(client, name, &opts).Desc())
	}

	return actions
}

// ResolveAction resolves an action with the given ID.
func (ga *GoogleAI) ResolveAction(atype api.ActionType, id string) api.Action {
	return resolveAction(ga.gclient, ga.catalog(), atype, id)
}

// ResolveAction resolves an action with the given ID.
func (v *VertexAI) ResolveAction(atype api.ActionType, id string) api.Action {
	return resolveAction(v.gclient, v.catalog(), atype, id)
}

// resolveAction is the shared implementation for resolving actions.
func resolveAction(client *genai.Client, c catalog, atype api.ActionType, id string) api.Action {
	mt := ClassifyModel(id)

	switch atype {
	case api.ActionTypeEmbedder:
		opts := c.embedderOptions(id)
		return newEmbedder(client, id, &opts)

	case api.ActionTypeModel:
		// Veo models should not be resolved as regular models
		if mt == ModelTypeVeo {
			return nil
		}
		return newModel(client, id, c.modelOptions(id))

	// A background model is a bundle: registering it registers both its start
	// and check actions, so the same value resolves either key. The registry
	// registers what we return and then looks up the key it was asked for.
	case api.ActionTypeBackgroundModel, api.ActionTypeCheckOperation:
		if mt != ModelTypeVeo {
			return nil
		}
		return newVeoModel(client, id, c.modelOptions(id))
	}

	return nil
}
