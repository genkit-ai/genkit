// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
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
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	// Imagen models
	for _, name := range models.imagen {
		opts := GetModelOptions(name, provider)
		model := newModel(client, name, opts)
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	// Veo models (background models)
	for _, name := range models.veo {
		opts := GetModelOptions(name, provider)
		veoModel := newVeoModel(client, name, opts)
		if actionDef, ok := veoModel.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}

	// Embedders
	for _, name := range models.embedders {
		opts := GetEmbedderOptions(name, provider)
		embedder := newEmbedder(client, name, &opts)
		if actionDef, ok := embedder.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
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
		return newEmbedder(client, name, &opts).(api.Action)

	case api.ActionTypeModel:
		// Veo models should not be resolved as regular models
		if mt == ModelTypeVeo {
			return nil
		}
		opts := GetModelOptions(name, provider)
		return newModel(client, name, opts).(api.Action)

	case api.ActionTypeBackgroundModel:
		if mt != ModelTypeVeo {
			return nil
		}
		return createVeoBackgroundAction(client, name, provider)

	case api.ActionTypeCheckOperation:
		if mt != ModelTypeVeo {
			return nil
		}
		return createVeoCheckAction(client, name, provider)
	}

	return nil
}

// createVeoBackgroundAction creates a background model action for Veo.
func createVeoBackgroundAction(client *genai.Client, name, provider string) api.Action {
	opts := GetModelOptions(name, provider)
	veoModel := newVeoModel(client, name, opts)
	actionName := api.NewName(provider, name)

	return core.NewAction(actionName, api.ActionTypeBackgroundModel, nil, nil,
		func(ctx context.Context, input *ai.ModelRequest) (*core.Operation[*ai.ModelResponse], error) {
			op, err := veoModel.Start(ctx, input)
			if err != nil {
				return nil, err
			}
			op.Action = api.KeyFromName(api.ActionTypeBackgroundModel, actionName)
			return op, nil
		})
}

// createVeoCheckAction creates a check operation action for Veo.
func createVeoCheckAction(client *genai.Client, name, provider string) api.Action {
	opts := GetModelOptions(name, provider)
	veoModel := newVeoModel(client, name, opts)
	actionName := api.NewName(provider, name)

	return core.NewAction(actionName, api.ActionTypeCheckOperation,
		map[string]any{"description": fmt.Sprintf("Check status of %s operation", name)}, nil,
		func(ctx context.Context, op *core.Operation[*ai.ModelResponse]) (*core.Operation[*ai.ModelResponse], error) {
			updatedOp, err := veoModel.Check(ctx, op)
			if err != nil {
				return nil, err
			}
			updatedOp.Action = api.KeyFromName(api.ActionTypeBackgroundModel, actionName)
			return updatedOp, nil
		})
}
