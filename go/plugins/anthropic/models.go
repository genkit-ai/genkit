// Copyright 2025 Google LLC
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

package anthropic

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal"
	ant "github.com/firebase/genkit/go/plugins/internal/anthropic"
)

// Capability sets shared by the entries below. claudeSupports is what every
// Claude model does; structuredClaudeSupports adds JSON output and native
// constrained generation.
//
// Only models on Anthropic's Structured Outputs list may claim Constrained.
// That list is narrower than the catalog, and the request path sends
// output_config solely when the model claims support, so a model claiming it
// wrongly would have output_config rejected while Genkit had already dropped
// the schema instructions from the prompt. See
// https://platform.claude.com/docs/en/build-with-claude/structured-outputs.
var (
	claudeSupports = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		ToolChoice: true,
		SystemRole: true,
		Media:      true,
	}
	structuredClaudeSupports = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		ToolChoice:  true,
		SystemRole:  true,
		Media:       true,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportAll,
	}
)

// structuredModel builds the ModelOptions for a Claude model on Anthropic's
// Structured Outputs list, so it can be constrained natively. The label prefix
// is shared by every entry, so it lives here rather than in each one.
func structuredModel(label string) ai.ModelOptions {
	return ai.ModelOptions{
		Label:    internal.ProviderLabel(anthropicLabelPrefix, label),
		Supports: &structuredClaudeSupports,
		Versions: []string{},
		Stage:    ai.ModelStageStable,
	}
}

// supportedModels curates capabilities for well-known Claude models, mirroring
// the JS plugin's KNOWN_MODELS. It is not the set of usable models: any Claude
// model resolves on demand and takes [dynamicModelOptions], so an ID absent
// here still works. Both ListActions and ResolveAction look it up through
// modelOptions.
//
// Catalog: https://docs.anthropic.com/en/docs/about-claude/models/all-models
// Retirements: https://platform.claude.com/docs/en/about-claude/model-deprecations
var supportedModels = map[string]ai.ModelOptions{
	"claude-fable-5":    structuredModel("Claude Fable 5"),
	"claude-opus-5":     structuredModel("Claude Opus 5"),
	"claude-sonnet-5":   structuredModel("Claude Sonnet 5"),
	"claude-opus-4-8":   structuredModel("Claude Opus 4.8"),
	"claude-opus-4-7":   structuredModel("Claude Opus 4.7"),
	"claude-opus-4-6":   structuredModel("Claude Opus 4.6"),
	"claude-opus-4-5":   structuredModel("Claude Opus 4.5"),
	"claude-sonnet-4-6": structuredModel("Claude Sonnet 4.6"),
	"claude-sonnet-4-5": structuredModel("Claude Sonnet 4.5"),
	"claude-haiku-4-5":  structuredModel("Claude Haiku 4.5"),
}

// dynamicModelOptions is advertised for Claude models that resolve dynamically
// rather than appearing in supportedModels. newModel fills in the label from
// the model ID. It claims no constrained generation: a model Anthropic serves
// but this list does not name may predate Structured Outputs, and schema
// instructions in the prompt work on every Claude model.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &claudeSupports,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// listModels returns a list of model names supported by the Anthropic client
func listModels(ctx context.Context, client *anthropic.Client) ([]string, error) {
	iter := client.Models.ListAutoPaging(ctx, anthropic.ModelListParams{})
	models := []string{}

	for iter.Next() {
		m := iter.Current()
		models = append(models, m.ID)
	}

	if err := iter.Err(); err != nil {
		return nil, ant.WrapAPIError(err)
	}

	return models, nil
}
