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
)

// Capability sets shared by the entries below, in widening order.
// advancedClaudeSupports mirrors the JS plugin's ADVANCED_MODEL_INFO: capable
// Claude models that additionally support JSON output.
//
// Only structuredClaudeSupports advertises constrained generation, and only
// the models on Anthropic's Structured Outputs list get it. That list is
// narrower than the catalog, and the request path sends output_config solely
// when the model claims support, so a model claiming it wrongly would have
// output_config rejected while Genkit had already dropped the schema
// instructions from the prompt. See
// https://platform.claude.com/docs/en/build-with-claude/structured-outputs.
var (
	claudeSupports = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		ToolChoice: true,
		SystemRole: true,
		Media:      true,
	}
	advancedClaudeSupports = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		ToolChoice: true,
		SystemRole: true,
		Media:      true,
		Output:     []string{"text", "json"},
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

// advancedModel builds the ModelOptions for a known, JSON-capable Claude model
// with the given display label. The label prefix is shared by every entry, so
// it lives here rather than in each one.
func advancedModel(label string) ai.ModelOptions {
	return modelWith(&advancedClaudeSupports, label)
}

// structuredModel builds the ModelOptions for a Claude model that also appears
// on Anthropic's Structured Outputs list, so it can be constrained natively.
func structuredModel(label string) ai.ModelOptions {
	return modelWith(&structuredClaudeSupports, label)
}

func modelWith(supports *ai.ModelSupports, label string) ai.ModelOptions {
	return ai.ModelOptions{
		Label:    anthropicLabelPrefix + " - " + label,
		Supports: supports,
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
var supportedModels = map[string]ai.ModelOptions{
	"claude-fable-5":  structuredModel("Claude Fable 5"),
	"claude-opus-5":   structuredModel("Claude Opus 5"),
	"claude-sonnet-5": structuredModel("Claude Sonnet 5"),
	"claude-opus-4-8": structuredModel("Claude Opus 4.8"),
	"claude-opus-4-7": structuredModel("Claude Opus 4.7"),
	"claude-opus-4-6": structuredModel("Claude Opus 4.6"),
	"claude-opus-4-5": structuredModel("Claude Opus 4.5"),
	// Absent from Anthropic's Structured Outputs list, so JSON output here is
	// produced from schema instructions in the prompt rather than natively.
	"claude-opus-4-1":   advancedModel("Claude Opus 4.1"),
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
		return nil, err
	}

	return models, nil
}
