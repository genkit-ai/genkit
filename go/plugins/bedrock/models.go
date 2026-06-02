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

package bedrock

import (
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// inferenceProfilePrefixes are AWS-recognised cross-region routing prefixes.
// They are passed through to Bedrock as part of ModelId, but stripped here
// when looking up capability metadata in [knownModels].
//
// See: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
var inferenceProfilePrefixes = []string{
	"global.",
	"us-gov.",
	"us.",
	"eu.",
	"apac.",
	"jp.",
	"au.",
}

// stripInferenceProfilePrefix removes a single recognised cross-region prefix
// from modelID. If no prefix matches, modelID is returned unchanged.
func stripInferenceProfilePrefix(modelID string) string {
	for _, p := range inferenceProfilePrefixes {
		if strings.HasPrefix(modelID, p) {
			return strings.TrimPrefix(modelID, p)
		}
	}
	return modelID
}

// modelSupports captures the per-model capability matrix. Multimodal models
// also flip [ai.ModelSupports.Media]; tool-capable models flip Tools and
// ToolChoice. Multiturn + SystemRole + Constrained are universal for the
// Converse API.
type modelSupports struct {
	Tools bool
	Media bool
}

// knownModels maps base Bedrock model IDs (no inference-profile prefix) to
// their capability flags. This list is the source of truth for the default
// [ai.ModelOptions] returned by [defaultModelOptions]. Callers can still
// register arbitrary model IDs by passing explicit opts to [DefineModel].
var knownModels = map[string]modelSupports{
	// Anthropic Claude 3 / 3.5 / 3.7
	"anthropic.claude-3-haiku-20240307-v1:0":    {Tools: true, Media: true},
	"anthropic.claude-3-sonnet-20240229-v1:0":   {Tools: true, Media: true},
	"anthropic.claude-3-opus-20240229-v1:0":     {Tools: true, Media: true},
	"anthropic.claude-3-5-haiku-20241022-v1:0":  {Tools: true, Media: false},
	"anthropic.claude-3-5-sonnet-20240620-v1:0": {Tools: true, Media: true},
	"anthropic.claude-3-5-sonnet-20241022-v2:0": {Tools: true, Media: true},
	"anthropic.claude-3-7-sonnet-20250219-v1:0": {Tools: true, Media: true},
	// Anthropic Claude 4 family
	"anthropic.claude-opus-4-20250514-v1:0":   {Tools: true, Media: true},
	"anthropic.claude-sonnet-4-20250514-v1:0": {Tools: true, Media: true},
	"anthropic.claude-opus-4-1-20250805-v1:0": {Tools: true, Media: true},
	// Amazon Nova
	"amazon.nova-micro-v1:0":   {Tools: true, Media: false},
	"amazon.nova-lite-v1:0":    {Tools: true, Media: true},
	"amazon.nova-pro-v1:0":     {Tools: true, Media: true},
	"amazon.nova-premier-v1:0": {Tools: true, Media: true},
	// Cohere Command R
	"cohere.command-r-v1:0":      {Tools: true, Media: false},
	"cohere.command-r-plus-v1:0": {Tools: true, Media: false},
	// Mistral
	"mistral.mistral-large-2402-v1:0": {Tools: true, Media: false},
	"mistral.mistral-large-2407-v1:0": {Tools: true, Media: false},
	"mistral.mistral-small-2402-v1:0": {Tools: true, Media: false},
	"mistral.pixtral-large-2502-v1:0": {Tools: true, Media: true},
	// AI21 Jamba
	"ai21.jamba-1-5-large-v1:0": {Tools: true, Media: false},
	"ai21.jamba-1-5-mini-v1:0":  {Tools: true, Media: false},
	// Meta Llama 3 / 3.1 / 3.2 / 3.3
	"meta.llama3-8b-instruct-v1:0":     {Tools: true, Media: false},
	"meta.llama3-70b-instruct-v1:0":    {Tools: true, Media: false},
	"meta.llama3-1-8b-instruct-v1:0":   {Tools: true, Media: false},
	"meta.llama3-1-70b-instruct-v1:0":  {Tools: true, Media: false},
	"meta.llama3-1-405b-instruct-v1:0": {Tools: true, Media: false},
	"meta.llama3-2-1b-instruct-v1:0":   {Tools: true, Media: false},
	"meta.llama3-2-3b-instruct-v1:0":   {Tools: true, Media: false},
	"meta.llama3-2-11b-instruct-v1:0":  {Tools: true, Media: true},
	"meta.llama3-2-90b-instruct-v1:0":  {Tools: true, Media: true},
	"meta.llama3-3-70b-instruct-v1:0":  {Tools: true, Media: false},
	// DeepSeek
	"deepseek.r1-v1:0": {Tools: true, Media: false},
	// Writer Palmyra
	"writer.palmyra-x4-v1:0": {Tools: true, Media: false},
	"writer.palmyra-x5-v1:0": {Tools: true, Media: false},
}

// defaultModelOptions returns the capability metadata for a Bedrock model ID.
// Cross-region inference-profile prefixes (e.g. "us.", "eu.") are stripped
// before lookup. If the resulting base ID isn't in [knownModels], a safe
// default with multiturn + tools + media is returned so the model is at least
// callable.
func defaultModelOptions(modelID string) *ai.ModelOptions {
	base := stripInferenceProfilePrefix(modelID)
	caps, ok := knownModels[base]
	stage := ai.ModelStageStable
	if !ok {
		// Unknown model — assume the conservative "modern Converse" baseline,
		// but flag it as unstable so callers know they're off the curated path.
		caps = modelSupports{Tools: true, Media: true}
		stage = ai.ModelStageUnstable
	}
	return &ai.ModelOptions{
		Label: "Bedrock - " + modelID,
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       caps.Tools,
			ToolChoice:  caps.Tools,
			SystemRole:  true,
			Media:       caps.Media,
			Constrained: ai.ConstrainedSupportNone,
		},
		Stage: stage,
	}
}

// defaultEmbedderOptions returns minimal capability metadata for an embedder.
// Multimodal embedders (Titan image, Nova multimodal, Cohere Embed v4) accept
// image input in addition to text; everything else is text-only.
func defaultEmbedderOptions(name string) *ai.EmbedderOptions {
	input := []string{"text"}
	if embedderAcceptsImage(name) {
		input = append(input, "image")
	}
	return &ai.EmbedderOptions{
		Label:    "Bedrock - " + name,
		Supports: &ai.EmbedderSupports{Input: input},
	}
}

// embedderAcceptsImage reports whether an embedder model ID is one of the
// multimodal families that accept image input. Cohere Embed v3 image support
// also works at the wire level but isn't advertised here because the same v3
// model IDs serve text-only callers.
func embedderAcceptsImage(name string) bool {
	switch {
	case strings.Contains(name, "titan-embed-image"):
		return true
	case strings.Contains(name, "multimodal-embed"):
		return true
	case strings.Contains(name, "cohere.embed-v4"):
		return true
	default:
		return false
	}
}
