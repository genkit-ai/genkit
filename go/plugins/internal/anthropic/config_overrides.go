// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal"
)

// The presentation layered onto the reflected [anthropic.MessageNewParams]
// schema before it reaches the dev UI: the SDK's params structs carry Go doc
// comments but no schema descriptions, and a few of their fields are owned by a
// Genkit option and rejected when supplied directly. See [internal.SchemaOverrides]
// for the path notation; a stale entry is a no-op, which
// TestConfigOverridePathsResolve catches.

// mncOverrides controls dev UI presentation of [anthropic.MessageNewParams].
var mncOverrides = internal.SchemaOverrides{
	Descriptions: map[string]string{
		"max_tokens": "Maximum number of tokens to generate before stopping. The model may stop on its own before reaching it, and the ceiling differs by model. Defaults to 4096 when unset.",
		// The parameter deprecation is worth stating here: the API rejects a
		// non-default value on Claude 4.7 and later rather than ignoring it.
		"temperature":            "Amount of randomness injected into the response, from 0.0 to 1.0. Lower is better for analytical and multiple-choice work, higher for creative work. Deprecated on Claude Opus 4.7 and later, which reject any value set; steer those models with the prompt instead.",
		"top_k":                  "Sample only from the top K options for each token, dropping the low-probability long tail. Advanced use only; temperature is usually enough. Deprecated on Claude Opus 4.7 and later, which reject any value set.",
		"top_p":                  "Nucleus sampling: consider tokens in decreasing probability order until the cumulative probability reaches this value. Set either temperature or top_p, not both. Advanced use only. Deprecated on Claude Opus 4.7 and later, which reject any value set.",
		"stop_sequences":         "Custom text sequences that stop generation. When one is emitted the response carries a stop_reason of stop_sequence and names the sequence that matched.",
		"service_tier":           "Whether the request may use priority capacity when it is available (auto) or standard capacity only (standard_only).",
		"container":              "Container identifier, used to reuse a container across requests.",
		"inference_geo":          "Geographic region to run inference in. Defaults to the workspace's configured region.",
		"thinking":               "Extended thinking controls. When enabled the response carries thinking blocks showing the model's reasoning before its answer. Requires a budget of at least 1024 tokens, which counts against max_tokens.",
		"thinking.budget_tokens": "Tokens the model may spend on internal reasoning. Must be at least 1024 and less than max_tokens. Larger budgets allow more thorough analysis at higher cost.",
		"tool_choice":            "How the model should use the tools available to it: a specific tool, any tool, its own choice, or none.",
		// Custom tools are appended by the plugin from ai.WithTools, so this
		// field is only useful for the server-side ones.
		"tools":                "Server-side tools to make available to the model: web search, web fetch, code execution, text editor, memory, and so on. Custom function tools must be registered with ai.WithTools() so the Genkit runtime can execute them and feed the results back.",
		"metadata":             "Metadata describing the request.",
		"metadata.user_id":     "Opaque identifier for the end user, which Anthropic may use to detect abuse. Use a UUID or hash, never identifying information such as a name, email address, or phone number.",
		"output_config":        "Controls the shape of the model's output.",
		"output_config.effort": "How much effort the model spends producing the output: low, medium, high, or max.",
	},
	Hidden: []string{
		// Owned by Genkit primitives; the plugin rejects each of these when
		// set, pointing at the option to use instead.
		"messages",             // ai.WithMessages / ai.WithPrompt
		"system",               // ai.WithSystem
		"model",                // ai.WithModel / ai.WithModelName
		"output_config.format", // ai.WithOutputType / ai.WithOutputSchema
	},
}

// stripParamObjArtifact removes the "any" property the reflector emits for
// every SDK params struct.
//
// The SDK embeds param.APIObject in each of them, which in turn embeds an
// anonymous `any` used to carry the raw message a value was decoded from. It is
// machinery rather than a request field, but it reflects as a property named
// "any" on every object at every depth, so the dev UI would render a junk field
// on each one.
func stripParamObjArtifact(schema map[string]any) {
	drop := func(s map[string]any) map[string]any {
		if props, ok := s["properties"].(map[string]any); ok {
			delete(props, "any")
		}
		return s
	}
	drop(schema)
	base.WalkSubschemas(schema, drop)
}
