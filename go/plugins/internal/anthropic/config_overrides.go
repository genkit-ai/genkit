// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"strings"

	"github.com/invopop/jsonschema"
)

// configOverrides describes per-property metadata layered onto the reflected
// [anthropic.MessageNewParams] schema before it is exposed to the Genkit
// Developer UI. The Anthropic SDK's params structs carry Go doc comments but
// no JSON Schema descriptions, and a few of their fields are owned by Genkit
// primitives and rejected when supplied directly, so we curate that here.
//
// All path application is best-effort: if the upstream SDK renames or removes
// a field, the corresponding entry silently no-ops rather than panicking. The
// cost is that stale entries quietly stop applying, which the schema tests
// catch rather than anything at runtime.
//
// This mirrors the same mechanism in the googlegenai plugin. The two are
// deliberately shaped alike so they can be lifted into one shared helper once
// a third plugin needs it.
type configOverrides struct {
	// descriptions maps a JSON property path to the help text shown as the
	// field's tooltip in the dev UI. Keys may be a top-level property name
	// ("temperature") or a dotted path ("output_config.effort").
	descriptions map[string]string
	// hidden lists JSON property paths to remove from the schema. Same
	// notation as descriptions. Use this for fields a Genkit primitive owns
	// and the plugin rejects at runtime.
	hidden []string
}

// mncOverrides controls dev UI presentation of [anthropic.MessageNewParams].
var mncOverrides = configOverrides{
	descriptions: map[string]string{
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
	hidden: []string{
		// Owned by Genkit primitives; the plugin rejects each of these when
		// set, pointing at the option to use instead.
		"messages",             // ai.WithMessages / ai.WithPrompt
		"system",               // ai.WithSystem
		"model",                // ai.WithModel / ai.WithModelName
		"output_config.format", // ai.WithOutputType / ai.WithOutputSchema
	},
}

// applyConfigOverrides mutates schema in place: hides the managed properties
// and writes descriptions onto the rest. Best-effort, so a path that no longer
// resolves silently no-ops.
//
// A hidden property is replaced by the permissive `true` schema rather than
// deleted. Both hide it from the dev UI, which renders only properties whose
// type it recognizes, but they differ on everything else. The config schema is
// enforced by input validation on every request, and a hidden field must still
// reach the plugin's own check, which names the Genkit option to use instead;
// deleting it would instead fail validation as an unknown property. Deleting
// also forces the parent open with additionalProperties: true to let the value
// back through, which gives up rejecting genuinely unknown fields. Replacing
// keeps additionalProperties: false intact, so a typo like maxTokens for
// max_tokens is still caught.
func applyConfigOverrides(schema *jsonschema.Schema, o configOverrides) {
	if schema == nil || schema.Properties == nil {
		return
	}
	hideTop := make(map[string]struct{})
	for _, path := range o.hidden {
		steps := parsePath(path)
		if len(steps) == 1 {
			hideTop[steps[0]] = struct{}{}
		}
		hideAtPath(schema, steps)
	}
	// A hidden property that the SDK marks required would otherwise leave the
	// schema demanding a field the dev UI cannot show and the plugin refuses.
	if len(hideTop) > 0 && len(schema.Required) > 0 {
		kept := schema.Required[:0]
		for _, r := range schema.Required {
			if _, drop := hideTop[r]; !drop {
				kept = append(kept, r)
			}
		}
		schema.Required = kept
	}
	for path, desc := range o.descriptions {
		if target := schemaAtPath(schema, parsePath(path)); target != nil {
			target.Description = desc
		}
	}
}

// stripParamObjArtifact removes the "any" property the reflector emits for
// every SDK params struct.
//
// The SDK embeds param.APIObject in each of them, which in turn embeds an
// anonymous `any` used to carry the raw message a value was decoded from. It
// is machinery rather than a request field, but it reflects as a property
// named "any" on every object at every depth, so the dev UI renders a junk
// field on each one. Nothing sends it and dropping it costs nothing.
func stripParamObjArtifact(schema *jsonschema.Schema) {
	if schema == nil {
		return
	}
	if schema.Properties != nil {
		schema.Properties.Delete("any")
		for pair := schema.Properties.Oldest(); pair != nil; pair = pair.Next() {
			stripParamObjArtifact(pair.Value)
		}
	}
	stripParamObjArtifact(schema.Items)
	for _, s := range schema.AnyOf {
		stripParamObjArtifact(s)
	}
	for _, s := range schema.OneOf {
		stripParamObjArtifact(s)
	}
	for _, s := range schema.AllOf {
		stripParamObjArtifact(s)
	}
}

// parsePath splits an override path into navigation steps. Each step is either
// a property name or the literal "[]" meaning "descend into an array's item
// schema". Examples:
//
//	"temperature"           -> ["temperature"]
//	"output_config.format"  -> ["output_config", "format"]
//	"tools[].name"          -> ["tools", "[]", "name"]
func parsePath(path string) []string {
	var steps []string
	for _, tok := range strings.Split(path, ".") {
		if name := strings.TrimSuffix(tok, "[]"); name != tok {
			steps = append(steps, name, "[]")
		} else {
			steps = append(steps, tok)
		}
	}
	return steps
}

// schemaAtPath descends a schema, walking Items for "[]" steps and Properties
// for named ones. Returns nil if any step does not resolve, which callers
// treat as a no-op rather than an error.
func schemaAtPath(schema *jsonschema.Schema, steps []string) *jsonschema.Schema {
	cur := schema
	for _, step := range steps {
		if cur == nil {
			return nil
		}
		if step == "[]" {
			cur = cur.Items
			continue
		}
		if cur.Properties == nil {
			return nil
		}
		next, ok := cur.Properties.Get(step)
		if !ok {
			return nil
		}
		cur = next
	}
	return cur
}

// hideAtPath replaces the leaf property at the given path with the permissive
// `true` schema, which accepts any value and declares no type. Reports whether
// there was a property to replace, so a stale path is a no-op rather than an
// error.
func hideAtPath(schema *jsonschema.Schema, steps []string) bool {
	if len(steps) == 0 {
		return false
	}
	leaf := steps[len(steps)-1]
	if leaf == "[]" {
		return false
	}
	parent := schemaAtPath(schema, steps[:len(steps)-1])
	if parent == nil || parent.Properties == nil {
		return false
	}
	if _, ok := parent.Properties.Get(leaf); !ok {
		return false
	}
	parent.Properties.Set(leaf, jsonschema.TrueSchema)
	return true
}
