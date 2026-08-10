// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
)

// configProps returns the advertised config schema's property map.
func configProps(t *testing.T) map[string]any {
	t.Helper()
	desc := NewModel(anthropic.Client{}, "anthropic", "claude-opus-4-8", "", ai.ModelOptions{}).Desc()
	cfg := desc.InputSchema["properties"].(map[string]any)["config"].(map[string]any)
	first := cfg["anyOf"].([]any)[0].(map[string]any)
	return first["properties"].(map[string]any)
}

// TestManagedFieldsAreHiddenButAccepted pins the two halves of hiding a field
// that a Genkit primitive owns: the dev UI must not offer it, and setting it in
// code must still reach the plugin's error rather than failing as an unknown
// property.
//
// A hidden field is replaced by the permissive `true` schema rather than
// deleted. The dev UI renders only properties whose type it recognizes, so a
// typeless one is skipped, while the schema still accepts the value. Deleting
// would force additionalProperties open on the parent to let the value back
// through, which is what gives up the unknown-field rejection below.
func TestManagedFieldsAreHiddenButAccepted(t *testing.T) {
	props := configProps(t)

	for _, path := range mncOverrides.hidden {
		name, _, nested := strings.Cut(path, ".")
		if nested {
			continue // checked separately below
		}
		got, ok := props[name]
		if !ok {
			t.Errorf("%s is absent from the schema, want the permissive true schema", name)
			continue
		}
		if got != true {
			t.Errorf("%s = %v, want true (typeless, so the dev UI skips it)", name, got)
		}
	}

	// output_config keeps effort and hides only format.
	oc := props["output_config"].(map[string]any)
	ocProps := oc["properties"].(map[string]any)
	if got := ocProps["format"]; got != true {
		t.Errorf("output_config.format = %v, want true", got)
	}
	if _, ok := ocProps["effort"].(map[string]any)["type"]; !ok {
		t.Error("output_config.effort lost its type; it is not managed by Genkit")
	}
}

// TestUnknownFieldsStillRejected guards the property that replacing rather than
// deleting exists to preserve. A misspelled field is the common mistake, and
// the SDK's wire names are snake_case, so camelCase must not slip through.
func TestUnknownFieldsStillRejected(t *testing.T) {
	desc := NewModel(anthropic.Client{}, "anthropic", "claude-opus-4-8", "", ai.ModelOptions{}).Desc()
	for _, cfg := range []map[string]any{
		{"nope": 1},
		{"maxTokens": 10},
	} {
		if err := validateConfig(t, desc.InputSchema, cfg); err == nil {
			t.Errorf("config %v was accepted, want it rejected as an unknown property", cfg)
		}
	}
}

// TestManagedConfigRejected pins that each field a Genkit primitive owns is
// refused with a message naming the option to use, rather than being silently
// overwritten while the request is built.
func TestManagedConfigRejected(t *testing.T) {
	tests := []struct {
		name   string
		config anthropic.MessageNewParams
		want   string
	}{
		{
			"system",
			anthropic.MessageNewParams{System: []anthropic.TextBlockParam{{Text: "be terse"}}},
			"ai.WithSystem()",
		},
		{
			"messages",
			anthropic.MessageNewParams{Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
			}},
			"ai.WithMessages()",
		},
		{
			"model",
			anthropic.MessageNewParams{Model: "claude-opus-4-8"},
			"ai.WithModel()",
		},
		{
			"output format",
			anthropic.MessageNewParams{OutputConfig: anthropic.OutputConfigParam{
				Format: anthropic.JSONOutputFormatParam{Schema: map[string]any{"type": "object"}},
			}},
			"ai.WithOutputType()",
		},
		{
			"custom function tool",
			anthropic.MessageNewParams{Tools: []anthropic.ToolUnionParam{
				{OfTool: &anthropic.ToolParam{Name: "myTool"}},
			}},
			"ai.WithTools()",
		},
	}

	desc := NewModel(anthropic.Client{}, "anthropic", "claude-opus-4-8", "", ai.ModelOptions{}).Desc()
	req := &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("hello"))}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The value must survive the action boundary, or the plugin's
			// message never gets the chance to explain itself.
			if err := validateConfig(t, desc.InputSchema, asMap(t, tt.config)); err != nil {
				t.Fatalf("config rejected before reaching the plugin: %v", err)
			}
			_, err := toAnthropicRequest("anthropic", req, tt.config)
			if err == nil {
				t.Fatal("config accepted, want it refused")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want it to name %s", err, tt.want)
			}
		})
	}
}

// TestConfigDescriptionsApplied pins that the curated help text reaches the
// schema. The SDK carries Go doc comments but no JSON Schema descriptions, so
// without this the dev UI shows every field bare.
func TestConfigDescriptionsApplied(t *testing.T) {
	props := configProps(t)

	if got := props["temperature"].(map[string]any)["description"]; got == nil {
		t.Error("temperature has no description")
	} else if !strings.Contains(got.(string), "4.7") {
		// The deprecation is the part a caller most needs: the API rejects a
		// value here rather than ignoring it.
		t.Errorf("temperature description does not mention the 4.7 deprecation: %q", got)
	}

	// A nested path must apply too, not just top-level ones.
	oc := props["output_config"].(map[string]any)["properties"].(map[string]any)
	if got := oc["effort"].(map[string]any)["description"]; got == nil {
		t.Error("output_config.effort has no description")
	}

	// Every description path must still resolve, or the entry is dead weight
	// that silently stopped applying when the SDK renamed something.
	desc := NewModel(anthropic.Client{}, "anthropic", "claude-opus-4-8", "", ai.ModelOptions{}).Desc()
	full := desc.InputSchema["properties"].(map[string]any)["config"].(map[string]any)
	blob, err := json.Marshal(full)
	if err != nil {
		t.Fatal(err)
	}
	for path, text := range mncOverrides.descriptions {
		if !strings.Contains(string(blob), text) {
			t.Errorf("description for %q never landed; the path no longer resolves", path)
		}
	}
}

// TestParamObjArtifactStripped pins that the SDK's embedded param.APIObject
// does not leak into the schema. It reflects as a property named "any" on every
// object at every depth, which the dev UI would render as a junk field on each
// one.
func TestParamObjArtifactStripped(t *testing.T) {
	desc := NewModel(anthropic.Client{}, "anthropic", "claude-opus-4-8", "", ai.ModelOptions{}).Desc()
	blob, err := json.Marshal(desc.InputSchema)
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(string(blob), `"any":`); n != 0 {
		t.Errorf(`schema carries %d "any" properties, want none`, n)
	}
}

func asMap(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}
