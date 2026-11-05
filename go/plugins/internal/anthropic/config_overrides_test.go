// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/plugins/internal"
)

// advertisedSchema is the config schema the dev UI reads, before the framework
// widens a copy of it for enforcement.
func advertisedSchema() map[string]any {
	return reflectConfigSchema(anthropic.MessageNewParams{})
}

// modelDesc is the descriptor a Claude model advertises, whose input schema is
// what requests are validated against.
func modelDesc() api.ActionDesc {
	return NewModel(anthropic.Client{}, "anthropic", "claude-opus-4-8", ai.ModelOptions{}).Desc()
}

// TestConfigOverridePathsResolve is the guard that keeps the override map
// honest. Applying a path the SDK no longer has is a silent no-op, so an entry
// left behind by a renamed field would quietly stop describing anything.
func TestConfigOverridePathsResolve(t *testing.T) {
	if stale := mncOverrides.UnresolvedPaths(advertisedSchema()); len(stale) > 0 {
		t.Errorf("override paths no longer in the SDK schema: %v", stale)
	}
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
	schema := advertisedSchema()
	for _, path := range mncOverrides.Hidden {
		if got := internal.SchemaAt(schema, path); got != true {
			t.Errorf("%q = %v, want true (typeless, so the dev UI skips it)", path, got)
		}
	}
	// Hiding is per-field: output_config.format goes, effort stays usable.
	if effort, ok := internal.SchemaAt(schema, "output_config.effort").(map[string]any); !ok || effort["type"] == nil {
		t.Errorf("output_config.effort lost its type; it is not managed by Genkit: %#v", effort)
	}
}

// TestUnknownFieldsStillRejected guards the property that replacing rather than
// deleting exists to preserve. A misspelled field is the common mistake, and
// the SDK's wire names are snake_case, so camelCase must not slip through.
func TestUnknownFieldsStillRejected(t *testing.T) {
	desc := modelDesc()
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

	desc := modelDesc()
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
			// The caller's request is what is wrong, so this must reach the
			// dev UI as a 400 rather than a 500.
			if !errors.Is(err, status.ErrInvalidArgument) {
				t.Errorf("error is not classified ErrInvalidArgument: %v", err)
			}
			if got := status.Of(err); got != status.InvalidArgument {
				t.Errorf("status.Of = %q, want %q", got, status.InvalidArgument)
			}
		})
	}
}

// TestConfigDescriptionsApplied pins that the curated help text reaches the
// schema. The SDK carries Go doc comments but no JSON Schema descriptions, so
// without this the dev UI shows every field bare.
func TestConfigDescriptionsApplied(t *testing.T) {
	schema := advertisedSchema()
	for path, want := range mncOverrides.Descriptions {
		target, ok := internal.SchemaAt(schema, path).(map[string]any)
		if !ok {
			continue // reported by TestConfigOverridePathsResolve
		}
		if got, _ := target["description"].(string); got != want {
			t.Errorf("description for %q\n got: %q\nwant: %q", path, got, want)
		}
	}
	// The 4.7 deprecation is the part a caller most needs: the API rejects a
	// value there rather than ignoring it.
	if !strings.Contains(mncOverrides.Descriptions["temperature"], "4.7") {
		t.Error("the temperature description no longer mentions the 4.7 deprecation")
	}
}

// TestParamObjArtifactStripped pins that the SDK's embedded param.APIObject
// does not leak into the schema. It reflects as a property named "any" on every
// object at every depth, which the dev UI would render as a junk field on each
// one.
func TestParamObjArtifactStripped(t *testing.T) {
	blob, err := json.Marshal(modelDesc().InputSchema)
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
