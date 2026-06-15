// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
)

func TestClassifyModelAntigravity(t *testing.T) {
	if got := ClassifyModel("antigravity-preview-05-2026"); got != ModelTypeAntigravity {
		t.Fatalf("ClassifyModel(antigravity-preview-05-2026) = %v, want %v", got, ModelTypeAntigravity)
	}
}

func TestAntigravityModelTypeDefaults(t *testing.T) {
	if got := ModelTypeAntigravity.ActionType(); got != api.ActionTypeModel {
		t.Fatalf("ModelTypeAntigravity.ActionType() = %v, want %v", got, api.ActionTypeModel)
	}

	supports := ModelTypeAntigravity.DefaultSupports()
	if supports == nil {
		t.Fatal("ModelTypeAntigravity.DefaultSupports() = nil")
	}
	if !supports.Multiturn {
		t.Error("Antigravity should support multiturn")
	}
	if !supports.Media {
		t.Error("Antigravity should support media input")
	}
	if supports.Tools {
		t.Error("Antigravity should not advertise tool support")
	}
	if supports.ToolChoice {
		t.Error("Antigravity should not advertise tool choice support")
	}
	// SystemRole is intentionally true: the Antigravity handler maps system
	// messages to user input itself, and advertising true bypasses the
	// framework's simulateSystemPrompt rewrite (which the model ignores).
	if !supports.SystemRole {
		t.Error("Antigravity should advertise system role so simulateSystemPrompt is bypassed")
	}
	if supports.LongRunning {
		t.Error("Antigravity should not be long-running")
	}
	if !reflect.DeepEqual(supports.Output, []string{"text"}) {
		t.Fatalf("Antigravity output = %v, want [text]", supports.Output)
	}

	// Antigravity is served by the interactions endpoint, so its config is the
	// interactions-shaped AntigravityConfig, not generateContent's config.
	if _, ok := ModelTypeAntigravity.DefaultConfig().(*AntigravityConfig); !ok {
		t.Fatalf("ModelTypeAntigravity.DefaultConfig() = %T, want *AntigravityConfig", ModelTypeAntigravity.DefaultConfig())
	}
}

func TestGetModelOptionsAntigravityKnownModel(t *testing.T) {
	opts := GetModelOptions(antigravityPreview052026, googleAIProvider)

	if opts.Label != "Google AI - Antigravity Preview 05 2026" {
		t.Fatalf("label = %q, want Google AI - Antigravity Preview 05 2026", opts.Label)
	}
	if opts.Stage != ai.ModelStageUnstable {
		t.Fatalf("stage = %v, want %v", opts.Stage, ai.ModelStageUnstable)
	}
	if opts.Supports != &AntigravitySupports {
		t.Fatalf("supports = %#v, want AntigravitySupports", opts.Supports)
	}
	if opts.ConfigSchema == nil {
		t.Fatal("ConfigSchema should be populated for Antigravity")
	}
}

func TestListModelsIncludesAntigravityForGoogleAI(t *testing.T) {
	googleModels, err := listModels(googleAIProvider)
	if err != nil {
		t.Fatalf("listModels(%q) error = %v", googleAIProvider, err)
	}
	if _, ok := googleModels[antigravityPreview052026]; !ok {
		t.Fatalf("Google AI models missing %q", antigravityPreview052026)
	}

	vertexModels, err := listModels(vertexAIProvider)
	if err != nil {
		t.Fatalf("listModels(%q) error = %v", vertexAIProvider, err)
	}
	if _, ok := vertexModels[antigravityPreview052026]; ok {
		t.Fatalf("Vertex AI models unexpectedly included %q", antigravityPreview052026)
	}
}
