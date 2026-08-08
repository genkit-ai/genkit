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

package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	openaiGo "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestInitIncludesImageModels(t *testing.T) {
	plugin := &OpenAI{APIKey: "test"}
	actions := plugin.Init(context.Background())

	for _, name := range []string{"openai/dall-e-3", "openai/gpt-image-1"} {
		var found api.Action
		for _, action := range actions {
			if action.Name() == name {
				found = action
				break
			}
		}
		if found == nil {
			t.Errorf("Init() did not register %q", name)
			continue
		}
		modelMetadata, ok := found.Desc().Metadata["model"].(map[string]any)
		if !ok {
			t.Errorf("%s metadata has no model entry", name)
			continue
		}
		supports, ok := modelMetadata["supports"].(map[string]any)
		if !ok {
			t.Errorf("%s metadata has no supports entry", name)
			continue
		}
		output, ok := supports["output"].([]string)
		if !ok || len(output) != 1 || output[0] != "media" {
			t.Errorf("%s supports.output = %#v, want [media]", name, supports["output"])
		}
	}
}

func TestImageModelRef(t *testing.T) {
	config := &compat_oai.ImageGenerationConfig{Quality: openaiGo.ImageGenerateParamsQualityHD}
	ref := ImageModelRef("dall-e-3", config)
	if got := ref.Name(); got != "openai/dall-e-3" {
		t.Errorf("Name() = %q, want %q", got, "openai/dall-e-3")
	}
	if got := ref.Config(); got != config {
		t.Errorf("Config() = %#v, want the supplied config", got)
	}
}

func TestImageConfigSchemasMatchModelCapabilities(t *testing.T) {
	t.Run("DALL-E 3", func(t *testing.T) {
		opts := supportedImageModels[openaiGo.ImageModelDallE3]
		if !slices.Equal(opts.Versions, []string{"dall-e-3"}) {
			t.Errorf("Versions = %#v, want [dall-e-3]", opts.Versions)
		}
		if got := opts.ConfigSchema["additionalProperties"]; got != false {
			t.Errorf("additionalProperties = %v, want false", got)
		}
		properties := opts.ConfigSchema["properties"].(map[string]any)
		assertIntegerSchema(t, properties, "n", 1, 10, 1)
		assertEnumSchema(t, properties, "size", "1024x1024", "1792x1024", "1024x1792")
		assertEnumSchema(t, properties, "quality", "standard", "hd")
		assertEnumSchema(t, properties, "style", "vivid", "natural")
		assertEnumSchema(t, properties, "response_format", "b64_json", "url")
		responseFormat := properties["response_format"].(map[string]any)
		if got := responseFormat["default"]; got != "b64_json" {
			t.Errorf("response_format.default = %v, want b64_json", got)
		}
		for _, unsupported := range []string{"background", "moderation", "output_compression", "output_format"} {
			if properties[unsupported] != nil {
				t.Errorf("DALL-E config schema includes GPT Image-only %s", unsupported)
			}
		}
	})

	t.Run("GPT Image 1", func(t *testing.T) {
		opts := supportedImageModels[openaiGo.ImageModelGPTImage1]
		if !slices.Equal(opts.Versions, []string{"gpt-image-1"}) {
			t.Errorf("Versions = %#v, want [gpt-image-1]", opts.Versions)
		}
		if got := opts.ConfigSchema["additionalProperties"]; got != false {
			t.Errorf("additionalProperties = %v, want false", got)
		}
		properties := opts.ConfigSchema["properties"].(map[string]any)
		assertIntegerSchema(t, properties, "n", 1, 10, 1)
		assertEnumSchema(t, properties, "size", "1024x1024", "1536x1024", "1024x1536", "auto")
		assertEnumSchema(t, properties, "quality", "low", "medium", "high")
		assertEnumSchema(t, properties, "background", "transparent", "opaque", "auto")
		assertEnumSchema(t, properties, "moderation", "low", "auto")
		assertIntegerSchema(t, properties, "output_compression", 1, 100, nil)
		assertEnumSchema(t, properties, "output_format", "png", "jpeg", "webp")
		if properties["style"] != nil {
			t.Error("GPT Image config schema includes DALL-E-only style")
		}
		if properties["response_format"] != nil {
			t.Error("GPT Image config schema includes unsupported response_format")
		}
	})
}

func assertEnumSchema(t *testing.T, properties map[string]any, name string, want ...string) {
	t.Helper()
	property, ok := properties[name].(map[string]any)
	if !ok {
		t.Fatalf("%s schema = %#v, want an object", name, properties[name])
	}
	got, ok := property["enum"].([]string)
	if !ok || !slices.Equal(got, want) {
		t.Errorf("%s.enum = %#v, want %#v", name, property["enum"], want)
	}
}

func assertIntegerSchema(t *testing.T, properties map[string]any, name string, minimum, maximum int, defaultValue any) {
	t.Helper()
	property, ok := properties[name].(map[string]any)
	if !ok {
		t.Fatalf("%s schema = %#v, want an object", name, properties[name])
	}
	if got := property["type"]; got != "integer" {
		t.Errorf("%s.type = %v, want integer", name, got)
	}
	if got := property["minimum"]; got != minimum {
		t.Errorf("%s.minimum = %v, want %d", name, got, minimum)
	}
	if got := property["maximum"]; got != maximum {
		t.Errorf("%s.maximum = %v, want %d", name, got, maximum)
	}
	if got := property["default"]; got != defaultValue {
		t.Errorf("%s.default = %v, want %v", name, got, defaultValue)
	}
}

func TestListActionsClassifiesImageModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("request path = %q, want %q", r.URL.Path, "/models")
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"object":"list","data":[{"id":"dall-e-3","object":"model","created":1,"owned_by":"openai"}],"has_more":false}`)
	}))
	defer server.Close()

	plugin := &OpenAI{
		APIKey: "test",
		Opts:   []option.RequestOption{option.WithBaseURL(server.URL)},
	}
	plugin.Init(context.Background())
	descriptions := plugin.ListActions(context.Background())
	if len(descriptions) != 1 {
		t.Fatalf("len(ListActions()) = %d, want 1", len(descriptions))
	}
	description := descriptions[0]
	if description.Name != "openai/dall-e-3" {
		t.Errorf("Name = %q, want %q", description.Name, "openai/dall-e-3")
	}
	properties, ok := description.InputSchema["properties"].(map[string]any)
	if !ok || properties["config"] == nil {
		t.Errorf("image action input schema has no config: %#v", description.InputSchema)
	}
}

func TestResolveActionClassifiesImageModels(t *testing.T) {
	plugin := &OpenAI{APIKey: "test"}
	plugin.Init(context.Background())
	action := plugin.ResolveAction(api.ActionTypeModel, "gpt-image-custom")
	if action == nil {
		t.Fatal("ResolveAction() returned nil")
	}
	modelMetadata := action.Desc().Metadata["model"].(map[string]any)
	supports := modelMetadata["supports"].(map[string]any)
	if supports["media"] != false {
		t.Errorf("supports.media = %v, want false", supports["media"])
	}
	if output := supports["output"].([]string); len(output) != 1 || output[0] != "media" {
		t.Errorf("supports.output = %#v, want [media]", output)
	}

	model, ok := action.(ai.Model)
	if !ok {
		t.Errorf("resolved action type = %T, want ai.Model", action)
		return
	}
	if got := model.Name(); got != "openai/gpt-image-custom" {
		t.Errorf("Name() = %q, want %q", got, "openai/gpt-image-custom")
	}
}
