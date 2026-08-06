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

package compat_oai

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/openai/openai-go"
)

// marshalParams round-trips params through the SDK marshaler so tests assert
// on the wire shape the provider sees.
func marshalParams(t *testing.T, params openai.ChatCompletionNewParams) map[string]any {
	t.Helper()
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var request map[string]any
	if err := json.Unmarshal(data, &request); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	return request
}

// TestRequestConfigApplyVersion pins what the shared config contributes: a
// Version takes over the request's model, and the credential it carries is
// never written onto the request.
func TestRequestConfigApplyVersion(t *testing.T) {
	params := openai.ChatCompletionNewParams{Model: "test-model"}
	RequestConfig{Version: "test-model-2026-01-01", APIKey: "secret-key"}.ApplyVersion(&params)
	request := marshalParams(t, params)

	if got := request["model"]; got != "test-model-2026-01-01" {
		t.Errorf("model = %v, want the pinned version", got)
	}
	for key, value := range request {
		if value == "secret-key" {
			t.Errorf("request carries the API key under %q", key)
		}
	}

	// An unset version leaves the model the request was built with.
	params = openai.ChatCompletionNewParams{Model: "test-model"}
	RequestConfig{}.ApplyVersion(&params)
	if params.Model != "test-model" {
		t.Errorf("model = %q, want it untouched by a version-less config", params.Model)
	}
}

// TestChatConfigZeroLeavesParamsUntouched pins that an absent config imposes
// nothing: fields the caller did not set stay unset on the request instead of
// arriving as zeroes.
func TestChatConfigZeroLeavesParamsUntouched(t *testing.T) {
	var params openai.ChatCompletionNewParams
	testChatConfig{}.ApplyToChatCompletion(&params)

	if !reflect.DeepEqual(params, openai.ChatCompletionNewParams{}) {
		t.Errorf("zero config wrote fields onto the params: %+v", params)
	}
}

// TestEmbeddingConfigApply pins the embedder config contract, including that
// the encoding format defaults to float and only changes when set.
func TestEmbeddingConfigApply(t *testing.T) {
	params := openai.EmbeddingNewParams{
		Model:          "text-embedding-3-small",
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}
	EmbeddingConfig{}.applyToEmbedding(&params)
	if params.EncodingFormat != openai.EmbeddingNewParamsEncodingFormatFloat {
		t.Errorf("EncodingFormat = %q, want the float default preserved", params.EncodingFormat)
	}
	if params.Dimensions.Valid() {
		t.Error("Dimensions set by the zero config, want unset")
	}

	EmbeddingConfig{
		Dimensions:     256,
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatBase64,
		User:           "user-1",
	}.applyToEmbedding(&params)
	if got := params.Dimensions.Or(0); got != 256 {
		t.Errorf("Dimensions = %d, want 256", got)
	}
	if params.EncodingFormat != openai.EmbeddingNewParamsEncodingFormatBase64 {
		t.Errorf("EncodingFormat = %q, want base64", params.EncodingFormat)
	}
	if got := params.User.Or(""); got != "user-1" {
		t.Errorf("User = %q, want user-1", got)
	}
}

// TestSDKConfigSchema pins that models taking the SDK params as config
// advertise the SDK's own wire fields, with the Opt wrappers and the stop
// union mapped to the shapes they marshal to.
func TestSDKConfigSchema(t *testing.T) {
	schema := sdkConfigSchema()
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("sdkConfigSchema has no properties: %v", schema)
	}

	for key, wantType := range map[string]string{
		"temperature":  "number",
		"max_tokens":   "integer",
		"top_p":        "number",
		"logprobs":     "boolean",
		"top_logprobs": "integer",
		"seed":         "integer",
	} {
		prop, ok := props[key].(map[string]any)
		if !ok {
			t.Errorf("property %q missing", key)
			continue
		}
		if got := prop["type"]; got != wantType {
			t.Errorf("property %q type = %v, want %q", key, got, wantType)
		}
	}

	stop, ok := props["stop"].(map[string]any)
	if !ok {
		t.Fatalf("stop property missing")
	}
	if _, ok := stop["anyOf"].([]any); !ok {
		t.Errorf("stop schema = %v, want anyOf of string and string array", stop)
	}
}

// testChatConfig declares a provider's fields the way plugin packages do: the
// shared request settings, two fields the SDK models, and one that rides as an
// extra field.
type testChatConfig struct {
	RequestConfig
	Temperature     *float64 `json:"temperature,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	EnableSearch    *bool    `json:"enableSearch,omitempty"`
}

func (c testChatConfig) ApplyToChatCompletion(params *openai.ChatCompletionNewParams) {
	c.ApplyVersion(params)
	if c.Temperature != nil {
		params.Temperature = openai.Float(*c.Temperature)
	}
	if c.MaxOutputTokens > 0 {
		params.MaxTokens = openai.Int(int64(c.MaxOutputTokens))
	}
	if c.EnableSearch != nil {
		AddExtraFields(params, map[string]any{"enable_search": *c.EnableSearch})
	}
}

// TestNewChatModelDescriptor pins what a custom-config model advertises: the
// flattened camelCase schema of the provider's config type (plus the
// framework's version key), and a label derived from the provider when the
// options carry none.
func TestNewChatModelDescriptor(t *testing.T) {
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "test-key"}
	o.Init(context.Background())

	desc := NewChatModel[testChatConfig](o, "test-model", ai.ModelOptions{}).Desc()
	if desc.Name != "testprovider/test-model" {
		t.Errorf("name = %q, want testprovider/test-model", desc.Name)
	}

	model, ok := desc.Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing, got %v", desc.Metadata)
	}
	if got := model["label"]; got != "testprovider - test-model" {
		t.Errorf("label = %v, want %q", got, "testprovider - test-model")
	}

	schema, ok := model["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions missing, got %v", model["customOptions"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions has no properties: %v", schema)
	}
	for _, key := range []string{"temperature", "maxOutputTokens", "version", "enableSearch"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}
	if props["max_tokens"] != nil {
		t.Error("config schema advertises SDK wire names, want the Genkit camelCase contract")
	}
	// A config declares only what its provider accepts, so fields it left out
	// must not reach the schema the Dev UI and callers program against.
	for _, key := range []string{"topP", "stopSequences", "frequencyPenalty", "presencePenalty", "logProbs", "topLogProbs"} {
		if props[key] != nil {
			t.Errorf("config schema advertises %q, which the config does not declare", key)
		}
	}
}

// TestConfigValidationAtBoundary pins the validation contract the framework
// enforces on every request against the schemas these models advertise: a
// partial typed config (SDK or provider-defined) validates, a camelCase map
// validates, and a map speaking wire names instead of the camelCase contract
// is rejected rather than silently dropped.
func TestConfigValidationAtBoundary(t *testing.T) {
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "test-key"}
	o.Init(context.Background())

	req := func(config any) *ai.ModelRequest {
		return &ai.ModelRequest{
			Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("hi"))},
			Config:   config,
		}
	}

	sdkSchema := newSDKModel(o.client, "testprovider", "sdk-model", ai.ModelOptions{}).Desc().InputSchema
	if err := base.ValidateValue(req(openai.ChatCompletionNewParams{Temperature: openai.Float(0.5)}), sdkSchema); err != nil {
		t.Errorf("partial typed SDK config rejected at the boundary: %v", err)
	}
	if err := base.ValidateValue(req(map[string]any{"max_tokens": "lots"}), sdkSchema); err == nil {
		t.Error("expected a mistyped max_tokens to be rejected")
	}

	chatSchema := NewChatModel[testChatConfig](o, "chat-model", ai.ModelOptions{}).Desc().InputSchema
	typed := testChatConfig{
		Temperature:     openai.Ptr(0.0),
		MaxOutputTokens: 5,
		EnableSearch:    openai.Ptr(true),
	}
	if err := base.ValidateValue(req(typed), chatSchema); err != nil {
		t.Errorf("typed provider config rejected at the boundary: %v", err)
	}
	if err := base.ValidateValue(req(map[string]any{"temperature": 0.2, "enableSearch": true, "version": "v"}), chatSchema); err != nil {
		t.Errorf("camelCase map config rejected at the boundary: %v", err)
	}
	if err := base.ValidateValue(req(map[string]any{"enable_search": true}), chatSchema); err == nil {
		t.Error("wire-name map config accepted, want the camelCase contract enforced")
	}
}

// TestAPIKeyNeverSerializes pins the credential contract: a request API key
// set on a config is invisible to every serialized surface, which is what
// keeps it out of the advertised schema, recorded traces, and the outgoing
// request body.
func TestAPIKeyNeverSerializes(t *testing.T) {
	cfg := testChatConfig{
		RequestConfig: RequestConfig{APIKey: "secret-key"},
		Temperature:   openai.Ptr(0.5),
	}

	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if strings.Contains(string(data), "secret-key") {
		t.Errorf("config marshal leaks the API key: %s", data)
	}

	var params openai.ChatCompletionNewParams
	cfg.ApplyToChatCompletion(&params)
	request := marshalParams(t, params)
	for key := range request {
		if key == "apiKey" || key == "api_key" {
			t.Errorf("request body carries the API key under %q", key)
		}
	}

	o := &OpenAICompatible{Provider: "testprovider", APIKey: "test-key"}
	o.Init(context.Background())
	model, _ := NewChatModel[testChatConfig](o, "keyed-model", ai.ModelOptions{}).Desc().Metadata["model"].(map[string]any)
	schema, _ := model["customOptions"].(map[string]any)
	props, _ := schema["properties"].(map[string]any)
	if props["apiKey"] != nil {
		t.Error("config schema advertises apiKey, want the credential kept out of serialized configs")
	}

	// The override reaches the model through the promoted RequestAPIKey,
	// which [ChatConfig] requires.
	var chatCfg ChatConfig = cfg
	if got := chatCfg.RequestAPIKey(); got != "secret-key" {
		t.Errorf("RequestAPIKey() = %q, want the key the config carries", got)
	}
}

// TestClientForKey pins that an empty key reuses the plugin's client and a
// set key derives a request-scoped one without mutating the plugin's options.
func TestClientForKey(t *testing.T) {
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "plugin-key"}
	o.Init(context.Background())
	optsLen := len(o.Opts)

	if got := o.clientForKey(""); got != o.client {
		t.Error("clientForKey(\"\") built a new client, want the plugin's")
	}
	if got := o.clientForKey("override"); got == o.client {
		t.Error("clientForKey(override) returned the plugin's client, want a request-scoped one")
	}
	if len(o.Opts) != optsLen {
		t.Errorf("plugin options grew from %d to %d, want them untouched", optsLen, len(o.Opts))
	}
}

// TestChatConfigDeclaration pins the pattern providers follow: the fields the
// SDK models land on the request and the provider's own ride as extra fields,
// all in one request.
func TestChatConfigDeclaration(t *testing.T) {
	cfg := testChatConfig{
		Temperature:  openai.Ptr(0.2),
		EnableSearch: openai.Ptr(true),
	}

	var params openai.ChatCompletionNewParams
	cfg.ApplyToChatCompletion(&params)
	request := marshalParams(t, params)

	if got := request["temperature"]; got != 0.2 {
		t.Errorf("temperature = %v, want 0.2", got)
	}
	if got := request["enable_search"]; got != true {
		t.Errorf("enable_search = %v, want true", got)
	}
}
