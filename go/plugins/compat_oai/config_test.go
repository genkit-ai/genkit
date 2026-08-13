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
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
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

	// The fields a Genkit option owns are hidden: replaced by the permissive
	// `true` schema so the dev UI does not render them, while a value still
	// passes boundary validation and reaches rejectManagedConfig, which names
	// the option to use. Spelled out rather than read from
	// sdkSchemaOverrides, which would make emptying that list pass this test.
	for _, field := range []string{"messages", "tools", "tool_choice", "functions", "function_call", "response_format", "n"} {
		if got, ok := props[field]; !ok || got != true {
			t.Errorf("property %q = %v, want the hidden field advertised as the permissive true schema", field, got)
		}
	}

	// The model stays visible and described: it is how a config pins the
	// version it is served by.
	model, ok := props["model"].(map[string]any)
	if !ok {
		t.Fatal("property \"model\" missing, want the version pin advertised")
	}
	if desc, _ := model["description"].(string); desc == "" {
		t.Error("property \"model\" has no description, want the SDK schema curated with help text")
	}
}

// TestSDKSchemaOverridePathsResolve pins the overrides against the linked
// SDK: an entry whose path no longer resolves stopped describing or hiding
// anything when the SDK renamed the field, and applying it is a silent no-op.
func TestSDKSchemaOverridePathsResolve(t *testing.T) {
	if stale := sdkSchemaOverrides.UnresolvedPaths(sdkConfigSchema()); len(stale) > 0 {
		t.Errorf("override paths do not resolve against the reflected SDK schema: %v", stale)
	}
}

// TestRejectManagedSDKConfig pins the curated rejection for each field a
// Genkit primitive owns: the message names the option to use, and a config
// carrying none of them passes.
func TestRejectManagedSDKConfig(t *testing.T) {
	cases := []struct {
		name   string
		config openai.ChatCompletionNewParams
		want   string
	}{
		{"messages", openai.ChatCompletionNewParams{Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")}}, "ai.WithMessages()"},
		{"tools", openai.ChatCompletionNewParams{Tools: []openai.ChatCompletionToolParam{{Function: shared.FunctionDefinitionParam{Name: "t"}}}}, "ai.WithTools()"},
		{"functions", openai.ChatCompletionNewParams{Functions: []openai.ChatCompletionNewParamsFunction{{Name: "f"}}}, "ai.WithTools()"},
		{"tool_choice", openai.ChatCompletionNewParams{ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("auto")}}, "ai.WithToolChoice()"},
		{"function_call", openai.ChatCompletionNewParams{FunctionCall: openai.ChatCompletionNewParamsFunctionCallUnion{OfFunctionCallMode: openai.String("none")}}, "ai.WithToolChoice()"},
		{"response_format", openai.ChatCompletionNewParams{ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONObject: &shared.ResponseFormatJSONObjectParam{}}}, "ai.WithOutputType()"},
		{"n", openai.ChatCompletionNewParams{N: openai.Int(2)}, "first candidate only"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := rejectManagedConfig(&tc.config)
			if err == nil {
				t.Fatalf("rejectManagedConfig(%s) = nil, want the managed field refused", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want it to point at %q", err, tc.want)
			}
		})
	}

	plain := openai.ChatCompletionNewParams{Temperature: openai.Float(0.5), Seed: openai.Int(7)}
	if err := rejectManagedConfig(&plain); err != nil {
		t.Errorf("rejectManagedConfig(plain) = %v, want the rest of the SDK surface untouched", err)
	}
}

// TestSDKConfigSchemaDropsParamPlumbing pins that the SDK's embedded
// param.APIObject, an anonymous any the reflector sees at every level of the
// struct, is not advertised as a field. It is not part of the wire format and
// nothing marshals from it.
func TestSDKConfigSchemaDropsParamPlumbing(t *testing.T) {
	var walk func(t *testing.T, path string, schema map[string]any)
	walk = func(t *testing.T, path string, schema map[string]any) {
		props, ok := schema["properties"].(map[string]any)
		if !ok {
			return
		}
		if _, ok := props["any"]; ok {
			t.Errorf("%s advertises the SDK's embedded param plumbing as an \"any\" property", path)
		}
		for name, sub := range props {
			if m, ok := sub.(map[string]any); ok {
				walk(t, path+"."+name, m)
			}
		}
		if items, ok := schema["items"].(map[string]any); ok {
			walk(t, path+"[]", items)
		}
	}
	walk(t, "config", sdkConfigSchema())
}

// TestEmbeddingConfigReleasedLayout pins the two fields openai.TextEmbeddingConfig
// carried through v1.11.0, which is now an alias for this type: their names,
// types, JSON tags, and position. Keyed literals and the wire format written
// against the released type keep working only while all four hold.
func TestEmbeddingConfigReleasedLayout(t *testing.T) {
	released := []struct{ name, jsonTag, kind string }{
		{"Dimensions", "dimensions,omitempty", "int"},
		{"EncodingFormat", "encodingFormat,omitempty", "openai.EmbeddingNewParamsEncodingFormat"},
	}

	typ := reflect.TypeOf(EmbeddingConfig{})
	if typ.NumField() < len(released) {
		t.Fatalf("EmbeddingConfig has %d fields, want at least the %d released ones", typ.NumField(), len(released))
	}
	for i, want := range released {
		field := typ.Field(i)
		if field.Name != want.name {
			t.Errorf("field %d = %q, want %q: the released fields must keep their position", i, field.Name, want.name)
		}
		if got := field.Tag.Get("json"); got != want.jsonTag {
			t.Errorf("field %q json tag = %q, want %q", field.Name, got, want.jsonTag)
		}
		if got := field.Type.String(); got != want.kind {
			t.Errorf("field %q type = %q, want %q", field.Name, got, want.kind)
		}
	}
}

// TestEmbeddingConfigSchema pins the constraints OpenAI documents for the
// embeddings endpoint riding on the schema, where the framework enforces
// them: the two encodings as an enum and a positive dimensions count. The
// embedder schema is not walked by the chat conformance test, so the help
// text on every field is checked here too.
func TestEmbeddingConfigSchema(t *testing.T) {
	schema := base.SchemaAsMap(base.InferJSONSchema(EmbeddingConfig{}))
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties: %v", schema)
	}

	encoding, _ := props["encodingFormat"].(map[string]any)
	if got, want := encoding["enum"], []any{"float", "base64"}; !reflect.DeepEqual(got, want) {
		t.Errorf("encodingFormat enum = %#v, want %#v", got, want)
	}
	dimensions, _ := props["dimensions"].(map[string]any)
	if got := dimensions["minimum"]; got != 1.0 {
		t.Errorf("dimensions minimum = %#v, want 1", got)
	}
	if props["apiKey"] != nil {
		t.Error("schema advertises apiKey, want the credential kept out of serialized configs")
	}
	if props["extra"] == nil {
		t.Error("schema is missing the extra passthrough")
	}
	for key, prop := range props {
		field, _ := prop.(map[string]any)
		if desc, _ := field["description"].(string); desc == "" {
			t.Errorf("%s has no description, want Dev UI help text on every config field", key)
		}
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
	for _, key := range []string{"temperature", "maxOutputTokens", "version", "enableSearch", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}
	if props["max_tokens"] != nil {
		t.Error("config schema advertises SDK wire names, want the Genkit camelCase contract")
	}
	// The passthrough advertises an open object: its contents are the
	// provider's to define, so constraining them would reject the exact
	// fields it exists to carry.
	extra, _ := props["extra"].(map[string]any)
	if got := extra["type"]; got != "object" {
		t.Errorf("extra type = %v, want object", got)
	}
	if got, has := extra["additionalProperties"]; has && got == false {
		t.Error("extra forbids additional properties, want arbitrary provider fields to validate")
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
	if err := base.ValidateValue(req(map[string]any{"extra": map[string]any{"safe_mode": true, "beta_features": []any{"x"}}}), chatSchema); err != nil {
		t.Errorf("extra passthrough config rejected at the boundary: %v", err)
	}
	if err := base.ValidateValue(req(map[string]any{"extra": "safe_mode"}), chatSchema); err == nil {
		t.Error("non-object extra accepted, want the passthrough constrained to an object")
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

// TestForwardRequestExtra pins the passthrough at the params level: fields
// merge at the top level of the wire request, a collision resolves toward the
// extra whether it hit a declared field's mapping or an extra the config
// already wrote, and each Genkit-built field is rejected by name.
func TestForwardRequestExtra(t *testing.T) {
	var params openai.ChatCompletionNewParams
	testChatConfig{MaxOutputTokens: 5, EnableSearch: openai.Ptr(false)}.ApplyToChatCompletion(&params)
	if err := forwardRequestExtra(&params, map[string]any{
		"max_tokens":    99,
		"enable_search": true,
		"safe_mode":     true,
	}); err != nil {
		t.Fatalf("forwardRequestExtra() error = %v", err)
	}
	request := marshalParams(t, params)

	if got := request["safe_mode"]; got != true {
		t.Errorf("safe_mode = %v, want the undeclared field forwarded verbatim", got)
	}
	if got := request["max_tokens"]; got != float64(99) {
		t.Errorf("max_tokens = %v, want the extra winning over the declared field's mapping", got)
	}
	if got := request["enable_search"]; got != true {
		t.Errorf("enable_search = %v, want the extra winning over the extra the config wrote", got)
	}

	// Spelled out rather than read from managedRequestFields, which would
	// make emptying that list pass this test.
	for _, field := range []string{"messages", "tools", "tool_choice", "functions", "function_call"} {
		var p openai.ChatCompletionNewParams
		err := forwardRequestExtra(&p, map[string]any{field: "hijack"})
		if err == nil || !strings.Contains(err.Error(), field) {
			t.Errorf("forwardRequestExtra(%s) error = %v, want the managed field named in a rejection", field, err)
		}
	}
}

// TestForwardEmbeddingExtra pins the embedder passthrough: fields merge into
// the wire request, and the input Genkit builds from the request is rejected.
func TestForwardEmbeddingExtra(t *testing.T) {
	params := openai.EmbeddingNewParams{Model: "text-embedding-3-small"}
	if err := forwardEmbeddingExtra(&params, map[string]any{"output_dtype": "int8"}); err != nil {
		t.Fatalf("forwardEmbeddingExtra() error = %v", err)
	}
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var request map[string]any
	if err := json.Unmarshal(data, &request); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if got := request["output_dtype"]; got != "int8" {
		t.Errorf("output_dtype = %v, want the undeclared field forwarded verbatim", got)
	}

	err = forwardEmbeddingExtra(&params, map[string]any{"input": "hijack"})
	if err == nil || !strings.Contains(err.Error(), "input") {
		t.Errorf("forwardEmbeddingExtra(input) error = %v, want the managed field named in a rejection", err)
	}
}

// TestExtraFieldsReachTheWire pins the passthrough end to end through the
// framework: a JSON config's extra object validates against the advertised
// schema, its fields ride the request verbatim at the top level, and they win
// the collisions the contract promises. A managed field inside extra fails
// the request before anything is sent.
func TestExtraFieldsReachTheWire(t *testing.T) {
	var mu sync.Mutex
	requests := 0
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"extra-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "test-key", BaseURL: server.URL}
	o.Init(ctx)
	g := genkit.Init(ctx)
	genkit.RegisterAction(g, NewChatModel[testChatConfig](o, "extra-model", ai.ModelOptions{}))

	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("testprovider/extra-model"),
		ai.WithPrompt("hi"),
		ai.WithConfig(map[string]any{
			"maxOutputTokens": 5,
			"enableSearch":    false,
			"extra": map[string]any{
				"max_tokens":    99,
				"enable_search": true,
				"safe_mode":     true,
			},
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	if got := body["safe_mode"]; got != true {
		t.Errorf("safe_mode = %v, want the undeclared field on the wire", got)
	}
	if got := body["max_tokens"]; got != float64(99) {
		t.Errorf("max_tokens = %v, want the extra winning over the declared field's mapping", got)
	}
	if got := body["enable_search"]; got != true {
		t.Errorf("enable_search = %v, want the extra winning over the extra the config wrote", got)
	}
	if _, ok := body["extra"]; ok {
		t.Error("request body carries an extra envelope, want its fields spliced at the top level")
	}
	if _, ok := body["messages"]; !ok {
		t.Error("request body has no messages, want the framework-built conversation intact")
	}
	mu.Unlock()

	_, err = genkit.Generate(ctx, g,
		ai.WithModelName("testprovider/extra-model"),
		ai.WithPrompt("hi"),
		ai.WithConfig(map[string]any{"extra": map[string]any{"messages": "mine now"}}),
	)
	if err == nil || !strings.Contains(err.Error(), "messages") {
		t.Fatalf("Generate() error = %v, want the managed field named in a rejection", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if requests != 1 {
		t.Errorf("requests = %d, want the rejected config to fail before anything is sent", requests)
	}
}

// TestDefineModelKeepsConfigUntyped pins the deprecated
// [OpenAICompatible.DefineModel] contract the not-yet-migrated plugins ride:
// config is not validated at the action boundary, so a map key the SDK does
// not model reaches the wire as a JSON extra. The same request against a
// model built by [OpenAICompatible.NewModel] fails the schema before
// anything is sent.
func TestDefineModelKeepsConfigUntyped(t *testing.T) {
	var mu sync.Mutex
	requests := 0
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"legacy-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "test-key", BaseURL: server.URL}
	o.Init(ctx)
	g := genkit.Init(ctx)
	genkit.RegisterAction(g, o.DefineModel("testprovider", "legacy-model", ai.ModelOptions{}).(api.Action))
	genkit.RegisterAction(g, o.NewModel("sdk-model", ai.ModelOptions{}))

	config := map[string]any{"enable_thinking": true, "topP": 0.3}
	if _, err := genkit.Generate(ctx, g,
		ai.WithModelName("testprovider/legacy-model"),
		ai.WithPrompt("hi"),
		ai.WithConfig(config),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	if got := body["enable_thinking"]; got != true {
		t.Errorf("enable_thinking = %v, want the unmodeled key on the wire", got)
	}
	if got := body["top_p"]; got != 0.3 {
		t.Errorf("top_p = %v, want the camelCase key rewritten to the wire name", got)
	}
	mu.Unlock()

	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("testprovider/sdk-model"),
		ai.WithPrompt("hi"),
		ai.WithConfig(config),
	)
	if err == nil || !strings.Contains(err.Error(), "did not match expected schema") {
		t.Fatalf("Generate() error = %v, want the SDK-typed model to reject the unmodeled key", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if requests != 1 {
		t.Errorf("requests = %d, want the rejected config to fail before anything is sent", requests)
	}
}
