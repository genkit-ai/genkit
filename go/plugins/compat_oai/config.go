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
	"maps"

	"github.com/openai/openai-go"
)

// ChatConfig is the constraint for a provider's chat model config: a type that
// can merge itself into the outgoing OpenAI chat completion request. A plugin
// declares every field its provider accepts, embeds [RequestConfig] for the
// settings Genkit owns, and writes SDK-modeled fields directly and
// anything else through [AddExtraFields]:
//
//	type ChatConfig struct {
//		compat_oai.RequestConfig
//
//		Temperature  *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 2."`
//		EnableSearch *bool    `json:"enableSearch,omitempty" jsonschema_description:"Lets the model consult web search, sent as the API's enable_search."`
//	}
//
//	func (c ChatConfig) ApplyToChatCompletion(params *openai.ChatCompletionNewParams) {
//		c.ApplyVersion(params)
//		if c.Temperature != nil {
//			params.Temperature = openai.Float(*c.Temperature)
//		}
//		if c.EnableSearch != nil {
//			compat_oai.AddExtraFields(params, map[string]any{"enable_search": *c.EnableSearch})
//		}
//	}
//
// The schema inferred from the config type is what the model advertises and
// what every request's config is validated against, so a field a provider
// does not accept is a field the Dev UI offers and the provider rejects.
// Declare what the provider's API reference lists, and use the same camelCase
// names other plugins use for the same setting so one config JSON keeps its
// meaning across providers and runtimes.
//
// Carry the provider's documentation on each field twice: a doc comment for
// Go readers, and a jsonschema_description tag with the same content for the
// Dev UI. Ranges, enums, and lengths the provider documents as hard limits
// belong in a jsonschema tag too, where validation rejects a violation before
// it is sent and billed. Leave a constraint out when the provider does not
// document one, and keep per-model limits (a context length, a level one
// model rejects) to the descriptions: the schema is shared by every model the
// plugin serves, so it may only say what holds for all of them.
//
// None of this leaves a new provider field stranded until the plugin declares
// it: every config inherits [RequestConfig.Extra], a passthrough
// [NewChatModel] forwards after the config's own merge, so a plugin never
// declares a passthrough of its own.
type ChatConfig interface {
	// ApplyToChatCompletion merges the config into params. Only fields the
	// config carries are written, so the zero config leaves params untouched.
	ApplyToChatCompletion(params *openai.ChatCompletionNewParams)
	// RequestAPIKey returns the API key overriding the plugin's for this
	// request, or "" for none (see [RequestConfig.APIKey]). Configs embedding
	// RequestConfig inherit it.
	RequestAPIKey() string
	// RequestExtra returns the request body fields the config carries beyond
	// the ones it declares, or nil for none (see [RequestConfig.Extra]).
	// Configs embedding RequestConfig inherit it.
	RequestExtra() map[string]any
}

// AddExtraFields merges fields into the extra request fields params carries,
// keeping any set earlier. [openai.ChatCompletionNewParams.SetExtraFields]
// alone replaces the map wholesale, which silently drops extras a config's
// embedded layers already set.
func AddExtraFields(params *openai.ChatCompletionNewParams, fields map[string]any) {
	if len(fields) == 0 {
		return
	}
	merged := make(map[string]any, len(params.ExtraFields())+len(fields))
	maps.Copy(merged, params.ExtraFields())
	maps.Copy(merged, fields)
	params.SetExtraFields(merged)
}

// RequestConfig holds the per-request settings Genkit owns rather than the
// provider: the credential the request is served with, the model version it
// is served by, and the passthrough for request fields the config does not
// declare. Every OpenAI-compatible provider implements them identically, so
// provider configs embed it and declare the rest themselves; see [ChatConfig].
//
// Nothing the provider owns belongs here. Sampling settings differ between
// providers in availability, name, and range, so a shared struct of them would
// force every config to advertise fields some provider rejects.
type RequestConfig struct {
	// APIKey overrides the plugin's API key for this request alone. It is a
	// client credential rather than a request parameter: it never serializes,
	// so it stays out of the advertised config schema, recorded traces, and
	// the outgoing request body, and it can only be set from a typed config in
	// code, never through a JSON or map config.
	APIKey string `json:"-"`
	// Version pins the exact model version the request is served by, e.g.
	// "gpt-4o-2024-11-20" for the "gpt-4o" family. It overrides the model ID
	// the request would otherwise carry.
	Version string `json:"version,omitempty" jsonschema_description:"Pins the exact model version the request is served by, e.g. gpt-4o-2024-11-20 for the gpt-4o family. Overrides the model ID the request would otherwise carry."`
	// Extra carries request body fields the config does not declare, sent
	// verbatim at the top level of the outgoing request. Keys are the
	// provider's wire names (usually snake_case), not the camelCase names
	// declared fields use, and a key that collides with anything the config
	// wrote wins, so a stale or missing mapping never blocks a request the
	// provider would accept. The fields Genkit builds from the request itself
	// (messages, tools and their variants) are rejected rather than forwarded.
	Extra map[string]any `json:"extra,omitempty" jsonschema_description:"Request body fields to send verbatim beyond the declared ones, keyed by the provider's wire names (usually snake_case). A colliding key wins over the declared field. Fields Genkit builds from the request, such as messages and tools, are rejected."`
}

// RequestAPIKey returns the API key the request overrides the plugin's with,
// or "" for none. Configs embedding RequestConfig inherit it, which is what
// makes the override reach [NewChatModel].
func (c RequestConfig) RequestAPIKey() string {
	return c.APIKey
}

// RequestExtra returns the request body fields the config carries beyond the
// ones it declares, or nil for none. Configs embedding RequestConfig inherit
// it, which is what makes the passthrough reach [NewChatModel].
func (c RequestConfig) RequestExtra() map[string]any {
	return c.Extra
}

// ApplyVersion writes Version onto the request's model, which is how a config
// pins the version it is served by. Provider configs call it first from their
// own ApplyToChatCompletion.
//
// It is deliberately not named ApplyToChatCompletion: a config embedding a
// complete apply method would satisfy [ChatConfig] while silently dropping
// every field the provider declared, so the interface stays unsatisfied until
// the plugin writes its own.
func (c RequestConfig) ApplyVersion(params *openai.ChatCompletionNewParams) {
	if c.Version != "" {
		params.Model = c.Version
	}
}

// EmbeddingConfig is the per-request config for OpenAI-compatible embedders.
//
// Dimensions and EncodingFormat lead because they are the two fields the
// released openai.TextEmbeddingConfig had, in the order it had them: that
// type is now an alias for this one, and keeping the order makes the two
// interchangeable everywhere a keyed literal is used.
type EmbeddingConfig struct {
	// Dimensions is the number of dimensions the output embeddings should
	// have, for models that support shortening.
	Dimensions int `json:"dimensions,omitempty" jsonschema:"minimum=1" jsonschema_description:"Number of dimensions the output embeddings should have, for models that support shortening."`
	// EncodingFormat selects the encoding of the returned embeddings, "float"
	// (the default) or "base64".
	EncodingFormat openai.EmbeddingNewParamsEncodingFormat `json:"encodingFormat,omitempty" jsonschema:"enum=float,enum=base64" jsonschema_description:"Encoding of the returned embeddings, float (the default) or base64."`
	// APIKey overrides the plugin's API key for this request alone. Like
	// [RequestConfig.APIKey], it never serializes and can only be set from a
	// typed config in code.
	APIKey string `json:"-"`
	// User is an end-user identifier the provider can use for abuse
	// monitoring.
	User string `json:"user,omitempty" jsonschema_description:"End-user identifier the provider can use to monitor and detect abuse."`
	// Extra carries request body fields this config does not declare, sent
	// verbatim at the top level of the outgoing request; the contract is
	// [RequestConfig.Extra]'s, with the input the only field Genkit protects.
	Extra map[string]any `json:"extra,omitempty" jsonschema_description:"Request body fields to send verbatim beyond the declared ones, keyed by the provider's wire names (usually snake_case). A colliding key wins over the declared field. The input Genkit builds from the request is rejected."`
}

// applyToEmbedding merges the config into params, leaving fields the config
// does not carry at their defaults.
func (c EmbeddingConfig) applyToEmbedding(params *openai.EmbeddingNewParams) {
	if c.Dimensions > 0 {
		params.Dimensions = openai.Int(int64(c.Dimensions))
	}
	if c.EncodingFormat != "" {
		params.EncodingFormat = c.EncodingFormat
	}
	if c.User != "" {
		params.User = openai.String(c.User)
	}
}
