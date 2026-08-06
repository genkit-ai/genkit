// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
)

// ModelRef names a Claude model and carries the config to generate with, so
// the config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(anthropic.ModelRef("claude-opus-4-5", &sdk.MessageNewParams{
//		MaxTokens: 1024,
//	}))
//
// id is the model ID, with or without the provider prefix: "claude-opus-4-5"
// and "anthropic/claude-opus-4-5" name the same model, as they do everywhere
// else in this package.
//
// This package and the Anthropic SDK are both named anthropic, so one of them
// needs an import alias; the example above aliases the SDK to sdk.
func ModelRef(id string, config *anthropic.MessageNewParams) ai.ModelRef {
	return ai.NewModelRef(modelName(id), config)
}
