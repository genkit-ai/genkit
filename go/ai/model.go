// Copyright 2025 Google LLC
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

package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
)

// This file holds the built-in request/response processing that every model
// defined via [NewModel] (and [NewBackgroundModel]) is wrapped with. These
// wrappers run at the model boundary so they apply on every model call,
// including direct [Model.Generate] invocations that bypass [Generate] and its
// user-supplied [Middleware]. They are internal to the model implementation and
// are intentionally not part of the public middleware surface (see
// middleware.go for the user-facing [Middleware] system).

// AugmentWithContextOptions configures how a request is augmented with context.
type AugmentWithContextOptions struct {
	Preface      *string                                                                // Preceding text to place before the rendered context documents.
	ItemTemplate func(d Document, index int, options *AugmentWithContextOptions) string // A function to render a document into a text part to be included in the message.
	CitationKey  *string                                                                // Metadata key to use for citation reference. Pass `nil` to provide no citations.
}

// contextPreface is the default preface for context augmentation.
const contextPreface = "\n\nUse the following information to complete your task:\n\n"

// CalculateInputOutputUsage populates resp.Usage with character and media
// counts derived from the request and response when those counts are not
// already set by the model. It is exported for background models and other
// providers that need to compute usage outside the normal telemetry wrapper.
func CalculateInputOutputUsage(req *ModelRequest, resp *ModelResponse) {
	if resp.Usage == nil {
		resp.Usage = &GenerationUsage{}
	}
	if resp.Usage.InputCharacters == 0 {
		resp.Usage.InputCharacters = countInputCharacters(req)
	}
	if resp.Usage.OutputCharacters == 0 {
		resp.Usage.OutputCharacters = countOutputCharacters(resp)
	}
	if resp.Usage.InputImages == 0 {
		resp.Usage.InputImages = countInputParts(req, func(part *Part) bool { return part.IsImage() })
	}
	if resp.Usage.OutputImages == 0 {
		resp.Usage.OutputImages = countOutputParts(resp, func(part *Part) bool { return part.IsImage() })
	}
	if resp.Usage.InputVideos == 0 {
		resp.Usage.InputVideos = countInputParts(req, func(part *Part) bool { return part.IsVideo() })
	}
	if resp.Usage.OutputVideos == 0 {
		resp.Usage.OutputVideos = countOutputParts(resp, func(part *Part) bool { return part.IsVideo() })
	}
	if resp.Usage.InputAudioFiles == 0 {
		resp.Usage.InputAudioFiles = countInputParts(req, func(part *Part) bool { return part.IsAudio() })
	}
	if resp.Usage.OutputAudioFiles == 0 {
		resp.Usage.OutputAudioFiles = countOutputParts(resp, func(part *Part) bool { return part.IsAudio() })
	}
}

// addAutomaticTelemetry wraps a model function to automatically measure latency
// and calculate character and media counts.
func addAutomaticTelemetry() func(next rawModelFunc) rawModelFunc {
	return func(fn rawModelFunc) rawModelFunc {
		return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			startTime := time.Now()

			// Call the underlying model function
			resp, err := fn(ctx, req, cb)
			if err != nil {
				return nil, err
			}

			// Calculate latency
			latencyMs := float64(time.Since(startTime).Nanoseconds()) / 1e6
			if resp.LatencyMs == 0 {
				resp.LatencyMs = latencyMs
			}

			CalculateInputOutputUsage(req, resp)

			return resp, nil
		}
	}
}

// countInputParts counts parts in the input request that match the given predicate.
func countInputParts(req *ModelRequest, predicate func(*Part) bool) int {
	if req == nil {
		return 0
	}

	count := 0
	for _, msg := range req.Messages {
		if msg == nil {
			continue
		}
		for _, part := range msg.Content {
			if part != nil && predicate(part) {
				count++
			}
		}
	}
	return count
}

// countInputCharacters counts the total characters in the input request.
func countInputCharacters(req *ModelRequest) int {
	if req == nil {
		return 0
	}

	total := 0
	for _, msg := range req.Messages {
		if msg == nil {
			continue
		}
		for _, part := range msg.Content {
			if part != nil && part.Text != "" {
				total += len(part.Text)
			}
		}
	}
	return total
}

// countOutputParts counts parts in the output response that match the given predicate.
func countOutputParts(resp *ModelResponse, predicate func(*Part) bool) int {
	if resp == nil || resp.Message == nil {
		return 0
	}

	count := 0
	for _, part := range resp.Message.Content {
		if part != nil && predicate(part) {
			count++
		}
	}
	return count
}

// countOutputCharacters counts the total characters in the output response.
func countOutputCharacters(resp *ModelResponse) int {
	if resp == nil || resp.Message == nil {
		return 0
	}

	total := 0
	for _, part := range resp.Message.Content {
		if part != nil && part.Text != "" {
			total += len(part.Text)
		}
	}
	return total
}

// simulateSystemPrompt provides a simulated system prompt for models that don't support it natively.
func simulateSystemPrompt(modelOpts *ModelOptions, opts map[string]string) func(next rawModelFunc) rawModelFunc {
	return func(next rawModelFunc) rawModelFunc {
		return func(ctx context.Context, input *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			// Short-circuiting middleware if system role is supported in model.
			if modelOpts.Supports.SystemRole {
				return next(ctx, input, cb)
			}
			preface := "SYSTEM INSTRUCTIONS:\n"
			acknowledgement := "Understood."

			if opts != nil {
				if p, ok := opts["preface"]; ok {
					preface = p
				}
				if a, ok := opts["acknowledgement"]; ok {
					acknowledgement = a
				}
			}
			modifiedMessages := make([]*Message, len(input.Messages))
			copy(modifiedMessages, input.Messages)
			for i, message := range input.Messages {
				if message.Role == "system" {
					systemPrompt := message.Content
					userMessage := &Message{
						Role:    "user",
						Content: append([]*Part{NewTextPart(preface)}, systemPrompt...),
					}
					modelMessage := NewModelTextMessage(acknowledgement)

					modifiedMessages = append(modifiedMessages[:i], append([]*Message{userMessage, modelMessage}, modifiedMessages[i+1:]...)...)
					break
				}
			}
			input.Messages = modifiedMessages
			return next(ctx, input, cb)
		}
	}
}

// validateSupport validates whether a model supports the features used in the model request.
func validateSupport(model string, opts *ModelOptions) func(next rawModelFunc) rawModelFunc {
	return func(next rawModelFunc) rawModelFunc {
		return func(ctx context.Context, input *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if opts == nil {
				opts = &ModelOptions{
					Supports: &ModelSupports{},
					Versions: []string{},
				}
			}

			if !opts.Supports.Media {
				for _, msg := range input.Messages {
					for _, part := range msg.Content {
						if part.IsMedia() {
							return nil, core.NewError(core.INVALID_ARGUMENT, "model %q does not support media, but media was provided. Request: %+v", model, input)
						}
					}
				}
			}

			if !opts.Supports.Tools && len(input.Tools) > 0 {
				return nil, core.NewError(core.INVALID_ARGUMENT, "model %q does not support tool use, but tools were provided. Request: %+v", model, input)
			}

			if !opts.Supports.Multiturn && len(input.Messages) > 1 {
				return nil, core.NewError(core.INVALID_ARGUMENT, "model %q does not support multiple messages, but %d were provided. Request: %+v", model, len(input.Messages), input)
			}

			if !opts.Supports.ToolChoice && input.ToolChoice != "" && input.ToolChoice != ToolChoiceAuto {
				return nil, core.NewError(core.INVALID_ARGUMENT, "model %q does not support tool choice, but tool choice was provided. Request: %+v", model, input)
			}

			if !opts.Supports.SystemRole {
				for _, msg := range input.Messages {
					if msg.Role == RoleSystem {
						return nil, core.NewError(core.INVALID_ARGUMENT, "model %q does not support system role, but system role was provided. Request: %+v", model, input)
					}
				}
			}

			if opts.Stage == ModelStageDeprecated {
				logger.FromContext(ctx).Warn("model is deprecated and may be removed in a future release", "model", model)
			}

			if (opts.Supports.Constrained == "" ||
				opts.Supports.Constrained == ConstrainedSupportNone ||
				(opts.Supports.Constrained == ConstrainedSupportNoTools && len(input.Tools) > 0)) &&
				input.Output != nil && input.Output.Constrained {
				return nil, core.NewError(core.INVALID_ARGUMENT, "model %q does not support native constrained output, but constrained output was requested. Request: %+v", model, input)
			}

			return next(ctx, input, cb)
		}
	}
}

// validateVersion validates that the requested model version is supported.
// It runs against the raw, pre-conversion config (see [normalizeConfig])
// because conversion into a Config type without a version field would
// silently drop the key.
func validateVersion(model string, versions []string, config any) error {
	if config == nil {
		return nil
	}

	var configMap map[string]any

	switch c := config.(type) {
	case map[string]any:
		configMap = c
	default:
		data, err := json.Marshal(config)
		if err != nil {
			return nil
		}
		if err := json.Unmarshal(data, &configMap); err != nil {
			return nil
		}
	}

	versionVal, exists := configMap["version"]
	if !exists {
		return nil
	}

	version, ok := versionVal.(string)
	if !ok {
		return core.NewError(core.INVALID_ARGUMENT, "version must be a string, got %T", versionVal)
	}

	if slices.Contains(versions, version) {
		return nil
	}

	return core.NewError(core.INVALID_ARGUMENT, "model %q does not support version %q, supported versions: %v", model, version, versions)
}

// contextItemTemplate is the default item template for context augmentation.
func contextItemTemplate(d Document, index int, options *AugmentWithContextOptions) string {
	out := "- "
	if options != nil && options.CitationKey != nil {
		out += fmt.Sprintf("[%v]: ", d.Metadata[*options.CitationKey])
	} else if options == nil || options.CitationKey == nil {
		if ref, ok := d.Metadata["ref"]; ok {
			out += fmt.Sprintf("[%v]: ", ref)
		} else if id, ok := d.Metadata["id"]; ok {
			out += fmt.Sprintf("[%v]: ", id)
		} else {
			out += fmt.Sprintf("[%v]: ", strconv.Itoa(index))
		}
	}
	out += d.concatText() + "\n"
	return out
}

// augmentWithContext augments a request with context documents.
func augmentWithContext(modelOpts *ModelOptions, augOpts *AugmentWithContextOptions) func(next rawModelFunc) rawModelFunc {
	return func(next rawModelFunc) rawModelFunc {
		return func(ctx context.Context, input *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			// Short-circuiting middleware if context is supported in model.
			if modelOpts.Supports.Context {
				return next(ctx, input, cb)
			}
			preface := contextPreface
			if augOpts != nil && augOpts.Preface != nil {
				preface = *augOpts.Preface
			}
			itemTemplate := contextItemTemplate
			if augOpts != nil && augOpts.ItemTemplate != nil {
				itemTemplate = augOpts.ItemTemplate
			}
			// if there is no context in the request, no-op
			if len(input.Docs) == 0 {
				return next(ctx, input, cb)
			}

			userMessage := lastUserMessage(input.Messages)
			// if there are no messages, no-op
			if userMessage == nil {
				return next(ctx, input, cb)
			}

			// if there is already a context part, no-op
			contextPartIndex := -1
			for i, part := range userMessage.Content {
				if part.Metadata != nil && part.Metadata[PartMetaPurpose] == PartPurposeContext {
					contextPartIndex = i
					break
				}
			}

			if contextPartIndex >= 0 && userMessage.Content[contextPartIndex].Metadata[PartMetaPending] == nil {
				return next(ctx, input, cb)
			}

			out := preface
			for i, doc := range input.Docs {
				out += itemTemplate(*doc, i, augOpts)
			}
			out += "\n"

			contextPart := NewTextPart(out)
			contextPart.Metadata = map[string]any{PartMetaPurpose: PartPurposeContext}
			if contextPartIndex >= 0 {
				userMessage.Content[contextPartIndex] = contextPart
			} else {
				userMessage.Content = append(userMessage.Content, contextPart)
			}
			return next(ctx, input, cb)
		}
	}
}

// lastUserMessage returns the last user message from a list of messages.
func lastUserMessage(messages []*Message) *Message {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i]
		}
	}
	return nil
}

// concatText returns the concatenated text parts of the document content.
func (d *Document) concatText() string {
	var builder strings.Builder
	for _, part := range d.Content {
		if part.IsText() {
			builder.WriteString(part.Text)
		}
	}
	return builder.String()
}
