// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"github.com/invopop/jsonschema"
	"google.golang.org/genai"
)

// configOverrides describes per-property metadata layered onto a reflected
// JSON schema before it is exposed to the Genkit Developer UI. The genai
// SDK structs do not carry JSON Schema descriptions, and a few of their
// fields are managed by Genkit primitives and rejected when supplied
// directly, so we curate that information here.
type configOverrides struct {
	// descriptions maps a JSON property name to the help text shown as the
	// field's tooltip in the dev UI.
	descriptions map[string]string
	// hidden lists JSON property names that should be removed from the
	// schema. Use this for fields the plugin manages on the user's behalf
	// (system prompts, tools, output schemas, caching) and would error on
	// if supplied directly through the config.
	hidden []string
}

// gccOverrides controls dev UI presentation of [genai.GenerateContentConfig].
var gccOverrides = configOverrides{
	descriptions: map[string]string{
		"temperature":                "Controls the degree of randomness in token selection. A lower value is good for a more predictable response. A higher value leads to more diverse or unexpected results.",
		"topP":                       "Decides how many possible words to consider. A higher value means that the model looks at more possible words, even the less likely ones, which makes the generated text more diverse.",
		"topK":                       "The maximum number of tokens to consider when sampling.",
		"maxOutputTokens":            "The maximum number of tokens to include in the response.",
		"stopSequences":              "Sequences (up to 5) that will stop output generation when encountered.",
		"presencePenalty":            "Positive values penalize tokens that already appear in the generated text, increasing the likelihood of generating more diverse content.",
		"frequencyPenalty":           "Positive values penalize tokens that repeatedly appear in the generated text, increasing the likelihood of generating more diverse content.",
		"seed":                       "When set to a specific number, the model makes a best effort to provide the same response for repeated requests. By default a random seed is used.",
		"responseLogprobs":           "Whether to return the log probabilities of the tokens chosen by the model at each step.",
		"logprobs":                   "Number of top candidate tokens to return log probabilities for at each generation step. Requires responseLogprobs.",
		"responseModalities":         "The set of response modalities the model is allowed to produce (e.g. TEXT, IMAGE, AUDIO).",
		"mediaResolution":            "If specified, the media resolution to use.",
		"audioTimestamp":             "If true, audio timestamps are included in the request to the model.",
		"thinkingConfig":             "Configures the model's thinking budget and behavior on supported models.",
		"imageConfig":                "Configures image generation when the model is asked to produce an image.",
		"speechConfig":               "Configures speech generation when the model is asked to produce audio.",
		"safetySettings":             "Adjust how likely you are to see responses that could be harmful. Content is blocked based on the probability that it is harmful.",
		"toolConfig":                 "Tool configuration shared across all tools the model has access to. Use this to constrain function calling mode or configure retrieval.",
		"tools":                      "Built-in API tools (GoogleSearch, Retrieval, CodeExecution, etc.) made available to the model. Custom function tools must be registered via ai.WithTools() instead, which wires them up to the Genkit runtime.",
		"labels":                     "User-defined metadata to break down billed charges.",
		"modelArmorConfig":           "Settings for prompt and response sanitization via Model Armor. If set, safetySettings must not be set.",
		"modelSelectionConfig":       "Configures model selection (e.g. feature priority).",
		"routingConfig":              "Configures requests through the model router.",
		"enableEnhancedCivicAnswers": "Enables enhanced civic answers. Not available on every model and not supported on Vertex AI.",
		"httpOptions":                "Per-request HTTP overrides for base URL, API version, headers, and timeout. These override the plugin-configured defaults.",
	},
	hidden: []string{
		// Managed by Genkit primitives; the plugin rejects these when set.
		"systemInstruction",  // ai.WithSystemPrompt
		"cachedContent",      // ai.WithCacheTTL
		"responseSchema",     // ai.WithOutputType / ai.WithOutputSchema
		"responseMimeType",   // ai.WithOutputType / ai.WithOutputSchema
		"responseJsonSchema", // ai.WithOutputSchema
		// Pinned to 1 by the plugin; the API only supports a single candidate.
		"candidateCount",
	},
}

// gicOverrides controls dev UI presentation of [genai.GenerateImagesConfig].
var gicOverrides = configOverrides{
	descriptions: map[string]string{
		"numberOfImages":           "Number of images to generate. Defaults to 4 when unset.",
		"aspectRatio":              "Aspect ratio of the generated images. Supported values include 1:1, 3:4, 4:3, 9:16, and 16:9.",
		"negativePrompt":           "Description of what to discourage in the generated images.",
		"guidanceScale":            "How strongly the model should adhere to the prompt. Higher values increase prompt alignment but may reduce image quality.",
		"seed":                     "Random seed for image generation. Not available when addWatermark is true.",
		"safetyFilterLevel":        "Filter level applied for safety filtering.",
		"personGeneration":         "Whether the model is allowed to generate people.",
		"outputMimeType":           "MIME type of the generated image.",
		"outputCompressionQuality": "JPEG compression quality (only applies to image/jpeg output).",
		"addWatermark":             "Whether to add a watermark to the generated images.",
		"imageSize":                "The size of the largest dimension of the generated image. Supported sizes are 1K and 2K (Imagen 3 does not support 2K).",
		"enhancePrompt":            "Whether to use prompt rewriting on the input.",
		"language":                 "Language of the text in the prompt.",
		"outputGcsUri":             "Cloud Storage URI for storing the generated images.",
		"labels":                   "User-defined metadata to break down billed charges.",
		"includeRaiReason":         "If true, includes the Responsible AI reason if an image is filtered out.",
		"includeSafetyAttributes":  "If true, returns safety scores for each generated image and the prompt.",
		"httpOptions":              "Per-request HTTP overrides for base URL, API version, headers, and timeout. These override the plugin-configured defaults.",
	},
}

// gvcOverrides controls dev UI presentation of [genai.GenerateVideosConfig].
var gvcOverrides = configOverrides{
	descriptions: map[string]string{
		"numberOfVideos":     "Number of videos to generate.",
		"fps":                "Frames per second for video generation.",
		"durationSeconds":    "Duration of the generated clip in seconds.",
		"seed":               "RNG seed. With identical inputs and the same seed, predictions are consistent across requests.",
		"aspectRatio":        "Aspect ratio for the generated video. 16:9 (landscape) and 9:16 (portrait) are supported.",
		"resolution":         "Resolution of the generated video. 720p and 1080p are supported.",
		"personGeneration":   "Whether to allow generating videos with people, and which ages are allowed.",
		"negativePrompt":     "Description of what to discourage in the generated videos.",
		"enhancePrompt":      "Whether to use prompt rewriting on the input.",
		"generateAudio":      "Whether to generate audio along with the video.",
		"compressionQuality": "Compression quality of the generated videos.",
		"outputGcsUri":       "Cloud Storage bucket for storing the generated videos.",
		"pubsubTopic":        "Pub/Sub topic for receiving video generation progress.",
		"httpOptions":        "Per-request HTTP overrides for base URL, API version, headers, and timeout. These override the plugin-configured defaults.",
	},
}

// applyConfigOverrides mutates schema in place: removes hidden properties
// and writes descriptions onto the remaining ones. Top-level only — nested
// object fields keep whatever the reflector produced.
func applyConfigOverrides(schema *jsonschema.Schema, o configOverrides) {
	if schema == nil || schema.Properties == nil {
		return
	}
	for _, name := range o.hidden {
		schema.Properties.Delete(name)
	}
	if len(o.hidden) > 0 && len(schema.Required) > 0 {
		hide := make(map[string]struct{}, len(o.hidden))
		for _, name := range o.hidden {
			hide[name] = struct{}{}
		}
		kept := schema.Required[:0]
		for _, r := range schema.Required {
			if _, drop := hide[r]; !drop {
				kept = append(kept, r)
			}
		}
		schema.Required = kept
	}
	for name, desc := range o.descriptions {
		if pair := schema.Properties.GetPair(name); pair != nil && pair.Value != nil {
			pair.Value.Description = desc
		}
	}
}

// overridesFor returns the overrides matching a given config struct value,
// or a zero (no-op) value for unknown types.
func overridesFor(config any) configOverrides {
	switch config.(type) {
	case genai.GenerateContentConfig, *genai.GenerateContentConfig:
		return gccOverrides
	case genai.GenerateImagesConfig, *genai.GenerateImagesConfig:
		return gicOverrides
	case genai.GenerateVideosConfig, *genai.GenerateVideosConfig:
		return gvcOverrides
	}
	return configOverrides{}
}
