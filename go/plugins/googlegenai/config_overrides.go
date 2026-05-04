// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"strings"

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
	// field's tooltip in the dev UI. Top-level only.
	descriptions map[string]string
	// hidden lists JSON property paths to remove from the schema. Each entry
	// is either a top-level property name ("systemInstruction") or a slash
	// path that descends through array `items` and nested `properties`
	// ("tools/items/functionDeclarations"). Use this for fields the plugin
	// rejects at runtime or that the Genkit framework manages directly.
	hidden []string
}

// gccOverrides controls dev UI presentation of [genai.GenerateContentConfig].
var gccOverrides = configOverrides{
	descriptions: map[string]string{
		"temperature":                "Controls the degree of randomness in token selection. Lower values produce more deterministic responses; higher values produce more diverse or creative ones.",
		"topP":                       "Considers tokens whose cumulative probability exceeds this value. Lower values constrain the model to high-probability tokens; higher values allow more variety.",
		"topK":                       "Limits sampling to the K most likely tokens at each step. Lower values constrain output; higher values allow more variety.",
		"maxOutputTokens":            "Maximum number of tokens the model may generate. When unset, the model picks a default that varies by model.",
		"stopSequences":              "Up to 5 strings; if the model emits any of them, generation stops immediately.",
		"presencePenalty":            "Positive values penalize tokens that already appear in the output, encouraging more diverse content.",
		"frequencyPenalty":           "Positive values penalize tokens that repeat frequently in the output, encouraging more diverse content.",
		"seed":                       "When set, the model makes a best effort to return the same response for repeated requests with identical inputs. Defaults to a random seed.",
		"responseLogprobs":           "Whether to return log probabilities for the tokens chosen at each generation step.",
		"logprobs":                   "Number of top candidate tokens to return log probabilities for at each step. Requires responseLogprobs.",
		"responseModalities":         "Modalities the model is permitted to produce (e.g. TEXT, IMAGE, AUDIO). Must be a subset of what the chosen model supports.",
		"mediaResolution":            "Resolution at which media inputs (images, video) are sampled. Higher resolutions capture more detail at the cost of more input tokens.",
		"audioTimestamp":             "Tags audio inputs with timestamps so the model can reference specific moments in its response.",
		"thinkingConfig":             "Extended-reasoning controls on Gemini 2.5+ thinking models — token budget, whether to surface thoughts, and reasoning level.",
		"imageConfig":                "Output image controls (aspect ratio, size, MIME type, person generation) used when the response includes an image modality.",
		"speechConfig":               "Voice and language settings used when the response includes an audio modality.",
		"safetySettings":             "Per-category thresholds controlling how aggressively the model blocks responses that may be harmful.",
		"toolConfig":                 "Shared configuration for the model's tool use — function calling mode, allowed function names, and retrieval settings.",
		"tools":                      "Built-in API tools made available to the model (GoogleSearch, Retrieval, CodeExecution, URLContext, FileSearch). Custom function tools must be registered via ai.WithTools() so they are wired into the Genkit runtime.",
		"labels":                     "User-defined key/value metadata used to break down billed charges.",
		"modelArmorConfig":           "Prompt and response sanitization via Google's Model Armor service. Mutually exclusive with safetySettings.",
		"modelSelectionConfig":       "Hints for model auto-selection, such as feature priority. Used when the request targets a model family rather than a specific model.",
		"routingConfig":              "Routes the request through Gemini's model router, either picking a model automatically or pinning to a specific one. Vertex AI only.",
		"enableEnhancedCivicAnswers": "Opts in to enhanced civic answers on supported models. Not available in Vertex AI.",
		"httpOptions":                "Per-request HTTP overrides — base URL, API version, headers, timeout — applied on top of plugin-level defaults.",
	},
	hidden: []string{
		// Managed by Genkit primitives; the plugin rejects these when set.
		"systemInstruction",                 // ai.WithSystemPrompt
		"cachedContent",                     // ai.WithCacheTTL
		"responseSchema",                    // ai.WithOutputType / ai.WithOutputSchema
		"responseMimeType",                  // ai.WithOutputType / ai.WithOutputSchema
		"responseJsonSchema",                // ai.WithOutputSchema
		"tools/items/functionDeclarations",  // ai.WithTools (built-in API tools on Tool stay visible)
		// Pinned to 1 by the plugin; the API only supports a single candidate.
		"candidateCount",
	},
}

// gicOverrides controls dev UI presentation of [genai.GenerateImagesConfig].
var gicOverrides = configOverrides{
	descriptions: map[string]string{
		"numberOfImages":           "Number of images to generate. Defaults to 4 when unset.",
		"aspectRatio":              "Aspect ratio of the generated images. Supported values: 1:1, 3:4, 4:3, 9:16, 16:9.",
		"negativePrompt":           "Free-form description of what to discourage in the generated images.",
		"guidanceScale":            "How strongly the model should adhere to the prompt. Higher values increase prompt alignment but may reduce image quality.",
		"seed":                     "Deterministic seed for image generation. Cannot be combined with addWatermark.",
		"safetyFilterLevel":        "How strictly to block unsafe content. Lower thresholds (e.g. BLOCK_LOW_AND_ABOVE) block more aggressively.",
		"personGeneration":         "Controls generation of people: ALLOW_ALL, ALLOW_ADULT (no minors), or DONT_ALLOW.",
		"outputMimeType":           "MIME type of the generated image (e.g. image/png, image/jpeg).",
		"outputCompressionQuality": "JPEG compression quality (only applies when outputMimeType is image/jpeg).",
		"addWatermark":             "Whether to embed a SynthID watermark in the generated images.",
		"imageSize":                "Size of the longest image dimension. Supported sizes are 1K and 2K (Imagen 3 does not support 2K).",
		"enhancePrompt":            "Lets the service rewrite the prompt for better results. Output may diverge slightly from the literal prompt.",
		"language":                 "Language of the text in the prompt.",
		"outputGcsUri":             "Cloud Storage URI to write generated images to. When unset, images are returned inline.",
		"labels":                   "User-defined key/value metadata used to break down billed charges.",
		"includeRaiReason":         "If true, includes the Responsible AI reason when an image is filtered out.",
		"includeSafetyAttributes":  "If true, returns per-image and per-prompt safety scores in the response.",
		"httpOptions":              "Per-request HTTP overrides — base URL, API version, headers, timeout — applied on top of plugin-level defaults.",
	},
}

// gvcOverrides controls dev UI presentation of [genai.GenerateVideosConfig].
var gvcOverrides = configOverrides{
	descriptions: map[string]string{
		"numberOfVideos":     "Number of videos to generate per request.",
		"fps":                "Frames per second for the generated video.",
		"durationSeconds":    "Length of the generated clip in seconds.",
		"seed":               "Deterministic RNG seed. Identical inputs with the same seed yield identical outputs.",
		"aspectRatio":        "Aspect ratio of the generated video. Supported values: 16:9 (landscape), 9:16 (portrait).",
		"resolution":         "Output video resolution. Supported values: 720p, 1080p.",
		"personGeneration":   "Controls generation of people: dont_allow or allow_adult (no minors).",
		"negativePrompt":     "Free-form description of what to discourage in the generated videos.",
		"enhancePrompt":      "Lets the service rewrite the prompt for better results. Output may diverge slightly from the literal prompt.",
		"generateAudio":      "If true, generates synchronized audio alongside the video.",
		"compressionQuality": "Trade off output file size against visual quality.",
		"outputGcsUri":       "Cloud Storage bucket to write generated videos to.",
		"pubsubTopic":        "Pub/Sub topic to publish progress notifications to during long-running generation.",
		"httpOptions":        "Per-request HTTP overrides — base URL, API version, headers, timeout — applied on top of plugin-level defaults.",
	},
}

// applyConfigOverrides mutates schema in place: removes hidden properties
// (top-level or via slash paths) and writes descriptions onto the
// remaining top-level ones.
func applyConfigOverrides(schema *jsonschema.Schema, o configOverrides) {
	if schema == nil || schema.Properties == nil {
		return
	}
	hideTop := make(map[string]struct{})
	for _, path := range o.hidden {
		if !strings.Contains(path, "/") {
			hideTop[path] = struct{}{}
			schema.Properties.Delete(path)
			continue
		}
		deleteAtPath(schema, strings.Split(path, "/"))
	}
	if len(hideTop) > 0 && len(schema.Required) > 0 {
		kept := schema.Required[:0]
		for _, r := range schema.Required {
			if _, drop := hideTop[r]; !drop {
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

// deleteAtPath descends through `items` and nested `properties` to remove a
// leaf property. Silently no-ops when the path doesn't exist (the upstream
// SDK may have renamed or removed the field).
func deleteAtPath(schema *jsonschema.Schema, parts []string) {
	cur := schema
	for _, part := range parts[:len(parts)-1] {
		if cur == nil {
			return
		}
		if part == "items" {
			cur = cur.Items
			continue
		}
		if cur.Properties == nil {
			return
		}
		next, ok := cur.Properties.Get(part)
		if !ok {
			return
		}
		cur = next
	}
	if cur != nil && cur.Properties != nil {
		cur.Properties.Delete(parts[len(parts)-1])
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
