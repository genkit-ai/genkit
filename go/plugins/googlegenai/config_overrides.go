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
//
// All path application is best-effort: if the upstream SDK renames or
// removes a field, the corresponding entry silently no-ops rather than
// panicking. The cost is that stale entries quietly stop applying — caught
// by the snapshot tests, not at runtime.
type configOverrides struct {
	// descriptions maps a JSON property path to the help text shown as the
	// field's tooltip in the dev UI. Keys may be a top-level property name
	// ("temperature") or a dotted path with "[]" denoting a descent into an
	// array's item shape ("safetySettings[].category").
	descriptions map[string]string
	// hidden lists JSON property paths to remove from the schema. Same
	// notation as descriptions. Use this for fields the plugin rejects at
	// runtime or that the Genkit framework manages directly.
	hidden []string
}

// gccOverrides controls dev UI presentation of [genai.GenerateContentConfig].
var gccOverrides = configOverrides{
	descriptions: map[string]string{
		// Top-level fields.
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

		// httpOptions sub-fields.
		"httpOptions.baseUrl":              "Overrides the plugin-configured API endpoint for this request.",
		"httpOptions.apiVersion":           "Overrides the plugin-configured API version for this request (e.g. v1, v1beta).",
		"httpOptions.timeout":              "Per-request timeout in milliseconds.",
		"httpOptions.headers":              "Additional HTTP headers to send with this request.",
		"httpOptions.extraBody":            "Extra fields merged into the request body. Must match the underlying REST API's shape.",
		"httpOptions.baseUrlResourceScope": "Scope at which baseUrl applies (PROJECT, LOCATION, etc.). Vertex AI only.",

		// safetySettings[] sub-fields.
		"safetySettings[].category":  "Harm category this setting applies to (e.g. HARM_CATEGORY_HATE_SPEECH, HARM_CATEGORY_DANGEROUS_CONTENT).",
		"safetySettings[].threshold": "Probability threshold at or above which the response is blocked (BLOCK_LOW_AND_ABOVE blocks the most; BLOCK_NONE blocks nothing).",
		"safetySettings[].method":    "Whether to score by probability or severity. Defaults to probability. Vertex AI only.",

		// toolConfig sub-fields.
		"toolConfig.functionCallingConfig":                      "Controls when the model may call tools.",
		"toolConfig.functionCallingConfig.mode":                 "AUTO: the model decides whether to call a tool. ANY: the model must call one of the allowed tools. NONE: the model never calls tools.",
		"toolConfig.functionCallingConfig.allowedFunctionNames": "Names the model is allowed to call. Only used when mode is ANY.",
		"toolConfig.retrievalConfig":                            "Geographic and language hints used by retrieval-grounded tools.",
		"toolConfig.retrievalConfig.latLng":                     "Caller's geographic location, used to bias retrieval toward locally relevant results.",
		"toolConfig.retrievalConfig.languageCode":               "Caller's language as an ISO 639-1 code, used to bias retrieval results.",
		"toolConfig.includeServerSideToolInvocations":           "If true, the response includes the server-side tool invocations (e.g. GoogleSearch queries) the model performed.",

		// thinkingConfig sub-fields.
		"thinkingConfig.includeThoughts": "Whether to surface the model's internal reasoning in the response. Only applies on models that expose thoughts.",
		"thinkingConfig.thinkingBudget":  "Maximum thinking tokens the model may spend. Higher values enable deeper reasoning at higher cost. Used by Gemini 2.5+.",
		"thinkingConfig.thinkingLevel":   "Discrete reasoning level (MINIMAL, LOW, MEDIUM, HIGH). Higher levels enable deeper reasoning at higher cost. Used by Gemini 3+.",

		// imageConfig sub-fields.
		"imageConfig.aspectRatio":              "Aspect ratio of generated images (e.g. 1:1, 3:4, 4:3, 9:16, 16:9, 21:9).",
		"imageConfig.imageSize":                "Size of the longest dimension (1K, 2K, 4K). Defaults to 1K.",
		"imageConfig.personGeneration":         "Controls generation of people: ALLOW_ALL, ALLOW_ADULT (no minors), or ALLOW_NONE.",
		"imageConfig.prominentPeople":          "Controls generation of celebrities specifically. Overridden to off if personGeneration is ALLOW_NONE.",
		"imageConfig.outputMimeType":           "MIME type of the generated image (e.g. image/png, image/jpeg).",
		"imageConfig.outputCompressionQuality": "JPEG compression quality (only applies when outputMimeType is image/jpeg).",

		// speechConfig sub-fields.
		"speechConfig.languageCode":            "ISO 639-1 language code for speech synthesis.",
		"speechConfig.voiceConfig":             "Voice for a single-speaker response. Mutually exclusive with multiSpeakerVoiceConfig.",
		"speechConfig.multiSpeakerVoiceConfig": "Per-speaker voices for a multi-speaker response. Mutually exclusive with voiceConfig.",

		// routingConfig sub-fields.
		"routingConfig.autoMode":   "Pick the model automatically from the request content. Mutually exclusive with manualMode.",
		"routingConfig.manualMode": "Pin the request to a specific model name. Mutually exclusive with autoMode.",

		// modelArmorConfig sub-fields.
		"modelArmorConfig.promptTemplateName":   "Resource name of the Model Armor template applied to the prompt (projects/.../locations/.../templates/...).",
		"modelArmorConfig.responseTemplateName": "Resource name of the Model Armor template applied to the response.",

		// modelSelectionConfig sub-fields.
		"modelSelectionConfig.featureSelectionPreference": "Preference for which model features to prioritize (PRIORITIZE_QUALITY, BALANCED, PRIORITIZE_COST, etc.).",
	},
	hidden: []string{
		// Managed by Genkit primitives; the plugin rejects these when set.
		"systemInstruction",            // ai.WithSystemPrompt
		"cachedContent",                // ai.WithCacheTTL
		"responseSchema",               // ai.WithOutputType / ai.WithOutputSchema
		"responseMimeType",             // ai.WithOutputType / ai.WithOutputSchema
		"responseJsonSchema",           // ai.WithOutputSchema
		"tools[].functionDeclarations", // ai.WithTools (built-in API tools on Tool stay visible)
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

		// httpOptions sub-fields.
		"httpOptions.baseUrl":              "Overrides the plugin-configured API endpoint for this request.",
		"httpOptions.apiVersion":           "Overrides the plugin-configured API version for this request (e.g. v1, v1beta).",
		"httpOptions.timeout":              "Per-request timeout in milliseconds.",
		"httpOptions.headers":              "Additional HTTP headers to send with this request.",
		"httpOptions.extraBody":            "Extra fields merged into the request body. Must match the underlying REST API's shape.",
		"httpOptions.baseUrlResourceScope": "Scope at which baseUrl applies (PROJECT, LOCATION, etc.). Vertex AI only.",
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

		// httpOptions sub-fields.
		"httpOptions.baseUrl":              "Overrides the plugin-configured API endpoint for this request.",
		"httpOptions.apiVersion":           "Overrides the plugin-configured API version for this request (e.g. v1, v1beta).",
		"httpOptions.timeout":              "Per-request timeout in milliseconds.",
		"httpOptions.headers":              "Additional HTTP headers to send with this request.",
		"httpOptions.extraBody":            "Extra fields merged into the request body. Must match the underlying REST API's shape.",
		"httpOptions.baseUrlResourceScope": "Scope at which baseUrl applies (PROJECT, LOCATION, etc.). Vertex AI only.",
	},
}

// applyConfigOverrides mutates schema in place: removes hidden properties
// and writes descriptions onto the remaining ones. Best-effort — paths that
// no longer resolve (because the upstream SDK renamed or removed a field)
// silently no-op rather than panicking.
func applyConfigOverrides(schema *jsonschema.Schema, o configOverrides) {
	if schema == nil || schema.Properties == nil {
		return
	}
	hideTop := make(map[string]struct{})
	for _, path := range o.hidden {
		steps := parsePath(path)
		if len(steps) == 1 {
			hideTop[steps[0]] = struct{}{}
		}
		deleteAtPath(schema, steps)
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
	for path, desc := range o.descriptions {
		if target := schemaAtPath(schema, parsePath(path)); target != nil {
			target.Description = desc
		}
	}
}

// parsePath splits an override path into navigation steps. Each step is
// either a property name or the literal "[]" meaning "descend into an
// array's item schema." Examples:
//
//	"systemInstruction"             -> ["systemInstruction"]
//	"tools[].functionDeclarations"  -> ["tools", "[]", "functionDeclarations"]
//	"foo[].bar[].baz"               -> ["foo", "[]", "bar", "[]", "baz"]
func parsePath(path string) []string {
	var steps []string
	for _, tok := range strings.Split(path, ".") {
		if name := strings.TrimSuffix(tok, "[]"); name != tok {
			steps = append(steps, name, "[]")
		} else {
			steps = append(steps, tok)
		}
	}
	return steps
}

// schemaAtPath descends a schema by walking `Items` for "[]" steps and
// `Properties` for named ones. Returns nil if any step doesn't resolve —
// callers should treat that as a no-op, not an error.
func schemaAtPath(schema *jsonschema.Schema, steps []string) *jsonschema.Schema {
	cur := schema
	for _, step := range steps {
		if cur == nil {
			return nil
		}
		if step == "[]" {
			cur = cur.Items
			continue
		}
		if cur.Properties == nil {
			return nil
		}
		next, ok := cur.Properties.Get(step)
		if !ok {
			return nil
		}
		cur = next
	}
	return cur
}

// deleteAtPath removes the leaf property at the given path from its parent's
// Properties. Silent no-op if the path doesn't resolve.
func deleteAtPath(schema *jsonschema.Schema, steps []string) {
	if len(steps) == 0 {
		return
	}
	leaf := steps[len(steps)-1]
	if leaf == "[]" {
		return
	}
	parent := schemaAtPath(schema, steps[:len(steps)-1])
	if parent == nil || parent.Properties == nil {
		return
	}
	parent.Properties.Delete(leaf)
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
