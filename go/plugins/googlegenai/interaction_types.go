// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

// Types for the Gemini "interactions" REST endpoint used by Antigravity
// preview models. These mirror the JS google-genai plugin
// (interaction-types.ts). The interactions API is wire-snake-cased, so the
// JSON tags use snake_case to match the REST contract directly.

// antigravityAPIRevision is the value sent in the Api-Revision header on every
// interactions request. It pins the request to a known API contract version.
const antigravityAPIRevision = "2026-05-20"

// defaultGoogleAIBaseURL and defaultGoogleAIAPIVersion are used when the genai
// client config does not override them.
const (
	defaultGoogleAIBaseURL    = "https://generativelanguage.googleapis.com"
	defaultGoogleAIAPIVersion = "v1beta"
)

// interactionContent is a single content block within a step. A block is a
// tagged union discriminated by Type; the "fat struct" below carries the union
// of all variants' fields and we switch on Type when decoding/encoding.
type interactionContent struct {
	Type string `json:"type"`

	// text
	Text        string `json:"text,omitempty"`
	Annotations []any  `json:"annotations,omitempty"`

	// image / audio / video / document
	Data       string `json:"data,omitempty"`
	URI        string `json:"uri,omitempty"`
	MimeType   string `json:"mime_type,omitempty"`
	Resolution string `json:"resolution,omitempty"`

	// thought
	Signature string               `json:"signature,omitempty"`
	Summary   []interactionContent `json:"summary,omitempty"`

	// function_call / function_result
	Name      string         `json:"name,omitempty"`
	Arguments map[string]any `json:"arguments,omitempty"`
	ID        string         `json:"id,omitempty"`
	Result    any            `json:"result,omitempty"`
	CallID    string         `json:"call_id,omitempty"`
	IsError   bool           `json:"is_error,omitempty"`
}

// interactionStep is one entry in an interaction's timeline. Like
// interactionContent it is a tagged union over Type; a step may also be a bare
// content block (Step = ModelOutputStep | UserInputStep | Content | ...), so
// the content fields are present here too.
type interactionStep struct {
	Type string `json:"type"`

	// model_output / user_input
	Content []interactionContent `json:"content,omitempty"`

	// thought
	Summary   []interactionContent `json:"summary,omitempty"`
	Signature string               `json:"signature,omitempty"`

	// function_call / google_search_call / code_execution_call
	Name      string         `json:"name,omitempty"`
	Arguments map[string]any `json:"arguments,omitempty"`
	ID        string         `json:"id,omitempty"`

	// function_result / google_search_result / code_execution_result
	Result  any    `json:"result,omitempty"`
	CallID  string `json:"call_id,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
}

// modalityTokens is a per-modality token count in the usage breakdown.
type modalityTokens struct {
	Modality string `json:"modality,omitempty"`
	Tokens   int    `json:"tokens,omitempty"`
}

// interactionUsage reports token usage for an interaction. Field names are
// forced snake_case by the REST API.
type interactionUsage struct {
	TotalInputTokens       int              `json:"total_input_tokens,omitempty"`
	InputTokensByModality  []modalityTokens `json:"input_tokens_by_modality,omitempty"`
	TotalCachedTokens      int              `json:"total_cached_tokens,omitempty"`
	TotalOutputTokens      int              `json:"total_output_tokens,omitempty"`
	OutputTokensByModality []modalityTokens `json:"output_tokens_by_modality,omitempty"`
	TotalThoughtTokens     int              `json:"total_thought_tokens,omitempty"`
	TotalTokens            int              `json:"total_tokens,omitempty"`
}

// geminiInteraction is the response from creating (or fetching) an interaction.
type geminiInteraction struct {
	Model                 string            `json:"model,omitempty"`
	Agent                 string            `json:"agent,omitempty"`
	EnvironmentID         string            `json:"environment_id,omitempty"`
	ID                    string            `json:"id,omitempty"`
	PreviousInteractionID string            `json:"previous_interaction_id,omitempty"`
	Status                string            `json:"status,omitempty"`
	Created               string            `json:"created,omitempty"`
	Updated               string            `json:"updated,omitempty"`
	Role                  string            `json:"role,omitempty"`
	Steps                 []interactionStep `json:"steps,omitempty"`
	Usage                 *interactionUsage `json:"usage,omitempty"`
}
