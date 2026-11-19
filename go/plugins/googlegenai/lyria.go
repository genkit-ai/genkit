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

package googlegenai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"google.golang.org/genai"
)

const (
	// lyria002 is the only Lyria model served by the legacy `:predict`
	// endpoint. The Lyria 3 models (lyria-3-clip-preview, lyria-3-pro-preview)
	// use the Interactions API instead (as the JS plugin does) and are not
	// yet supported here: they are neither listed nor resolved.
	lyriaPredictModel = lyria002

	// maxLyriaSampleCount is a conservative client-side cap on SampleCount.
	// The Vertex docs state no maximum, so this is a plugin safeguard that
	// bounds response size rather than a documented limit.
	maxLyriaSampleCount = 4

	// maxLyriaResponseBytes caps the response body read into memory. A 30s
	// 48kHz WAV clip is roughly 7MB base64 encoded.
	maxLyriaResponseBytes = 64 << 20

	// maxLyriaErrorMessage truncates server error messages we surface.
	maxLyriaErrorMessage = 300
)

// isLyriaPredictModel reports whether name is a Lyria model served by the
// legacy `:predict` endpoint, the only kind generateMusic supports.
func isLyriaPredictModel(name string) bool {
	return name == lyriaPredictModel
}

// lyriaSupported reports whether a Lyria model can be listed, resolved or
// defined on the given client: only :predict models, and only on Vertex AI.
func lyriaSupported(client *genai.Client, name string) bool {
	return isLyriaPredictModel(name) && client.ClientConfig().Backend == genai.BackendVertexAI
}

// LyriaConfig controls a Vertex AI Lyria music generation request. It mirrors
// the JS LyriaConfigSchema used for the lyria-002 legacy endpoint.
type LyriaConfig struct {
	// NegativePrompt optionally describes what to exclude from the generated audio.
	NegativePrompt string `json:"negativePrompt,omitempty"`
	// Seed, when set, makes generation deterministic. Mutually exclusive with
	// SampleCount: the request is rejected if both are set, and sampleCount is
	// not sent when Seed is set.
	Seed *int `json:"seed,omitempty"`
	// SampleCount is the number of audio clips to generate (1 to 4). Defaults
	// to 1 when Seed is not set.
	SampleCount int `json:"sampleCount,omitempty"`
}

type lyriaInstance struct {
	Prompt         string `json:"prompt"`
	NegativePrompt string `json:"negativePrompt,omitempty"`
	Seed           *int   `json:"seed,omitempty"`
}

type lyriaParameters struct {
	SampleCount int `json:"sampleCount,omitempty"`
}

type lyriaPredictRequest struct {
	Instances []lyriaInstance `json:"instances"`
	// Parameters is omitted when seed is used: seed and sampleCount are
	// mutually exclusive.
	Parameters *lyriaParameters `json:"parameters,omitempty"`
}

type lyriaPrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	// AudioContent is the field name the Vertex reference documents for the
	// audio; the JS plugin reads bytesBase64Encoded. Accept either.
	AudioContent string `json:"audioContent"`
	MimeType     string `json:"mimeType"`
}

func (p lyriaPrediction) audio() string {
	if p.BytesBase64Encoded != "" {
		return p.BytesBase64Encoded
	}
	return p.AudioContent
}

type lyriaPredictResponse struct {
	Predictions []lyriaPrediction `json:"predictions"`
}

// defaultLyriaMimeType is the audio mime type assumed when Vertex does not
// include one in a prediction. lyria-002 returns WAV (RIFF) bytes.
const defaultLyriaMimeType = "audio/wav"

// translateLyriaResponse converts a raw Lyria predict response into an
// *ai.ModelResponse with one media part per returned audio clip. Predictions
// without audio (typically content-filtered) are skipped.
func translateLyriaResponse(resp *lyriaPredictResponse, input *ai.ModelRequest) *ai.ModelResponse {
	msg := &ai.Message{Role: ai.RoleModel}
	for _, p := range resp.Predictions {
		audio := p.audio()
		if audio == "" {
			continue
		}
		mime := p.MimeType
		if mime == "" {
			mime = defaultLyriaMimeType
		}
		msg.Content = append(msg.Content, ai.NewMediaPart(mime, "data:"+mime+";base64,"+audio))
	}
	return &ai.ModelResponse{
		FinishReason: ai.FinishReasonStop,
		Message:      msg,
		Request:      input,
		Usage: &ai.GenerationUsage{
			Custom: map[string]float64{"generations": float64(len(msg.Content))},
		},
	}
}

// warnLyriaCountMismatch logs a warning when Lyria returns fewer audio
// predictions than were requested via SampleCount. Vertex silently truncates
// in some cases (content filter, quota), and a silent drop is easy to miss.
func warnLyriaCountMismatch(ctx context.Context, requested, received int) {
	if received >= requested {
		return
	}
	logger.FromContext(ctx).Warn(
		"lyria returned fewer predictions than requested",
		"requested", requested,
		"received", received,
	)
}

// lyriaPredictURL builds the `:predict` URL for a Lyria model from the
// client's resolved base URL (which the genai SDK derives from the location:
// global host, multi-regional rep host or regional host, unless the plugin's
// BaseURL overrides it) and API version.
func lyriaPredictURL(baseURL, apiVersion, project, location, model string) string {
	return fmt.Sprintf(
		"%s/%s/projects/%s/locations/%s/publishers/google/models/%s:predict",
		strings.TrimRight(baseURL, "/"), apiVersion, project, location, model,
	)
}

// lyriaPrompt joins the text of the user messages. It rejects non-text parts
// rather than silently dropping them.
func lyriaPrompt(input *ai.ModelRequest) (string, error) {
	var parts []string
	for _, m := range input.Messages {
		if m.Role != ai.RoleUser {
			continue
		}
		for _, p := range m.Content {
			if !p.IsText() {
				return "", status.Errorf(status.ErrInvalidArgument, "lyria only accepts text prompts; got a non-text part")
			}
		}
		if text := m.Text(); text != "" {
			parts = append(parts, text)
		}
	}
	prompt := strings.Join(parts, "\n")
	if prompt == "" {
		return "", status.Errorf(status.ErrInvalidArgument, "lyria requires a non-empty text prompt")
	}
	return prompt, nil
}

// buildLyriaRequest validates cfg and builds the predict request. Seed and
// sampleCount are mutually exclusive on the API, so sampleCount is omitted
// when a seed is set and the default of 1 applies only without a seed.
func buildLyriaRequest(prompt string, cfg *LyriaConfig) (lyriaPredictRequest, int, error) {
	if cfg == nil {
		cfg = &LyriaConfig{}
	}
	if cfg.SampleCount < 0 || cfg.SampleCount > maxLyriaSampleCount {
		return lyriaPredictRequest{}, 0, status.Errorf(status.ErrInvalidArgument, "lyria sampleCount must be between 1 and %d, got %d", maxLyriaSampleCount, cfg.SampleCount)
	}
	if cfg.Seed != nil && cfg.SampleCount != 0 {
		return lyriaPredictRequest{}, 0, status.Errorf(status.ErrInvalidArgument, "lyria seed and sampleCount are mutually exclusive; set only one")
	}
	req := lyriaPredictRequest{
		Instances: []lyriaInstance{{
			Prompt:         prompt,
			NegativePrompt: cfg.NegativePrompt,
			Seed:           cfg.Seed,
		}},
	}
	want := 1
	if cfg.Seed == nil {
		if cfg.SampleCount != 0 {
			want = cfg.SampleCount
		}
		req.Parameters = &lyriaParameters{SampleCount: want}
	}
	return req, want, nil
}

// generateMusic calls the Vertex AI `predict` endpoint for Lyria music models.
// The `google.golang.org/genai` SDK does not expose a music-specific method, so
// this function issues the HTTPS call directly, reusing the authenticated
// *http.Client, base URL, API version and headers that the plugin configured
// on the genai.Client.
func generateMusic(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	cfg *LyriaConfig,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if cb != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "streaming mode is not supported for Lyria music generation")
	}
	if !isLyriaPredictModel(model) {
		return nil, status.Errorf(status.ErrUnimplemented, "lyria model %q is not supported: only %q is available (Lyria 3 uses the Interactions API, which this plugin does not implement yet)", model, lyriaPredictModel)
	}

	cc := client.ClientConfig()
	if cc.Backend != genai.BackendVertexAI {
		return nil, status.Errorf(status.ErrFailedPrecondition, "Lyria is only available through the Vertex AI backend")
	}
	if cc.HTTPClient == nil {
		return nil, status.Errorf(status.ErrFailedPrecondition, "lyria: genai.Client has no HTTP client configured")
	}
	if cc.Project == "" || cc.Location == "" {
		return nil, status.Errorf(status.ErrFailedPrecondition, "lyria requires a Vertex AI project and location; Express Mode (API key) and custom endpoints without them are not supported")
	}

	prompt, err := lyriaPrompt(input)
	if err != nil {
		return nil, err
	}
	req, want, err := buildLyriaRequest(prompt, cfg)
	if err != nil {
		return nil, err
	}

	version := cc.HTTPOptions.APIVersion
	if version == "" {
		version = "v1beta1"
	}
	url := lyriaPredictURL(cc.HTTPOptions.BaseURL, version, cc.Project, cc.Location, model)
	lp, err := doLyriaPredict(ctx, cc.HTTPClient, url, cc.HTTPOptions.Headers, req)
	if err != nil {
		return nil, err
	}

	resp := translateLyriaResponse(lp, input)
	if len(resp.Message.Content) == 0 {
		return nil, status.Errorf(status.ErrFailedPrecondition, "lyria returned no audio; the prompt was likely blocked by content filters")
	}
	warnLyriaCountMismatch(ctx, want, len(resp.Message.Content))
	return resp, nil
}

// doLyriaPredict issues an authenticated POST to the Vertex AI Lyria
// `:predict` endpoint at url. Headers are the plugin's configured headers;
// the Genkit client header is added when they do not already carry one.
func doLyriaPredict(ctx context.Context, httpClient *http.Client, url string, headers http.Header, req lyriaPredictRequest) (*lyriaPredictResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("lyria: marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("lyria: building request: %w", err)
	}
	for k, vs := range genkitClientHeader {
		httpReq.Header[k] = append([]string(nil), vs...)
	}
	for k, vs := range headers {
		httpReq.Header[http.CanonicalHeaderKey(k)] = append([]string(nil), vs...)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("lyria: request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read one byte past the cap to detect an oversize body.
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxLyriaResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("lyria: reading response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, wrapAPIError(lyriaAPIError(resp.StatusCode, data))
	}
	if len(data) > maxLyriaResponseBytes {
		return nil, status.Errorf(status.ErrInternal, "lyria: response exceeds the %d byte limit; try a smaller sampleCount", maxLyriaResponseBytes)
	}

	var lp lyriaPredictResponse
	if err := json.Unmarshal(data, &lp); err != nil {
		return nil, status.Errorf(status.ErrInternal, "lyria: unmarshaling response: %w", err)
	}
	return &lp, nil
}

// lyriaAPIError converts a non-2xx response into a genai.APIError so that
// wrapAPIError maps it to the right status (and retry delay). Only the
// structured error message is surfaced, truncated; the raw body is not.
func lyriaAPIError(code int, data []byte) genai.APIError {
	apiErr := genai.APIError{Code: code}
	var env struct {
		Error genai.APIError `json:"error"`
	}
	if err := json.Unmarshal(data, &env); err == nil {
		apiErr.Status = env.Error.Status
		apiErr.Details = env.Error.Details
		apiErr.Message = env.Error.Message
	}
	if len(apiErr.Message) > maxLyriaErrorMessage {
		apiErr.Message = apiErr.Message[:maxLyriaErrorMessage] + "..."
	}
	if apiErr.Message == "" {
		apiErr.Message = http.StatusText(code)
	}
	return apiErr
}
