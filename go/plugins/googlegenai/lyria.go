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
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/internal/base"
	"google.golang.org/genai"
)

// LyriaConfig controls a Vertex AI Lyria music generation request. It mirrors
// the JS LyriaConfigSchema used for the lyria-002 legacy endpoint.
type LyriaConfig struct {
	// NegativePrompt optionally describes what to exclude from the generated audio.
	NegativePrompt string `json:"negativePrompt,omitempty"`
	// Seed, when set, makes generation deterministic. Mutually exclusive with SampleCount.
	Seed *int `json:"seed,omitempty"`
	// SampleCount is the number of audio clips to generate. Defaults to 1.
	SampleCount int `json:"sampleCount,omitempty"`
	// Location overrides the plugin-level Vertex AI location for this request.
	// Lyria is typically only available in specific regions (e.g. "global").
	Location string `json:"location,omitempty"`
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
	Instances  []lyriaInstance `json:"instances"`
	Parameters lyriaParameters `json:"parameters"`
}

type lyriaPrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	MimeType           string `json:"mimeType"`
}

type lyriaPredictResponse struct {
	Predictions []lyriaPrediction `json:"predictions"`
}

// lyriaConfigFromRequest extracts a *LyriaConfig from a model request. The
// config may arrive as LyriaConfig, *LyriaConfig, map[string]any, or nil.
func lyriaConfigFromRequest(input *ai.ModelRequest) (*LyriaConfig, error) {
	var result LyriaConfig
	switch config := input.Config.(type) {
	case LyriaConfig:
		result = config
	case *LyriaConfig:
		if config != nil {
			result = *config
		}
	case map[string]any:
		r, err := base.MapToStruct[LyriaConfig](config)
		if err != nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("The Lyria configuration settings are not in the correct format. Check that the names and values match what the model expects: %v", err), nil)
		}
		result = r
	case nil:
		// empty but valid config
	default:
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("Invalid Lyria configuration type: %T. Expected *googlegenai.LyriaConfig.", input.Config), nil)
	}
	return &result, nil
}

// defaultLyriaMimeType is the audio mime type assumed when Vertex does not
// include one in a prediction. lyria-002 returns WAV (RIFF) bytes without
// labeling them.
const defaultLyriaMimeType = "audio/wav"

// translateLyriaResponse converts a raw Lyria predict response into an
// *ai.ModelResponse with one media part per returned audio clip.
func translateLyriaResponse(resp *lyriaPredictResponse, input *ai.ModelRequest) *ai.ModelResponse {
	msg := &ai.Message{Role: ai.RoleModel}
	for _, p := range resp.Predictions {
		mime := p.MimeType
		if mime == "" {
			mime = defaultLyriaMimeType
		}
		url := fmt.Sprintf("data:%s;base64,%s", mime, p.BytesBase64Encoded)
		msg.Content = append(msg.Content, ai.NewMediaPart(mime, url))
	}
	return &ai.ModelResponse{
		FinishReason: ai.FinishReasonStop,
		Message:      msg,
		Request:      input,
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

// lyriaPredictURL builds the Vertex AI `:predict` endpoint URL for a Lyria
// music model in the given project + location. Mirrors the host + version
// scheme in js/plugins/google-genai/src/vertexai/client.ts: the bare
// aiplatform host is used for the "global" location and a regional prefix
// is added otherwise.
func lyriaPredictURL(location, project, model string) string {
	host := "aiplatform.googleapis.com"
	if location != "global" {
		host = location + "-" + host
	}
	return fmt.Sprintf(
		"https://%s/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		host, project, location, model,
	)
}

// generateMusic calls the Vertex AI `predict` endpoint for Lyria music models.
// The `google.golang.org/genai` SDK does not expose a music-specific method, so
// this function issues the HTTPS call directly, reusing the authenticated
// *http.Client that the plugin configured on the genai.Client (credentials,
// quota project header, OpenTelemetry tracing).
func generateMusic(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if cb != nil {
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, "streaming mode is not supported for Lyria music generation", nil)
	}

	cc := client.ClientConfig()
	if cc.Backend != genai.BackendVertexAI {
		return nil, core.NewPublicError(core.FAILED_PRECONDITION, "Lyria is only available through the Vertex AI backend", nil)
	}
	if cc.HTTPClient == nil {
		return nil, core.NewPublicError(core.FAILED_PRECONDITION, "lyria: genai.Client has no HTTP client configured", nil)
	}

	cfg, err := lyriaConfigFromRequest(input)
	if err != nil {
		return nil, err
	}

	var parts []string
	for _, m := range input.Messages {
		if m.Role != ai.RoleUser {
			continue
		}
		if text := m.Text(); text != "" {
			parts = append(parts, text)
		}
	}
	userPrompt := strings.Join(parts, "\n")
	if userPrompt == "" {
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, "lyria requires a non-empty text prompt", nil)
	}

	req := lyriaPredictRequest{
		Instances: []lyriaInstance{{
			Prompt:         userPrompt,
			NegativePrompt: cfg.NegativePrompt,
			Seed:           cfg.Seed,
		}},
		Parameters: lyriaParameters{SampleCount: cfg.SampleCount},
	}
	if req.Parameters.SampleCount == 0 {
		req.Parameters.SampleCount = 1
	}

	location := cc.Location
	if cfg.Location != "" {
		location = cfg.Location
	}
	if location == "" {
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, "lyria requires a Vertex AI location", nil)
	}
	if cc.Project == "" {
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, "lyria requires a Vertex AI project id", nil)
	}

	url := lyriaPredictURL(location, cc.Project, model)
	lp, err := doLyriaPredict(ctx, cc.HTTPClient, url, req)
	if err != nil {
		return nil, err
	}
	if len(lp.Predictions) == 0 {
		return nil, core.NewPublicError(core.INTERNAL, "lyria: no predictions returned (possibly content-filtered)", nil)
	}
	warnLyriaCountMismatch(ctx, req.Parameters.SampleCount, len(lp.Predictions))

	return translateLyriaResponse(lp, input), nil
}

// doLyriaPredict issues an authenticated POST to the Vertex AI Lyria
// `:predict` endpoint at url. It attaches the standard Genkit client
// header on top of whatever the supplied httpClient already injects
// (credentials, quota project, OTel tracing).
func doLyriaPredict(ctx context.Context, httpClient *http.Client, url string, req lyriaPredictRequest) (*lyriaPredictResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("lyria: marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	for k, vs := range genkitClientHeader {
		for _, v := range vs {
			httpReq.Header.Add(k, v)
		}
	}

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("lyria: request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, core.NewPublicError(core.INTERNAL, fmt.Sprintf("lyria: %s: %s", resp.Status, string(data)), nil)
	}

	var lp lyriaPredictResponse
	if err := json.Unmarshal(data, &lp); err != nil {
		return nil, fmt.Errorf("lyria: unmarshaling response: %w", err)
	}
	return &lp, nil
}
