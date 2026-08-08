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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestImageGenerateParams(t *testing.T) {
	request := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("a cat wearing a hat")},
		Config: map[string]any{
			"version":         "dall-e-3-2026-01-01",
			"n":               2,
			"quality":         "hd",
			"response_format": "url",
			"size":            "1024x1024",
			"temperature":     0.5,
		},
	}

	params, err := imageGenerateParams("dall-e-3", request)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]any{
		"model":           "dall-e-3-2026-01-01",
		"prompt":          "a cat wearing a hat",
		"n":               float64(2),
		"quality":         "hd",
		"response_format": "url",
		"size":            "1024x1024",
	} {
		if value := got[key]; value != want {
			t.Errorf("params[%q] = %v, want %v", key, value, want)
		}
	}
	if _, ok := got["temperature"]; ok {
		t.Error("standard Genkit config field temperature was sent to the Images API")
	}
}

func TestImageGenerateParamsDefaults(t *testing.T) {
	for _, tc := range []struct {
		model              string
		wantResponseFormat string
	}{
		{model: "dall-e-3", wantResponseFormat: "b64_json"},
		{model: "gpt-image-1", wantResponseFormat: ""},
	} {
		t.Run(tc.model, func(t *testing.T) {
			params, err := imageGenerateParams(tc.model, &ai.ModelRequest{
				Messages: []*ai.Message{ai.NewUserTextMessage("a landscape")},
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := string(params.ResponseFormat); got != tc.wantResponseFormat {
				t.Errorf("ResponseFormat = %q, want %q", got, tc.wantResponseFormat)
			}
		})
	}
}

func TestImageGenerateParamsRejectsInvalidPrompt(t *testing.T) {
	for _, request := range []*ai.ModelRequest{
		nil,
		{},
		{Messages: []*ai.Message{ai.NewUserMessage(ai.NewMediaPart("image/png", "https://example.com/input.png"))}},
	} {
		if _, err := imageGenerateParams("dall-e-3", request); err == nil {
			t.Error("imageGenerateParams() succeeded without a text prompt")
		}
	}
}

func TestGenerateImageRejectsNilClient(t *testing.T) {
	_, err := generateImage(
		context.Background(),
		nil,
		"dall-e-3",
		&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("a mountain")}},
		nil,
	)
	if err == nil {
		t.Fatal("generateImage() succeeded with a nil client")
	}
}

func TestGenerateImageRejectsNilResult(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `null`)
	}))
	defer server.Close()

	client := openai.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	_, err := generateImage(
		context.Background(),
		&client,
		"dall-e-3",
		&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("a mountain")}},
		nil,
	)
	if err == nil {
		t.Fatal("generateImage() succeeded with a nil Images API result")
	}
}

func TestGenerateImage(t *testing.T) {
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/generations" {
			t.Errorf("request path = %q, want %q", r.URL.Path, "/images/generations")
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
			return
		}
		if err := json.Unmarshal(body, &requestBody); err != nil {
			t.Errorf("decoding request body: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"created":1,"data":[{"b64_json":"aGVsbG8="},{"url":"https://example.com/image.png"}]}`)
	}))
	defer server.Close()

	client := openai.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	request := &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("a mountain")}}
	response, err := generateImage(context.Background(), &client, "dall-e-3", request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := requestBody["prompt"]; got != "a mountain" {
		t.Errorf("prompt = %v, want %q", got, "a mountain")
	}
	if response.Request != request {
		t.Error("response did not preserve the originating request")
	}
	if response.Raw == nil {
		t.Error("response did not preserve the raw Images API response")
	}
	if response.FinishReason != ai.FinishReasonStop {
		t.Errorf("FinishReason = %q, want %q", response.FinishReason, ai.FinishReasonStop)
	}
	if got := len(response.Message.Content); got != 2 {
		t.Fatalf("len(Content) = %d, want 2", got)
	}
	if got := response.Message.Content[0].Text; got != "data:image/png;base64,aGVsbG8=" {
		t.Errorf("first media URL = %q, want base64 data URI", got)
	}
	if got := response.Message.Content[1].Text; got != "https://example.com/image.png" {
		t.Errorf("second media URL = %q, want URL", got)
	}
}

func TestGenerateImageUsesRequestedContentType(t *testing.T) {
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"created":1,"data":[{"b64_json":"aGVsbG8="}]}`)
	}))
	defer server.Close()

	client := openai.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	response, err := generateImage(context.Background(), &client, "gpt-image-1", &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("a mountain")},
		Config: ImageGenerationConfig{
			OutputFormat: openai.ImageGenerateParamsOutputFormatWebP,
			Style:        openai.ImageGenerateParamsStyleVivid,
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := requestBody["output_format"]; got != "webp" {
		t.Errorf("output_format = %v, want webp", got)
	}
	if _, ok := requestBody["style"]; ok {
		t.Error("GPT Image request contains DALL-E-only style")
	}
	part := response.Message.Content[0]
	if got := part.ContentType; got != "image/webp" {
		t.Errorf("content type = %q, want image/webp", got)
	}
	if got := part.Text; got != "data:image/webp;base64,aGVsbG8=" {
		t.Errorf("media URL = %q, want WebP data URI", got)
	}
}

func TestGenerateImageRejectsStreaming(t *testing.T) {
	_, err := generateImage(
		context.Background(),
		nil,
		"dall-e-3",
		&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("a mountain")}},
		func(context.Context, *ai.ModelResponseChunk) error { return nil },
	)
	if err == nil {
		t.Fatal("generateImage() succeeded in streaming mode")
	}
}
