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

package googlegenai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"google.golang.org/genai"
)

func TestLyriaConfigFromRequest(t *testing.T) {
	t.Parallel()

	seed := 42
	tests := []struct {
		name           string
		request        *ai.ModelRequest
		expectError    bool
		expectSeed     *int
		expectCount    int
		expectNegative string
		expectLocation string
	}{
		{
			name: "valid config struct value",
			request: &ai.ModelRequest{
				Config: LyriaConfig{SampleCount: 2, NegativePrompt: "drums"},
			},
			expectCount:    2,
			expectNegative: "drums",
		},
		{
			name: "valid config struct pointer",
			request: &ai.ModelRequest{
				Config: &LyriaConfig{Seed: &seed, Location: "global"},
			},
			expectSeed:     &seed,
			expectLocation: "global",
		},
		{
			name: "nil config pointer",
			request: &ai.ModelRequest{
				Config: (*LyriaConfig)(nil),
			},
		},
		{
			name: "valid map config",
			request: &ai.ModelRequest{
				Config: map[string]any{
					"sampleCount":    3,
					"negativePrompt": "vocals",
				},
			},
			expectCount:    3,
			expectNegative: "vocals",
		},
		{
			name:    "nil config",
			request: &ai.ModelRequest{Config: nil},
		},
		{
			name: "invalid config type",
			request: &ai.ModelRequest{
				Config: &genai.GenerateContentConfig{},
			},
			expectError: true,
		},
		{
			name: "invalid map values",
			request: &ai.ModelRequest{
				Config: map[string]any{"sampleCount": "not-a-number"},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := lyriaConfigFromRequest(tt.request)
			if (err != nil) != tt.expectError {
				t.Fatalf("lyriaConfigFromRequest() error = %v, expectError %v", err, tt.expectError)
			}
			if tt.expectError {
				return
			}
			if got.SampleCount != tt.expectCount {
				t.Errorf("SampleCount = %d, want %d", got.SampleCount, tt.expectCount)
			}
			if got.NegativePrompt != tt.expectNegative {
				t.Errorf("NegativePrompt = %q, want %q", got.NegativePrompt, tt.expectNegative)
			}
			if got.Location != tt.expectLocation {
				t.Errorf("Location = %q, want %q", got.Location, tt.expectLocation)
			}
			if (got.Seed == nil) != (tt.expectSeed == nil) {
				t.Errorf("Seed nil mismatch: got %v want %v", got.Seed, tt.expectSeed)
			}
			if got.Seed != nil && tt.expectSeed != nil && *got.Seed != *tt.expectSeed {
				t.Errorf("Seed = %d, want %d", *got.Seed, *tt.expectSeed)
			}
		})
	}
}

func TestLyriaPredictURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		location string
		project  string
		model    string
		want     string
	}{
		{
			name:     "global location uses bare aiplatform host",
			location: "global",
			project:  "my-proj",
			model:    "lyria-002",
			want:     "https://aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/global/publishers/google/models/lyria-002:predict",
		},
		{
			name:     "regional location uses host prefix",
			location: "us-central1",
			project:  "my-proj",
			model:    "lyria-002",
			want:     "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/us-central1/publishers/google/models/lyria-002:predict",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := lyriaPredictURL(tt.location, tt.project, tt.model)
			if got != tt.want {
				t.Errorf("lyriaPredictURL() =\n  %s\nwant\n  %s", got, tt.want)
			}
		})
	}
}

func TestWarnLyriaCountMismatch(t *testing.T) {
	prev := slog.Default()
	defer slog.SetDefault(prev)

	var buf strings.Builder
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))

	t.Run("fewer than requested triggers warn", func(t *testing.T) {
		buf.Reset()
		warnLyriaCountMismatch(context.Background(), 4, 1)
		out := buf.String()
		if !strings.Contains(out, "level=WARN") {
			t.Errorf("expected WARN log, got: %s", out)
		}
		if !strings.Contains(out, "requested=4") || !strings.Contains(out, "received=1") {
			t.Errorf("log missing requested/received counts: %s", out)
		}
	})

	t.Run("matching counts emit nothing", func(t *testing.T) {
		buf.Reset()
		warnLyriaCountMismatch(context.Background(), 2, 2)
		if buf.Len() != 0 {
			t.Errorf("expected no log, got: %s", buf.String())
		}
	})

	t.Run("more than requested emits nothing", func(t *testing.T) {
		buf.Reset()
		warnLyriaCountMismatch(context.Background(), 1, 3)
		if buf.Len() != 0 {
			t.Errorf("expected no log, got: %s", buf.String())
		}
	})
}

func TestGenerateMusic_StreamingReturnsPublicError(t *testing.T) {
	t.Parallel()

	cb := func(context.Context, *ai.ModelResponseChunk) error { return nil }
	_, err := generateMusic(context.Background(), nil, "lyria-002", &ai.ModelRequest{}, cb)
	if err == nil {
		t.Fatal("expected error for streaming callback, got nil")
	}
	var ufe *core.UserFacingError
	if !errors.As(err, &ufe) {
		t.Fatalf("error %T is not *core.UserFacingError: %v", err, err)
	}
	if ufe.Status != core.INVALID_ARGUMENT {
		t.Errorf("status = %s, want INVALID_ARGUMENT", ufe.Status)
	}
}

func TestDoLyriaPredict_AttachesGenkitClientHeader(t *testing.T) {
	t.Parallel()

	var gotHeader string
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get(xGoogApiClientHeader)
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"predictions":[{"bytesBase64Encoded":"AAAA","mimeType":"audio/wav"}]}`)
	}))
	defer srv.Close()

	req := lyriaPredictRequest{
		Instances:  []lyriaInstance{{Prompt: "jazz"}},
		Parameters: lyriaParameters{SampleCount: 1},
	}
	resp, err := doLyriaPredict(context.Background(), srv.Client(), srv.URL, req)
	if err != nil {
		t.Fatalf("doLyriaPredict() error = %v", err)
	}
	if len(resp.Predictions) != 1 {
		t.Fatalf("expected 1 prediction, got %d", len(resp.Predictions))
	}
	if !strings.HasPrefix(gotHeader, "genkit-go/") {
		t.Errorf("x-goog-api-client = %q, want prefix %q", gotHeader, "genkit-go/")
	}

	var sent lyriaPredictRequest
	if err := json.Unmarshal(gotBody, &sent); err != nil {
		t.Fatalf("server got non-JSON body: %v", err)
	}
	if sent.Instances[0].Prompt != "jazz" {
		t.Errorf("prompt forwarded = %q, want %q", sent.Instances[0].Prompt, "jazz")
	}
}

func TestDoLyriaPredict_NonOKStatusReturnsError(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":"bad request"}`)
	}))
	defer srv.Close()

	_, err := doLyriaPredict(context.Background(), srv.Client(), srv.URL, lyriaPredictRequest{})
	if err == nil {
		t.Fatal("expected error for 400 response, got nil")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("error = %v, want it to mention 400", err)
	}
}

func TestTranslateLyriaResponse(t *testing.T) {
	t.Parallel()

	t.Run("empty predictions", func(t *testing.T) {
		resp := &lyriaPredictResponse{}
		res := translateLyriaResponse(resp, &ai.ModelRequest{})
		if res.FinishReason != ai.FinishReasonStop {
			t.Errorf("FinishReason = %s, want Stop", res.FinishReason)
		}
		if res.Message == nil || res.Message.Role != ai.RoleModel {
			t.Errorf("expected Role=model, got %+v", res.Message)
		}
		if len(res.Message.Content) != 0 {
			t.Errorf("expected 0 parts, got %d", len(res.Message.Content))
		}
	})

	t.Run("single prediction", func(t *testing.T) {
		resp := &lyriaPredictResponse{
			Predictions: []lyriaPrediction{
				{BytesBase64Encoded: "AAAA", MimeType: "audio/wav"},
			},
		}
		res := translateLyriaResponse(resp, &ai.ModelRequest{})
		if len(res.Message.Content) != 1 {
			t.Fatalf("expected 1 part, got %d", len(res.Message.Content))
		}
		p := res.Message.Content[0]
		if p.Kind != ai.PartMedia {
			t.Errorf("Kind = %v, want PartMedia", p.Kind)
		}
		if p.ContentType != "audio/wav" {
			t.Errorf("ContentType = %q, want audio/wav", p.ContentType)
		}
		if p.Text != "data:audio/wav;base64,AAAA" {
			t.Errorf("data URL = %q", p.Text)
		}
	})

	t.Run("multiple predictions", func(t *testing.T) {
		resp := &lyriaPredictResponse{
			Predictions: []lyriaPrediction{
				{BytesBase64Encoded: "A", MimeType: "audio/wav"},
				{BytesBase64Encoded: "B", MimeType: "audio/wav"},
				{BytesBase64Encoded: "C", MimeType: "audio/mpeg"},
			},
		}
		res := translateLyriaResponse(resp, &ai.ModelRequest{})
		if len(res.Message.Content) != 3 {
			t.Fatalf("expected 3 parts, got %d", len(res.Message.Content))
		}
		if res.Message.Content[2].ContentType != "audio/mpeg" {
			t.Errorf("3rd part mime = %q, want audio/mpeg", res.Message.Content[2].ContentType)
		}
	})
}
