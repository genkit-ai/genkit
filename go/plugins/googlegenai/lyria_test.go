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
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"google.golang.org/genai"
)

// lyriaServer is a fake Vertex AI endpoint that records the last request.
type lyriaServer struct {
	*httptest.Server
	mu     sync.Mutex
	path   string
	header http.Header
	body   map[string]any
}

func newLyriaServer(t *testing.T, code int, resp string) *lyriaServer {
	t.Helper()
	ls := &lyriaServer{}
	ls.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		ls.mu.Lock()
		ls.path = r.URL.Path
		ls.header = r.Header.Clone()
		ls.body = nil
		_ = json.Unmarshal(data, &ls.body)
		ls.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_, _ = io.WriteString(w, resp)
	}))
	t.Cleanup(ls.Close)
	return ls
}

func vertexTestClient(t *testing.T, baseURL string, mutate func(*genai.ClientConfig)) *genai.Client {
	t.Helper()
	cfg := &genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    "my-proj",
		Location:   "us-central1",
		HTTPClient: http.DefaultClient,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: baseURL,
			Headers: http.Header{"X-Custom": []string{"yes"}},
		},
	}
	if mutate != nil {
		mutate(cfg)
	}
	c, err := genai.NewClient(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return c
}

func lyriaReq(prompt string) *ai.ModelRequest {
	return &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage(prompt)}}
}

const okLyriaResp = `{"predictions":[{"bytesBase64Encoded":"AAAA","mimeType":"audio/wav"}]}`

func intPtr(i int) *int { return &i }

func TestGenerateMusic_Request(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		cfg        *LyriaConfig
		wantParams any // nil means "parameters" must be absent
		wantInst   map[string]any
	}{
		{
			name:       "default sampleCount is 1",
			cfg:        &LyriaConfig{},
			wantParams: map[string]any{"sampleCount": float64(1)},
			wantInst:   map[string]any{"prompt": "jazz"},
		},
		{
			name:       "explicit sampleCount and negative prompt",
			cfg:        &LyriaConfig{SampleCount: 3, NegativePrompt: "vocals"},
			wantParams: map[string]any{"sampleCount": float64(3)},
			wantInst:   map[string]any{"prompt": "jazz", "negativePrompt": "vocals"},
		},
		{
			name:     "seed omits sampleCount",
			cfg:      &LyriaConfig{Seed: intPtr(7)},
			wantInst: map[string]any{"prompt": "jazz", "seed": float64(7)},
		},
		{
			name:     "zero seed is still sent and omits sampleCount",
			cfg:      &LyriaConfig{Seed: intPtr(0)},
			wantInst: map[string]any{"prompt": "jazz", "seed": float64(0)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := newLyriaServer(t, 200, okLyriaResp)
			client := vertexTestClient(t, srv.URL, nil)
			resp, err := generateMusic(context.Background(), client, "lyria-002", lyriaReq("jazz"), tt.cfg, nil)
			if err != nil {
				t.Fatalf("generateMusic: %v", err)
			}
			if len(resp.Message.Content) != 1 || resp.Message.Content[0].Text != "data:audio/wav;base64,AAAA" {
				t.Errorf("content = %+v", resp.Message.Content)
			}
			if resp.Usage == nil || resp.Usage.Custom["generations"] != 1 {
				t.Errorf("usage = %+v, want generations=1", resp.Usage)
			}

			wantPath := "/v1beta1/projects/my-proj/locations/us-central1/publishers/google/models/lyria-002:predict"
			if srv.path != wantPath {
				t.Errorf("path = %q, want %q", srv.path, wantPath)
			}
			if got := srv.header.Get(xGoogApiClientHeader); !strings.HasPrefix(got, "genkit-go/") {
				t.Errorf("x-goog-api-client = %q", got)
			}
			if srv.header.Get("X-Custom") != "yes" {
				t.Errorf("plugin header not forwarded: %v", srv.header)
			}
			insts := srv.body["instances"].([]any)
			if len(insts) != 1 {
				t.Fatalf("instances = %v", insts)
			}
			inst := insts[0].(map[string]any)
			if len(inst) != len(tt.wantInst) {
				t.Errorf("instance = %v, want %v", inst, tt.wantInst)
			}
			for k, v := range tt.wantInst {
				if inst[k] != v {
					t.Errorf("instance[%s] = %v, want %v", k, inst[k], v)
				}
			}
			params, has := srv.body["parameters"]
			if tt.wantParams == nil {
				if has {
					t.Errorf("parameters = %v, want absent", params)
				}
			} else if m := params.(map[string]any); m["sampleCount"] != tt.wantParams.(map[string]any)["sampleCount"] {
				t.Errorf("parameters = %v, want %v", params, tt.wantParams)
			}
		})
	}
}

func TestGenerateMusic_APIVersionAndBaseURL(t *testing.T) {
	t.Parallel()
	srv := newLyriaServer(t, 200, okLyriaResp)
	client := vertexTestClient(t, srv.URL+"/gateway/", func(c *genai.ClientConfig) {
		c.HTTPOptions.APIVersion = "v1"
	})
	if _, err := generateMusic(context.Background(), client, "lyria-002", lyriaReq("x"), &LyriaConfig{}, nil); err != nil {
		t.Fatal(err)
	}
	want := "/gateway/v1/projects/my-proj/locations/us-central1/publishers/google/models/lyria-002:predict"
	if srv.path != want {
		t.Errorf("path = %q, want %q", srv.path, want)
	}
}

func TestGenerateMusic_InvalidInputs(t *testing.T) {
	t.Parallel()
	srv := newLyriaServer(t, 200, okLyriaResp)
	client := vertexTestClient(t, srv.URL, nil)

	media := &ai.ModelRequest{Messages: []*ai.Message{{
		Role:    ai.RoleUser,
		Content: []*ai.Part{ai.NewTextPart("hi"), ai.NewMediaPart("image/png", "data:image/png;base64,AA==")},
	}}}
	tests := []struct {
		name string
		req  *ai.ModelRequest
		cfg  *LyriaConfig
		cb   ai.ModelStreamCallback
		want error
	}{
		{"streaming", lyriaReq("x"), nil, func(context.Context, *ai.ModelResponseChunk) error { return nil }, status.ErrInvalidArgument},
		{"empty prompt", lyriaReq(""), nil, nil, status.ErrInvalidArgument},
		{"non-text part", media, nil, nil, status.ErrInvalidArgument},
		{"seed and sampleCount", lyriaReq("x"), &LyriaConfig{Seed: intPtr(1), SampleCount: 2}, nil, status.ErrInvalidArgument},
		{"negative sampleCount", lyriaReq("x"), &LyriaConfig{SampleCount: -1}, nil, status.ErrInvalidArgument},
		{"sampleCount too large", lyriaReq("x"), &LyriaConfig{SampleCount: maxLyriaSampleCount + 1}, nil, status.ErrInvalidArgument},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := generateMusic(context.Background(), client, "lyria-002", tt.req, tt.cfg, tt.cb)
			if !errors.Is(err, tt.want) {
				t.Errorf("err = %v, want %v", err, tt.want)
			}
		})
	}
	if srv.path != "" {
		t.Errorf("server was called (%s) for an invalid request", srv.path)
	}
}

func TestGenerateMusic_UnsupportedBackendOrModel(t *testing.T) {
	t.Parallel()

	t.Run("google ai backend", func(t *testing.T) {
		client := newTestClient(t, "")
		_, err := generateMusic(context.Background(), client, "lyria-002", lyriaReq("x"), nil, nil)
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Errorf("err = %v, want FAILED_PRECONDITION", err)
		}
	})
	t.Run("lyria 3", func(t *testing.T) {
		client := vertexTestClient(t, "http://127.0.0.1:1", nil)
		_, err := generateMusic(context.Background(), client, "lyria-3-clip-preview", lyriaReq("x"), nil, nil)
		if !errors.Is(err, status.ErrUnimplemented) {
			t.Errorf("err = %v, want UNIMPLEMENTED", err)
		}
	})
	t.Run("express mode has no project or location", func(t *testing.T) {
		client := vertexTestClient(t, "http://127.0.0.1:1", func(c *genai.ClientConfig) {
			c.Project, c.Location, c.APIKey = "", "", "secret-key-value"
		})
		_, err := generateMusic(context.Background(), client, "lyria-002", lyriaReq("x"), nil, nil)
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Fatalf("err = %v, want FAILED_PRECONDITION", err)
		}
		if strings.Contains(err.Error(), "secret-key-value") {
			t.Errorf("error leaks the API key: %v", err)
		}
	})
}

func TestGenerateMusic_Responses(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		code      int
		body      string
		want      error // nil means success
		wantParts int
	}{
		{"no predictions", 200, `{"predictions":[]}`, status.ErrFailedPrecondition, 0},
		{"empty audio only", 200, `{"predictions":[{"bytesBase64Encoded":"","mimeType":"audio/wav"}]}`, status.ErrFailedPrecondition, 0},
		{"empty audio skipped", 200, `{"predictions":[{"bytesBase64Encoded":""},{"bytesBase64Encoded":"QQ=="}]}`, nil, 1},
		{"audioContent alias", 200, `{"predictions":[{"audioContent":"QQ==","mimeType":"audio/wav"}]}`, nil, 1},
		{"malformed json", 200, `{"predictions":`, status.ErrInternal, 0},
		{"400", 400, `{"error":{"code":400,"message":"bad","status":"INVALID_ARGUMENT"}}`, status.ErrInvalidArgument, 0},
		{"403", 403, `{"error":{"code":403,"message":"nope","status":"PERMISSION_DENIED"}}`, status.ErrPermissionDenied, 0},
		{"404 without body", 404, ``, status.ErrNotFound, 0},
		{"429", 429, `{"error":{"code":429,"message":"slow down","status":"RESOURCE_EXHAUSTED"}}`, status.ErrResourceExhausted, 0},
		{"500", 500, `not json`, status.ErrInternal, 0},
		{"503", 503, `{"error":{"code":503,"status":"UNAVAILABLE"}}`, status.ErrUnavailable, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := newLyriaServer(t, tt.code, tt.body)
			client := vertexTestClient(t, srv.URL, nil)
			resp, err := generateMusic(context.Background(), client, "lyria-002", lyriaReq("x"), nil, nil)
			if tt.want == nil {
				if err != nil {
					t.Fatalf("err = %v", err)
				}
				if got := len(resp.Message.Content); got != tt.wantParts {
					t.Errorf("parts = %d, want %d", got, tt.wantParts)
				}
				return
			}
			if !errors.Is(err, tt.want) {
				t.Errorf("err = %v, want %v", err, tt.want)
			}
			if tt.code >= 400 && strings.Contains(err.Error(), "not json") {
				t.Errorf("error leaks the raw body: %v", err)
			}
		})
	}
}

func TestGenerateMusic_ErrorMessageTruncated(t *testing.T) {
	t.Parallel()
	long := strings.Repeat("a", 5000)
	srv := newLyriaServer(t, 400, `{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"`+long+`"}}`)
	client := vertexTestClient(t, srv.URL, nil)
	_, err := generateMusic(context.Background(), client, "lyria-002", lyriaReq("x"), nil, nil)
	if err == nil || len(err.Error()) > 1000 {
		t.Errorf("error length = %d, want truncated", len(err.Error()))
	}
}

func TestGenerateMusic_OversizeResponse(t *testing.T) {
	t.Parallel()
	big := `{"predictions":[{"bytesBase64Encoded":"` + strings.Repeat("A", maxLyriaResponseBytes) + `"}]}`
	srv := newLyriaServer(t, 200, big)
	client := vertexTestClient(t, srv.URL, nil)
	_, err := generateMusic(context.Background(), client, "lyria-002", lyriaReq("x"), nil, nil)
	if !errors.Is(err, status.ErrInternal) || !strings.Contains(err.Error(), "limit") {
		t.Errorf("err = %v, want INTERNAL mentioning the limit", err)
	}
}

func TestGenerateMusic_TransportErrorWrapped(t *testing.T) {
	t.Parallel()
	srv := newLyriaServer(t, 200, okLyriaResp)
	client := vertexTestClient(t, srv.URL, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := generateMusic(ctx, client, "lyria-002", lyriaReq("x"), nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want it to wrap context.Canceled", err)
	}
}

func TestLyriaPredictURL(t *testing.T) {
	t.Parallel()
	got := lyriaPredictURL("https://aiplatform.googleapis.com/", "v1beta1", "p", "global", "lyria-002")
	want := "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/publishers/google/models/lyria-002:predict"
	if got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

// TestLyriaSDKBaseURLs pins the hosts the SDK derives for each location kind,
// which generateMusic relies on instead of building its own.
func TestLyriaSDKBaseURLs(t *testing.T) {
	t.Parallel()
	for loc, want := range map[string]string{
		"global":      "https://aiplatform.googleapis.com/",
		"us":          "https://aiplatform.us.rep.googleapis.com/",
		"us-central1": "https://us-central1-aiplatform.googleapis.com/",
	} {
		c, err := genai.NewClient(context.Background(), &genai.ClientConfig{
			Backend: genai.BackendVertexAI, Project: "p", Location: loc, HTTPClient: http.DefaultClient,
		})
		if err != nil {
			t.Fatal(err)
		}
		if got := c.ClientConfig().HTTPOptions.BaseURL; got != want {
			t.Errorf("location %q: base URL = %q, want %q", loc, got, want)
		}
	}
}

func TestLyriaClassificationAndSupports(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"lyria-002", "lyria-3-clip-preview", "lyria-3-pro-preview"} {
		if ClassifyModel(name) != ModelTypeLyria {
			t.Errorf("ClassifyModel(%q) != ModelTypeLyria", name)
		}
	}
	if !isLyriaPredictModel("lyria-002") || isLyriaPredictModel("lyria-3-clip-preview") {
		t.Error("only lyria-002 is a :predict model")
	}
	opts := GetModelOptions("lyria-002", vertexAIProvider)
	if opts.Supports == nil || opts.Supports.Media {
		t.Errorf("lyria-002 supports = %+v, want text-only input", opts.Supports)
	}
}

// lyriaModelListServer serves model listings for both backends.
func lyriaModelListServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"models":[{"name":"models/lyria-002"},{"name":"models/gemini-2.5-flash"}],`+
			`"publisherModels":[{"name":"publishers/google/models/lyria-002"},`+
			`{"name":"publishers/google/models/lyria-3-clip-preview"},`+
			`{"name":"publishers/google/models/lyria-3-pro-preview"},`+
			`{"name":"publishers/google/models/gemini-2.5-flash"}]}`)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func actionNames(descs []api.ActionDesc) []string {
	var out []string
	for _, d := range descs {
		out = append(out, d.Name)
	}
	return out
}

func TestLyriaListActions(t *testing.T) {
	t.Parallel()
	srv := lyriaModelListServer(t)

	t.Run("vertex lists only lyria-002", func(t *testing.T) {
		client := vertexTestClient(t, srv.URL, nil)
		names := actionNames(listActions(context.Background(), client, catalog{provider: vertexAIProvider}))
		var lyria []string
		for _, n := range names {
			if strings.Contains(n, "lyria") {
				lyria = append(lyria, n)
			}
		}
		if len(lyria) != 1 || lyria[0] != "vertexai/lyria-002" {
			t.Errorf("lyria actions = %v, want [vertexai/lyria-002]", lyria)
		}
	})
	t.Run("google ai lists none", func(t *testing.T) {
		client := newTestClient(t, srv.URL)
		for _, n := range actionNames(listActions(context.Background(), client, catalog{provider: googleAIProvider})) {
			if strings.Contains(n, "lyria") {
				t.Errorf("Google AI listed %q", n)
			}
		}
	})
}

func TestLyriaResolveAction(t *testing.T) {
	t.Parallel()
	vertex := vertexTestClient(t, "http://127.0.0.1:1", nil)
	googleAI := newTestClient(t, "")

	tests := []struct {
		name   string
		client *genai.Client
		prov   string
		id     string
		want   bool
	}{
		{"vertex lyria-002", vertex, vertexAIProvider, "lyria-002", true},
		{"vertex lyria-3 clip", vertex, vertexAIProvider, "lyria-3-clip-preview", false},
		{"vertex lyria-3 pro", vertex, vertexAIProvider, "lyria-3-pro-preview", false},
		{"google ai lyria-002", googleAI, googleAIProvider, "lyria-002", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveAction(tt.client, catalog{provider: tt.prov}, api.ActionTypeModel, tt.id)
			if (got != nil) != tt.want {
				t.Errorf("resolveAction() = %v, want resolved=%v", got, tt.want)
			}
		})
	}
}

func TestLyriaRejectUnsupported(t *testing.T) {
	t.Parallel()
	vertex := vertexTestClient(t, "http://127.0.0.1:1", nil)
	googleAI := newTestClient(t, "")
	if err := rejectUnsupportedLyria(vertex, vertexAIProvider, "lyria-002"); err != nil {
		t.Errorf("vertex lyria-002 rejected: %v", err)
	}
	if err := rejectUnsupportedLyria(vertex, vertexAIProvider, "gemini-2.5-flash"); err != nil {
		t.Errorf("non-lyria rejected: %v", err)
	}
	for _, c := range []struct {
		client *genai.Client
		id     string
	}{{vertex, "lyria-3-clip-preview"}, {googleAI, "lyria-002"}} {
		if err := rejectUnsupportedLyria(c.client, "p", c.id); !errors.Is(err, status.ErrInvalidArgument) {
			t.Errorf("rejectUnsupportedLyria(%q) = %v, want INVALID_ARGUMENT", c.id, err)
		}
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

	t.Run("missing mime type defaults to audio/wav", func(t *testing.T) {
		resp := &lyriaPredictResponse{
			Predictions: []lyriaPrediction{
				{BytesBase64Encoded: "UklGRg==", MimeType: ""},
			},
		}
		res := translateLyriaResponse(resp, &ai.ModelRequest{})
		if len(res.Message.Content) != 1 {
			t.Fatalf("expected 1 part, got %d", len(res.Message.Content))
		}
		p := res.Message.Content[0]
		if p.ContentType != "audio/wav" {
			t.Errorf("ContentType = %q, want audio/wav (Vertex omits mimeType in real responses)", p.ContentType)
		}
		if p.Text != "data:audio/wav;base64,UklGRg==" {
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
