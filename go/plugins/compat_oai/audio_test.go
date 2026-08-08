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
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func newAudioPlugin(t *testing.T, handler http.HandlerFunc) *OpenAICompatible {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	plugin := &OpenAICompatible{
		Provider: "test",
		Opts: []option.RequestOption{
			option.WithAPIKey("test-key"),
			option.WithBaseURL(server.URL),
		},
	}
	plugin.Init(context.Background())
	return plugin
}

func TestSpeechModel(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/speech" {
			t.Errorf("path = %q, want /audio/speech", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		for key, want := range map[string]any{
			"model":           "tts-1-2026-01-01",
			"input":           "Hello",
			"voice":           "echo",
			"speed":           1.25,
			"response_format": "wav",
		} {
			if got := body[key]; got != want {
				t.Errorf("request[%q] = %v, want %v", key, got, want)
			}
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write([]byte{1, 2, 3, 4})
	})

	model := plugin.DefineSpeechModel("test", "tts-1", ai.ModelOptions{
		Versions: []string{"tts-1", "tts-1-2026-01-01"},
	})
	req := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello")},
		Config: SpeechConfig{
			Voice:          openai.AudioSpeechNewParamsVoiceEcho,
			Speed:          1.25,
			ResponseFormat: openai.AudioSpeechNewParamsResponseFormatWAV,
			Version:        "tts-1-2026-01-01",
		},
	}
	resp, err := model.Generate(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Request != req {
		t.Error("response did not preserve its request")
	}
	if got := resp.Message.Content[0].ContentType; got != "audio/wav" {
		t.Errorf("content type = %q, want audio/wav", got)
	}
	if got := resp.Message.Content[0].Text; got != "data:audio/wav;base64,AQIDBA==" {
		t.Errorf("media URL = %q, want encoded WAV data", got)
	}
}

func TestSpeechModelDefaultsVoiceAndFormat(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if got := body["voice"]; got != "alloy" {
			t.Errorf("voice = %v, want alloy", got)
		}
		if _, ok := body["response_format"]; ok {
			t.Error("default response_format should be omitted")
		}
		_, _ = w.Write([]byte("audio"))
	})

	model := plugin.DefineSpeechModel("test", "tts-1", ai.ModelOptions{})
	resp, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello")},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Message.Content[0].ContentType; got != "audio/mpeg" {
		t.Errorf("content type = %q, want audio/mpeg", got)
	}
}

func TestSpeechModelRejectsEmptyText(t *testing.T) {
	for _, input := range []string{"", " \t\n"} {
		plugin := newAudioPlugin(t, func(http.ResponseWriter, *http.Request) {
			t.Fatal("server should not be called")
		})
		model := plugin.DefineSpeechModel("test", "tts-1", ai.ModelOptions{})
		_, err := model.Generate(context.Background(), &ai.ModelRequest{
			Messages: []*ai.Message{ai.NewUserTextMessage(input)},
		}, nil)
		if err == nil || !strings.Contains(err.Error(), "non-empty text") {
			t.Fatalf("Generate() error = %v, want non-empty-text error", err)
		}
	}
}

func TestSpeechModelUnknownFormatUsesBinaryContentType(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("audio"))
	})
	model := plugin.DefineSpeechModel("test", "tts-1", ai.ModelOptions{
		ConfigSchema: map[string]any{"type": "object"},
	})
	resp, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello")},
		Config: SpeechConfig{
			ResponseFormat: openai.AudioSpeechNewParamsResponseFormat("future-format"),
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	part := resp.Message.Content[0]
	if got := part.ContentType; got != "application/octet-stream" {
		t.Errorf("content type = %q, want application/octet-stream", got)
	}
	if !strings.HasPrefix(part.Text, "data:application/octet-stream;base64,") {
		t.Errorf("media URL = %q, want binary data URI", part.Text)
	}
}

func TestSpeechModelSendsInstructions(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if got := body["instructions"]; got != "Speak warmly." {
			t.Errorf("instructions = %v, want Speak warmly.", got)
		}
		_, _ = w.Write([]byte("audio"))
	})
	model := plugin.DefineSpeechModel("test", "gpt-4o-mini-tts", ai.ModelOptions{
		ConfigSchema: map[string]any{"type": "object"},
	})
	_, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello")},
		Config:   map[string]any{"instructions": "Speak warmly."},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAudioModelsRejectStreaming(t *testing.T) {
	plugin := newAudioPlugin(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("server should not be called for streaming requests")
	})
	models := map[string]ai.Model{
		"speech":  plugin.DefineSpeechModel("test", "tts-1", ai.ModelOptions{}),
		"whisper": plugin.DefineWhisperModel("test", "whisper-1", ai.ModelOptions{}),
		"transcription": plugin.DefineTranscriptionModel(
			"test",
			"gpt-4o-transcribe",
			ai.ModelOptions{},
		),
	}
	requests := map[string]*ai.ModelRequest{
		"speech": {
			Messages: []*ai.Message{ai.NewUserTextMessage("Hello")},
		},
		"transcription": {
			Messages: []*ai.Message{ai.NewUserMessage(
				ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
			)},
		},
		"whisper": {
			Messages: []*ai.Message{ai.NewUserMessage(
				ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
			)},
		},
	}
	for name, model := range models {
		t.Run(name, func(t *testing.T) {
			_, err := model.Generate(
				context.Background(),
				requests[name],
				func(context.Context, *ai.ModelResponseChunk) error { return nil },
			)
			if err == nil || !strings.Contains(err.Error(), "streaming") {
				t.Fatalf("Generate() error = %v, want streaming-not-supported error", err)
			}
		})
	}
}

func TestTranscriptionModel(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/transcriptions" {
			t.Errorf("path = %q, want /audio/transcriptions", r.URL.Path)
		}
		reader, err := r.MultipartReader()
		if err != nil {
			t.Fatalf("MultipartReader: %v", err)
		}
		fields := readMultipartFields(t, reader)
		for key, want := range map[string]string{
			"model":           "whisper-1",
			"prompt":          "Transcribe this",
			"language":        "en",
			"response_format": "text",
			"file":            "audio bytes",
		} {
			if got := fields[key]; got != want {
				t.Errorf("request[%q] = %q, want %q", key, got, want)
			}
		}
		w.Header().Set("Content-Type", "text/plain")
		_, _ = io.WriteString(w, "Hello world")
	})

	model := plugin.DefineTranscriptionModel("test", "whisper-1", ai.ModelOptions{})
	req := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewTextPart("Transcribe this"),
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8gYnl0ZXM="),
		)},
		Config: map[string]any{"language": "en"},
		Output: &ai.ModelOutputConfig{Format: "text"},
	}
	resp, err := model.Generate(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Text(); got != "Hello world" {
		t.Errorf("response text = %q, want Hello world", got)
	}
}

func TestTranscriptionModelJSONResponse(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"Hello JSON","language":"en"}`)
	})
	model := plugin.DefineTranscriptionModel("test", "gpt-4o-transcribe", ai.ModelOptions{})
	resp, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
		)},
		Output: &ai.ModelOutputConfig{Format: "json"},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Text(); got != "Hello JSON" {
		t.Errorf("response text = %q, want Hello JSON", got)
	}
}

func TestGPTTranscriptionModelDefaultsToJSON(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		reader, err := r.MultipartReader()
		if err != nil {
			t.Fatalf("MultipartReader: %v", err)
		}
		fields := readMultipartFields(t, reader)
		if got := fields["response_format"]; got != "json" {
			t.Errorf("response_format = %q, want json", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"Hello JSON"}`)
	})

	model := plugin.DefineTranscriptionModel("test", "gpt-4o-transcribe", ai.ModelOptions{})
	resp, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
		)},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Text(); got != "Hello JSON" {
		t.Errorf("response text = %q, want Hello JSON", got)
	}
}

func TestVersionedGPTTranscriptionModelDefaultsToJSON(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		reader, err := r.MultipartReader()
		if err != nil {
			t.Fatalf("MultipartReader: %v", err)
		}
		fields := readMultipartFields(t, reader)
		if got := fields["model"]; got != "gpt-4o-transcribe-2026-01-01" {
			t.Errorf("model = %q, want versioned GPT transcription model", got)
		}
		if got := fields["response_format"]; got != "json" {
			t.Errorf("response_format = %q, want json", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"Versioned JSON"}`)
	})

	model := plugin.DefineTranscriptionModel("test", "custom-transcribe", ai.ModelOptions{
		Versions: []string{"custom-transcribe", "gpt-4o-transcribe-2026-01-01"},
	})
	resp, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
		)},
		Config: TranscriptionConfig{Version: "gpt-4o-transcribe-2026-01-01"},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Text(); got != "Versioned JSON" {
		t.Errorf("response text = %q, want Versioned JSON", got)
	}
}

func TestGPTTranscriptionModelRejectsNonJSONResponseFormat(t *testing.T) {
	plugin := newAudioPlugin(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("server should not be called")
	})
	model := plugin.DefineTranscriptionModel("test", "gpt-4o-transcribe", ai.ModelOptions{})
	_, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
		)},
		Output: &ai.ModelOutputConfig{Format: "text"},
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "only supports json") {
		t.Fatalf("Generate() error = %v, want JSON-only error", err)
	}
}

func TestTranscriptionModelRejectsIncompatibleOutput(t *testing.T) {
	plugin := newAudioPlugin(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("server should not be called")
	})
	model := plugin.DefineTranscriptionModel("test", "whisper-1", ai.ModelOptions{})
	_, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
		)},
		Config: TranscriptionConfig{ResponseFormat: openai.AudioResponseFormatSRT},
		Output: &ai.ModelOutputConfig{Format: "json"},
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "not compatible") {
		t.Fatalf("Generate() error = %v, want incompatible output error", err)
	}
}

func TestTranscriptionModelRejectsNonAudioMedia(t *testing.T) {
	plugin := newAudioPlugin(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("server should not be called")
	})
	model := plugin.DefineTranscriptionModel("test", "whisper-1", ai.ModelOptions{})
	_, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("image/png", "data:image/png;base64,aW1hZ2U="),
		)},
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "audio") {
		t.Fatalf("Generate() error = %v, want missing-audio error", err)
	}
}

func TestTranscriptionModelRejectsRemoteAudio(t *testing.T) {
	plugin := newAudioPlugin(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("server should not be called")
	})
	model := plugin.DefineTranscriptionModel("test", "whisper-1", ai.ModelOptions{})
	_, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("audio/wav", "https://example.com/audio.wav"),
		)},
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "data URI") {
		t.Fatalf("Generate() error = %v, want data-URI-required error", err)
	}
}

func TestTranscriptionModelRejectsUnsupportedAudioType(t *testing.T) {
	plugin := newAudioPlugin(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("server should not be called")
	})
	model := plugin.DefineTranscriptionModel("test", "whisper-1", ai.ModelOptions{})
	_, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("audio/aac", "data:audio/aac;base64,YXVkaW8="),
		)},
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "unsupported transcription media type") {
		t.Fatalf("Generate() error = %v, want unsupported-media-type error", err)
	}
}

func TestTranslationIgnoresTranscriptionChunkingStrategy(t *testing.T) {
	plugin := newAudioPlugin(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/translations" {
			t.Errorf("path = %q, want /audio/translations", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/plain")
		_, _ = io.WriteString(w, "Translated")
	})
	req := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
		)},
	}
	resp, err := plugin.generateTranscription(context.Background(), req, "whisper-1", TranscriptionConfig{
		ChunkingStrategy: make(chan int),
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Text(); got != "Translated" {
		t.Errorf("response text = %q, want Translated", got)
	}
}

func TestToChunkingStrategy(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
		want  string
	}{
		{name: "auto", value: "auto", want: `"auto"`},
		{
			name: "server VAD",
			value: TranscriptionChunkingStrategy{
				Type:              "server_vad",
				PrefixPaddingMS:   300,
				SilenceDurationMS: 500,
				Threshold:         0.5,
			},
			want: `{"type":"server_vad","prefix_padding_ms":300,"silence_duration_ms":500,"threshold":0.5}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			strategy, err := toChunkingStrategy(tc.value)
			if err != nil {
				t.Fatal(err)
			}
			got, err := json.Marshal(strategy)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tc.want {
				t.Errorf("chunking strategy = %s, want %s", got, tc.want)
			}
		})
	}
}

func TestToChunkingStrategyPreservesTypedValue(t *testing.T) {
	vad := &openai.AudioTranscriptionNewParamsChunkingStrategyVadConfig{
		Type: "server_vad",
	}
	typed := openai.AudioTranscriptionNewParamsChunkingStrategyUnion{
		OfAudioTranscriptionNewsChunkingStrategyVadConfig: vad,
	}

	got, err := toChunkingStrategy(typed)
	if err != nil {
		t.Fatal(err)
	}
	if got.OfAudioTranscriptionNewsChunkingStrategyVadConfig != vad {
		t.Error("toChunkingStrategy copied an already typed strategy")
	}
}

func TestAudioFilenameNormalizesContentType(t *testing.T) {
	for _, tc := range []struct {
		contentType string
		want        string
	}{
		{contentType: "audio/wav; codecs=1", want: "input.wav"},
		{contentType: " Audio/X-Wav ", want: "input.wav"},
		{contentType: "audio/wave", want: "input.wav"},
		{contentType: "audio/x-mp3", want: "input.mp3"},
		{contentType: "audio/m4a", want: "input.m4a"},
		{contentType: "audio/x-m4a", want: "input.m4a"},
		{contentType: "audio/x-ogg", want: "input.ogg"},
		{contentType: "audio/x-flac", want: "input.flac"},
		{contentType: "audio/x-webm", want: "input.webm"},
		{contentType: "audio/mpga", want: "input.mpga"},
	} {
		t.Run(tc.contentType, func(t *testing.T) {
			got, err := audioFilename(tc.contentType)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Errorf("audioFilename(%q) = %q, want %q", tc.contentType, got, tc.want)
			}
		})
	}

	if _, err := audioFilename("application/octet-stream"); err == nil {
		t.Error("audioFilename(application/octet-stream) succeeded, want error")
	}
}

func TestJSONOnlyTranscriptionSchemaHandlesUnexpectedStructure(t *testing.T) {
	for _, schema := range []map[string]any{
		nil,
		{"type": "object"},
		{"properties": "unexpected"},
		{"properties": map[string]any{"response_format": "unexpected"}},
	} {
		if got := jsonOnlyTranscriptionConfigSchema(schema); !reflect.DeepEqual(got, schema) {
			t.Errorf("jsonOnlyTranscriptionConfigSchema(%#v) = %#v, want unchanged schema", schema, got)
		}
	}
}

func readMultipartFields(t *testing.T, reader *multipart.Reader) map[string]string {
	t.Helper()
	fields := map[string]string{}
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("NextPart: %v", err)
		}
		data, err := io.ReadAll(part)
		if err != nil {
			t.Fatalf("read multipart field: %v", err)
		}
		fields[part.FormName()] = string(data)
		if part.FormName() == "file" && part.FileName() != "input.wav" {
			t.Errorf("file name = %q, want input.wav", part.FileName())
		}
	}
	return fields
}
