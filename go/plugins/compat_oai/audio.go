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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"github.com/openai/openai-go"
)

var (
	// SpeechSupports describes text-to-speech model capabilities.
	SpeechSupports = ai.ModelSupports{
		Output: []string{"media"},
	}

	// TranscriptionSupports describes speech-to-text model capabilities.
	TranscriptionSupports = ai.ModelSupports{
		Media:  true,
		Output: []string{"text", "json"},
	}

	gptTranscriptionSupports = ai.ModelSupports{
		Media:  true,
		Output: []string{"json"},
	}
)

// SpeechConfig configures an OpenAI-compatible text-to-speech request.
type SpeechConfig struct {
	Voice          openai.AudioSpeechNewParamsVoice          `json:"voice,omitempty" jsonschema:"enum=alloy,enum=ash,enum=ballad,enum=coral,enum=echo,enum=fable,enum=onyx,enum=nova,enum=sage,enum=shimmer,enum=verse,default=alloy"`
	Instructions   string                                    `json:"instructions,omitempty"`
	Speed          float64                                   `json:"speed,omitempty" jsonschema:"minimum=0.25,maximum=4"`
	ResponseFormat openai.AudioSpeechNewParamsResponseFormat `json:"response_format,omitempty" jsonschema:"enum=mp3,enum=opus,enum=aac,enum=flac,enum=wav,enum=pcm"`
	Version        string                                    `json:"version,omitempty"`
}

// TranscriptionChunkingStrategy configures server-side voice activity detection.
type TranscriptionChunkingStrategy struct {
	Type              string  `json:"type"`
	PrefixPaddingMS   int     `json:"prefix_padding_ms,omitempty"`
	SilenceDurationMS int     `json:"silence_duration_ms,omitempty"`
	Threshold         float64 `json:"threshold,omitempty" jsonschema:"minimum=0,maximum=1"`
}

// TranscriptionConfig configures an OpenAI-compatible speech-to-text request.
type TranscriptionConfig struct {
	Temperature float64 `json:"temperature,omitempty"`
	// ChunkingStrategy is either the string "auto" or a server VAD configuration object.
	ChunkingStrategy       any                           `json:"chunking_strategy,omitempty"`
	Include                []openai.TranscriptionInclude `json:"include,omitempty"`
	Language               string                        `json:"language,omitempty"`
	TimestampGranularities []string                      `json:"timestamp_granularities,omitempty"`
	ResponseFormat         openai.AudioResponseFormat    `json:"response_format,omitempty" jsonschema:"enum=json,enum=text,enum=srt,enum=verbose_json,enum=vtt"`
	Version                string                        `json:"version,omitempty"`
}

// WhisperConfig configures Whisper transcription or translation requests.
type WhisperConfig struct {
	TranscriptionConfig
	Translate bool `json:"translate,omitempty" jsonschema:"default=false"`
}

var responseFormatMediaTypes = map[openai.AudioSpeechNewParamsResponseFormat]string{
	openai.AudioSpeechNewParamsResponseFormatMP3:  "audio/mpeg",
	openai.AudioSpeechNewParamsResponseFormatOpus: "audio/opus",
	openai.AudioSpeechNewParamsResponseFormatAAC:  "audio/aac",
	openai.AudioSpeechNewParamsResponseFormatFLAC: "audio/flac",
	openai.AudioSpeechNewParamsResponseFormatWAV:  "audio/wav",
	openai.AudioSpeechNewParamsResponseFormatPCM:  "audio/L16",
}

// DefineSpeechModel defines an OpenAI-compatible text-to-speech model.
func (o *OpenAICompatible) DefineSpeechModel(provider, id string, opts ai.ModelOptions) ai.Model {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("OpenAICompatible.Init not called")
	}

	if opts.Supports == nil {
		opts.Supports = &SpeechSupports
	}
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = core.InferSchemaMap(SpeechConfig{})
	}
	if opts.Versions == nil {
		opts.Versions = []string{id}
	}

	return ai.NewModel(api.NewName(provider, id), &opts, func(
		ctx context.Context,
		req *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		if cb != nil {
			return nil, errors.New("streaming is not supported for speech models")
		}
		config, err := parseAudioConfig[SpeechConfig](req.Config)
		if err != nil {
			return nil, fmt.Errorf("invalid speech config: %w", err)
		}
		if len(req.Messages) == 0 || req.Messages[0] == nil {
			return nil, errors.New("speech request requires a message")
		}
		input := req.Messages[0].Text()
		if strings.TrimSpace(input) == "" {
			return nil, errors.New("speech request requires non-empty text")
		}

		voice := config.Voice
		if voice == "" {
			voice = openai.AudioSpeechNewParamsVoiceAlloy
		}
		model := id
		if config.Version != "" {
			model = config.Version
		}
		params := openai.AudioSpeechNewParams{
			Input:          input,
			Model:          model,
			Voice:          voice,
			ResponseFormat: config.ResponseFormat,
		}
		if config.Speed != 0 {
			params.Speed = openai.Float(config.Speed)
		}
		if config.Instructions != "" {
			params.Instructions = openai.String(config.Instructions)
		}

		res, err := o.client.Audio.Speech.New(ctx, params)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
		audio, err := io.ReadAll(res.Body)
		if err != nil {
			return nil, fmt.Errorf("read speech response: %w", err)
		}

		format := config.ResponseFormat
		if format == "" {
			format = openai.AudioSpeechNewParamsResponseFormatMP3
		}
		contentType := responseFormatMediaTypes[format]
		if contentType == "" {
			contentType = "application/octet-stream"
		}
		dataURI := fmt.Sprintf("data:%s;base64,%s", contentType, base64.StdEncoding.EncodeToString(audio))
		return &ai.ModelResponse{
			Message:      ai.NewModelMessage(ai.NewMediaPart(contentType, dataURI)),
			FinishReason: ai.FinishReasonStop,
			Raw:          audio,
			Request:      req,
		}, nil
	})
}

// DefineTranscriptionModel defines an OpenAI-compatible speech-to-text model.
func (o *OpenAICompatible) DefineTranscriptionModel(provider, id string, opts ai.ModelOptions) ai.Model {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("OpenAICompatible.Init not called")
	}

	if opts.Supports == nil {
		if isGPTTranscriptionModel(id) {
			opts.Supports = &gptTranscriptionSupports
		} else {
			opts.Supports = &TranscriptionSupports
		}
	}
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = transcriptionConfigSchema(id)
	}
	if opts.Versions == nil {
		opts.Versions = []string{id}
	}

	return ai.NewModel(api.NewName(provider, id), &opts, func(
		ctx context.Context,
		req *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		if cb != nil {
			return nil, errors.New("streaming is not supported for transcription models")
		}
		config, err := parseAudioConfig[TranscriptionConfig](req.Config)
		if err != nil {
			return nil, fmt.Errorf("invalid transcription config: %w", err)
		}
		return o.generateTranscription(ctx, req, id, config, false)
	})
}

// DefineWhisperModel defines a Whisper model that supports transcription and translation.
func (o *OpenAICompatible) DefineWhisperModel(provider, id string, opts ai.ModelOptions) ai.Model {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("OpenAICompatible.Init not called")
	}

	if opts.Supports == nil {
		opts.Supports = &TranscriptionSupports
	}
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = core.InferSchemaMap(WhisperConfig{})
	}
	if opts.Versions == nil {
		opts.Versions = []string{id}
	}

	return ai.NewModel(api.NewName(provider, id), &opts, func(
		ctx context.Context,
		req *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		if cb != nil {
			return nil, errors.New("streaming is not supported for Whisper models")
		}
		config, err := parseAudioConfig[WhisperConfig](req.Config)
		if err != nil {
			return nil, fmt.Errorf("invalid Whisper config: %w", err)
		}
		return o.generateTranscription(ctx, req, id, config.TranscriptionConfig, config.Translate)
	})
}

func (o *OpenAICompatible) generateTranscription(
	ctx context.Context,
	req *ai.ModelRequest,
	id string,
	config TranscriptionConfig,
	translate bool,
) (*ai.ModelResponse, error) {
	media, prompt, err := transcriptionInput(req)
	if err != nil {
		return nil, err
	}
	if !strings.HasPrefix(media.Text, "data:") {
		return nil, errors.New("transcription audio must use a data URI")
	}
	contentType, audio, err := uri.Data(media)
	if err != nil {
		return nil, fmt.Errorf("read transcription media: %w", err)
	}
	filename, err := audioFilename(contentType)
	if err != nil {
		return nil, err
	}

	model := id
	if config.Version != "" {
		model = config.Version
	}
	format, err := transcriptionResponseFormat(
		req.Output,
		config.ResponseFormat,
		defaultTranscriptionResponseFormat(model),
	)
	if err != nil {
		return nil, err
	}
	if isGPTTranscriptionModel(model) && format != openai.AudioResponseFormatJSON {
		return nil, fmt.Errorf("model %s only supports json responses", model)
	}

	file := &audioFile{
		Reader:      bytes.NewReader(audio),
		filename:    filename,
		contentType: contentType,
	}
	if translate {
		params := audioTranslationParams(file, model, prompt, config, format)
		text, raw, err := o.audioTextResponse(ctx, "audio/translations", params, format)
		if err != nil {
			return nil, err
		}
		return transcriptionResponse(req, text, raw), nil
	}

	params, err := audioTranscriptionParams(file, model, prompt, config, format)
	if err != nil {
		return nil, err
	}
	text, raw, err := o.audioTextResponse(ctx, "audio/transcriptions", params, format)
	if err != nil {
		return nil, err
	}
	return transcriptionResponse(req, text, raw), nil
}

func audioTranslationParams(
	file *audioFile,
	model string,
	prompt string,
	config TranscriptionConfig,
	format openai.AudioResponseFormat,
) openai.AudioTranslationNewParams {
	params := openai.AudioTranslationNewParams{
		File:           file,
		Model:          model,
		ResponseFormat: openai.AudioTranslationNewParamsResponseFormat(format),
	}
	if prompt != "" {
		params.Prompt = openai.String(prompt)
	}
	if config.Temperature != 0 {
		params.Temperature = openai.Float(config.Temperature)
	}
	return params
}

func audioTranscriptionParams(
	file *audioFile,
	model string,
	prompt string,
	config TranscriptionConfig,
	format openai.AudioResponseFormat,
) (openai.AudioTranscriptionNewParams, error) {
	chunkingStrategy, err := toChunkingStrategy(config.ChunkingStrategy)
	if err != nil {
		return openai.AudioTranscriptionNewParams{}, fmt.Errorf("invalid chunking strategy: %w", err)
	}
	params := openai.AudioTranscriptionNewParams{
		File:                   file,
		Model:                  model,
		ChunkingStrategy:       chunkingStrategy,
		Include:                config.Include,
		ResponseFormat:         format,
		TimestampGranularities: config.TimestampGranularities,
	}
	if config.Language != "" {
		params.Language = openai.String(config.Language)
	}
	if prompt != "" {
		params.Prompt = openai.String(prompt)
	}
	if config.Temperature != 0 {
		params.Temperature = openai.Float(config.Temperature)
	}

	return params, nil
}

func toChunkingStrategy(value any) (openai.AudioTranscriptionNewParamsChunkingStrategyUnion, error) {
	var result openai.AudioTranscriptionNewParamsChunkingStrategyUnion
	if value == nil {
		return result, nil
	}
	switch value := value.(type) {
	case openai.AudioTranscriptionNewParamsChunkingStrategyUnion:
		return value, nil
	case *openai.AudioTranscriptionNewParamsChunkingStrategyUnion:
		if value == nil {
			return result, nil
		}
		return *value, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return result, err
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return result, err
	}
	return result, nil
}

func (o *OpenAICompatible) audioTextResponse(ctx context.Context, path string, params any, format openai.AudioResponseFormat) (string, any, error) {
	if format == openai.AudioResponseFormatJSON || format == openai.AudioResponseFormatVerboseJSON {
		var raw json.RawMessage
		if err := o.client.Post(ctx, path, params, &raw); err != nil {
			return "", nil, err
		}
		var result struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(raw, &result); err != nil {
			return "", nil, fmt.Errorf("parse transcription response: %w", err)
		}
		return result.Text, raw, nil
	}

	var result string
	if err := o.client.Post(ctx, path, params, &result); err != nil {
		return "", nil, err
	}
	return result, result, nil
}

func transcriptionResponse(req *ai.ModelRequest, text string, raw any) *ai.ModelResponse {
	return &ai.ModelResponse{
		Message:      ai.NewModelTextMessage(text),
		FinishReason: ai.FinishReasonStop,
		Raw:          raw,
		Request:      req,
	}
}

func parseAudioConfig[T any](config any) (T, error) {
	var zero T
	switch config := config.(type) {
	case nil:
		return zero, nil
	case T:
		return config, nil
	case *T:
		if config == nil {
			return zero, nil
		}
		return *config, nil
	case map[string]any:
		return base.MapToStruct[T](config)
	default:
		return zero, fmt.Errorf("unexpected config type: %T", config)
	}
}

func transcriptionInput(req *ai.ModelRequest) (*ai.Part, string, error) {
	if len(req.Messages) == 0 || req.Messages[0] == nil {
		return nil, "", errors.New("transcription request requires a message")
	}
	var media *ai.Part
	var prompt strings.Builder
	for _, part := range req.Messages[0].Content {
		if part == nil {
			continue
		}
		if media == nil && part.IsAudio() {
			media = part
		}
		if part.IsText() || part.IsData() {
			prompt.WriteString(part.Text)
		}
	}
	if media == nil {
		return nil, "", errors.New("no audio found in the transcription request")
	}
	return media, prompt.String(), nil
}

func transcriptionResponseFormat(
	output *ai.ModelOutputConfig,
	custom openai.AudioResponseFormat,
	defaultFormat openai.AudioResponseFormat,
) (openai.AudioResponseFormat, error) {
	outputFormat := ""
	if output != nil {
		outputFormat = output.Format
	}
	if outputFormat == "media" {
		return "", errors.New("output format media is not supported")
	}
	if outputFormat == "json" && custom != "" && custom != openai.AudioResponseFormatJSON && custom != openai.AudioResponseFormatVerboseJSON {
		return "", fmt.Errorf("custom response format %s is not compatible with output format json", custom)
	}
	if custom != "" {
		return custom, nil
	}
	if outputFormat != "" {
		return openai.AudioResponseFormat(outputFormat), nil
	}
	return defaultFormat, nil
}

func defaultTranscriptionResponseFormat(model string) openai.AudioResponseFormat {
	if isGPTTranscriptionModel(model) {
		return openai.AudioResponseFormatJSON
	}
	return openai.AudioResponseFormatText
}

func transcriptionConfigSchema(model string) map[string]any {
	schema := core.InferSchemaMap(TranscriptionConfig{})
	if !isGPTTranscriptionModel(model) {
		return schema
	}
	// The pinned OpenAI SDK supports only JSON responses for GPT transcription
	// models, which is intentionally stricter than the current canonical JS schema.
	return jsonOnlyTranscriptionConfigSchema(schema)
}

func jsonOnlyTranscriptionConfigSchema(schema map[string]any) map[string]any {
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		return schema
	}
	responseFormat, ok := properties["response_format"].(map[string]any)
	if !ok {
		return schema
	}

	schema = maps.Clone(schema)
	properties = maps.Clone(properties)
	schema["properties"] = properties
	responseFormat = maps.Clone(responseFormat)
	properties["response_format"] = responseFormat
	responseFormat["enum"] = []any{string(openai.AudioResponseFormatJSON)}
	responseFormat["default"] = string(openai.AudioResponseFormatJSON)
	return schema
}

func isGPTTranscriptionModel(model string) bool {
	for _, base := range []string{
		openai.AudioModelGPT4oTranscribe,
		openai.AudioModelGPT4oMiniTranscribe,
	} {
		if model == base || strings.HasPrefix(model, base+"-") {
			return true
		}
	}
	return false
}

type audioFile struct {
	*bytes.Reader
	filename    string
	contentType string
}

func (f *audioFile) Filename() string    { return f.filename }
func (f *audioFile) ContentType() string { return f.contentType }

func audioFilename(contentType string) (string, error) {
	if idx := strings.IndexByte(contentType, ';'); idx >= 0 {
		contentType = contentType[:idx]
	}
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	extensions := map[string]string{
		"audio/mpeg":   ".mp3",
		"audio/mp3":    ".mp3",
		"audio/x-mp3":  ".mp3",
		"audio/mpga":   ".mpga",
		"audio/x-mpga": ".mpga",
		"audio/mp4":    ".mp4",
		"audio/m4a":    ".m4a",
		"audio/x-m4a":  ".m4a",
		"audio/wav":    ".wav",
		"audio/x-wav":  ".wav",
		"audio/wave":   ".wav",
		"audio/ogg":    ".ogg",
		"audio/x-ogg":  ".ogg",
		"audio/flac":   ".flac",
		"audio/x-flac": ".flac",
		"audio/webm":   ".webm",
		"audio/x-webm": ".webm",
	}
	extension := extensions[contentType]
	if extension == "" {
		return "", fmt.Errorf("unsupported transcription media type %q", contentType)
	}
	return "input" + extension, nil
}
