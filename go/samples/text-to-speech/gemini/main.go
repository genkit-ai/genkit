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

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

type textToSpeechInput struct {
	Text       string `json:"text,omitempty"`
	Model      string `json:"model,omitempty"`
	VoiceName  string `json:"voiceName,omitempty"`
	OutputPath string `json:"outputPath,omitempty"`
}

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the recommended
	// practice.
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)

	// Define a simple flow that generates audio from text.
	// Gemini TTS commonly returns raw PCM as audio/L16. The flow writes a
	// playable WAV file by wrapping those PCM bytes with a WAV header.
	genkit.DefineFlow(g, "text-to-speech-flow", func(ctx context.Context, input textToSpeechInput) (string, error) {
		text := input.Text
		if text == "" {
			text = "Say: Genkit is the best Gen AI library!"
		}
		model := input.Model
		if model == "" {
			model = "googleai/gemini-2.5-flash-preview-tts"
		}
		voiceName := input.VoiceName
		if voiceName == "" {
			voiceName = "Algenib"
		}
		outputPath := input.OutputPath
		if outputPath == "" {
			outputPath = "./generated-tts.wav"
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName(model),
			ai.WithConfig(&genai.GenerateContentConfig{
				SpeechConfig: &genai.SpeechConfig{
					VoiceConfig: &genai.VoiceConfig{
						PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
							VoiceName: voiceName,
						},
					},
				},
			}),
			ai.WithPrompt(text))
		if err != nil {
			return "", err
		}

		part := firstAudioPart(resp)
		if part == nil {
			return "", errors.New("no audio media part found in model response")
		}
		if err := writePlayableAudio(outputPath, part); err != nil {
			return "", err
		}
		return fmt.Sprintf("wrote playable audio to %s", outputPath), nil
	})

	// Define a simple flow that generates audio transcripts from a given audio
	genkit.DefineFlow(g, "speech-to-text-flow", func(ctx context.Context, input any) (string, error) {
		audio, err := os.Open("./genkit.wav")
		if err != nil {
			return "", err
		}
		defer audio.Close()

		audioBytes, err := io.ReadAll(audio)
		if err != nil {
			return "", err
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-2.5-flash"),
			ai.WithMessages(ai.NewUserMessage(
				ai.NewTextPart("Can you transcribe the next audio?"),
				ai.NewMediaPart("audio/wav", "data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(audioBytes)))),
		)
		if err != nil {
			return "", err
		}

		return resp.Text(), nil
	})

	<-ctx.Done()
}

func firstAudioPart(resp *ai.ModelResponse) *ai.Part {
	if resp == nil || resp.Message == nil {
		return nil
	}
	for _, part := range resp.Message.Content {
		if part.IsAudio() {
			return part
		}
	}
	return nil
}

func writePlayableAudio(path string, part *ai.Part) error {
	contentType, data, err := decodeMediaData(part)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		mediaType = contentType
	}

	switch strings.ToLower(mediaType) {
	case "audio/l16":
		// Gemini TTS returns 16-bit little-endian PCM at 24 kHz. Raw PCM is valid
		// audio data but not directly playable by most browsers or media players.
		return os.WriteFile(path, wavFromPCM(data, 1, 24000, 16), 0644)
	default:
		return os.WriteFile(path, data, 0644)
	}
}

func decodeMediaData(part *ai.Part) (string, []byte, error) {
	if part == nil || !part.IsMedia() {
		return "", nil, errors.New("part is not media")
	}
	contentType := part.ContentType
	if payload, ok := strings.CutPrefix(part.Text, "data:"); ok {
		header, encoded, ok := strings.Cut(payload, ",")
		if !ok {
			return "", nil, errors.New("invalid media data URI")
		}
		if mediaType, _, ok := strings.Cut(header, ";"); ok && mediaType != "" {
			contentType = mediaType
		}
		data, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return "", nil, err
		}
		return contentType, data, nil
	}
	return "", nil, errors.New("media part is not inline data")
}

func wavFromPCM(pcm []byte, channels, sampleRate, bitsPerSample int) []byte {
	var buf bytes.Buffer
	dataSize := uint32(len(pcm))
	byteRate := uint32(sampleRate * channels * bitsPerSample / 8)
	blockAlign := uint16(channels * bitsPerSample / 8)

	buf.WriteString("RIFF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(36)+dataSize)
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(channels))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(&buf, binary.LittleEndian, byteRate)
	_ = binary.Write(&buf, binary.LittleEndian, blockAlign)
	_ = binary.Write(&buf, binary.LittleEndian, uint16(bitsPerSample))
	buf.WriteString("data")
	_ = binary.Write(&buf, binary.LittleEndian, dataSize)
	buf.Write(pcm)
	return buf.Bytes()
}
