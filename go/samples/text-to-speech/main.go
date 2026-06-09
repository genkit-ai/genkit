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

type ttsInput struct {
	Text string `json:"text"`
}

type ttsOutput struct {
	AudioPath string `json:"audioPath"`
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
		return os.WriteFile(path, wavFromPCM(data, 1, 24000, 16), 0644)
	default:
		return os.WriteFile(path, data, 0644)
	}
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

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the recommended
	// practice.
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithDefaultModel("googleai/gemini-2.5-flash-preview-tts"),
	)

	// Define a simple flow that generates an audio from a given text
	genkit.DefineFlow(g, "text-to-speech-flow", func(ctx context.Context, input *ttsInput) (string, error) {
		prompt := "Say: Genkit is the best Gen AI library!"
		if input != nil && input.Text != "" {
			prompt = fmt.Sprintf("Say: %s", input.Text)
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature:        genai.Ptr[float32](1.0),
				ResponseModalities: []string{"AUDIO"},
				SpeechConfig: &genai.SpeechConfig{
					VoiceConfig: &genai.VoiceConfig{
						PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
							VoiceName: "Algenib",
						},
					},
				},
			}),
			ai.WithPrompt(prompt))
		if err != nil {
			return "", err
		}

		// Inline audio is exposed as a data URI string.
		return resp.Media(), nil
	})

	// Define a simple flow that generates an audio from input text
	genkit.DefineFlow(g, "text-to-speech-flow-with-gemini-3.1-model", func(ctx context.Context, input *ttsInput) (*ttsOutput, error) {
		prompt := "Say: Genkit is the best Gen AI library!"
		if input != nil && input.Text != "" {
			prompt = fmt.Sprintf("Say: %s", input.Text)
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-3.1-flash-tts-preview"),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature:        genai.Ptr[float32](1.0),
				ResponseModalities: []string{"AUDIO"},
				SpeechConfig: &genai.SpeechConfig{
					VoiceConfig: &genai.VoiceConfig{
						PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
							VoiceName: "Algenib",
						},
					},
				},
			}),
			ai.WithPrompt(prompt))

		if err != nil {
			return nil, err
		}

		audio := firstAudioPart(resp)
		if audio == nil {
			return nil, errors.New("no audio part in response")
		}

		tmpFile, err := os.CreateTemp("", "genkit-tts-*.wav")
		if err != nil {
			return nil, err
		}
		path := tmpFile.Name()
		tmpFile.Close()

		if err := writePlayableAudio(path, audio); err != nil {
			return nil, err
		}

		return &ttsOutput{AudioPath: path}, nil
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
