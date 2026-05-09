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
	"context"
	"encoding/base64"
	"io"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

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
	genkit.DefineFlow(g, "text-to-speech-flow", func(ctx context.Context, input any) (string, error) {
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
			ai.WithPrompt("Say: Genkit is the best Gen AI library!"))
		if err != nil {
			return "", err
		}

		// base64 encoded audio
		text := resp.Text()
		return text, nil
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
