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
	"errors"
	"fmt"

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
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Define a simple flow that generates an image of a given topic
	genkit.DefineFlow(g, "imageFlow", func(ctx context.Context, input string) ([]string, error) {
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash-image")
		if m == nil {
			return nil, errors.New("imageFlow: failed to find model")
		}

		if input == "" {
			input = `A little blue gopher with big eyes trying to learn Python,
				use a cartoon style, the story should be tragic because he
				chose the wrong programming language, the proper programing
				language for a gopher should be Go`
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature:        genai.Ptr[float32](0.5),
				ResponseModalities: []string{"IMAGE", "TEXT"},
			}),
			ai.WithPrompt(fmt.Sprintf(`generate a story about %s and for each scene, generate an image for it`, input)))
		if err != nil {
			return nil, err
		}

		story := []string{}
		for _, p := range resp.Message.Content {
			if p.IsMedia() || p.IsText() {
				story = append(story, p.Text)
			}
		}

		return story, nil
	})

	<-ctx.Done()
}
