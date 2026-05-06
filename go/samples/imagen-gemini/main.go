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
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

type imageFlowResult struct {
	Model   string   `json:"model"`
	Prompt  string   `json:"prompt"`
	Caption string   `json:"caption"`
	Images  []string `json:"images"`
}

func generateGemini31Lite(ctx context.Context, g *genkit.Genkit, modelName, input string) (string, error) {
	prompt := input
	if prompt == "" {
		prompt = "bananas"
	}

	return genkit.GenerateText(ctx, g,
		ai.WithModel(googlegenai.ModelRef(modelName, &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{
				ThinkingLevel: genai.ThinkingLevelMinimal,
			},
		})),
		ai.WithPrompt("Write one funny sentence about %s.", prompt),
	)
}

func generateGemini31Image(ctx context.Context, g *genkit.Genkit, modelName, input string) (*imageFlowResult, error) {
	prompt := input
	if prompt == "" {
		prompt = "a robot cat wearing sunglasses"
	}

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(googlegenai.ModelRef(modelName, &genai.GenerateContentConfig{
			ResponseModalities: []string{"IMAGE", "TEXT"},
			ThinkingConfig: &genai.ThinkingConfig{
				ThinkingLevel: genai.ThinkingLevelMinimal,
			},
		})),
		ai.WithPrompt("Generate an image of %s and include a short caption.", prompt),
	)
	if err != nil {
		return nil, fmt.Errorf("could not generate image response: %w", err)
	}

	out := &imageFlowResult{
		Model:   modelName,
		Prompt:  prompt,
		Caption: resp.Text(),
		Images:  []string{},
	}
	for _, p := range resp.Message.Content {
		if !p.IsMedia() || p.Text == "" {
			continue
		}
		imageData := p.Text
		// Preserve existing URIs (data:, http(s), gs://, etc.); wrap only raw base64
		// image content for preview clients.
		if !strings.HasPrefix(imageData, "data:") &&
			!strings.Contains(imageData, "://") {
			imageData = fmt.Sprintf("data:%s;base64,%s", p.ContentType, imageData)
		}
		out.Images = append(out.Images, imageData)
	}

	return out, nil
}

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the recommended
	// practice. The Vertex AI plugin is added conditionally when GOOGLE_CLOUD_PROJECT
	// and a location are set, so the gemini-3.1 Vertex flows only register when
	// that config is present. Vertex's gemini-3.1-flash-*-preview models are
	// Global-only — set GOOGLE_CLOUD_LOCATION=global.
	plugins := []api.Plugin{&googlegenai.GoogleAI{}}
	hasVertex := hasVertexAIConfig()
	if hasVertex {
		plugins = append(plugins, &googlegenai.VertexAI{})
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugins...))

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

	genkit.DefineFlow(g, "gemini31LiteGoogleAIFlow", func(ctx context.Context, input string) (string, error) {
		return generateGemini31Lite(ctx, g, "googleai/gemini-3.1-flash-lite-preview", input)
	})

	genkit.DefineFlow(g, "gemini31ImageGoogleAIFlow", func(ctx context.Context, input string) (*imageFlowResult, error) {
		return generateGemini31Image(ctx, g, "googleai/gemini-3.1-flash-image-preview", input)
	})

	if hasVertex {
		genkit.DefineFlow(g, "gemini31LiteVertexAIFlow", func(ctx context.Context, input string) (string, error) {
			return generateGemini31Lite(ctx, g, "vertexai/gemini-3.1-flash-lite-preview", input)
		})

		genkit.DefineFlow(g, "gemini31ImageVertexAIFlow", func(ctx context.Context, input string) (*imageFlowResult, error) {
			return generateGemini31Image(ctx, g, "vertexai/gemini-3.1-flash-image-preview", input)
		})
	}

	<-ctx.Done()
}

func hasVertexAIConfig() bool {
	if os.Getenv("GOOGLE_CLOUD_PROJECT") == "" {
		return false
	}
	return os.Getenv("GOOGLE_CLOUD_LOCATION") != "" || os.Getenv("GOOGLE_CLOUD_REGION") != ""
}
