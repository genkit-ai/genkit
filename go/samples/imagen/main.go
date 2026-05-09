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
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.VertexAI{}))

	genkit.DefineFlow(g, "image-generation", func(ctx context.Context, input string) ([]string, error) {
		r, err := genkit.Generate(ctx, g,
			ai.WithModelName("vertexai/imagen-3.0-generate-001"),
			ai.WithPrompt("Generate an image of %s", input),
			ai.WithConfig(&genai.GenerateImagesConfig{
				NumberOfImages:    2,
				NegativePrompt:    "night",
				AspectRatio:       "9:16",
				SafetyFilterLevel: genai.SafetyFilterLevelBlockLowAndAbove,
				PersonGeneration:  genai.PersonGenerationAllowAll,
				Language:          genai.ImagePromptLanguageEn,
				AddWatermark:      true,
				OutputMIMEType:    "image/jpeg",
			}),
		)
		if err != nil {
			log.Fatal(err)
		}

		var images []string
		for _, m := range r.Message.Content {
			images = append(images, m.Text)
		}
		return images, nil
	})

	<-ctx.Done()
}
