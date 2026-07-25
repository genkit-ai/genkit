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

	genkit "github.com/firebase/genkit/go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	g, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
	if err != nil {
		log.Fatalf("failed to initialize Genkit: %v", err)
	}

	// Define a tool that returns media alongside its output.
	// This simulates a tool that takes a screenshot: the structured output
	// reports success, and tool.AttachParts adds the image itself to the
	// tool's multipart response.
	screenshot := g.DefineTool("screenshot", "Takes a screenshot",
		func(ctx context.Context, input any) (map[string]any, error) {
			rectangle := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHIAAABUAQMAAABk5vEVAAAABlBMVEX///8AAABVwtN+" +
				"AAAAI0lEQVR4nGNgGHaA/z8UHIDwOWASDqP8Uf7w56On/1FAQwAAVM0exw1hqwkAAAAASUVORK5CYII="
			tool.AttachParts(ctx, ai.NewMediaPart("image/png", rectangle))
			return map[string]any{"success": true}, nil
		},
	)

	// Define a simple flow that uses the multipart tool
	g.DefineStreamingFlow("cardFlow", func(ctx context.Context, input any, cb ai.ModelStreamCallback) (string, error) {
		resp, err := g.Generate(ctx,
			ai.WithModelName("googleai/gemini-3-pro-preview"),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](1.0),
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingLevel: genai.ThinkingLevelHigh,
				},
			}),
			ai.WithTools(screenshot),
			ai.WithStreaming(cb),
			ai.WithPrompt("Tell me what I'm seeing in the screen"),
		)
		if err != nil {
			return "", err
		}

		return resp.Text(), nil
	})

	<-ctx.Done()
}
