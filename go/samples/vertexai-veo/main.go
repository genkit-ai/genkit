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
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize with Vertex AI plugin. Ensure GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.VertexAI{}))

	genkit.DefineFlow(g, "text-to-video", func(ctx context.Context, input string) (string, error) {
		if input == "" {
			input = "A futuristic city at sunset, flying cars, cyberpunk style"
		}
		operation, err := genkit.GenerateOperation(ctx, g,
			ai.WithMessages(ai.NewUserTextMessage(input)),
			ai.WithModelName("vertexai/veo-3.1-generate-preview"),
			ai.WithConfig(&genai.GenerateVideosConfig{
				NumberOfVideos:  1,
				AspectRatio:     "16:9",
				DurationSeconds: genai.Ptr(int32(4)),
			}),
		)
		if err != nil {
			log.Fatalf("Failed to start video generation: %v", err)
		}
		printStatus(operation)

		operation, err = waitForCompletion(ctx, g, operation)
		if err != nil {
			log.Fatalf("Operation failed: %v", err)
		}
		log.Println("Video generation completed successfully!")

		if err := saveGeneratedVideo(operation, "veo3_vertexai_video.mp4"); err != nil {
			log.Fatalf("Failed to save video: %v", err)
		}

		return "Video successfully saved to veo3_vertexai_video.mp4", nil
	})

	<-ctx.Done()
}

// waitForCompletion polls the operation status until it completes.
func waitForCompletion(ctx context.Context, g *genkit.Genkit, op *core.Operation[*ai.ModelResponse]) (*core.Operation[*ai.ModelResponse], error) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for !op.Done {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		case <-ticker.C:
			updatedOp, err := genkit.CheckModelOperation(ctx, g, op)
			if err != nil {
				return nil, fmt.Errorf("failed to check status: %w", err)
			}

			if updatedOp.Error != nil {
				return nil, fmt.Errorf("operation error: %w", updatedOp.Error)
			}

			printStatus(updatedOp)
			op = updatedOp
		}
	}

	return op, nil
}

// printStatus prints the current status message from the operation.
func printStatus(op *core.Operation[*ai.ModelResponse]) {
	if op.Output != nil && !op.Done && op.Output.Message != nil && len(op.Output.Message.Content) > 0 {
		log.Printf("Status Message: %s", op.Output.Message.Content[0].Text)
	}
}

// saveGeneratedVideo extracts the video from the operation output and saves it to disk.
// Vertex AI returns raw VideoBytes encoded as a data URI by the Genkit plugin.
func saveGeneratedVideo(operation *core.Operation[*ai.ModelResponse], filename string) error {
	if operation.Output == nil || operation.Output.Message == nil {
		return fmt.Errorf("operation output is empty")
	}

	for _, part := range operation.Output.Message.Content {
		if part.IsMedia() && part.Text != "" {
			if strings.HasPrefix(part.Text, "data:") {
				// Vertex AI returns the raw video encoded as base64 in the text field
				commaIndex := strings.Index(part.Text, ",")
				if commaIndex == -1 {
					return fmt.Errorf("invalid data URI format")
				}

				base64Data := part.Text[commaIndex+1:]
				videoBytes, err := base64.StdEncoding.DecodeString(base64Data)
				if err != nil {
					return fmt.Errorf("failed to decode base64 video data: %w", err)
				}

				return os.WriteFile(filename, videoBytes, 0o644)
			} else {
				// If it's a direct Cloud Storage URI (less common without output GCS bucket config)
				return fmt.Errorf("received URI instead of raw bytes: %s. Use HTTP client to download", part.Text)
			}
		}
	}

	return fmt.Errorf("no video found in the operation output")
}
