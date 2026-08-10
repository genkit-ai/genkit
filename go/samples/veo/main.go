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
	"io"
	"log"
	"net/http"
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

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	genkit.DefineFlow(g, "text-to-video", func(ctx context.Context, input string) (string, error) {
		if input == "" {
			input = "Cat racing mouse"
		}
		operation, err := genkit.GenerateOperation(ctx, g,
			ai.WithMessages(ai.NewUserTextMessage(input)),
			ai.WithModelName("googleai/veo-3.1-generate-preview"),
			ai.WithConfig(&genai.GenerateVideosConfig{
				NumberOfVideos:  1,
				AspectRatio:     "16:9",
				DurationSeconds: genai.Ptr(int32(8)),
				Resolution:      "720p",
			}),
		)
		if err != nil {
			return "", fmt.Errorf("failed to start video generation: %w", err)
		}
		printStatus(operation)

		operation, err = waitForCompletion(ctx, g, operation)
		if err != nil {
			return "", fmt.Errorf("operation failed: %w", err)
		}
		log.Println("Video generation completed successfully!")

		if err := downloadGeneratedVideo(ctx, operation); err != nil {
			return "", fmt.Errorf("failed to download video: %w", err)
		}

		// Return the video URI for chaining
		uri, err := extractVideoURL(operation)
		if err != nil {
			return "", err
		}
		return uri, nil
	})

	genkit.DefineFlow(g, "image-to-video", func(ctx context.Context, input any) (string, error) {
		imgb64, err := fetchImgAsBase64()
		if err != nil {
			return "", fmt.Errorf("unable to download image: %w", err)
		}
		operation, err := genkit.GenerateOperation(ctx, g,
			ai.WithModelName("googleai/veo-3.1-generate-preview"),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Generate a video of the following image, the cat should wake up and start accelerating the go-kart as if it just acquired a mushroom from Mario Kart"),
				ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,"+imgb64),
			)),
			ai.WithConfig(&genai.GenerateVideosConfig{
				NumberOfVideos:  1,
				AspectRatio:     "16:9",
				DurationSeconds: genai.Ptr(int32(8)),
			}),
		)
		if err != nil {
			return "", fmt.Errorf("failed to start video generation: %w", err)
		}
		printStatus(operation)

		operation, err = waitForCompletion(ctx, g, operation)
		if err != nil {
			return "", fmt.Errorf("operation failed: %w", err)
		}
		log.Println("Video generation completed successfully!")

		if err := downloadGeneratedVideo(ctx, operation); err != nil {
			return "", fmt.Errorf("failed to download video: %w", err)
		}

		return "Video successfully downloaded to veo3_video.mp4", nil
	})

	genkit.DefineFlow(g, "video-to-video", func(ctx context.Context, inputURI string) (string, error) {
		if inputURI == "" {
			return "", fmt.Errorf("input URI is required for video extension")
		}

		log.Printf("Extending video from URI: %s", inputURI)

		operation, err := genkit.GenerateOperation(ctx, g,
			ai.WithModelName("googleai/veo-3.1-generate-preview"),
			ai.WithMessages(ai.NewUserMessage(
				ai.NewTextPart("Edit the original video background to be a rainforest, also change the video style to be a cartoon from 1950, make the transition smooth. You must keep the characters from the original video"),
				ai.NewMediaPart("video/mp4", inputURI),
			)),
			ai.WithConfig(&genai.GenerateVideosConfig{
				NumberOfVideos:  1,
				AspectRatio:     "16:9",
				DurationSeconds: genai.Ptr(int32(8)),
			}),
		)
		if err != nil {
			return "", fmt.Errorf("failed to start video generation: %w", err)
		}
		printStatus(operation)

		operation, err = waitForCompletion(ctx, g, operation)
		if err != nil {
			return "", fmt.Errorf("operation failed: %w", err)
		}
		log.Println("Video extension completed successfully!")

		if err := downloadGeneratedVideo(ctx, operation); err != nil {
			return "", fmt.Errorf("failed to download video: %w", err)
		}

		return "Video successfully downloaded to veo3_video.mp4", nil
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
	if op.Output != nil && op.Output.Message != nil && len(op.Output.Message.Content) > 0 {
		log.Printf("Status: %s", op.Output.Message.Content[0].Text)
	}
}

// downloadGeneratedVideo downloads the generated video from the operation result.
func downloadGeneratedVideo(ctx context.Context, operation *core.Operation[*ai.ModelResponse]) error {
	apiKey, err := getAPIKey()
	if err != nil {
		return err
	}

	videoURL, err := extractVideoURL(operation)
	if err != nil {
		return err
	}

	downloadURL := buildDownloadURL(videoURL, apiKey)

	return downloadVideo(ctx, downloadURL, "veo3_video.mp4")
}

// getAPIKey retrieves the API key from environment variables.
func getAPIKey() (string, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		return "", fmt.Errorf("no API key found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
	}
	return apiKey, nil
}

// extractVideoURL extracts the video URL from the operation output.
func extractVideoURL(operation *core.Operation[*ai.ModelResponse]) (string, error) {
	if operation.Output == nil {
		return "", fmt.Errorf("operation output is nil")
	}

	if operation.Output.Message == nil {
		return "", fmt.Errorf("model response message is nil")
	}

	for _, part := range operation.Output.Message.Content {
		if part.IsMedia() && part.Text != "" {
			return part.Text, nil
		}
	}

	return "", fmt.Errorf("no video URL found in the operation output")
}

// buildDownloadURL appends the API key to the video URL if not already present.
func buildDownloadURL(videoURL, apiKey string) string {
	if strings.Contains(videoURL, "key=") {
		return videoURL
	}

	separator := "?"
	if strings.Contains(videoURL, "?") {
		separator = "&"
	}

	return fmt.Sprintf("%s%skey=%s", videoURL, separator, apiKey)
}

// downloadVideo downloads a file from a URL and saves it to the specified filename.
func downloadVideo(ctx context.Context, url, filename string) error {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download video: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download video: HTTP %d", resp.StatusCode)
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	if _, err := io.Copy(file, resp.Body); err != nil {
		return fmt.Errorf("failed to write video data to file: %w", err)
	}

	return nil
}

// fetchImgAsBase64 downloads a predefined image and returns the image encoded in a base64 string
func fetchImgAsBase64() (string, error) {
	// CC0 license image
	imgURL := "https://pd.w.org/2025/07/896686fbbcd9990c9.84605288-2048x1365.jpg"
	resp, err := http.Get(imgURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download image: HTTP %d", resp.StatusCode)
	}

	imageBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	base64string := base64.StdEncoding.EncodeToString(imageBytes)
	return base64string, nil
}
