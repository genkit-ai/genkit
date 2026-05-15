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
//
// SPDX-License-Identifier: Apache-2.0

// This sample shows how to upload a file to Gemini Files API and use it directly with Genkit.
//
// Usage:
//   1. Set GEMINI_API_KEY environment variable
//   2. Run: go run main.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Create Files API client
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  os.Getenv("GEMINI_API_KEY"),
	})
	if err != nil {
		log.Fatal("Failed to create client:", err)
	}

	// Upload image to Files API
	fmt.Println("Uploading image to Files API...")
	file, err := client.Files.UploadFromPath(ctx, "test.jpg", &genai.UploadFileConfig{
		MIMEType:    "image/jpeg",
		DisplayName: "Test Image",
	})
	if err != nil {
		log.Fatal("Failed to upload:", err)
	}
	fmt.Printf("Uploaded! File URI: %s\n", file.URI)

	// Use Files API URI directly with Genkit (now supported!)
	fmt.Println("Analyzing image with Genkit using Files API URI...")
	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("googleai/gemini-2.5-flash"),
		ai.WithMessages(
			ai.NewUserMessage(
				ai.NewTextPart("What do you see in this image?"),
				ai.NewMediaPart("image/jpeg", file.URI),
			),
		),
	)
	if err != nil {
		log.Fatal("Failed to analyze:", err)
	}

	fmt.Printf("Analysis: %s\n", resp.Text())

	// Clean up
	client.Files.Delete(ctx, file.Name, nil)
	fmt.Println("Cleaned up uploaded file")
}
