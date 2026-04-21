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
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/ollama"
)

// getContentTypeFromExtension returns a MIME type based on file extension
func getContentTypeFromExtension(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".bmp":
		return "image/bmp"
	case ".svg":
		return "image/svg+xml"
	default:
		return "image/png" // Default fallback
	}
}

func main() {
	// Get the image path from command line argument or use a default path
	imagePath := "test.png"
	if len(os.Args) > 1 {
		imagePath = os.Args[1]
	}

	// Check if image exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		log.Fatalf("Image file not found: %s", imagePath)
	}

	// Read the image file
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		log.Fatalf("Failed to read image file: %v", err)
	}

	// Detect content type (MIME type) from the file's binary signature
	contentType := http.DetectContentType(imageData)

	// If content type is generic/unknown, try to infer from file extension
	if contentType == "application/octet-stream" {
		contentType = getContentTypeFromExtension(imagePath)
	}

	// Encode image to base64
	base64Image := base64.StdEncoding.EncodeToString(imageData)
	dataURI := fmt.Sprintf("data:%s;base64,%s", contentType, base64Image)

	// Create a new Genkit instance
	g := genkit.Init(context.Background())

	// Initialize the Ollama plugin
	ollamaPlugin := &ollama.Ollama{
		ServerAddress: "http://localhost:11434", // Default Ollama server address
	}

	// Initialize the plugin
	actions := ollamaPlugin.Init(context.Background())
	log.Printf("Initialized Ollama plugin with %d actions", len(actions))

	// Define a model that supports images (llava is one of the supported models)
	modelName := "llava"
	model := ollamaPlugin.DefineModel(g, ollama.ModelDefinition{
		Name: modelName,
		Type: "chat",
	}, nil)

	// Create a context
	ctx := context.Background()

	// Create a request with text and image
	request := &ai.ModelRequest{
		Config: &ollama.GenerateContentConfig{
			Temperature: ollama.Ptr(0.4),
		},
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("Describe what you see in this image:"),
					ai.NewMediaPart(contentType, dataURI),
				},
			},
		},
	}

	// Call the model
	fmt.Printf("Sending request to %s model...\n", modelName)
	response, err := model.Generate(ctx, request, nil)
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	// Print the response
	fmt.Println("\nModel Response:")
	for _, part := range response.Message.Content {
		if part.IsText() {
			fmt.Println(part.Text)
		}
	}
}
