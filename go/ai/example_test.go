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

// Package ai_test provides examples for ai package helper functions.
//
// The ai package contains helper types and functions used with genkit.
// Most generation and definition functions are in the genkit package;
// see that package for the primary API documentation.
package ai_test

import (
	"fmt"

	"github.com/firebase/genkit/go/ai"
)

// This example demonstrates creating different types of message parts.
func ExampleNewTextPart() {
	// Create a text part
	part := ai.NewTextPart("Hello, world!")
	fmt.Println(part.Text)
	// Output: Hello, world!
}

// This example demonstrates creating a message with text content.
func ExampleNewUserTextMessage() {
	// Create a user message with text
	msg := ai.NewUserTextMessage("What is the capital of France?")
	fmt.Println("Role:", msg.Role)
	fmt.Println("Text:", msg.Content[0].Text)
	// Output:
	// Role: user
	// Text: What is the capital of France?
}

// This example demonstrates creating system and model messages.
func ExampleNewSystemTextMessage() {
	// Create a system message
	sysMsg := ai.NewSystemTextMessage("You are a helpful assistant.")
	fmt.Println("System role:", sysMsg.Role)

	// Create a model response message
	modelMsg := ai.NewModelTextMessage("I'm here to help!")
	fmt.Println("Model role:", modelMsg.Role)
	// Output:
	// System role: system
	// Model role: model
}

// This example demonstrates creating a data part for raw string content.
func ExampleNewDataPart() {
	// Create a data part with raw string content
	part := ai.NewDataPart(`{"name": "Alice", "age": 30}`)
	fmt.Println("Is data part:", part.IsData())
	fmt.Println("Content:", part.Text)
	// Output:
	// Is data part: true
	// Content: {"name": "Alice", "age": 30}
}

// This example demonstrates accessing text from a Part.
func ExamplePart_Text() {
	// Create a part with text
	part := ai.NewTextPart("Sample text content")

	// Access the text field directly
	fmt.Println(part.Text)
	// Output: Sample text content
}

// This example demonstrates the Document type used in RAG applications.
func ExampleDocument() {
	// Create a document with text content
	doc := &ai.Document{
		Content: []*ai.Part{
			ai.NewTextPart("This is the document content."),
		},
		Metadata: map[string]any{
			"source": "knowledge-base",
			"page":   42,
		},
	}

	fmt.Println("Content:", doc.Content[0].Text)
	fmt.Println("Source:", doc.Metadata["source"])
	// Output:
	// Content: This is the document content.
	// Source: knowledge-base
}

// This example demonstrates creating an Embedding for vector search.
func ExampleEmbedding() {
	// Create an embedding (typically returned by an embedder)
	embedding := &ai.Embedding{
		Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
		Metadata: map[string]any{
			"source": "document-1",
		},
	}

	fmt.Printf("Embedding dimensions: %d\n", len(embedding.Embedding))
	fmt.Printf("First value: %.1f\n", embedding.Embedding[0])
	// Output:
	// Embedding dimensions: 5
	// First value: 0.1
}

// This example demonstrates creating a media part for images or other media.
func ExampleNewMediaPart() {
	// Create a media part with base64-encoded image data
	// In practice, you would encode actual image bytes
	imageData := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ..."
	part := ai.NewMediaPart("image/png", imageData)

	fmt.Println("Is media:", part.IsMedia())
	fmt.Println("Content type:", part.ContentType)
	// Output:
	// Is media: true
	// Content type: image/png
}

// This example demonstrates creating a model reference with configuration.
func ExampleNewModelRef() {
	// Create a reference to a model with custom configuration
	// The config type depends on the model provider
	modelRef := ai.NewModelRef("googleai/gemini-2.5-flash", map[string]any{
		"temperature": 0.7,
	})

	fmt.Println("Model name:", modelRef.Name())
	// Output: Model name: googleai/gemini-2.5-flash
}

// This example demonstrates building a multi-turn conversation.
func ExampleNewUserMessage() {
	// Build a conversation with multiple parts
	userMsg := ai.NewUserMessage(
		ai.NewTextPart("What's in this image?"),
		ai.NewMediaPart("image/jpeg", "base64data..."),
	)

	fmt.Println("Role:", userMsg.Role)
	fmt.Println("Parts:", len(userMsg.Content))
	// Output:
	// Role: user
	// Parts: 2
}
