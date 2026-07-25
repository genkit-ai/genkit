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

// This sample demonstrates how to use embedded prompts with genkit.
// Prompts are embedded directly into the binary using Go's embed package,
// which allows you to ship a self-contained binary without needing to
// distribute prompt files separately.

package main

import (
	"context"
	"embed"
	"errors"
	"log"

	genkit "github.com/firebase/genkit/go"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// Embed the prompts directory into the binary.
// The //go:embed directive makes the prompts available at compile time.
//
//go:embed prompts/*
var promptsFS embed.FS

func main() {
	ctx := context.Background()

	g, err := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithPromptFS(promptsFS),
	)
	if err != nil {
		log.Fatalf("failed to initialize Genkit: %v", err)
	}

	g.DefineFlow("sayHello", func(ctx context.Context, name string) (string, error) {
		prompt := g.LookupPrompt("example")
		if prompt == nil {
			return "", errors.New("prompt not found")
		}

		text, _, err := prompt.Execute(ctx, nil)
		if err != nil {
			return "", err
		}

		return text, nil
	})

	<-ctx.Done()
}
