// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0

// This program can be manually tested like so:
// Start the server listening on port 3100:
//
//	genkit start -o -- go run .

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	oai "github.com/firebase/genkit/go/plugins/compat_oai/openai"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/openai/openai-go"
)

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("no OPENAI_API_KEY environment variable set")
	}
	oai := &oai.OpenAI{
		APIKey: apiKey,
	}
	g := genkit.Init(ctx, genkit.WithPlugins(oai))

	genkit.DefineFlow(g, "basic", func(ctx context.Context, subject string) (string, error) {
		gpt4o := oai.Model(g, "gpt-4o")

		prompt := fmt.Sprintf("tell me a joke about %s", subject)
		config := &openai.ChatCompletionNewParams{Temperature: openai.Float(0.5), MaxTokens: openai.Int(100)}
		resp, err := genkit.Generate(ctx, g, ai.WithModel(gpt4o), ai.WithPrompt(prompt), ai.WithConfig(config))
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("resp: %s", resp.Text()), nil
	})

	genkit.DefineFlow(g, "defined-model", func(ctx context.Context, subject string) (string, error) {
		gpt4oMini := oai.Model(g, "gpt-4o-mini")
		prompt := fmt.Sprintf("tell me a joke about %s", subject)
		config := &openai.ChatCompletionNewParams{Temperature: openai.Float(0.5)}
		resp, err := genkit.Generate(ctx, g, ai.WithModel(gpt4oMini), ai.WithPrompt(prompt), ai.WithConfig(config))
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	genkit.DefineFlow(g, "media", func(ctx context.Context, subject string) (string, error) {
		gpt4oMini := oai.Model(g, "gpt-4o-mini")
		config := &openai.ChatCompletionNewParams{Temperature: openai.Float(0.5)}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(gpt4oMini),
			ai.WithConfig(config),
			ai.WithMessages(
				ai.NewUserMessage(ai.NewTextPart("Hi, I'll provide you a quick request in the following message")),
				ai.NewUserMessage(
					ai.NewTextPart("can you tell me which animal is in the provided image?"),
					ai.NewMediaPart("image/jpg", "https://pd.w.org/2025/07/58268765f177911d4.13750400-2048x1365.jpg"),
				)))
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
