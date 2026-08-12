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

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/openai/openai-go"
)

func main() {
	ctx := context.Background()

	ds := &dashscope.DashScope{}
	g := genkit.Init(ctx, genkit.WithPlugins(ds))

	genkit.DefineFlow(g, "dashscope", func(ctx context.Context, subject string) (string, error) {
		model := dashscope.ModelRef("qwen-plus", &dashscope.ChatConfig{
			Temperature:    openai.Ptr(0.7),
			EnableThinking: openai.Ptr(false),
		})

		prompt := fmt.Sprintf("tell me a joke about %s", subject)
		foo, err := genkit.Generate(ctx, g, ai.WithModel(model), ai.WithPrompt(prompt))
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("foo: %s", foo.Text()), nil
	})

	// getWeather returns a deliberately made-up temperature. If the model's
	// final answer mentions it, that proves the model actually called the
	// tool rather than answering from its own knowledge - confirming that
	// Tools: true is correct for this model.
	getWeather := genkit.DefineTool(g, "getWeather", "Get the current temperature for a city",
		func(ctx *ai.ToolContext, input struct {
			City string `json:"city"`
		}) (struct {
			TempC int `json:"tempC"`
		}, error) {
			return struct {
				TempC int `json:"tempC"`
			}{TempC: 22}, nil
		},
	)

	type ToolTestInput struct {
		City string `json:"city"`
		// Model is a dashscope model id, e.g. "qwen-plus" or "qwen3-coder-plus".
		// Defaults to "qwen-plus" if left blank.
		Model string `json:"model,omitempty"`
	}

	genkit.DefineFlow(g, "dashscope-tool-test", func(ctx context.Context, input ToolTestInput) (string, error) {
		modelID := input.Model
		if modelID == "" {
			modelID = "qwen-plus"
		}
		model := dashscope.ModelRef(modelID, nil)

		prompt := fmt.Sprintf("What's the current temperature in %s? Use the tool to check, don't guess.", input.City)
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(model),
			ai.WithPrompt(prompt),
			ai.WithTools(getWeather),
		)
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	type VisionTestInput struct {
		// ImageURL is a publicly reachable image URL to describe.
		ImageURL string `json:"imageUrl"`
		// Model is a dashscope model id, e.g. "qwen3-vl-plus".
		// Defaults to "qwen3-vl-plus" if left blank.
		Model string `json:"model,omitempty"`
	}

	genkit.DefineFlow(g, "dashscope-vision-test", func(ctx context.Context, input VisionTestInput) (string, error) {
		modelID := input.Model
		if modelID == "" {
			modelID = "qwen3-vl-plus"
		}
		model := dashscope.ModelRef(modelID, nil)

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(model),
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewTextPart("Describe exactly what is in this image, in one sentence."),
					ai.NewMediaPart("", input.ImageURL),
				),
			),
		)
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
