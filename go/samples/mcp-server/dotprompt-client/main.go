// Copyright 2026 Google LLC
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

// Run from go/samples/mcp-server with GEMINI_API_KEY set:
//
//	go run ./dotprompt-client
package main

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/mcp"
)

//go:embed prompts/*
var promptsFS embed.FS

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithPromptFS(promptsFS),
	)

	client, err := mcp.NewGenkitMCPClient(mcp.MCPClientOptions{
		Name: "demo",
		Stdio: &mcp.StdioConfig{
			Command: "go",
			Args:    []string{"run", "server.go"},
		},
	})
	if err != nil {
		return fmt.Errorf("connect to MCP server: %w", err)
	}
	defer func() {
		if err := client.Disconnect(); err != nil {
			log.Printf("disconnect from MCP server: %v", err)
		}
	}()

	tools, err := client.GetActiveTools(ctx, g)
	if err != nil {
		return fmt.Errorf("list MCP tools: %w", err)
	}
	if len(tools) == 0 {
		return errors.New("MCP server returned no tools")
	}
	for _, tool := range tools {
		// DotPrompt lists tool names, so its registry must contain each MCP tool.
		// The same tool value can be passed directly to ai.WithTools when
		// defining or executing a prompt in Go code.
		genkit.RegisterAction(g, tool)
	}

	prompt := genkit.LookupPrompt(g, "encode")
	if prompt == nil {
		return errors.New("could not find encode.prompt")
	}
	response, err := prompt.Execute(ctx, ai.WithInput(map[string]any{"text": "Hello World"}))
	if err != nil {
		return fmt.Errorf("execute encode.prompt: %w", err)
	}
	fmt.Println(response.Text())
	return nil
}
