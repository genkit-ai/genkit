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
//
// SPDX-License-Identifier: Apache-2.0

// Native conformance executor for Go plugins.
//
// Protocol: JSONL-over-stdio.
//
//  1. Receives --plugin <name> as a CLI argument.
//  2. Initializes the matching plugin via a registry map.
//  3. Prints {"ready": true}\n to stdout.
//  4. Reads one JSON line from stdin per request.
//  5. Calls genkit.Generate() natively.
//  6. Writes one JSON line to stdout with the response.
//  7. Repeats until stdin closes.
//
// Driven by the Python `conform` tool:
//
//	conform check-model --runtime go --runner native
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	anthropicSDK "github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/anthropic"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/ollama"
)

// pluginInitFunc returns the genkit.GenkitOption to initialize a specific plugin.
type pluginInitFunc func() genkit.GenkitOption

// pluginRegistry maps conform plugin names to their init functions.
var pluginRegistry = map[string]pluginInitFunc{
	"google-genai": func() genkit.GenkitOption { return genkit.WithPlugins(&googlegenai.GoogleAI{}) },
	"vertex-ai":    func() genkit.GenkitOption { return genkit.WithPlugins(&googlegenai.VertexAI{}) },
	"anthropic":    func() genkit.GenkitOption { return genkit.WithPlugins(&anthropic.Anthropic{}) },
	"ollama": func() genkit.GenkitOption {
		return genkit.WithPlugins(&ollama.Ollama{
			ServerAddress: "http://localhost:11434",
		})
	},
}

// nativeRequest is the JSON structure received from conform on stdin.
type nativeRequest struct {
	Model  string         `json:"model"`
	Input  map[string]any `json:"input"`
	Stream bool           `json:"stream"`
}

// nativeResponse is the JSON structure sent back to conform on stdout.
type nativeResponse struct {
	Response map[string]any   `json:"response"`
	Chunks   []map[string]any `json:"chunks"`
	Error    string           `json:"error,omitempty"`
}

// buildMessages converts raw YAML/JSON messages into Genkit messages.
func buildMessages(raw []any) []*ai.Message {
	var msgs []*ai.Message
	for _, m := range raw {
		mmap, ok := m.(map[string]any)
		if !ok {
			continue
		}
		role, _ := mmap["role"].(string)
		contentRaw, _ := mmap["content"].([]any)

		var parts []*ai.Part
		for _, c := range contentRaw {
			cmap, ok := c.(map[string]any)
			if !ok {
				continue
			}
			if text, ok := cmap["text"].(string); ok {
				parts = append(parts, ai.NewTextPart(text))
			}
			if media, ok := cmap["media"].(map[string]any); ok {
				url, _ := media["url"].(string)
				ct, _ := media["contentType"].(string)
				parts = append(parts, ai.NewMediaPart(ct, url))
			}
		}

		msg := &ai.Message{
			Role:    ai.Role(role),
			Content: parts,
		}
		msgs = append(msgs, msg)
	}
	return msgs
}

// toolCache tracks already-defined tools to avoid re-registration panics.
// The registry doesn't allow defining the same action name twice.
var toolCache = map[string]ai.ToolRef{}

// buildTools creates placeholder tools from raw tool definitions.
// Tools are defined once and cached; subsequent requests with the same
// tool name reuse the existing registration.
func buildTools(g *genkit.Genkit, raw []any) []ai.ToolRef {
	var tools []ai.ToolRef
	for _, t := range raw {
		tmap, ok := t.(map[string]any)
		if !ok {
			continue
		}
		name, _ := tmap["name"].(string)
		if name == "" {
			continue
		}

		if cached, exists := toolCache[name]; exists {
			tools = append(tools, cached)
			continue
		}

		description, _ := tmap["description"].(string)

		type ToolInput struct {
			City string `json:"city,omitempty"`
		}

		tool := genkit.DefineTool(
			g,
			name,
			description,
			func(ctx *ai.ToolContext, input *ToolInput) (string, error) {
				return "21C", nil
			},
		)
		toolCache[name] = tool
		tools = append(tools, tool)
	}
	return tools
}

// serializeResponse converts a ModelResponse to a JSON-friendly map.
func serializeResponse(resp *ai.ModelResponse) map[string]any {
	if resp == nil {
		return map[string]any{}
	}

	result := map[string]any{}

	if resp.Message != nil {
		msgMap := map[string]any{
			"role": string(resp.Message.Role),
		}
		var contentList []map[string]any
		for _, p := range resp.Message.Content {
			part := map[string]any{}
			if p.IsText() {
				part["text"] = p.Text
			}
			if p.IsToolRequest() && p.ToolRequest != nil {
				part["toolRequest"] = map[string]any{
					"name":  p.ToolRequest.Name,
					"input": p.ToolRequest.Input,
					"ref":   p.ToolRequest.Ref,
				}
			}
			if p.IsMedia() {
				part["media"] = map[string]any{
					"url":         p.Text,
					"contentType": p.ContentType,
				}
			}
			contentList = append(contentList, part)
		}
		msgMap["content"] = contentList
		result["message"] = msgMap
	}

	if resp.FinishReason != "" {
		result["finishReason"] = string(resp.FinishReason)
	}

	if resp.Usage != nil {
		result["usage"] = map[string]any{
			"inputTokens":  resp.Usage.InputTokens,
			"outputTokens": resp.Usage.OutputTokens,
			"totalTokens":  resp.Usage.TotalTokens,
		}
	}

	return result
}

// serializeChunk converts a ModelResponseChunk to a JSON-friendly map.
func serializeChunk(c *ai.ModelResponseChunk) map[string]any {
	if c == nil {
		return map[string]any{}
	}
	result := map[string]any{}
	var contentList []map[string]any
	for _, p := range c.Content {
		part := map[string]any{}
		if p.IsText() {
			part["text"] = p.Text
		}
		if p.IsToolRequest() && p.ToolRequest != nil {
			part["toolRequest"] = map[string]any{
				"name":  p.ToolRequest.Name,
				"input": p.ToolRequest.Input,
				"ref":   p.ToolRequest.Ref,
			}
		}
		contentList = append(contentList, part)
	}
	result["content"] = contentList
	return result
}

// handleRequest processes a single native request and returns the response.
func handleRequest(ctx context.Context, g *genkit.Genkit, req *nativeRequest) *nativeResponse {
	// Build generate options.
	opts := []ai.GenerateOption{
		ai.WithModelName(req.Model),
		ai.WithReturnToolRequests(true),
	}

	// Config — plugin-specific handling.
	// Each plugin validates config against its own schema, so we can't use
	// a generic GenerationCommonConfig for all plugins.
	if strings.HasPrefix(req.Model, "anthropic/") {
		// Anthropic requires MaxTokens.  Use the SDK's native type.
		cfg := &anthropicSDK.MessageNewParams{MaxTokens: 4096}
		if cfgRaw, ok := req.Input["config"].(map[string]any); ok {
			if v, ok := cfgRaw["maxOutputTokens"].(float64); ok {
				cfg.MaxTokens = int64(v)
			}
		}
		opts = append(opts, ai.WithConfig(cfg))
	} else if cfgRaw, ok := req.Input["config"].(map[string]any); ok {
		// For other plugins, pass the raw map.  The framework will
		// validate it against the model's registered schema.
		opts = append(opts, ai.WithConfig(cfgRaw))
	}

	// Messages.
	if msgs, ok := req.Input["messages"].([]any); ok {
		builtMsgs := buildMessages(msgs)
		if len(builtMsgs) > 0 {
			opts = append(opts, ai.WithMessages(builtMsgs...))
		}
	}

	// Tools.
	if toolsRaw, ok := req.Input["tools"].([]any); ok {
		tools := buildTools(g, toolsRaw)
		if len(tools) > 0 {
			opts = append(opts, ai.WithTools(tools...))
		}
	}

	// Output format.
	if output, ok := req.Input["output"].(map[string]any); ok {
		if format, ok := output["format"].(string); ok && format == "json" {
			opts = append(opts, ai.WithOutputFormat(ai.OutputFormatJSON))
		}
	}

	// Streaming.
	var chunks []*ai.ModelResponseChunk
	if req.Stream {
		opts = append(opts, ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
			chunks = append(chunks, c)
			return nil
		}))
	}

	// Execute.
	resp, err := genkit.Generate(ctx, g, opts...)
	if err != nil {
		return &nativeResponse{
			Error: err.Error(),
		}
	}

	if resp == nil {
		return &nativeResponse{
			Error: "generate returned nil response",
		}
	}

	// Serialize chunks.
	var chunkMaps []map[string]any
	for _, c := range chunks {
		chunkMaps = append(chunkMaps, serializeChunk(c))
	}

	return &nativeResponse{
		Response: serializeResponse(resp),
		Chunks:   chunkMaps,
	}
}

func main() {
	pluginName := flag.String("plugin", "", "Plugin name (e.g. google-genai, anthropic)")
	flag.Parse()

	if *pluginName == "" {
		fmt.Fprintf(os.Stderr, "error: --plugin is required\n")
		fmt.Fprintf(os.Stderr, "available plugins: %s\n", strings.Join(availablePlugins(), ", "))
		os.Exit(1)
	}

	initFn, ok := pluginRegistry[*pluginName]
	if !ok {
		fmt.Fprintf(os.Stderr, "error: unknown plugin %q\n", *pluginName)
		fmt.Fprintf(os.Stderr, "available plugins: %s\n", strings.Join(availablePlugins(), ", "))
		os.Exit(1)
	}

	ctx := context.Background()

	// Initialize the plugin.
	g := genkit.Init(ctx, initFn())

	// Signal readiness.
	readyLine, _ := json.Marshal(map[string]any{"ready": true})
	fmt.Fprintf(os.Stdout, "%s\n", readyLine)
	os.Stdout.Sync()

	// Read requests from stdin, one JSON line per request.
	scanner := bufio.NewScanner(os.Stdin)
	// Increase scanner buffer for large requests (e.g. base64 images).
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var req nativeRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			resp := nativeResponse{Error: fmt.Sprintf("invalid request JSON: %v", err)}
			out, _ := json.Marshal(resp)
			fmt.Fprintf(os.Stdout, "%s\n", out)
			os.Stdout.Sync()
			continue
		}

		result := handleRequest(ctx, g, &req)
		out, err := json.Marshal(result)
		if err != nil {
			errResp := nativeResponse{Error: fmt.Sprintf("failed to marshal response: %v", err)}
			out, _ = json.Marshal(errResp)
		}
		fmt.Fprintf(os.Stdout, "%s\n", out)
		os.Stdout.Sync()
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "stdin read error: %v\n", err)
		os.Exit(1)
	}
}

// availablePlugins returns sorted list of registered plugin names.
func availablePlugins() []string {
	var names []string
	for k := range pluginRegistry {
		names = append(names, k)
	}
	return names
}
