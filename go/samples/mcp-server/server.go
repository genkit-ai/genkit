// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//
// Run with: go run server.go

package main

import (
	"context"
	"crypto/md5"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/mcp"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	g := genkit.Init(ctx)

	// Tool 1: Encode/decode text
	genkit.DefineTool(g, "text_encode", "Encode or decode text using various methods",
		func(ctx *ai.ToolContext, input struct {
			Text   string `json:"text" description:"Text to encode/decode"`
			Method string `json:"method" description:"Method: base64_encode, base64_decode, url_encode"`
		}) (map[string]interface{}, error) {
			logger.FromContext(ctx.Context).Debug("Executing text_encode tool", "method", input.Method, "textLength", len(input.Text))

			switch input.Method {
			case "base64_encode":
				encoded := base64.StdEncoding.EncodeToString([]byte(input.Text))
				return map[string]interface{}{
					"original": input.Text,
					"method":   input.Method,
					"result":   encoded,
				}, nil
			case "base64_decode":
				decoded, err := base64.StdEncoding.DecodeString(input.Text)
				if err != nil {
					return nil, fmt.Errorf("invalid base64: %v", err)
				}
				return map[string]interface{}{
					"original": input.Text,
					"method":   input.Method,
					"result":   string(decoded),
				}, nil
			case "url_encode":
				encoded := strings.ReplaceAll(input.Text, " ", "%20")
				encoded = strings.ReplaceAll(encoded, "&", "%26")
				return map[string]interface{}{
					"original": input.Text,
					"method":   input.Method,
					"result":   encoded,
				}, nil
			default:
				return nil, fmt.Errorf("unsupported method: %s", input.Method)
			}
		})

	// Tool 2: Generate hashes
	genkit.DefineTool(g, "hash_generate", "Generate hash values for text",
		func(ctx *ai.ToolContext, input struct {
			Text string `json:"text" description:"Text to hash"`
			Type string `json:"type" description:"Hash type: md5, sha256"`
		}) (map[string]interface{}, error) {
			logger.FromContext(ctx.Context).Debug("Executing hash_generate tool", "type", input.Type, "textLength", len(input.Text))

			switch input.Type {
			case "md5":
				hash := md5.Sum([]byte(input.Text))
				return map[string]interface{}{
					"original": input.Text,
					"type":     input.Type,
					"hash":     hex.EncodeToString(hash[:]),
				}, nil
			case "sha256":
				hash := sha256.Sum256([]byte(input.Text))
				return map[string]interface{}{
					"original": input.Text,
					"type":     input.Type,
					"hash":     hex.EncodeToString(hash[:]),
				}, nil
			default:
				return nil, fmt.Errorf("unsupported hash type: %s", input.Type)
			}
		})

	// Tool 3: Fetch URL content
	genkit.DefineTool(g, "fetch_url", "Fetch content from a URL",
		func(ctx *ai.ToolContext, input struct {
			URL string `json:"url" description:"URL to fetch content from"`
		}) (map[string]interface{}, error) {
			logger.FromContext(ctx.Context).Debug("Executing fetch_url tool", "url", input.URL)

			resp, err := http.Get(input.URL)
			if err != nil {
				return nil, fmt.Errorf("failed to fetch URL: %v", err)
			}
			defer resp.Body.Close()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("failed to read response body: %v", err)
			}

			logger.FromContext(ctx.Context).Debug("Successfully fetched URL content", "url", input.URL, "status", resp.StatusCode, "contentLength", len(body))

			return map[string]interface{}{
				"url":     input.URL,
				"status":  resp.StatusCode,
				"content": string(body),
				"headers": resp.Header,
				"length":  len(body),
			}, nil
		})

	// Start MCP server
	server := mcp.NewMCPServer(g, mcp.MCPServerOptions{
		Name: "text-utilities",
	})

	logger.FromContext(ctx).Info("Starting MCP server", "name", "text-utilities", "tools", server.ListRegisteredTools())
	logger.FromContext(ctx).Info("Ready! Run: go run client.go")

	if err := server.ServeStdio(); err != nil && err != context.Canceled {
		logger.FromContext(ctx).Error("MCP server error", "error", err)
		os.Exit(1)
	}
}
