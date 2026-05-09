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
// Run with: export GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here && go run . -repo owner/name

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/mcp"
)

func main() {
	var repo string
	var pr int
	flag.StringVar(&repo, "repo", "firebase/genkit", "GitHub repo in the form owner/name")
	flag.IntVar(&pr, "pr", 0, "Pull request number")
	flag.Parse()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	ghToken := os.Getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
	if ghToken == "" {
		logger.FromContext(ctx).Error("GITHUB_PERSONAL_ACCESS_TOKEN is required")
		os.Exit(1)
	}

	serverCmd := os.Getenv("GITHUB_MCP_CMD")
	if serverCmd == "" {
		logger.FromContext(ctx).Error("GITHUB_MCP_CMD is required")
		os.Exit(1)
	}
	if pr <= 0 {
		logger.FromContext(ctx).Error("-pr <number> is required")
		os.Exit(1)
	}

	toolsets := "pull_requests"
	args := []string{"stdio", "--toolsets", toolsets, "--read-only"}

	client, err := mcp.NewGenkitMCPClient(mcp.MCPClientOptions{
		Name:    "github",
		Version: "1.0.0",
		Stdio: &mcp.StdioConfig{
			Command: serverCmd,
			Env: []string{
				"GITHUB_PERSONAL_ACCESS_TOKEN=" + ghToken,
				"GITHUB_TOOLSETS=" + toolsets,
				"GITHUB_READ_ONLY=1",
			},
			Args: args,
		},
	})
	if err != nil {
		logger.FromContext(ctx).Error("failed to start GitHub MCP server", "error", err)
		os.Exit(1)
	}
	defer client.Disconnect()

	tools, err := client.GetActiveTools(ctx, g)
	if err != nil {
		logger.FromContext(ctx).Error("failed to list GitHub tools", "error", err)
		os.Exit(1)
	}
	var toolRefs []ai.ToolRef
	for _, t := range tools {
		toolRefs = append(toolRefs, t)
	}

	owner, name := splitRepo(repo)

	prompt := fmt.Sprintf(`Summarize and explain what pull request #%d in %s is doing.

Instructions:
1) Use pull request tools to retrieve details and diffs (e.g., get_pull_request, list_pull_request_files, get_pull_request_diff).
2) Provide:
   - TL;DR (one sentence)
   - What changed (3–6 bullets)
   - Why it changed (intent)
3) Keep it concise. Return markdown only.

Always pass owner='%s' repo='%s' pull_number='%d'.`,
		pr, repo, owner, name, pr,
	)

	m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt(prompt),
		ai.WithTools(toolRefs...),
		ai.WithToolChoice(ai.ToolChoiceAuto),
	)
	if err != nil {
		logger.FromContext(ctx).Error("generation failed", "error", err)
		os.Exit(1)
	}
	fmt.Println(resp.Text())
}

func splitRepo(repo string) (string, string) {
	owner := repo
	name := repo
	if i := strings.Index(repo, "/"); i > 0 {
		owner = repo[:i]
		name = repo[i+1:]
	}
	return owner, name
}
