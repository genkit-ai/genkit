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

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/mcp"
)

func main() {
	g := genkit.Init(context.Background())
	genkit.DefineResource(g, "server-b-files", &ai.ResourceOptions{
		Template: "b://files/{path}",
	}, func(ctx context.Context, input *ai.ResourceInput) (*ai.ResourceOutput, error) {
		return &ai.ResourceOutput{
			Content: []*ai.Part{ai.NewTextPart("Content from Server B")},
		}, nil
	})
	server := mcp.NewMCPServer(g, mcp.MCPServerOptions{Name: "server-b"})
	server.ServeStdio()
}
