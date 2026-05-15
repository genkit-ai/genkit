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
	"errors"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit with Google AI plugin
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Define a flow to demonstrate code execution
	genkit.DefineFlow(g, "codeExecutionFlow", func(ctx context.Context, _ any) (string, error) {
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
		if m == nil {
			return "", errors.New("failed to find model")
		}

		problem := "find the sum of first 5 prime numbers"
		fmt.Printf("Problem: %s\n", problem)

		// Generate response with code execution enabled
		fmt.Println("Sending request to Gemini...")
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.2),
				Tools: []*genai.Tool{
					{
						CodeExecution: &genai.ToolCodeExecution{},
					},
				},
			}),
			ai.WithPrompt(problem))
		if err != nil {
			return "", err
		}

		// You can also use the helper function for simpler code
		fmt.Println("\n=== INTERNAL CODE EXECUTION ===")
		displayCodeExecution(resp.Message)

		fmt.Println("\n=== COMPLETE INTERNAL CODE EXECUTION ===")
		text := resp.Text()
		fmt.Println(text)

		return text, nil
	})
	<-ctx.Done()
}

// DisplayCodeExecution prints the code execution results from a message in a formatted way.
// This is a helper for applications that want to display code execution results to users.
func displayCodeExecution(msg *ai.Message) {
	// Extract and display executable code
	code := googlegenai.GetExecutableCode(msg)
	fmt.Printf("Language: %s\n", code.Language)
	fmt.Printf("```%s\n%s\n```\n", code.Language, code.Code)

	// Extract and display execution results
	result := googlegenai.GetCodeExecutionResult(msg)
	fmt.Printf("\nExecution result:\n")
	fmt.Printf("Status: %s\n", result.Outcome)
	fmt.Printf("Output:\n")
	if strings.TrimSpace(result.Output) == "" {
		fmt.Printf("  <no output>\n")
	} else {
		lines := strings.SplitSeq(result.Output, "\n")
		for line := range lines {
			fmt.Printf("  %s\n", line)
		}
	}

	// Display any explanatory text
	for _, part := range msg.Content {
		if part.IsText() {
			fmt.Printf("\nExplanation:\n%s\n", part.Text)
		}
	}
}
