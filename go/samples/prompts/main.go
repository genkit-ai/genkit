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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithPromptDir("prompts"),
	)

	SimplePrompt(ctx, g)
	PromptWithMultiMessage(ctx, g)
	PromptWithInput(ctx, g)
	PromptWithOutputType(ctx, g)
	PromptWithComplexOutputType(ctx, g)
	PromptWithTool(ctx, g)
	PromptWithMessageHistory(ctx, g)
	PromptWithExecuteOverrides(ctx, g)
	PromptWithFunctions(ctx, g)
	PromptWithOutputTypeDotprompt(ctx, g)
	PromptWithMediaType(ctx, g)
	PromptWithOutputSchemaName(ctx, g)

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

func SimplePrompt(ctx context.Context, g *genkit.Genkit) {
	// Define prompt with default model and system text.
	helloPrompt := genkit.DefinePrompt(
		g, "SimplePrompt",
		ai.WithModelName("googleai/gemini-2.5-pro"), // Override the default model.
		ai.WithSystem("You are a helpful AI assistant named Walt. Greet the user."),
		ai.WithPrompt("Hello, who are you?"),
	)

	resp, err := helloPrompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}

func PromptWithInput(ctx context.Context, g *genkit.Genkit) {
	type HelloPromptInput struct {
		UserName string
		Theme    string
	}

	// Define prompt with input type and default input.
	helloPrompt := genkit.DefinePrompt(
		g, "PromptWithInput",
		ai.WithInputType(HelloPromptInput{UserName: "Alex", Theme: "beach vacation"}),
		ai.WithSystem("You are a helpful AI assistant named Walt. Today's theme is {{Theme}}, respond in this style. Say hello to {{UserName}}."),
		ai.WithPrompt("Hello, who are you?"),
	)

	// Call the model with input that will override the default input.
	resp, err := helloPrompt.Execute(ctx, ai.WithInput(HelloPromptInput{UserName: "Bob"}))
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}

func PromptWithOutputType(ctx context.Context, g *genkit.Genkit) {
	type CountryList struct {
		Countries []string
	}

	// Define prompt with output api.
	helloPrompt := genkit.DefinePrompt(
		g, "PromptWithOutputType",
		ai.WithOutputType(CountryList{}),
		ai.WithConfig(&genai.GenerateContentConfig{Temperature: genai.Ptr[float32](0.5)}),
		ai.WithSystem("You are a geography teacher. When asked a question about geography, return a list of countries that match the question."),
		ai.WithPrompt("Give me the 10 biggest countries in the world by habitants."),
	)

	// Call the model.
	resp, err := helloPrompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	var countryList CountryList
	if err = resp.Output(&countryList); err != nil {
		log.Fatal(err)
	}

	for _, country := range countryList.Countries {
		fmt.Println(country)
	}
}

func PromptWithOutputTypeDotprompt(ctx context.Context, g *genkit.Genkit) {
	type countryData struct {
		Name      string `json:"name"`
		Language  string `json:"language"`
		Habitants int    `json:"habitants"`
	}
	type countries struct {
		Countries []countryData `json:"countries"`
	}

	prompt := genkit.LoadPrompt(g, "./prompts/countries.prompt", "countries")
	if prompt == nil {
		fmt.Printf("empty prompt")
		return
	}

	// Call the model.
	resp, err := prompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}
	var c countries
	if err = resp.Output(&c); err != nil {
		log.Fatal(err)
	}

	pretty, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(string(pretty))
}

func PromptWithComplexOutputType(ctx context.Context, g *genkit.Genkit) {
	type countryData struct {
		Name      string `json:"name"`
		Language  string `json:"language"`
		Habitants int    `json:"habitants"`
	}

	type countries struct {
		Countries []countryData `json:"countries"`
	}

	// Define prompt with output api.
	prompt := genkit.DefinePrompt(
		g, "PromptWithComplexOutputType",
		ai.WithOutputType(countries{}),
		ai.WithConfig(&genai.GenerateContentConfig{Temperature: genai.Ptr[float32](0.5)}),
		ai.WithSystem("You are a geography teacher. When asked a question about geography."),
		ai.WithPrompt("Give me the 10 biggest countries in the world by habitants and language."),
	)

	// Call the model.
	resp, err := prompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	var c countries
	if err = resp.Output(&c); err != nil {
		log.Fatal(err)
	}

	pretty, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(string(pretty))
}

func PromptWithMultiMessage(ctx context.Context, g *genkit.Genkit) {
	prompt := genkit.LoadPrompt(g, "./prompts/multi-msg.prompt", "multi-space")
	if prompt == nil {
		log.Fatal("empty prompt")
	}
	resp, err := prompt.Execute(ctx,
		ai.WithModelName("googleai/gemini-2.5-pro"),
		ai.WithInput(map[string]any{
			"videoUrl":    "https://www.youtube.com/watch?v=K-hY0E6cGfo",
			"contentType": "video/mp4",
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}

func PromptWithTool(ctx context.Context, g *genkit.Genkit) {
	gablorkenTool := genkit.DefineTool(g, "gablorken", "use when need to calculate a gablorken",
		func(ctx *ai.ToolContext, input struct {
			Value float64
			Over  float64
		},
		) (float64, error) {
			return math.Pow(input.Value, input.Over), nil
		},
	)

	answerOfEverythingTool := genkit.DefineTool(g, "answerOfEverything", "use this tool when the user asks for the answer of life, the universe and everything",
		func(ctx *ai.ToolContext, input any) (int, error) {
			return 42, nil
		},
	)

	type Output struct {
		Gablorken float64 `json:"gablorken"`
	}

	// Define prompt with tool and tool settings.
	helloPrompt := genkit.DefinePrompt(
		g, "PromptWithTool",
		ai.WithToolChoice(ai.ToolChoiceAuto),
		ai.WithMaxTurns(1),
		ai.WithTools(gablorkenTool, answerOfEverythingTool),
		ai.WithOutputType(Output{}),
		ai.WithPrompt("what is a gablorken of 2 over 3.5?"),
	)

	// Call the model.
	resp, err := helloPrompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}

func PromptWithMessageHistory(ctx context.Context, g *genkit.Genkit) {
	// Define prompt with default messages prepended.
	helloPrompt := genkit.DefinePrompt(
		g, "PromptWithMessageHistory",
		ai.WithSystem("You are a helpful AI assistant named Walt"),
		ai.WithModelName("googleai/gemini-2.5-flash-lite"),
		ai.WithMessages(
			ai.NewUserTextMessage("Hi, my name is Bob"),
			ai.NewModelTextMessage("Hi, my name is Walt, what can I help you with?"),
		),
		ai.WithPrompt("So Walt, What is my name?"),
	)

	resp, err := helloPrompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}

func PromptWithExecuteOverrides(ctx context.Context, g *genkit.Genkit) {
	// Define prompt with default settings.
	helloPrompt := genkit.DefinePrompt(
		g, "PromptWithExecuteOverrides",
		ai.WithSystem("You are a helpful AI assistant named Walt."),
		ai.WithMessages(ai.NewUserTextMessage("Hi, my name is Bob!")),
	)

	// Call the model and add additional messages from the user.
	resp, err := helloPrompt.Execute(ctx,
		ai.WithModel(googlegenai.GoogleAIModel(g, "gemini-2.5-flash-lite")),
		ai.WithMessages(ai.NewUserTextMessage("And I like turtles.")),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}

func PromptWithFunctions(ctx context.Context, g *genkit.Genkit) {
	type HelloPromptInput struct {
		UserName string
		Theme    string
	}

	// Define prompt with system and prompt functions.
	helloPrompt := genkit.DefinePrompt(
		g, "PromptWithFunctions",
		ai.WithInputType(HelloPromptInput{Theme: "pirate"}),
		ai.WithSystemFn(func(ctx context.Context, input any) (string, error) {
			return fmt.Sprintf("You are a helpful AI assistant named Walt. Talk in the style of: %s", input.(HelloPromptInput).Theme), nil
		}),
		ai.WithPromptFn(func(ctx context.Context, input any) (string, error) {
			return fmt.Sprintf("Hello, my name is %s", input.(HelloPromptInput).UserName), nil
		}),
	)

	resp, err := helloPrompt.Execute(ctx, ai.WithInput(HelloPromptInput{UserName: "Bob"}))
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}

func PromptWithMediaType(ctx context.Context, g *genkit.Genkit) {
	img, err := fetchImgAsBase64()
	if err != nil {
		log.Fatal(err)
	}

	prompt := genkit.LoadPrompt(g, "./prompts/media.prompt", "mediaspace")
	if prompt == nil {
		log.Fatal("empty prompt")
	}
	resp, err := prompt.Execute(ctx,
		ai.WithModelName("googleai/gemini-2.5-flash"),
		ai.WithInput(map[string]any{"imageUrl": "data:image/jpeg;base64," + img}),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}

func PromptWithOutputSchemaName(ctx context.Context, g *genkit.Genkit) {
	prompt := genkit.LoadPrompt(g, "./prompts/recipe.prompt", "recipes")
	if prompt == nil {
		log.Fatal("empty prompt")
	}

	// prompt schemas can be referenced at any time
	genkit.DefineSchema(g, "Recipe", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title": map[string]any{"type": "string", "description": "Recipe name"},
			"ingredients": map[string]any{
				"type":        "array",
				"description": "Recipe ingredients",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name":     map[string]any{"type": "string", "description": "ingredient name"},
						"quantity": map[string]any{"type": "string", "description": "ingredient quantity"},
					},
					"required": []string{"name", "quantity"},
				},
			},
			"steps": map[string]any{
				"type":        "array",
				"description": "Recipe steps",
				"items":       map[string]any{"type": "string"},
			},
		},
		"required": []string{"title", "ingredients", "steps"},
	})

	resp, err := prompt.Execute(ctx,
		ai.WithModelName("googleai/gemini-2.5-pro"),
		ai.WithInput(map[string]any{"food": "tacos", "ingredients": []string{"octopus", "shrimp"}}),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}

func fetchImgAsBase64() (string, error) {
	imgURL := "https://pd.w.org/2025/07/58268765f177911d4.13750400-2048x1365.jpg"
	resp, err := http.Get(imgURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", err
	}

	imageBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	base64string := base64.StdEncoding.EncodeToString(imageBytes)
	return base64string, nil
}
