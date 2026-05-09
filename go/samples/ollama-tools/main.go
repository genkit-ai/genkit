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
	"fmt"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/ollama"
)

// WeatherInput defines the input structure for the weather tool
type WeatherInput struct {
	Location string `json:"location"`
}

// WeatherData represents weather information
type WeatherData struct {
	Location  string  `json:"location"`
	TempC     float64 `json:"temp_c"`
	TempF     float64 `json:"temp_f"`
	Condition string  `json:"condition"`
}

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Ollama plugin
	ollamaPlugin := &ollama.Ollama{
		ServerAddress: "http://localhost:11434", // Default Ollama server address
		Timeout:       60,                       // Response timeout in seconds
	}

	g := genkit.Init(ctx, genkit.WithPlugins(ollamaPlugin))

	// Define the Ollama model
	model := ollamaPlugin.DefineModel(g,
		ollama.ModelDefinition{
			Name: "llama3.1", // Choose an appropriate model
			Type: "chat",     // Must be chat for tool support
		},
		nil)

	// Define tools
	weatherTool := genkit.DefineTool(g, "weather", "Get current weather for a location",
		func(ctx *ai.ToolContext, input WeatherInput) (WeatherData, error) {
			// Get weather data (simulated)
			return simulateWeather(input.Location), nil
		},
	)

	// Create system message
	systemMsg := ai.NewSystemTextMessage(
		"You are a helpful assistant that can look up weather. " +
			"When providing weather information, use the appropriate tool.")

	// Create user message
	userMsg := ai.NewUserTextMessage("I'd like to know the weather in Tokyo.")

	// Generate response with tools
	fmt.Println("Generating response with weather tool...")

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithMessages(systemMsg, userMsg),
		ai.WithTools(weatherTool),
		ai.WithToolChoice(ai.ToolChoiceAuto),
		ai.WithConfig(&ollama.GenerateContentConfig{
			Temperature: ollama.Ptr(0.5),
		}),
	)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Print the final response
	fmt.Println("\n----- Final Response -----")
	fmt.Printf("%s\n", resp.Text())
	fmt.Println("--------------------------")
}

// simulateWeather returns simulated weather data for a location
func simulateWeather(location string) WeatherData {
	// In a real app, this would call a weather API
	// For demonstration, we'll return mock data
	tempC := 22.5
	if location == "Tokyo" || location == "Tokyo, Japan" {
		tempC = 24.0
	} else if location == "Paris" || location == "Paris, France" {
		tempC = 18.5
	} else if location == "New York" || location == "New York, USA" {
		tempC = 15.0
	}

	conditions := []string{"Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy"}
	condition := conditions[time.Now().Unix()%int64(len(conditions))]

	return WeatherData{
		Location:  location,
		TempC:     tempC,
		TempF:     tempC*9/5 + 32,
		Condition: condition,
	}
}
