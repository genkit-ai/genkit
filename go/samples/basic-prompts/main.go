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

// This sample demonstrates prompts, each defined twice: once inline in code
// with DefinePrompt, and once in a .prompt file (Dotprompt) looked up by name.
// Every pair behaves identically, so the files show what moves out of code.
//
//   - joke: the simplest prompt, one interpolated field.
//   - structured-joke: DefineDataPrompt, typed both ways from its parameters.
//   - recipe: Handlebars conditionals and loops, plus a shared partial.
//   - assistant: middleware attached to the prompt itself.
//   - chat: a conversation passed per execution rather than declared.
//
// The prompts directory is compiled into the binary with go:embed, so the
// program ships as one file. Partials and helpers are registered on the Genkit
// instance in main and are then usable from any prompt.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, whose Prompts page runs each .prompt file against edits
// to its template, alongside a trace of every run at
// http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Streaming needs ?stream=true, and swapping PromptFlow for
// DotpromptFlow in any name below runs the .prompt file instead:
//
//	curl -N -X POST 'http://localhost:8080/recipePromptFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"dish": "tacos", "cuisine": "Mexican", "servingSize": 4, "maxPrepMinutes": 30}}'
//
//	curl -N -X POST 'http://localhost:8080/chatPromptFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"question": "What did I just ask you?", "history": [{"role": "user", "text": "How do I read a file in Go?"}, {"role": "model", "text": "Use os.ReadFile."}]}}'
package main

import (
	"context"
	"embed"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/middleware"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// The directive below compiles the prompts directory into the binary, so the
// program ships as one file with no .prompt files beside it.
//
//go:embed prompts/*
var promptsFS embed.FS

type JokeRequest struct {
	Topic string `json:"topic" jsonschema:"default=airplane food" jsonschema_description:"What the joke should be about"`
}

type Joke struct {
	Joke     string `json:"joke" jsonschema_description:"The joke text"`
	Category string `json:"category" jsonschema_description:"The joke category"`
}

// A jsonschema "default" is form fill for the Dev UI, not a value applied on
// the way through: every field here without omitempty is required.
type RecipeRequest struct {
	Dish                string   `json:"dish" jsonschema:"default=pasta" jsonschema_description:"The dish to cook"`
	Cuisine             string   `json:"cuisine" jsonschema:"default=Italian" jsonschema_description:"The cuisine to cook it in"`
	ServingSize         int      `json:"servingSize" jsonschema:"default=4" jsonschema_description:"How many people it should feed"`
	MaxPrepMinutes      int      `json:"maxPrepMinutes" jsonschema:"default=30" jsonschema_description:"The longest the recipe may take to prepare"`
	DietaryRestrictions []string `json:"dietaryRestrictions,omitempty" jsonschema_description:"Any dietary restrictions to respect"`
}

type Ingredient struct {
	Name     string `json:"name" jsonschema_description:"The ingredient name"`
	Amount   string `json:"amount" jsonschema_description:"The ingredient amount (e.g. 1 cup, 2 tablespoons, etc.)"`
	Optional bool   `json:"optional,omitempty" jsonschema_description:"Whether the ingredient is optional in the recipe"`
}

type Recipe struct {
	Title        string       `json:"title" jsonschema_description:"The recipe title (e.g. 'Spicy Chicken Tacos')"`
	Description  string       `json:"description,omitempty" jsonschema_description:"The recipe description (under 100 characters)"`
	Ingredients  []Ingredient `json:"ingredients" jsonschema_description:"The recipe ingredients (group by type and order by importance)"`
	Instructions []string     `json:"instructions" jsonschema_description:"The recipe instructions (step by step)"`
	PrepTime     string       `json:"prepTime" jsonschema_description:"The recipe preparation time (e.g. 10 minutes, 30 minutes, etc.)"`
	Difficulty   string       `json:"difficulty" jsonschema:"enum=easy,enum=medium,enum=hard"`
}

type AssistantRequest struct {
	Query string `json:"query" jsonschema:"default=what files are in my current directory?" jsonschema_description:"The user's query or request"`
}

// ChatRequest is the chat prompt's input: just the new question. The
// conversation is deliberately not in here, since it is passed to Execute and
// placed by the prompt itself.
type ChatRequest struct {
	Question string `json:"question" jsonschema:"default=What did I just ask you?" jsonschema_description:"The new question to answer"`
}

// ChatTurn is one exchange in the conversation the client keeps.
type ChatTurn struct {
	Role string `json:"role" jsonschema:"enum=user,enum=model" jsonschema_description:"Who said it"`
	Text string `json:"text" jsonschema_description:"What was said"`
}

// ChatSession is what the chat flows receive: the new question plus the
// conversation it continues.
type ChatSession struct {
	Question string     `json:"question" jsonschema:"default=What did I just ask you?" jsonschema_description:"The new question to answer"`
	History  []ChatTurn `json:"history,omitempty" jsonschema_description:"The conversation so far oldest first"`
}

// model is shared by every flow below, so switching models or thinking levels
// for the whole sample is a one-line change.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

func main() {
	ctx := context.Background()

	// The Google AI plugin reads the API key from GEMINI_API_KEY or
	// GOOGLE_API_KEY, which is the recommended practice. WithPromptFS then reads
	// the .prompt files from the embedded FS instead of from disk, using the same
	// "prompts" directory, so nothing about the lookups below changes.
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}, &middleware.Middleware{}),
		genkit.WithPromptFS(promptsFS),
	)

	// Registering schemas lets a .prompt file reference one by name, as in
	// "schema: JokeRequest". The alternative is writing the shape out inline in
	// the file's frontmatter.
	genkit.DefineSchemasFor(g, JokeRequest{}, Joke{}, RecipeRequest{}, Recipe{}, AssistantRequest{}, ChatRequest{})

	// A partial is a template fragment and a helper is a Go function, both
	// living on the Genkit instance rather than on one prompt: "chefPersona"
	// backs both recipe prompts below. Register them before those prompts.
	genkit.DefinePartial(g, "chefPersona", "You are an experienced chef who specializes in {{uppercase cuisine}} cooking. Come up with easy, creative recipes.")

	// A helper receives the value at the Go type the input arrived as, so prefer
	// scalars: a slice arrives as []string from a struct but []any from a map.
	genkit.DefineHelper(g, "uppercase", strings.ToUpper)

	DefineSimpleJokeWithInlinePrompt(g)
	DefineSimpleJokeWithDotprompt(g)
	DefineStructuredJokeWithInlinePrompt(g)
	DefineStructuredJokeWithDotprompt(g)
	DefineRecipeWithInlinePrompt(g)
	DefineRecipeWithDotprompt(g)
	DefineAssistantWithInlinePrompt(g)
	DefineAssistantWithDotprompt(g)
	DefineChatWithInlinePrompt(g)
	DefineChatWithDotprompt(g)

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineSimpleJokeWithInlinePrompt defines a prompt in code with DefinePrompt.
// With no output schema it always returns a string.
func DefineSimpleJokeWithInlinePrompt(g *genkit.Genkit) {
	jokePrompt := genkit.DefinePrompt(
		g, "joke.code",
		ai.WithModel(model),
		// WithInputType's values override the defaults in the jsonschema tags.
		ai.WithInputType(JokeRequest{Topic: "rush hour traffic"}),
		ai.WithPrompt("Share a long joke about {{topic}}."),
	)

	genkit.DefineStreamingFlow(g, "simpleJokePromptFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[string]) (string, error) {
			// A map is one way to pass input, useful when there is no input type.
			// Forwarding every chunk is all this flow does, so WithStreaming
			// says it in one option; ExecuteStream is for a flow that has to act
			// on the chunks.
			resp, err := jokePrompt.Execute(ctx,
				ai.WithInput(map[string]any{"topic": input.Topic}),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not generate joke: %w", err)
			}
			return resp.Text(), nil
		},
	)
}

// DefineSimpleJokeWithDotprompt looks up the same prompt loaded from a .prompt
// file at startup, where the model, input schema, and defaults now live.
func DefineSimpleJokeWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "simpleJokeDotpromptFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[string]) (string, error) {
			jokePrompt := genkit.LookupPrompt(g, "joke")
			// A map is one way to pass input, useful when there is no input type.
			resp, err := jokePrompt.Execute(ctx,
				ai.WithInput(map[string]any{"topic": input.Topic}),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not generate joke: %w", err)
			}
			return resp.Text(), nil
		},
	)
}

// DefineStructuredJokeWithInlinePrompt uses DefineDataPrompt, whose type
// parameters set the input and output schemas and the JSON output format.
func DefineStructuredJokeWithInlinePrompt(g *genkit.Genkit) {
	jokePrompt := genkit.DefineDataPrompt[JokeRequest, Joke](
		g, "structured-joke.code",
		ai.WithModel(model),
		ai.WithPrompt("Share a long joke about {{topic}}."),
	)

	genkit.DefineStreamingFlow(g, "structuredJokePromptFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[Joke]) (Joke, error) {
			for val, err := range jokePrompt.ExecuteStream(ctx, input) {
				if err != nil {
					return Joke{}, fmt.Errorf("could not generate joke: %w", err)
				}
				if val.Done {
					return val.Output, nil
				}
				sendChunk(ctx, val.Chunk)
			}

			return Joke{}, status.Errorf(status.ErrInternal, "the stream ended without a final result")
		},
	)
}

// DefineStructuredJokeWithDotprompt wraps a .prompt file in the same Go types
// with LookupDataPrompt.
func DefineStructuredJokeWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "structuredJokeDotpromptFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[Joke]) (Joke, error) {
			jokePrompt := genkit.LookupDataPrompt[JokeRequest, Joke](g, "structured-joke")
			for val, err := range jokePrompt.ExecuteStream(ctx, input) {
				if err != nil {
					return Joke{}, fmt.Errorf("could not generate joke: %w", err)
				}
				if val.Done {
					return val.Output, nil
				}
				sendChunk(ctx, val.Chunk)
			}
			return Joke{}, status.Errorf(status.ErrInternal, "the stream ended without a final result")
		},
	)
}

// DefineRecipeWithInlinePrompt adds nested types and Handlebars conditionals
// and loops, and streams ingredients as they arrive.
func DefineRecipeWithInlinePrompt(g *genkit.Genkit) {
	recipePrompt := genkit.DefineDataPrompt[RecipeRequest, Recipe](
		g, "recipe.code",
		ai.WithModel(model),
		// {{> chefPersona}} pulls in the partial registered in main, which
		// calls the uppercase helper. A partial is compiled against the same
		// input as the template including it.
		ai.WithSystem("{{> chefPersona}}"),
		ai.WithPrompt("Create a {{cuisine}} {{dish}} recipe for {{servingSize}} people that takes under {{maxPrepMinutes}} minutes to prepare. "+
			"{{#if dietaryRestrictions}}Dietary restrictions: {{#each dietaryRestrictions}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}.{{/if}}"),
	)

	genkit.DefineStreamingFlow(g, "recipePromptFlow",
		func(ctx context.Context, input RecipeRequest, sendChunk core.StreamCallback[Ingredient]) (Recipe, error) {
			// Not required, but it shows the flow choosing what to stream.
			filterNew := newIngredientFilter()
			for val, err := range recipePrompt.ExecuteStream(ctx, input) {
				if err != nil {
					return Recipe{}, fmt.Errorf("could not generate recipe: %w", err)
				}
				if val.Done {
					return val.Output, nil
				}
				for _, i := range filterNew(val.Chunk.Ingredients) {
					sendChunk(ctx, i)
				}
			}
			return Recipe{}, status.Errorf(status.ErrInternal, "the stream ended without a final result")
		},
	)
}

// DefineRecipeWithDotprompt is the same prompt in a .prompt file, which writes
// its system and user turns out as a multi-message template.
func DefineRecipeWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "recipeDotpromptFlow",
		func(ctx context.Context, input RecipeRequest, sendChunk core.StreamCallback[Ingredient]) (Recipe, error) {
			filterNew := newIngredientFilter()
			recipePrompt := genkit.LookupDataPrompt[RecipeRequest, Recipe](g, "recipe")
			for val, err := range recipePrompt.ExecuteStream(ctx, input) {
				if err != nil {
					return Recipe{}, fmt.Errorf("could not generate recipe: %w", err)
				}
				if val.Done {
					return val.Output, nil
				}
				for _, i := range filterNew(val.Chunk.Ingredients) {
					sendChunk(ctx, i)
				}
			}
			return Recipe{}, status.Errorf(status.ErrInternal, "the stream ended without a final result")
		},
	)
}

// newIngredientFilter drops ingredients already streamed, so a caller sees each
// one once even though every chunk restates the whole list.
func newIngredientFilter() func([]Ingredient) []Ingredient {
	seen := map[string]struct{}{}
	return func(ings []Ingredient) (newIngs []Ingredient) {
		for _, ing := range ings {
			if _, ok := seen[ing.Name]; !ok {
				seen[ing.Name] = struct{}{}
				newIngs = append(newIngs, ing)
			}
		}
		return
	}
}

// DefineAssistantWithInlinePrompt attaches middleware (Retry, Fallback,
// Filesystem, Skills) to the prompt itself, so every execution gets it.
func DefineAssistantWithInlinePrompt(g *genkit.Genkit) {
	assistantPrompt := genkit.DefinePrompt(
		g, "assistant.code",
		ai.WithModel(model),
		ai.WithPrompt("{{query}}"),
		ai.WithInputType(AssistantRequest{}),
		ai.WithUse(
			&middleware.Retry{MaxRetries: 2},
			&middleware.Fallback{
				Models: []ai.ModelRef{
					googlegenai.ModelRef("googleai/gemini-3.5-flash", nil),
					googlegenai.ModelRef("googleai/gemini-3.1-pro-preview", &genai.GenerateContentConfig{
						Temperature: genai.Ptr[float32](2.0),
					}),
				},
			},
			&middleware.Filesystem{RootDir: "."},
			&middleware.Skills{SkillPaths: []string{"./skills"}},
		),
	)

	genkit.DefineStreamingFlow(g, "assistantPromptFlow",
		func(ctx context.Context, input AssistantRequest, sendChunk core.StreamCallback[string]) (string, error) {
			resp, err := assistantPrompt.Execute(ctx,
				ai.WithInput(&input),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not answer the query: %w", err)
			}
			return resp.Text(), nil
		},
	)
}

// DefineAssistantWithDotprompt configures the same middleware in the .prompt
// file's YAML frontmatter instead.
func DefineAssistantWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "assistantDotpromptFlow",
		func(ctx context.Context, input AssistantRequest, sendChunk core.StreamCallback[string]) (string, error) {
			// The "assistant" prompt file includes all middleware in its frontmatter.
			assistantPrompt := genkit.LookupPrompt(g, "assistant")
			resp, err := assistantPrompt.Execute(ctx,
				ai.WithInput(&input),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not answer the query: %w", err)
			}
			return resp.Text(), nil
		},
	)
}

// DefineChatWithInlinePrompt is the simplest multi-turn prompt: it declares no
// conversation, so the messages passed to Execute fill the middle of the
// request, between the system message and the user prompt.
func DefineChatWithInlinePrompt(g *genkit.Genkit) {
	chatPrompt := genkit.DefinePrompt(
		g, "chat.code",
		ai.WithModel(model),
		ai.WithInputType(ChatRequest{}),
		ai.WithSystem("You are a helpful AI assistant named Walt. Keep replies to a few sentences."),
		ai.WithPrompt("{{question}}"),
	)

	genkit.DefineStreamingFlow(g, "chatPromptFlow",
		func(ctx context.Context, session ChatSession, sendChunk core.StreamCallback[string]) (string, error) {
			resp, err := chatPrompt.Execute(ctx,
				ai.WithInput(ChatRequest{Question: session.Question}),
				// Placed for us, because the prompt claims no conversation
				// slot. One that does, like chat.prompt, decides instead.
				ai.WithMessages(chatMessages(session.History)...),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not answer the question: %w", err)
			}
			return resp.Text(), nil
		},
	)
}

// DefineChatWithDotprompt is the other end: chat.prompt scripts an opening
// exchange and marks where the real conversation goes with {{history}}. Claiming
// the slot means placing it, so a template without the marker drops the
// caller's messages.
func DefineChatWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "chatDotpromptFlow",
		func(ctx context.Context, session ChatSession, sendChunk core.StreamCallback[string]) (string, error) {
			chatPrompt := genkit.LookupPrompt(g, "chat")
			resp, err := chatPrompt.Execute(ctx,
				ai.WithInput(ChatRequest{Question: session.Question}),
				ai.WithMessages(chatMessages(session.History)...),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not answer the question: %w", err)
			}
			return resp.Text(), nil
		},
	)
}

// chatMessages adapts the app's own turns to Genkit messages. Taking
// []*ai.Message on the wire instead would skip this, at the cost of a schema
// covering every part kind a message can hold. Message text is verbatim, so a
// user who typed {{#if}} is quoted rather than compiled.
func chatMessages(turns []ChatTurn) []*ai.Message {
	messages := make([]*ai.Message, 0, len(turns))
	for _, t := range turns {
		messages = append(messages, ai.NewTextMessage(ai.Role(t.Role), t.Text))
	}
	return messages
}
