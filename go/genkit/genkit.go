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
//
// SPDX-License-Identifier: Apache-2.0

// Package genkit provides Genkit functionality for application developers.
package genkit

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"iter"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/internal/registry"
)

// genkitCtxKey is the context key for the Genkit instance.
var genkitCtxKey = base.NewContextKey[*Genkit]()

// FromContext returns the [*Genkit] instance stored in the context.
// This is set automatically by [Generate] and related functions.
// Middleware implementations can use this to access the Genkit instance
// during generation.
func FromContext(ctx context.Context) *Genkit {
	return genkitCtxKey.FromContext(ctx)
}

// Genkit encapsulates a Genkit instance, providing access to its registry,
// configuration, and core functionalities. It serves as the central hub for
// defining and managing Genkit resources like flows, models, tools, and prompts.
//
// A Genkit instance is created using [Init].
type Genkit struct {
	reg *registry.Registry // Registry for actions, values, and other resources.
}

// genkitOptions are options for configuring the Genkit instance.
type genkitOptions struct {
	DefaultModel string       // Default model to use if no other model is specified.
	PromptDir    string       // Directory where dotprompts are stored. Will be loaded automatically on initialization.
	PromptFS     fs.FS        // Embedded filesystem containing prompts (alternative to PromptDir).
	Plugins      []api.Plugin // Plugin to initialize automatically.
}

type GenkitOption interface {
	apply(g *genkitOptions) error
}

// apply applies the options to the Genkit options.
func (o *genkitOptions) apply(gOpts *genkitOptions) error {
	if o.DefaultModel != "" {
		if gOpts.DefaultModel != "" {
			return errors.New("cannot set default model more than once (WithDefaultModel)")
		}
		gOpts.DefaultModel = o.DefaultModel
	}

	if o.PromptDir != "" {
		if gOpts.PromptDir != "" {
			return errors.New("cannot set prompt directory more than once (WithPromptDir)")
		}
		gOpts.PromptDir = o.PromptDir
	}

	if o.PromptFS != nil {
		if gOpts.PromptFS != nil {
			return errors.New("cannot set prompt filesystem more than once (WithPromptFS)")
		}
		gOpts.PromptFS = o.PromptFS
	}

	if len(o.Plugins) > 0 {
		if gOpts.Plugins != nil {
			return errors.New("cannot set plugins more than once (WithPlugins)")
		}
		gOpts.Plugins = o.Plugins
	}

	return nil
}

// WithPlugins provides a list of plugins to initialize when creating the Genkit instance.
// Each plugin's [Plugin.Init] method will be called sequentially during [Init].
// This option can only be applied once.
func WithPlugins(plugins ...api.Plugin) GenkitOption {
	return &genkitOptions{Plugins: plugins}
}

// WithDefaultModel sets the default model name to use for generation tasks
// when no specific model is provided in the request options. The name should
// correspond to a model registered either by a plugin or via [DefineModel].
// This option can only be applied once.
func WithDefaultModel(model string) GenkitOption {
	return &genkitOptions{DefaultModel: model}
}

// WithPromptDir specifies the directory where `.prompt` files are located.
// Prompts are automatically loaded from this directory during [Init].
// The default directory is "prompts" relative to the project root where
// [Init] is called.
//
// When used with [WithPromptFS], this directory serves as the root path within
// the embedded filesystem instead of a local disk path. For example, if using
// `//go:embed prompts/*`, set the directory to "prompts" to match.
//
// Invalid prompt files will result in logged errors during initialization,
// while valid files that define invalid prompts will cause [Init] to panic.
func WithPromptDir(dir string) GenkitOption {
	return &genkitOptions{PromptDir: dir}
}

// WithPromptFS specifies an embedded filesystem ([fs.FS]) containing `.prompt` files.
// This is useful for embedding prompts directly into the binary using Go's [embed] package,
// eliminating the need to distribute prompt files separately.
//
// The `fsys` parameter should be an [fs.FS] implementation (e.g., [embed.FS]).
// Use [WithPromptDir] to specify the root directory within the filesystem where
// prompts are located (defaults to "prompts").
//
// Example:
//
//	import "embed"
//
//	//go:embed prompts/*
//	var promptsFS embed.FS
//
//	func main() {
//		g := genkit.Init(ctx,
//			genkit.WithPromptFS(promptsFS),
//			genkit.WithPromptDir("prompts"),
//		)
//	}
//
// Invalid prompt files will result in logged errors during initialization,
// while valid files that define invalid prompts will cause [Init] to panic.
func WithPromptFS(fsys fs.FS) GenkitOption {
	return &genkitOptions{PromptFS: fsys}
}

// Init creates and initializes a new [Genkit] instance with the provided options.
// It sets up the registry, initializes plugins ([WithPlugins]), loads prompts
// ([WithPromptDir]), and configures other settings like the default model
// ([WithDefaultModel]).
//
// During local development (when the `GENKIT_ENV` environment variable is set to `dev`),
// Init also starts the Reflection API server as a background goroutine. This server
// provides metadata about registered actions and is used by developer tools.
// By default, it listens on port 3100.
//
// The provided context should handle application shutdown signals (like SIGINT, SIGTERM)
// to ensure graceful termination of background processes, including the reflection server.
//
// Example:
//
//	package main
//
//	import (
//		"context"
//		"log"
//
//		"github.com/firebase/genkit/go/ai"
//		"github.com/firebase/genkit/go/genkit"
//		"github.com/firebase/genkit/go/plugins/googlegenai" // Example plugin
//	)
//
//	func main() {
//		ctx := context.Background()
//
//		// Assumes a prompt file at ./prompts/jokePrompt.prompt
//		g := genkit.Init(ctx,
//			genkit.WithPlugins(&googlegenai.GoogleAI{}),
//			genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
//			genkit.WithPromptDir("./prompts"),
//		)
//
//		// Generate text using the default model
//		funFact, err := genkit.GenerateText(ctx, g, ai.WithPrompt("Tell me a fake fun fact!"))
//		if err != nil {
//			log.Fatalf("GenerateText failed: %v", err)
//		}
//		log.Println("Generated Fact:", funFact)
//
//		// Look up and execute a loaded prompt
//		jokePrompt := genkit.LookupPrompt(g, "jokePrompt")
//		if jokePrompt == nil {
//			log.Fatalf("Prompt 'jokePrompt' not found.")
//		}
//
//		resp, err := jokePrompt.Execute(ctx, nil) // Execute with default input (if any)
//		if err != nil {
//			log.Fatalf("jokePrompt.Execute failed: %v", err)
//		}
//		log.Println("Generated joke:", resp.Text())
//	}
func Init(ctx context.Context, opts ...GenkitOption) *Genkit {
	ctx, _ = signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)

	gOpts := &genkitOptions{}
	for _, opt := range opts {
		if err := opt.apply(gOpts); err != nil {
			panic(fmt.Errorf("genkit.Init: error applying options: %w", err))
		}
	}

	r := registry.New()
	g := &Genkit{reg: r}

	for _, plugin := range gOpts.Plugins {
		actions := plugin.Init(ctx)
		for _, action := range actions {
			action.Register(r)
		}
		r.RegisterPlugin(plugin.Name(), plugin)

		if mp, ok := plugin.(ai.MiddlewarePlugin); ok {
			descs, err := mp.Middlewares(ctx)
			if err != nil {
				panic(fmt.Errorf("genkit.Init: plugin %q Middlewares failed: %w", plugin.Name(), err))
			}
			for _, d := range descs {
				d.Register(r)
			}
		}
	}

	ai.ConfigureFormats(r)
	ai.DefineGenerateAction(ctx, r)
	if gOpts.PromptFS != nil {
		dir := gOpts.PromptDir
		if dir == "" {
			dir = "prompts"
		}
		ai.LoadPromptDirFromFS(r, gOpts.PromptFS, dir, "")
	} else {
		loadPromptDirOS(r, gOpts.PromptDir, "")
	}

	r.RegisterValue(api.DefaultModelKey, gOpts.DefaultModel)
	r.RegisterValue(api.PromptDirKey, gOpts.PromptDir)

	if api.CurrentEnvironment() == api.EnvironmentDev {
		errCh := make(chan error, 1)
		serverStartCh := make(chan struct{})

		if v2URL := os.Getenv("GENKIT_REFLECTION_V2_SERVER"); v2URL != "" {
			// V2: connect to the CLI's WebSocket server.
			go startReflectionServerV2(ctx, g, reflectionServerV2Options{URL: v2URL}, errCh, serverStartCh)
		} else {
			// V1: start an HTTP reflection server.
			go func() {
				if s := startReflectionServer(ctx, g, errCh, serverStartCh); s == nil {
					return
				}
				if err := <-errCh; err != nil {
					slog.Error("reflection server error", "err", err)
				}
			}()
		}

		select {
		case err := <-errCh:
			panic(fmt.Errorf("genkit.Init: reflection server startup failed: %w", err))
		case <-serverStartCh:
			slog.Debug("reflection server started successfully")
		case <-ctx.Done():
			panic(ctx.Err())
		}
	}

	return g
}

// RegisterAction registers a [api.Action] that was previously created by calling
// NewX instead of DefineX.
//
// Example:
//
//	model := ai.NewModel(...)
//	genkit.RegisterAction(g, model)
func RegisterAction(g *Genkit, action api.Registerable) {
	action.Register(g.reg)
}

// DefineFlow defines a non-streaming flow, registers it as a [core.Action] of type Flow,
// and returns a [core.Flow] runner.
// The provided function `fn` takes an input of type `In` and returns an output of type `Out`.
// Flows are the primary mechanism for orchestrating multi-step AI tasks in Genkit.
// Each run of a flow is traced, and steps within the flow can be traced using [Run].
//
// Example:
//
//	myFlow := genkit.DefineFlow(g, "mySimpleFlow",
//		func(ctx context.Context, name string) (string, error) {
//			greeting := fmt.Sprintf("Hello, %s!", name)
//			// You could add more steps here, potentially using genkit.Run()
//			return greeting, nil
//		},
//	)
//
//	// Later, run the flow:
//	result, err := myFlow.Run(ctx, "World")
//	if err != nil {
//		// handle error
//	}
//	fmt.Println(result) // Output: Hello, World!
func DefineFlow[In, Out any](g *Genkit, name string, fn core.Func[In, Out]) *core.Flow[In, Out, struct{}] {
	return core.DefineFlow(g.reg, name, fn)
}

// DefineStreamingFlow defines a streaming flow, registers it as a [core.Action] of type Flow,
// and returns a [core.Flow] runner capable of streaming.
//
// The provided function `fn` takes an input of type `In`. It can optionally stream
// intermediate results of type `Stream` by invoking the provided callback function.
// Finally, it returns a final output of type `Out`.
//
// If the function supports streaming and the callback is non-nil when the flow is run,
// it should invoke the callback periodically with `Stream` values. The final `Out` value,
// typically an aggregation of the streamed data, is returned at the end.
// If the callback is nil or the function doesn't support streaming for a given input,
// it should simply compute and return the `Out` value directly.
//
// Example:
//
//	counterFlow := genkit.DefineStreamingFlow(g, "counter",
//		func(ctx context.Context, limit int, stream core.StreamCallback[int]) (string, error) {
//			if stream == nil { // Non-streaming case
//				return fmt.Sprintf("Counted up to %d", limit), nil
//			}
//			// Streaming case
//			for i := 1; i <= limit; i++ {
//				if err := stream(ctx, i); err != nil {
//					return "", fmt.Errorf("streaming error: %w", err)
//				}
//				// time.Sleep(100 * time.Millisecond) // Optional delay
//			}
//			return fmt.Sprintf("Finished counting to %d", limit), nil
//		},
//	)
//
//	// Later, run the flow with streaming:
//	streamCh, err := counterFlow.Stream(ctx, 5)
//	if err != nil {
//		// handle error
//	}
//	for result := range streamCh {
//		if result.Err != nil {
//			log.Printf("Stream error: %v", result.Err)
//			break
//		}
//		if result.Done {
//			fmt.Println("Final Output:", result.Output) // Output: Finished counting to 5
//		} else {
//			fmt.Println("Stream Chunk:", result.Stream) // Outputs: 1, 2, 3, 4, 5
//		}
//	}
func DefineStreamingFlow[In, Out, Stream any](g *Genkit, name string, fn core.StreamingFunc[In, Out, Stream]) *core.Flow[In, Out, Stream] {
	return core.DefineStreamingFlow(g.reg, name, fn)
}

// NewFlow creates a [core.Flow] without registering it as an action.
// To register the flow later, call [RegisterAction].
func NewFlow[In, Out any](name string, fn core.Func[In, Out]) *core.Flow[In, Out, struct{}] {
	return core.NewFlow(name, fn)
}

// NewStreamingFlow creates a streaming [core.Flow] without registering it as an action.
// To register the flow later, call [RegisterAction].
func NewStreamingFlow[In, Out, Stream any](name string, fn core.StreamingFunc[In, Out, Stream]) *core.Flow[In, Out, Stream] {
	return core.NewStreamingFlow(name, fn)
}

// Run executes the given function `fn` within the context of the current flow run,
// creating a distinct trace span for this step. It's used to add observability
// to specific sub-operations within a flow defined by [DefineFlow] or [DefineStreamingFlow].
// The `name` parameter provides a label for the trace span.
// It returns the output of `fn` and any error it produces.
//
// Example (within a DefineFlow function):
//
//	complexFlow := genkit.DefineFlow(g, "complexTask",
//		func(ctx context.Context, input string) (string, error) {
//			// Step 1: Process input (traced as "process-input")
//			processedInput, err := genkit.Run(ctx, "process-input", func() (string, error) {
//				// ... some processing ...
//				return strings.ToUpper(input), nil
//			})
//			if err != nil {
//				return "", err
//			}
//
//			// Step 2: Generate response (traced as "generate-response")
//			response, err := genkit.Run(ctx, "generate-response", func() (string, error) {
//				// ... call an AI model or another service ...
//				return "Response for " + processedInput, nil
//			})
//			if err != nil {
//				return "", err
//			}
//
//			return response, nil
//		},
//	)
func Run[Out any](ctx context.Context, name string, fn func() (Out, error)) (Out, error) {
	return core.Run(ctx, name, fn)
}

// ListFlows returns a slice of all [api.Action] instances that represent
// flows registered with the Genkit instance `g`.
// This is useful for introspection or for dynamically exposing flow endpoints,
// for example, in an HTTP server.
func ListFlows(g *Genkit) []api.Action {
	acts := listActions(g)
	flows := []api.Action{}
	for _, act := range acts {
		if act.Type == api.ActionTypeFlow {
			flows = append(flows, g.reg.LookupAction(act.Key))
		}
	}
	return flows
}

// ListTools returns a slice of all [ai.Tool] instances that are registered
// with the Genkit instance `g`. This is useful for introspection and for
// exposing tools to external systems like MCP servers.
func ListTools(g *Genkit) []ai.Tool {
	acts := g.reg.ListActions()
	tools := []ai.Tool{}
	for _, action := range acts {
		tool := LookupTool(g, action.Desc().Name)
		if tool != nil {
			tools = append(tools, tool)
		}
	}
	return tools
}

// DefineModel defines a custom model implementation, registers it as a [core.Action]
// of type Model, and returns an [ai.Model] interface.
//
// The `name` argument is the unique identifier for the model (e.g., "myProvider/myModel").
// The `opts` argument provides metadata about the model's capabilities ([ai.ModelOptions]).
// The `fn` argument ([ai.ModelFunc]) implements the actual generation logic, handling
// input requests ([ai.ModelRequest]) and producing responses ([ai.ModelResponse]),
// potentially streaming chunks ([ai.ModelResponseChunk]) via the callback.
//
// For models that don't need to be registered (e.g., for plugin development or testing),
// use [ai.NewModel] instead.
//
// Example:
//
//	echoModel := genkit.DefineModel(g, "custom/echo",
//		&ai.ModelOptions{
//			Label:    "Echo Model",
//			Supports: &ai.ModelSupports{Multiturn: true},
//		},
//		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
//			// Simple echo implementation
//			resp := &ai.ModelResponse{
//				Message: &ai.Message{
//					Role:    ai.RoleModel,
//					Content: []*ai.Part{},
//				},
//			}
//			// Combine content from the last user message
//			var responseText strings.Builder
//			if len(req.Messages) > 0 {
//				lastMsg := req.Messages[len(req.Messages)-1]
//				if lastMsg.Role == ai.RoleUser {
//					for _, part := range lastMsg.Content {
//						if part.IsText() {
//							responseText.WriteString(part.Text)
//						}
//					}
//				}
//			}
//			if responseText.Len() == 0 {
//				responseText.WriteString("...")
//			}
//
//			resp.Message.Content = append(resp.Message.Content, ai.NewTextPart(responseText.String()))
//
//			// Example of streaming (optional)
//			if cb != nil {
//				chunk := &ai.ModelResponseChunk{ Index: 0, Content: resp.Message.Content }
//				if err := cb(ctx, chunk); err != nil {
//					return nil, err // Handle streaming error
//				}
//			}
//
//			resp.FinishReason = ai.FinishReasonStop
//			return resp, nil
//		},
//	)
func DefineModel(g *Genkit, name string, opts *ai.ModelOptions, fn ai.ModelFunc) ai.Model {
	return ai.DefineModel(g.reg, name, opts, fn)
}

// DefineBackgroundModel defines a background model, registers it as a [ai.BackgroundModel],
// and returns an [ai.BackgroundModel].
//
// The `name` is the identifier the model uses to request the background model. The `opts`
// are the options for the background model. The `startFn` is the function that starts the background model.
// The `checkFn` is the function that checks the status of the background model.
func DefineBackgroundModel(g *Genkit, name string, opts *ai.BackgroundModelOptions, startFn ai.StartModelOpFunc, checkFn ai.CheckModelOpFunc) ai.BackgroundModel {
	return ai.DefineBackgroundModel(g.reg, name, opts, startFn, checkFn)
}

// LookupModel retrieves a registered [ai.Model] by its provider and name.
// It returns the model instance if found, or `nil` if no model with the
// given identifier is registered (e.g., via [DefineModel] or a plugin).
// It will try to resolve the model dynamically by matching the provider name;
// this does not necessarily mean the model is valid.
func LookupModel(g *Genkit, name string) ai.Model {
	return ai.LookupModel(g.reg, name)
}

// LookupBackgroundModel retrieves a registered background model by its provider and name.
// It returns the background action instance if found, or `nil` if no background model with the
// given identifier is registered.
func LookupBackgroundModel(g *Genkit, name string) ai.BackgroundModel {
	return ai.LookupBackgroundModel(g.reg, name)
}

// DefineTool defines a tool that can be used by models during generation,
// registers it as a [core.Action] of type Tool, and returns an [ai.Tool].
// Tools allow models to interact with external systems or perform specific computations.
//
// The `name` is the identifier the model uses to request the tool. The `description`
// helps the model understand when to use the tool. The function `fn` implements
// the tool's logic, taking an [ai.ToolContext] and an input of type `In`, and
// returning an output of type `Out`. The input and output types determine the
// `inputSchema` and `outputSchema` in the tool's definition, which guide the model
// on how to provide input and interpret output.
//
// For tools that don't need to be registered (e.g., dynamically created tools),
// use [ai.NewTool] instead.
//
// # Options
//
//   - [ai.WithInputSchema]: Provide a custom JSON schema instead of inferring from the type parameter
//   - [ai.WithInputSchemaName]: Reference a pre-registered schema by name
//
// Example:
//
//	weatherTool := genkit.DefineTool(g, "getWeather", "Fetches the weather for a given city",
//		func(ctx *ai.ToolContext, city string) (string, error) {
//			// In a real scenario, call a weather API
//			log.Printf("Tool: Fetching weather for %s", city)
//			if city == "Paris" {
//				return "Sunny, 25°C", nil
//			}
//			return "Cloudy, 18°C", nil
//		},
//	)
//
//	// Use the tool in a generation request:
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithPrompt("What's the weather like in Paris?"),
//		ai.WithTools(weatherTool), // Make the tool available
//		// Optionally use ai.WithToolChoice(...)
//	)
//	if err != nil {
//		log.Fatalf("Generate failed: %v", err)
//	}
//
//	fmt.Println(resp.Text()) // Might output something like "The weather in Paris is Sunny, 25°C."
func DefineTool[In, Out any](g *Genkit, name, description string, fn ai.ToolFunc[In, Out], opts ...ai.ToolOption) *ai.ToolDef[In, Out] {
	return ai.DefineTool(g.reg, name, description, fn, opts...)
}

// DefineToolWithInputSchema defines a tool with a custom input schema that can be used by models during generation,
// registers it as a [core.Action] of type Tool, and returns an [*ai.ToolDef].
//
// This variant of [DefineTool] allows specifying a JSON Schema for the tool's input, providing more
// control over input validation and model guidance. The input parameter to the tool function will be
// of type `any` and should be validated/processed according to the schema.
//
// The `name` is the identifier the model uses to request the tool. The `description` helps the model
// understand when to use the tool. The `inputSchema` defines the expected structure and constraints
// of the input. The function `fn` implements the tool's logic, taking an [ai.ToolContext] and an
// input of type `any`, and returning an output of type `Out`.
//
// Deprecated: Use [DefineTool] with [ai.WithInputSchema] instead.
//
// Example:
//
//	// Define a custom input schema
//	inputSchema := map[string]any{
//		"type": "object",
//		"properties": map[string]any{
//			"city": map[string]any{"type": "string"},
//			"unit": map[string]any{
//				"type": "string",
//				"enum": []any{"C", "F"},
//			},
//		},
//		"required": []string{"city"},
//	}
//
//	// Define the tool with the schema
//	weatherTool := genkit.DefineTool(g, "getWeather",
//		"Fetches the weather for a given city with unit preference",
//		func(ctx *ai.ToolContext, input any) (string, error) {
//			// Parse and validate input
//			data := input.(map[string]any)
//			city := data["city"].(string)
//			unit := "C" // default
//			if u, ok := data["unit"].(string); ok {
//				unit = u
//			}
//			// Implementation...
//			return fmt.Sprintf("Weather in %s: 25°%s", city, unit), nil
//		},
//		ai.WithToolInputSchema(inputSchema),
//	)
func DefineToolWithInputSchema[Out any](g *Genkit, name, description string, inputSchema map[string]any, fn ai.ToolFunc[any, Out]) *ai.ToolDef[any, Out] {
	return ai.DefineTool(g.reg, name, description, fn, ai.WithInputSchema(inputSchema))
}

// DefineMultipartTool defines a multipart tool that can be used by models during generation,
// registers it as a [core.Action] of type Tool, and returns an [*ai.ToolDef].
// Unlike regular tools that return just an output value, multipart tools can return
// both an output value and additional content parts (like images or other media).
//
// The `name` is the identifier the model uses to request the tool. The `description`
// helps the model understand when to use the tool. The function `fn` implements
// the tool's logic, taking an [ai.ToolContext] and an input of type `In`, and
// returning an [ai.MultipartToolResponse] which contains both the output and optional
// content parts.
//
// For multipart tools that don't need to be registered (e.g., dynamically created tools),
// use [ai.NewMultipartTool] instead.
//
// # Options
//
//   - [ai.WithInputSchema]: Provide a custom JSON schema instead of inferring from the type parameter
//   - [ai.WithInputSchemaName]: Reference a pre-registered schema by name
//
// Example:
//
//	type ImageGenInput struct {
//		Prompt string `json:"prompt"`
//		Style  string `json:"style,omitempty"`
//	}
//
//	imageGenTool := genkit.DefineMultipartTool(g, "generateImage", "Generates an image from a text prompt",
//		func(ctx *ai.ToolContext, input ImageGenInput) (*ai.MultipartToolResponse, error) {
//			// In a real scenario, call an image generation API
//			log.Printf("Tool: Generating image for prompt: %s", input.Prompt)
//
//			// Generate image bytes (placeholder)
//			imageBytes := []byte{...}
//
//			return &ai.MultipartToolResponse{
//				Output: map[string]any{
//					"status": "success",
//					"prompt": input.Prompt,
//				},
//				Content: []*ai.Part{
//					ai.NewMediaPart("image/png", string(imageBytes)),
//				},
//			}, nil
//		},
//	)
//
//	// Use the tool in a generation request:
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithPrompt("Create an image of a sunset over mountains"),
//		ai.WithTools(imageGenTool),
//	)
//	if err != nil {
//		log.Fatalf("Generate failed: %v", err)
//	}
//
//	fmt.Println(resp.Text())
func DefineMultipartTool[In any](g *Genkit, name, description string, fn ai.MultipartToolFunc[In], opts ...ai.ToolOption) *ai.ToolDef[In, *ai.MultipartToolResponse] {
	return ai.DefineMultipartTool(g.reg, name, description, fn, opts...)
}

// LookupTool retrieves a registered tool by its name.
// It returns the tool instance if found, or `nil` if no tool with the
// given name is registered (e.g., via [DefineTool]).
// Since the types are not known at lookup time, it returns a type-erased tool.
func LookupTool(g *Genkit, name string) ai.Tool {
	return ai.LookupTool(g.reg, name)
}

// DefineMiddleware registers a middleware descriptor with the Genkit instance
// and returns the resulting [*ai.MiddlewareDesc]. Registered middleware is
// surfaced to the Dev UI and addressable by name for cross-runtime dispatch.
//
// This is the path for application code that declares its own middleware
// directly. Plugins should instead construct descriptors with [ai.NewMiddleware]
// (no registration) and return them from [ai.MiddlewarePlugin.Middlewares];
// [Init] registers those descriptors during plugin setup.
//
// The `description` is a human-readable explanation shown in the Dev UI. The
// `prototype` is a value of a type that implements [ai.Middleware]. Its
// [ai.Middleware.Name] method supplies the registered name, and its fields
// (both exported JSON config and unexported plugin-level state) are captured
// by a value-copy inside the descriptor so JSON-dispatched invocations
// preserve prototype state across calls.
//
// For pure Go use, registration is not strictly required: passing a middleware
// config directly to [ai.WithUse] invokes its [ai.Middleware.New] method on
// the local fast path without a registry lookup. Registration is what makes
// the middleware visible to the Dev UI and callable from other runtimes. For
// ad-hoc one-off middleware that doesn't need Dev UI visibility, use
// [ai.MiddlewareFunc] instead of defining a type.
//
// Example:
//
//	type Trace struct {
//		Label string `json:"label,omitempty"`
//	}
//
//	func (Trace) Name() string { return "mine/trace" }
//
//	func (t Trace) New(ctx context.Context) (*ai.Hooks, error) {
//		return &ai.Hooks{
//			WrapModel: func(ctx context.Context, p *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
//				start := time.Now()
//				resp, err := next(ctx, p)
//				log.Printf("[%s] model call took %s", t.Label, time.Since(start))
//				return resp, err
//			},
//		}, nil
//	}
//
//	// Register so it appears in the Dev UI and can be called by name:
//	genkit.DefineMiddleware(g, "logs model call latency", Trace{})
//
//	// Use it per-call:
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithPrompt("hello"),
//		ai.WithUse(Trace{Label: "debug"}),
//	)
func DefineMiddleware[M ai.Middleware](g *Genkit, description string, prototype M) *ai.MiddlewareDesc {
	return ai.DefineMiddleware(g.reg, description, prototype)
}

// LookupMiddleware retrieves a registered middleware descriptor by its name.
// It returns the descriptor if found, or `nil` if no middleware with the
// given name is registered (e.g., via [DefineMiddleware] or through a
// plugin's [ai.MiddlewarePlugin.Middlewares] method).
func LookupMiddleware(g *Genkit, name string) *ai.MiddlewareDesc {
	return ai.LookupMiddleware(g.reg, name)
}

// DefinePrompt defines a prompt programmatically, registers it as a [core.Action]
// of type Prompt, and returns an executable [ai.Prompt].
//
// This provides an alternative to defining prompts in `.prompt` files, offering
// more flexibility through Go code. Prompts encapsulate configuration (model, parameters),
// message templates (system, user, history), input/output schemas, and associated tools.
//
// Prompts can be executed in two main ways:
//  1. Render + Generate: Call [ai.Prompt.Render] to get [ai.GenerateActionOptions],
//     modify them if needed, and pass them to [GenerateWithRequest].
//  2. Execute: Call [ai.Prompt.Execute] directly, passing input and execution options.
//
// For prompts that don't need to be registered (e.g., for single-use or testing),
// use [ai.NewPrompt] instead.
//
// # Options
//
// Model and Configuration:
//   - [ai.WithModel]: Specify the model (accepts [ai.Model] or [ai.ModelRef])
//   - [ai.WithModelName]: Specify model by name string
//   - [ai.WithConfig]: Set generation parameters (temperature, max tokens, etc.)
//
// Prompt Content:
//   - [ai.WithPrompt]: Set the user prompt template (supports {{variable}} syntax)
//   - [ai.WithPromptFn]: Set a function that generates the user prompt dynamically
//   - [ai.WithSystem]: Set system instructions template
//   - [ai.WithSystemFn]: Set a function that generates system instructions dynamically
//   - [ai.WithMessages]: Provide static conversation history
//   - [ai.WithMessagesFn]: Provide a function that generates conversation history
//
// Input Schema:
//   - [ai.WithInputType]: Set input schema from a Go type (provides default values)
//   - [ai.WithInputSchema]: Provide a custom JSON schema for input
//   - [ai.WithInputSchemaName]: Reference a pre-registered schema by name
//
// Output Schema:
//   - [ai.WithOutputType]: Set output schema from a Go type
//   - [ai.WithOutputSchema]: Provide a custom JSON schema for output
//   - [ai.WithOutputSchemaName]: Reference a pre-registered schema by name
//   - [ai.WithOutputFormat]: Specify output format (json, text, etc.)
//
// Tools and Resources:
//   - [ai.WithTools]: Enable tools the model can call
//   - [ai.WithToolChoice]: Control whether tool calls are required, optional, or disabled
//   - [ai.WithMaxTurns]: Set maximum tool call iterations
//   - [ai.WithResources]: Attach resources available during generation
//
// Metadata:
//   - [ai.WithDescription]: Set a description for the prompt
//   - [ai.WithMetadata]: Set arbitrary metadata
//
// Example:
//
//	type GeoInput struct {
//		Country string `json:"country"`
//	}
//
//	type GeoOutput struct {
//		Capital string `json:"capital"`
//	}
//
//	// Define the prompt
//	capitalPrompt := genkit.DefinePrompt(g, "findCapital",
//		ai.WithDescription("Finds the capital of a country."),
//		ai.WithModelName("googleai/gemini-2.5-flash"),
//		ai.WithSystem("You are a helpful geography assistant."),
//		ai.WithPrompt("What is the capital of {{country}}?"),
//		ai.WithInputType(GeoInput{Country: "USA"}),
//		ai.WithOutputType(GeoOutput{}),
//		// Config is provider-specific, e.g., genai.GenerateContentConfig for Google AI
//	)
//
//	// Option 1: Render + Generate (using default input "USA")
//	actionOpts, err := capitalPrompt.Render(ctx, nil) // nil input uses default
//	if err != nil {
//		log.Fatalf("Render failed: %v", err)
//	}
//	resp1, err := genkit.GenerateWithRequest(ctx, g, actionOpts, nil, nil)
//	if err != nil {
//		log.Fatalf("GenerateWithRequest failed: %v", err)
//	}
//	var out1 GeoOutput
//	if err = resp1.Output(&out1); err != nil {
//		log.Fatalf("Output failed: %v", err)
//	}
//	fmt.Printf("Capital of USA: %s\n", out1.Capital) // Output: Capital of USA: Washington D.C.
//
//	// Option 2: Execute directly (with new input)
//	resp2, err := capitalPrompt.Execute(ctx, ai.WithInput(GeoInput{Country: "France"}))
//	if err != nil {
//		log.Fatalf("Execute failed: %v", err)
//	}
//	var out2 GeoOutput
//	if err = resp2.Output(&out2); err != nil {
//		log.Fatalf("Output failed: %v", err)
//	}
//	fmt.Printf("Capital of France: %s\n", out2.Capital) // Output: Capital of France: Paris
func DefinePrompt(g *Genkit, name string, opts ...ai.PromptOption) ai.Prompt {
	return ai.DefinePrompt(g.reg, name, opts...)
}

// LookupPrompt retrieves a registered [ai.Prompt] by its name.
// Prompts can be registered via [DefinePrompt] or loaded automatically from
// `.prompt` files in the directory specified by [WithPromptDir] or [LoadPromptDir].
// It returns the prompt instance if found, or `nil` otherwise.
func LookupPrompt(g *Genkit, name string) ai.Prompt {
	return ai.LookupPrompt(g.reg, name)
}

// DefineSchema defines a named JSON schema and registers it in the registry.
//
// Registered schemas can be referenced by name in prompts (both `.prompt` files
// and programmatic definitions) to define input or output structures.
// The `schema` argument must be a JSON schema definition represented as a map.
//
// Example:
//
//	genkit.DefineSchema(g, "User", map[string]any{
//	    "type": "object",
//	    "properties": map[string]any{
//	        "name": map[string]any{"type": "string"},
//	        "age":  map[string]any{"type": "integer"},
//	    },
//	    "required": []string{"name"}
//	})
//
//	genkit.Generate(ctx, g, ai.WithOutputSchemaName("User"), ai.WithPrompt("What is your name?"))
func DefineSchema(g *Genkit, name string, schema map[string]any) {
	core.DefineSchema(g.reg, name, schema)
}

// DefineSchemaFor defines a named JSON schema derived from a Go type
// and registers it in the registry.
//
// This is an alternative to [DefineSchema].
//
// Example:
//
//	type User struct {
//	    Name string `json:"name"`
//	    Age int `json:"age"`
//	}
//
//	genkit.DefineSchemaFor[User](g)
//
//	genkit.Generate(ctx, g, ai.WithOutputSchemaName("User"), ai.WithPrompt("What is your name?"))
func DefineSchemaFor[T any](g *Genkit) {
	core.DefineSchemaFor[T](g.reg)
}

// DefineDataPrompt creates a new [ai.DataPrompt] with strongly-typed input and output.
// It automatically infers input schema from the In type parameter and configures
// output schema and JSON format from the Out type parameter (unless Out is string).
//
// This is a convenience wrapper around [DefinePrompt] that provides compile-time
// type safety for both input and output. For prompts that don't need to be registered,
// use [ai.NewDataPrompt] instead.
//
// DefineDataPrompt accepts the same options as [DefinePrompt]. See [DefinePrompt] for
// the full list of available options. Note that input and output schemas are automatically
// inferred from the type parameters.
//
// Example:
//
//	type GeoInput struct {
//		Country string `json:"country"`
//	}
//
//	type GeoOutput struct {
//		Capital string `json:"capital"`
//	}
//
//	capitalPrompt := genkit.DefineDataPrompt[GeoInput, GeoOutput](g, "findCapital",
//		ai.WithModelName("googleai/gemini-2.5-flash"),
//		ai.WithSystem("You are a helpful geography assistant."),
//		ai.WithPrompt("What is the capital of {{country}}?"),
//	)
//
//	output, resp, err := capitalPrompt.Execute(ctx, GeoInput{Country: "France"})
//	if err != nil {
//		log.Fatalf("Execute failed: %v", err)
//	}
//	fmt.Printf("Capital: %s\n", output.Capital)
func DefineDataPrompt[In, Out any](g *Genkit, name string, opts ...ai.PromptOption) *ai.DataPrompt[In, Out] {
	return ai.DefineDataPrompt[In, Out](g.reg, name, opts...)
}

// LookupDataPrompt looks up a prompt by name and wraps it with type information.
// This is useful for wrapping prompts loaded from .prompt files with strong types.
// It returns nil if the prompt was not found.
func LookupDataPrompt[In, Out any](g *Genkit, name string) *ai.DataPrompt[In, Out] {
	return ai.LookupDataPrompt[In, Out](g.reg, name)
}

// GenerateWithRequest performs a model generation request using explicitly provided
// [ai.GenerateActionOptions]. This function is typically used in conjunction with
// prompts defined via [DefinePrompt], where [ai.prompt.Render] produces the
// `actionOpts`. It allows fine-grained control over the request sent to the model.
//
// It accepts optional model middleware (`mw`) for intercepting/modifying the request/response,
// and an optional streaming callback (`cb`) of type [ai.ModelStreamCallback] to receive
// response chunks as they arrive.
//
// Example (using options rendered from a prompt):
//
//	myPrompt := genkit.LookupPrompt(g, "myDefinedPrompt")
//	actionOpts, err := myPrompt.Render(ctx, map[string]any{"topic": "go programming"})
//	if err != nil {
//		// handle error
//	}
//
//	// Optional: Modify actionOpts here if needed (config is provider-specific)
//
//	resp, err := genkit.GenerateWithRequest(ctx, g, actionOpts, nil, nil) // No middleware or streaming
//	if err != nil {
//		// handle error
//	}
//	fmt.Println(resp.Text())
func GenerateWithRequest(ctx context.Context, g *Genkit, actionOpts *ai.GenerateActionOptions, mw []ai.ModelMiddleware, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	return ai.GenerateWithRequest(ctx, g.reg, actionOpts, mw, cb)
}

// Generate performs a model generation request using a flexible set of options
// provided via [ai.GenerateOption] arguments. It's a convenient way to make
// generation calls without pre-defining a prompt object.
//
// # Options
//
// Model and Configuration:
//   - [ai.WithModel]: Specify the model (accepts [ai.Model] or [ai.ModelRef])
//   - [ai.WithModelName]: Specify model by name string (e.g., "googleai/gemini-2.5-flash")
//   - [ai.WithConfig]: Set generation parameters (temperature, max tokens, etc.)
//
// Prompting:
//   - [ai.WithPrompt]: Set the user prompt (supports format strings)
//   - [ai.WithPromptFn]: Set a function that generates the user prompt dynamically
//   - [ai.WithSystem]: Set system instructions
//   - [ai.WithSystemFn]: Set a function that generates system instructions dynamically
//   - [ai.WithMessages]: Provide conversation history
//   - [ai.WithMessagesFn]: Provide a function that generates conversation history
//
// Tools and Resources:
//   - [ai.WithTools]: Enable tools the model can call
//   - [ai.WithToolChoice]: Control whether tool calls are required, optional, or disabled
//   - [ai.WithMaxTurns]: Set maximum tool call iterations
//   - [ai.WithReturnToolRequests]: Return tool requests instead of executing them
//   - [ai.WithResources]: Attach resources available during generation
//
// Output:
//   - [ai.WithOutputType]: Request structured output matching a Go type
//   - [ai.WithOutputSchema]: Provide a custom JSON schema for output
//   - [ai.WithOutputSchemaName]: Reference a pre-registered schema by name
//   - [ai.WithOutputFormat]: Specify output format (json, text, etc.)
//   - [ai.WithOutputEnums]: Constrain output to specific enum values
//
// Context and Streaming:
//   - [ai.WithDocs]: Provide context documents
//   - [ai.WithTextDocs]: Provide context as text strings
//   - [ai.WithStreaming]: Enable streaming with a callback function
//   - [ai.WithMiddleware]: Apply middleware to the model request/response
//
// Tool Continuation:
//   - [ai.WithToolResponses]: Resume generation with tool response parts
//   - [ai.WithToolRestarts]: Resume generation by restarting tool requests
//
// Example:
//
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithModelName("googleai/gemini-2.5-flash"),
//		ai.WithPrompt("Write a short poem about clouds."),
//	)
//	if err != nil {
//		log.Fatalf("Generate failed: %v", err)
//	}
//
//	fmt.Println(resp.Text())
func Generate(ctx context.Context, g *Genkit, opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
	return ai.Generate(genkitCtxKey.NewContext(ctx, g), g.reg, opts...)
}

// GenerateStream generates a model response and streams the output.
// It returns an iterator that yields streaming results.
//
// If the yield function is passed a non-nil error, generation has failed with that
// error; the yield function will not be called again.
//
// If the yield function's [ai.ModelStreamValue] argument has Done == true, the value's
// Response field contains the final response; the yield function will not be called again.
//
// Otherwise the Chunk field of the passed [ai.ModelStreamValue] holds a streamed chunk.
//
// GenerateStream accepts the same options as [Generate]. See [Generate] for the full
// list of available options.
//
// Example:
//
//	for result, err := range genkit.GenerateStream(ctx, g,
//		ai.WithPrompt("Tell me a story about a brave knight."),
//	) {
//		if err != nil {
//			log.Fatalf("Stream error: %v", err)
//		}
//		if result.Done {
//			fmt.Println("\nFinal response:", result.Response.Text())
//		} else {
//			fmt.Print(result.Chunk.Text())
//		}
//	}
func GenerateStream(ctx context.Context, g *Genkit, opts ...ai.GenerateOption) iter.Seq2[*ai.ModelStreamValue, error] {
	return ai.GenerateStream(genkitCtxKey.NewContext(ctx, g), g.reg, opts...)
}

// GenerateOperation performs a model generation request using a flexible set of options
// provided via [ai.GenerateOption] arguments. It's designed for long-running generation
// tasks that may not complete immediately.
//
// Unlike [Generate], this function returns a [ai.ModelOperation] which can be used to
// check the status of the operation and get the result. Use [CheckModelOperation] to
// poll for completion.
//
// GenerateOperation accepts the same options as [Generate]. See [Generate] for the full
// list of available options.
//
// Example:
//
//	op, err := genkit.GenerateOperation(ctx, g,
//		ai.WithModelName("googleai/veo-2.0-generate-001"),
//		ai.WithPrompt("A banana riding a bicycle."),
//	)
//	if err != nil {
//		log.Fatalf("GenerateOperation failed: %v", err)
//	}
//
//	fmt.Println(op.ID)
//
//	// Check the status of the operation
//	op, err = genkit.CheckModelOperation(ctx, g, op)
//	if err != nil {
//		log.Fatalf("failed to check operation status: %v", err)
//	}
//
//	fmt.Println(op.Done)
//
//	// Get the result of the operation
//	fmt.Println(op.Output.Text())
func GenerateOperation(ctx context.Context, g *Genkit, opts ...ai.GenerateOption) (*ai.ModelOperation, error) {
	return ai.GenerateOperation(genkitCtxKey.NewContext(ctx, g), g.reg, opts...)
}

// CheckModelOperation checks the status of a background model operation by looking up the model and calling its Check method.
func CheckModelOperation(ctx context.Context, g *Genkit, op *ai.ModelOperation) (*ai.ModelOperation, error) {
	return ai.CheckModelOperation(ctx, g.reg, op)
}

// GenerateText performs a model generation request similar to [Generate], but
// directly returns the generated text content as a string. It's a convenience
// wrapper for cases where only the textual output is needed.
//
// GenerateText accepts the same options as [Generate]. See [Generate] for the full
// list of available options.
//
// Example:
//
//	joke, err := genkit.GenerateText(ctx, g,
//		ai.WithPrompt("Tell me a funny programming joke."),
//	)
//	if err != nil {
//		log.Fatalf("GenerateText failed: %v", err)
//	}
//	fmt.Println(joke)
func GenerateText(ctx context.Context, g *Genkit, opts ...ai.GenerateOption) (string, error) {
	return ai.GenerateText(genkitCtxKey.NewContext(ctx, g), g.reg, opts...)
}

// GenerateData performs a model generation request, expecting structured output
// (typically JSON) that conforms to the schema inferred from the Out type parameter.
// It automatically sets output type and JSON format, unmarshals the response, and
// returns the typed result.
//
// GenerateData accepts the same options as [Generate]. See [Generate] for the full
// list of available options. Note that output options like [ai.WithOutputType] are
// automatically applied based on the Out type parameter.
//
// Example:
//
//	type BookInfo struct {
//		Title  string `json:"title"`
//		Author string `json:"author"`
//		Year   int    `json:"year"`
//	}
//
//	book, _, err := genkit.GenerateData[BookInfo](ctx, g,
//		ai.WithPrompt("Tell me about 'The Hitchhiker's Guide to the Galaxy'."),
//	)
//	if err != nil {
//		log.Fatalf("GenerateData failed: %v", err)
//	}
//
//	log.Printf("Book: %+v\n", book) // Output: Book: {Title:The Hitchhiker's Guide to the Galaxy Author:Douglas Adams Year:1979}
func GenerateData[Out any](ctx context.Context, g *Genkit, opts ...ai.GenerateOption) (*Out, *ai.ModelResponse, error) {
	return ai.GenerateData[Out](genkitCtxKey.NewContext(ctx, g), g.reg, opts...)
}

// GenerateDataStream generates a model response with streaming and returns strongly-typed output.
// It returns an iterator that yields streaming results.
//
// If the yield function is passed a non-nil error, generation has failed with that
// error; the yield function will not be called again.
//
// If the yield function's [ai.StreamValue] argument has Done == true, the value's
// Output and Response fields contain the final typed output and response; the yield function
// will not be called again.
//
// Otherwise the Chunk field of the passed [ai.StreamValue] holds a streamed chunk.
//
// GenerateDataStream accepts the same options as [Generate]. See [Generate] for the full
// list of available options. Note that output options are automatically applied based on
// the Out type parameter.
//
// Example:
//
//	type Story struct {
//		Title   string `json:"title"`
//		Content string `json:"content"`
//	}
//
//	for result, err := range genkit.GenerateDataStream[Story](ctx, g,
//		ai.WithPrompt("Write a short story about a brave knight."),
//	) {
//		if err != nil {
//			log.Fatalf("Stream error: %v", err)
//		}
//		if result.Done {
//			fmt.Printf("Story: %+v\n", result.Output)
//		} else {
//			fmt.Print(result.Chunk.Text())
//		}
//	}
func GenerateDataStream[Out any](ctx context.Context, g *Genkit, opts ...ai.GenerateOption) iter.Seq2[*ai.StreamValue[Out, Out], error] {
	return ai.GenerateDataStream[Out](genkitCtxKey.NewContext(ctx, g), g.reg, opts...)
}

// Retrieve performs a document retrieval request using a flexible set of options
// provided via [ai.RetrieverOption] arguments. It's a convenient way to retrieve
// relevant documents from registered retrievers without directly calling the
// retriever instance.
//
// # Options
//
//   - [ai.WithRetriever]: Specify the retriever (accepts [ai.Retriever] or [ai.RetrieverRef])
//   - [ai.WithRetrieverName]: Specify retriever by name string
//   - [ai.WithConfig]: Set retriever-specific configuration
//   - [ai.WithTextDocs]: Provide query text as documents
//   - [ai.WithDocs]: Provide query as [ai.Document] instances
//
// Example:
//
//	resp, err := genkit.Retrieve(ctx, g,
//		ai.WithRetrieverName("myRetriever"),
//		ai.WithTextDocs("What is the capital of France?"),
//	)
//	if err != nil {
//		log.Fatalf("Retrieve failed: %v", err)
//	}
//
//	for _, doc := range resp.Documents {
//		fmt.Printf("Document: %+v\n", doc)
//	}
func Retrieve(ctx context.Context, g *Genkit, opts ...ai.RetrieverOption) (*ai.RetrieverResponse, error) {
	return ai.Retrieve(ctx, g.reg, opts...)
}

// Embed performs an embedding request using a flexible set of options
// provided via [ai.EmbedderOption] arguments. It's a convenient way to generate
// embeddings from registered embedders without directly calling the embedder instance.
//
// # Options
//
//   - [ai.WithEmbedder]: Specify the embedder (accepts [ai.Embedder] or [ai.EmbedderRef])
//   - [ai.WithEmbedderName]: Specify embedder by name string
//   - [ai.WithConfig]: Set embedder-specific configuration
//   - [ai.WithTextDocs]: Provide text to embed
//   - [ai.WithDocs]: Provide [ai.Document] instances to embed
//
// Example:
//
//	resp, err := genkit.Embed(ctx, g,
//		ai.WithEmbedderName("myEmbedder"),
//		ai.WithTextDocs("Hello, world!"),
//	)
//	if err != nil {
//		log.Fatalf("Embed failed: %v", err)
//	}
//
//	for i, embedding := range resp.Embeddings {
//		fmt.Printf("Embedding %d: %v\n", i, embedding.Embedding)
//	}
func Embed(ctx context.Context, g *Genkit, opts ...ai.EmbedderOption) (*ai.EmbedResponse, error) {
	return ai.Embed(ctx, g.reg, opts...)
}

// DefineRetriever defines a custom retriever implementation, registers it as a
// [core.Action] of type Retriever, and returns an [ai.Retriever].
// Retrievers are used to find documents relevant to a given query, often by
// performing similarity searches in a vector database.
//
// The `name` is the unique identifier for the retriever. The `fn` function
// contains the logic to process an [ai.RetrieverRequest] (containing the query)
// and return an [ai.RetrieverResponse] (containing the relevant documents).
//
// For retrievers that don't need to be registered (e.g., for plugin development),
// use [ai.NewRetriever] instead.
func DefineRetriever(g *Genkit, name string, opts *ai.RetrieverOptions, fn ai.RetrieverFunc) ai.Retriever {
	return ai.DefineRetriever(g.reg, name, opts, fn)
}

// LookupRetriever retrieves a registered [ai.Retriever] by its provider and name.
// It returns the retriever instance if found, or `nil` if no retriever with the
// given identifier is registered (e.g., via [DefineRetriever] or a plugin).
func LookupRetriever(g *Genkit, name string) ai.Retriever {
	return ai.LookupRetriever(g.reg, name)
}

// DefineEmbedder defines a custom text embedding implementation, registers it as a
// [core.Action] of type Embedder, and returns an [ai.Embedder].
// Embedders convert text documents or queries into numerical vector representations (embeddings).
//
// The `name` is the unique identifier for the embedder.
// The `fn` function contains the logic to process an [ai.EmbedRequest] (containing documents or a query)
// and return an [ai.EmbedResponse] (containing the corresponding embeddings).
//
// For embedders that don't need to be registered (e.g., for plugin development),
// use [ai.NewEmbedder] instead.
func DefineEmbedder(g *Genkit, name string, opts *ai.EmbedderOptions, fn ai.EmbedderFunc) ai.Embedder {
	return ai.DefineEmbedder(g.reg, name, opts, fn)
}

// LookupEmbedder retrieves a registered [ai.Embedder] by its provider and name.
// It returns the embedder instance if found, or `nil` if no embedder with the
// given identifier is registered (e.g., via [DefineEmbedder] or a plugin).
// It will try to resolve the embedder dynamically if the embedder is not found.
func LookupEmbedder(g *Genkit, name string) ai.Embedder {
	return ai.LookupEmbedder(g.reg, name)
}

// LookupPlugin retrieves a registered plugin instance by its name.
// Plugins are registered during initialization via [WithPlugins].
// It returns the plugin instance as `Plugin` if found, or `nil` otherwise.
// The caller is responsible for type-asserting the returned value to the
// specific plugin api.
func LookupPlugin(g *Genkit, name string) api.Plugin {
	return g.reg.LookupPlugin(name)
}

// DefineEvaluator defines an evaluator that processes test cases one by one,
// registers it as a [core.Action] of type Evaluator, and returns an [ai.Evaluator].
// Evaluators are used to assess the quality or performance of AI models or flows
// based on a dataset of test cases.
//
// This variant calls the provided `eval` function for each individual test case
// ([ai.EvaluatorCallbackRequest]) in the evaluation dataset.
//
// The `provider` and `name` form the unique identifier. `options` provide
// metadata about the evaluator ([ai.EvaluatorOptions]). The `eval` function
// implements the logic to score a single test case and returns the results
// in an [ai.EvaluatorCallbackResponse].
func DefineEvaluator(g *Genkit, name string, opts *ai.EvaluatorOptions, fn ai.EvaluatorFunc) ai.Evaluator {
	return ai.DefineEvaluator(g.reg, name, opts, fn)
}

// DefineBatchEvaluator defines an evaluator that processes the entire dataset at once,
// registers it as a [core.Action] of type Evaluator, and returns an [ai.Evaluator].
//
// This variant provides the full evaluation request ([ai.EvaluatorRequest]), including
// the entire dataset, to the `eval` function. This allows for more flexible processing,
// such as batching calls to external services or parallelizing computations.
//
// The `provider` and `name` form the unique identifier. `options` provide
// metadata about the evaluator ([ai.EvaluatorOptions]). The `eval` function
// implements the logic to score the dataset and returns the aggregated results
// in an [ai.EvaluatorResponse].
func DefineBatchEvaluator(g *Genkit, name string, opts *ai.EvaluatorOptions, fn ai.BatchEvaluatorFunc) ai.Evaluator {
	return ai.DefineBatchEvaluator(g.reg, name, opts, fn)
}

// LookupEvaluator retrieves a registered [ai.Evaluator] by its provider and name.
// It returns the evaluator instance if found, or `nil` if no evaluator with the
// given identifier is registered (e.g., via [DefineEvaluator], [DefineBatchEvaluator],
// or a plugin).
func LookupEvaluator(g *Genkit, name string) ai.Evaluator {
	return ai.LookupEvaluator(g.reg, name)
}

// Evaluate performs an evaluation request using a flexible set of options
// provided via [ai.EvaluatorOption] arguments. It's a convenient way to run
// evaluations using registered evaluators without directly calling the
// evaluator instance.
//
// # Options
//
//   - [ai.WithEvaluator]: Specify the evaluator (accepts [ai.Evaluator] or [ai.EvaluatorRef])
//   - [ai.WithEvaluatorName]: Specify evaluator by name string
//   - [ai.WithDataset]: Provide the dataset of examples to evaluate
//   - [ai.WithID]: Set a unique identifier for this evaluation run
//   - [ai.WithConfig]: Set evaluator-specific configuration
//
// Example:
//
//	dataset := []*ai.Example{
//		{
//			Input: "What is the capital of France?",
//			Reference: "Paris",
//		},
//	}
//
//	resp, err := genkit.Evaluate(ctx, g,
//		ai.WithEvaluatorName("myEvaluator"),
//		ai.WithDataset(dataset...),
//	)
//	if err != nil {
//		log.Fatalf("Evaluate failed: %v", err)
//	}
//
//	for _, result := range *resp {
//		fmt.Printf("Evaluation result: %+v\n", result)
//	}
func Evaluate(ctx context.Context, g *Genkit, opts ...ai.EvaluatorOption) (*ai.EvaluatorResponse, error) {
	return ai.Evaluate(ctx, g.reg, opts...)
}

// LoadPromptDir loads all `.prompt` files from the specified directory `dir`
// into the registry, associating them with the given `namespace`.
// Files starting with `_` are treated as partials and are not registered as
// executable prompts but can be included in other prompts.
//
// If `dir` is empty, it defaults to "./prompts". If the directory doesn't exist,
// it logs a debug message (if using the default) or panics (if specified).
// The `namespace` acts as a prefix to the prompt name (e.g., namespace "myApp" and
// file "greeting.prompt" results in prompt name "myApp/greeting"). Use an empty
// string for no namespace.
//
// This function is often called implicitly by [Init] using the directory specified
// by [WithPromptDir], but can be called explicitly to load prompts from other
// locations or with different namespaces.
func LoadPromptDir(g *Genkit, dir, namespace string) {
	loadPromptDirOS(g.reg, dir, namespace)
}

// loadPromptDirOS loads prompts from an OS directory by converting to os.DirFS.
func loadPromptDirOS(r api.Registry, dir, namespace string) {
	useDefaultDir := false
	if dir == "" {
		dir = "./prompts"
		useDefaultDir = true
	}

	absPath, err := filepath.Abs(dir)
	if err != nil {
		if !useDefaultDir {
			panic(fmt.Errorf("failed to resolve prompt directory %q: %w", dir, err))
		}
		slog.Debug("default prompt directory not found, skipping loading .prompt files", "dir", dir)
		return
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		if !useDefaultDir {
			panic(fmt.Errorf("failed to resolve prompt directory %q: %w", dir, err))
		}
		slog.Debug("Default prompt directory not found, skipping loading .prompt files", "dir", dir)
		return
	}

	ai.LoadPromptDirFromFS(r, os.DirFS(absPath), ".", namespace)
}

// LoadPromptDirFromFS loads all `.prompt` files from the specified embedded filesystem `fsys`
// into the registry, associating them with the given `namespace`.
// Files starting with `_` are treated as partials and are not registered as
// executable prompts but can be included in other prompts.
//
// The `fsys` parameter should be an [fs.FS] implementation (e.g., [embed.FS]).
// The `dir` parameter specifies the directory within the filesystem where
// prompts are located (e.g., "prompts" if using `//go:embed prompts/*`).
// The `namespace` acts as a prefix to the prompt name (e.g., namespace "myApp" and
// file "greeting.prompt" results in prompt name "myApp/greeting"). Use an empty
// string for no namespace.
//
// This function provides an alternative to [LoadPromptDir] for loading prompts
// from embedded filesystems, enabling self-contained binaries without external
// prompt files.
//
// Example:
//
//	import "embed"
//
//	//go:embed prompts/*
//	var promptsFS embed.FS
//
//	func main() {
//		g := genkit.Init(ctx)
//		genkit.LoadPromptDirFromFS(g, promptsFS, "prompts", "myNamespace")
//	}
func LoadPromptDirFromFS(g *Genkit, fsys fs.FS, dir, namespace string) {
	ai.LoadPromptDirFromFS(g.reg, fsys, dir, namespace)
}

// LoadPrompt loads a single `.prompt` file specified by `path` into the registry,
// associating it with the given `namespace`, and returns the resulting [ai.prompt].
//
// The `path` should be the full path to the `.prompt` file.
// The `namespace` acts as a prefix to the prompt name (e.g., namespace "myApp" and
// path "/path/to/greeting.prompt" results in prompt name "myApp/greeting"). Use an
// empty string for no namespace.
//
// This provides a way to load specific prompt files programmatically, outside of the
// automatic loading done by [Init] or [LoadPromptDir].
//
// Example:
//
//	// Load a specific prompt file with a namespace
//	customPrompt := genkit.LoadPrompt(g, "./prompts/analyzer.prompt", "analysis")
//	if customPrompt == nil {
//		log.Fatal("Custom prompt not found or failed to parse.")
//	}
//
//	// Execute the loaded prompt
//	resp, err := customPrompt.Execute(ctx, ai.WithInput(map[string]any{"text": "some data"}))
//	// ... handle response and error ...
func LoadPrompt(g *Genkit, path, namespace string) ai.Prompt {
	dir, filename := filepath.Split(path)
	if dir == "" {
		dir = "."
	} else {
		dir = filepath.Clean(dir)
	}

	return ai.LoadPromptFromFS(g.reg, os.DirFS(dir), ".", filename, namespace)
}

// LoadPromptFromSource loads a prompt from raw `.prompt` file content (frontmatter + template)
// into the registry and returns the resulting [ai.Prompt].
//
// The `source` parameter should contain the complete `.prompt` file text, including
// the YAML frontmatter (delimited by `---`) and the template body.
// The `name` parameter is the prompt name, which may include a variant suffix
// (e.g., "greeting" or "greeting.formal").
// The `namespace` acts as a prefix to the prompt name. Use an empty string for no namespace.
//
// This is useful for loading prompts from sources other than the filesystem,
// such as databases, environment variables, or embedded strings.
//
// Example:
//
//	promptSource := `---
//	model: googleai/gemini-2.5-flash
//	input:
//	  schema:
//	    name: string
//	---
//	Hello, {{name}}!
//	`
//
//	prompt, err := genkit.LoadPromptFromSource(g, promptSource, "greeting", "myApp")
//	if err != nil {
//		log.Fatalf("Failed to load prompt: %v", err)
//	}
//
//	resp, err := prompt.Execute(ctx, ai.WithInput(map[string]any{"name": "World"}))
//	// ...
func LoadPromptFromSource(g *Genkit, source, name, namespace string) (ai.Prompt, error) {
	return ai.LoadPromptFromSource(g.reg, source, name, namespace)
}

// DefinePartial wraps DefinePartial to register a partial template with the given name and source.
// Partials can be referenced in templates with the syntax {{>partialName}}.
func DefinePartial(g *Genkit, name string, source string) {
	g.reg.RegisterPartial(name, source)
}

// DefineHelper wraps DefineHelper to register a helper function with the given name.
// This allows for extending the templating capabilities with custom logic.
//
// Example usage:
//
//	genkit.DefineHelper(g, "uppercase", func(s string) string {
//		return strings.ToUpper(s)
//	})
//
// In a template, you would use it as:
//
//	{{uppercase "hello"}} => "HELLO"
func DefineHelper(g *Genkit, name string, fn any) {
	g.reg.RegisterHelper(name, fn)
}

// DefineFormat defines a new [ai.Formatter] and registers it in the registry.
// Formatters control how model responses are structured and parsed.
//
// Formatters can be used with [ai.WithOutputFormat] to inject specific formatting
// instructions into prompts and automatically format the model response according
// to the desired output structure.
//
// Built-in formatters include:
//   - "text": Plain text output (default if no format specified)
//   - "json": Structured JSON output (default when an output schema is provided)
//   - "jsonl": JSON Lines format for streaming structured data
//
// Example:
//
//	// Define a custom formatter
//	type csvFormatter struct{}
//	func (f csvFormatter) Name() string { return "csv" }
//	func (f csvFormatter) Handler(schema map[string]any) (ai.FormatHandler, error) {
//		// Implementation details...
//	}
//
//	// Register the formatter
//	genkit.DefineFormat(g, "csv", csvFormatter{})
//
//	// Use the formatter in a generation request
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithPrompt("List 3 countries and their capitals"),
//		ai.WithOutputFormat("csv"), // Use the custom formatter
//	)
func DefineFormat(g *Genkit, name string, formatter ai.Formatter) {
	ai.DefineFormat(g.reg, name, formatter)
}

// IsDefinedFormat checks if a formatter with the given name is registered in the registry.
func IsDefinedFormat(g *Genkit, name string) bool {
	return g.reg.LookupValue("/format/"+name) != nil
}

// DefineResource defines a resource and registers it with the Genkit instance.
// Resources provide content that can be referenced in prompts via URI.
//
// Example:
//
//	DefineResource(g, "company-docs", &ai.ResourceOptions{
//	  URI: "file:///docs/handbook.pdf",
//	  Description: "Company handbook",
//	}, func(ctx context.Context, input *ai.ResourceInput) (*ai.ResourceOutput, error) {
//	  content, err := os.ReadFile("/docs/handbook.pdf")
//	  if err != nil {
//	    return nil, err
//	  }
//	  return &ai.ResourceOutput{
//	    Content: []*ai.Part{ai.NewTextPart(string(content))},
//	  }, nil
//	})
func DefineResource(g *Genkit, name string, opts *ai.ResourceOptions, fn ai.ResourceFunc) ai.Resource {
	return ai.DefineResource(g.reg, name, opts, fn)
}

// FindMatchingResource finds a resource that matches the given URI.
func FindMatchingResource(g *Genkit, uri string) (ai.Resource, *ai.ResourceInput, error) {
	return ai.FindMatchingResource(g.reg, uri)
}

// NewResource creates an unregistered resource action that can be temporarily
// attached during generation via WithResources option.
//
// Example:
//
//	resource := NewResource("user-data", &ai.ResourceOptions{
//	  Template: "user://profile/{id}",
//	}, func(ctx context.Context, input *ai.ResourceInput) (*ai.ResourceOutput, error) {
//	  userID := input.Variables["id"]
//	  // Load user data dynamically...
//	  return &ai.ResourceOutput{Content: []*ai.Part{ai.NewTextPart(userData)}}, nil
//	})
//
//	// Use in generation:
//	ai.Generate(ctx, g,
//	  ai.WithPrompt([]*ai.Part{
//	    ai.NewTextPart("Analyze this user:"),
//	    ai.NewResourcePart("user://profile/123"),
//	  }),
//	  ai.WithResources(resource),
//	)
func NewResource(name string, opts *ai.ResourceOptions, fn ai.ResourceFunc) ai.Resource {
	return ai.NewResource(name, opts, fn)
}

// ListResources returns a slice of all resource actions
func ListResources(g *Genkit) []ai.Resource {
	acts := g.reg.ListActions()
	resources := []ai.Resource{}
	for _, action := range acts {
		actionDesc := action.Desc()
		if actionDesc.Type == api.ActionTypeResource {
			resource := ai.LookupResource(g.reg, actionDesc.Name)
			if resource != nil {
				resources = append(resources, resource)
			}
		}
	}
	return resources
}
