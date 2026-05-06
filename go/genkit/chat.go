// Copyright 2024 Google LLC
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

package genkit

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit/session"
)

type Chat struct {
	Genkit     *Genkit                   `json:"genkit,omitempty"`
	Model      ai.Model                  `json:"model,omitempty"`      // The model to query
	ThreadName string                    `json:"threadName,omitempty"` // The chats threadname
	Session    *session.Session          `json:"session,omitempty"`    // The chats session
	SystemText string                    `json:"systemtext,omitempty"` // Message sent to the model as system instructions
	Prompt     *ai.Prompt                `json:"prompt,omitempty"`     // Optional prompt
	Input      any                       `json:"input,omitempty"`      // Optional input fields for the chat. This should be a struct, a pointer to a struct that matches the input schema, or a string.
	Request    *ChatRequest              `json:"request,omitempty"`
	Stream     ai.ModelStreamingCallback `json:"stream,omitempty"`
}

type ChatRequest struct {
	Config   any             `json:"config,omitempty"`
	Context  []*ai.Document  `json:"context,omitempty"`
	Messages []*ai.Message   `json:"messages,omitempty"`
	Schema   any             `json:"schema,omitempty"` // Defines the output format and schema
	Format   ai.OutputFormat `json:"outputformat,omitempty"`
	Tools    []ai.Tool       `json:"tools,omitempty"` // Tools lists the available tools that the model can ask the client to run.
}

type ChatOption func(c *Chat) error // ChatOption configures params for the chat

// NewChat creates a new chat instance with the provided AI model and options.
// If no session or thread name is provided, it adds a new session and default thread.
func NewChat(ctx context.Context, g *Genkit, opts ...ChatOption) (chat *Chat, err error) {
	chat = &Chat{
		Genkit:  g,
		Request: &ChatRequest{},
	}

	for _, with := range opts {
		err := with(chat)
		if err != nil {
			return nil, err
		}
	}

	if chat.Session == nil {
		s, err := session.New(ctx)
		if err != nil {
			return nil, err
		}
		chat.Session = s
	}

	if chat.ThreadName == "" {
		chat.ThreadName = "default"
	}

	return chat, nil
}

// Send sends a message to the chat, generating a response from the AI model.
// It retrieves the chat history from the session store, adds the new message
// to the history, and sends the entire conversation to the AI model for
// generating a response. If a system message is set for the chat, it is
// included in the conversation before the history.
func (c *Chat) Send(ctx context.Context, msg any) (resp *ai.ModelResponse, err error) {
	// Load history
	data, err := c.Session.GetData()
	if err != nil {
		return nil, err
	}

	// Set session details in context
	ctx = c.Session.SetContext(ctx)

	var messages []*ai.Message
	if c.SystemText != "" {
		// Add system message before history
		messages = append(messages, ai.NewSystemTextMessage(c.SystemText))
	}

	// Handle prompt if set
	if c.Prompt != nil {
		mr, err := c.Prompt.Render(ctx, c.Input)
		if err != nil {
			return nil, err
		}
		messages = append(messages, mr.Messages...)
	}

	// Add history
	messages = append(messages, data.Threads[c.ThreadName]...)

	// Add new message after history
	aiMsgs, err := getChatMessages(msg)
	if err != nil {
		return nil, err
	}
	messages = append(messages, aiMsgs...)

	// Assemble options
	var generateOptions []ai.GenerateOption
	generateOptions = append(generateOptions, ai.WithModel(c.Model))
	generateOptions = append(generateOptions, ai.WithMessages(messages...))
	generateOptions = append(generateOptions, ai.WithConfig(c.Request.Config))
	generateOptions = append(generateOptions, ai.WithStreaming(c.Stream))
	generateOptions = append(generateOptions, ai.WithTools(c.Request.Tools...))
	generateOptions = append(generateOptions, ai.WithContext(c.Request.Context...))

	if c.Request.Format != "" {
		generateOptions = append(generateOptions, ai.WithOutputFormat(c.Request.Format))
	}
	if c.Request.Schema != nil {
		generateOptions = append(generateOptions, ai.WithOutputSchema(c.Request.Schema))
	}

	// Call generate
	resp, err = ai.Generate(ctx,
		c.Genkit.reg,
		generateOptions...,
	)
	if err != nil {
		return nil, err
	}

	// Update history
	messages = append(messages, resp.Message)
	err = c.UpdateMessages(c.ThreadName, messages)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// SendText sends a text message to the chat, generating a response from the AI model and return the text.
// It retrieves the chat history from the session store, adds the new message
// to the history, and sends the entire conversation to the AI model for
// generating a response. If a system message is set for the chat, it is
// included in the conversation before the history.
func (c *Chat) SendText(ctx context.Context, msgText string) (string, error) {
	msg := ai.NewUserTextMessage(msgText)
	resp, err := c.Send(ctx, msg)
	if err != nil {
		return "", err
	}

	return resp.Text(), nil
}

// UpdateMessages updates the messages for the chat.
func (c *Chat) UpdateMessages(thread string, msgs []*ai.Message) error {
	c.Request.Messages = msgs
	return c.Session.UpdateMessages(thread, msgs)
}

// WithModel sets the model for the chat.
func WithModel(model ai.Model) ChatOption {
	return func(c *Chat) error {
		if c.Model != nil {
			return errors.New("genkit.WithModel: cannot set model more than once")
		}
		c.Model = model
		return nil
	}
}

// WithSession sets a session for the chat.
func WithSession(session *session.Session) ChatOption {
	return func(c *Chat) error {
		if c.Session != nil {
			return errors.New("genkit.WithSession: cannot set session more than once")
		}
		c.Session = session
		return nil
	}
}

// WithThreadName sets a thread name for the chat.
// This is used to seperate different conversions within the same session.
func WithThreadName(name string) ChatOption {
	return func(c *Chat) error {
		if c.ThreadName != "" {
			return errors.New("genkit.WithThreadName: cannot set threadname more than once")
		}

		c.ThreadName = name
		return nil
	}
}

// WithStreaming adds a streaming callback to the chat request.
func WithStreaming(cb ai.ModelStreamingCallback) ChatOption {
	return func(c *Chat) error {
		if c.Stream != nil {
			return errors.New("genkit.WithStreaming: cannot set streaming callback more than once")
		}
		c.Stream = cb
		return nil
	}
}

// WithSystemText sets a system message for the chat.
func WithSystemText(msg string) ChatOption {
	return func(c *Chat) error {
		if c.SystemText != "" {
			return errors.New("genkit.WithSystemText: cannot set systemText more than once")
		}
		c.SystemText = msg
		return nil
	}
}

// WithConfig adds provided config to chat.
func WithConfig(config ai.GenerationCommonConfig) ChatOption {
	return func(c *Chat) error {
		if c.Request.Config != nil {
			return errors.New("genkit.WithConfig: cannot set config more than once")
		}
		c.Request.Config = &config
		return nil
	}
}

// WithContext adds provided documents to chat.
func WithContext(docs ...*ai.Document) ChatOption {
	return func(c *Chat) error {
		if len(c.Request.Context) > 0 {
			return errors.New("genkit.WithContext: cannot set context more than once")
		}
		c.Request.Context = append(c.Request.Context, docs...)
		return nil
	}
}

// WithTools adds provided tools to the chat.
func WithTools(tools ...ai.Tool) ChatOption {
	return func(c *Chat) error {
		if len(c.Request.Tools) != 0 {
			return errors.New("genkit.WithTools: cannot set tools more than once")
		}
		c.Request.Tools = tools
		return nil
	}
}

// WithOutputSchema adds provided output schema to the chat.
func WithOutputSchema(schema any) ChatOption {
	return func(c *Chat) error {
		if c.Request.Schema != nil {
			return errors.New("genkit.WithOutputSchema: cannot set outputSchema more than once")
		}

		c.Request.Schema = schema
		return nil
	}
}

// WithOutputFormat adds provided output format to the chat.
func WithOutputFormat(format ai.OutputFormat) ChatOption {
	return func(c *Chat) error {
		if c.Request.Format != "" {
			return errors.New("genkit.WithOutputFormat: cannot set outputFormat more than once")
		}

		c.Request.Format = format
		return nil
	}
}

// WithPrompt sets a prompt for the chat.
func WithPrompt(prompt *ai.Prompt) ChatOption {
	return func(c *Chat) error {
		if c.Prompt != nil {
			return errors.New("genkit.WithPrompt: cannot set prompt more than once")
		}
		c.Prompt = prompt
		return nil
	}
}

// WithInput adds input to pass to the model.
func WithInput(input any) ChatOption {
	return func(c *Chat) error {
		if c.Input != nil {
			return errors.New("genkit.WithInput: cannot set input more than once")
		}
		c.Input = input
		return nil
	}
}

// getChatMessages takes a msg and returns a slice of messages.
func getChatMessages(msg any) (messages []*ai.Message, err error) {
	switch v := msg.(type) {
	case int:
		messages = append(messages, ai.NewUserTextMessage(strconv.Itoa(v)))
	case float32:
	case float64:
		messages = append(messages, ai.NewUserTextMessage(fmt.Sprintf("%f", v)))
	case string:
		messages = append(messages, ai.NewUserTextMessage(v))
	case ai.Message:
		messages = append(messages, &v)
	case *ai.Message:
		messages = append(messages, v)
	case []ai.Message:
		for _, m := range v {
			messages = append(messages, &m)
		}
	case []*ai.Message:
		messages = append(messages, v...)
	default:
		return messages, fmt.Errorf("genkit.getChatMessages: unknown message type %T", v)
	}

	return messages, nil
}
