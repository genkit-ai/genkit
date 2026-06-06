// Copyright 2024 Google LLC
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
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/genkit/session"
	"github.com/firebase/genkit/go/plugins/vertexai"
)

func main() {
	ctx := context.Background()
	g, err := genkit.Init(ctx)
	if err != nil {
		log.Fatal(err)
	}

	if err := vertexai.Init(ctx, g, nil); err != nil {
		log.Fatal(err)
	}

	SimpleChat(ctx, g)
	StatefulChat(ctx, g)
	MultiThreadChat(ctx, g)
	PersistentStorageChat(ctx, g)
}

func SimpleChat(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-1.5-flash")

	chat, err := genkit.NewChat(
		ctx,
		g,
		genkit.WithModel(m),
		genkit.WithSystemText("You're a pirate first mate. Address the user as Captain and assist them however you can."),
		genkit.WithConfig(ai.GenerationCommonConfig{Temperature: 1.3}),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := chat.Send(ctx, "Hello")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}

func StatefulChat(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-1.5-pro")

	// To include state in a session, you need to instantiate a session explicitly
	s, err := session.New(ctx,
		session.WithData(session.Data{
			State: map[string]any{
				"username": "Michael",
			},
		},
		),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Modify the state using tools
	chatTool := genkit.DefineTool(
		g,
		"updateName",
		"use this tool to update the name of the user",
		func(ctx *ai.ToolContext, input struct {
			Name string
		}) (string, error) {
			// Set name in state
			s, err := session.FromContext(ctx)
			if err != nil {
				return "", err
			}

			err = s.UpdateState(map[string]any{
				"username": input.Name,
			})
			if err != nil {
				return "", err
			}

			return "changed username to " + input.Name, nil
		},
	)

	chat, err := genkit.NewChat(
		ctx,
		g,
		genkit.WithModel(m),
		genkit.WithSystemText("You're Kitt from Knight Rider. Address the user as Kitt would and always introduce yourself."),
		genkit.WithConfig(ai.GenerationCommonConfig{Temperature: 1}),
		genkit.WithSession(s),
		genkit.WithTools(chatTool),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := chat.Send(ctx, "Hello, my name is Earl")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
	data, err := s.GetData()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(data.State["username"])
}

func MultiThreadChat(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-1.5-flash")

	pirateChat, err := genkit.NewChat(
		ctx,
		g,
		genkit.WithModel(m),
		genkit.WithSystemText("You're a pirate first mate. Address the user as Captain and assist them however you can."),
		genkit.WithConfig(ai.GenerationCommonConfig{Temperature: 1.3}),
		genkit.WithThreadName("pirate"),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := pirateChat.Send(ctx, "Hello")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())

	lawyerChat, err := genkit.NewChat(
		ctx,
		g,
		genkit.WithModel(m),
		genkit.WithSystemText("You're a lawyer. Give unsolicited advice no matter what is asked."),
		genkit.WithConfig(ai.GenerationCommonConfig{Temperature: 1.3}),
		genkit.WithThreadName("lawyer"),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err = lawyerChat.Send(ctx, "Hello")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}

func PersistentStorageChat(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-1.5-pro")

	// To override default in-mem session storage
	store := &MyOwnSessionStore{
		SessionData: make(map[string]session.Data),
	}
	s, err := session.New(ctx,
		session.WithStore(store),
	)
	if err != nil {
		log.Fatal(err)
	}

	chat, err := genkit.NewChat(
		ctx,
		g,
		genkit.WithModel(m),
		genkit.WithSystemText("You're a helpful chatbox. Help the user."),
		genkit.WithConfig(ai.GenerationCommonConfig{Temperature: 1}),
		genkit.WithSession(s),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := chat.Send(ctx, "Hello, my name is Earl")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())

	// Load and use existing session
	session2, err := session.Load(ctx, s.GetID(), store)
	if err != nil {
		log.Fatal(err)
	}

	chat2, err := genkit.NewChat(
		ctx,
		g,
		genkit.WithModel(m),
		genkit.WithSystemText("You're a helpful chatbox. Help the user."),
		genkit.WithConfig(ai.GenerationCommonConfig{Temperature: 1}),
		genkit.WithSession(session2),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err = chat2.Send(ctx, "What's my name?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}

type MyOwnSessionStore struct {
	SessionData map[string]session.Data
}

func (s *MyOwnSessionStore) Get(sessionId string) (data session.Data, err error) {
	d, err := os.ReadFile("/tmp/" + sessionId)
	if err != nil {
		return data, err
	}
	err = json.Unmarshal(d, &data)
	if err != nil {
		return data, err
	}

	if data.Threads == nil {
		data.Threads = make(map[string][]*ai.Message)
	}

	s.SessionData[sessionId] = data
	return s.SessionData[sessionId], nil
}

func (s *MyOwnSessionStore) Save(sessionId string, data session.Data) error {
	s.SessionData[sessionId] = data
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	err = os.WriteFile("/tmp/"+sessionId, []byte(d), 0644)
	if err != nil {
		return err
	}
	return nil
}
