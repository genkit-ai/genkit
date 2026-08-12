// Copyright 2026 Google LLC
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

// This sample demonstrates how a prompt turns your own data into a request. It
// fills all four of a prompt's content slots (system, conversation, user, and
// context documents) from one typed input, then shows the template forms of the
// same slots alongside.
//
// The rule it is built around: in a prompt, template text is yours, so it is
// compiled, while anything a function returns is content you already produced,
// so it is sent verbatim. Reach for a template when the wording is fixed and
// only values vary, and for a function when the content itself depends on the
// data: branching on a field, trimming a conversation, attaching an image, or
// choosing documents.
//
// To run:
//
//	go run .
//
// In another terminal, answer a question with a conversation behind it:
//
//	curl -N -X POST http://localhost:8080/supportFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"request": {"area": "billing", "tier": "pro", "question": "Why was I charged twice this month?"}, "history": [{"role": "user", "text": "Hi, I have a question about my bill."}, {"role": "model", "text": "Happy to help. What is going on?"}]}}'
//
// Triage a ticket into a structured verdict. The question contains handlebars
// syntax, which reaches the model as written:
//
//	curl -X POST http://localhost:8080/triageFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"request": {"area": "api", "tier": "free", "question": "How do I write {{#if}} in a prompt template?"}}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
)

// maxHistoryMessages is how many recent messages the support prompt keeps.
// Everything older is replaced with a one-line note.
const maxHistoryMessages = 6

// SupportRequest is the prompt's input: the app's own shape, not a map. Every
// content function below receives exactly this type, whether the prompt was
// called in process, from the Dev UI, or over HTTP.
type SupportRequest struct {
	Area           string `json:"area" jsonschema:"description=The product area the question is about,enum=billing,enum=setup,enum=api,default=api"`
	Tier           string `json:"tier" jsonschema:"description=The customer's support tier,enum=free,enum=pro,enum=enterprise,default=free"`
	Question       string `json:"question" jsonschema:"description=The customer's question in their own words"`
	Screenshot     string `json:"screenshot,omitempty" jsonschema:"description=Optional data: or https: URL of a screenshot the customer attached"`
	ScreenshotType string `json:"screenshotType,omitempty" jsonschema:"description=Media type of the screenshot for example image/png"`
}

// Turn is one exchange in the conversation the client keeps. A real app would
// read these from a session or a database; taking them as data keeps the flows
// callable over HTTP.
type Turn struct {
	Role string `json:"role" jsonschema:"enum=user,enum=model"`
	Text string `json:"text"`
}

// SupportTicket is what the flows receive. The conversation is separate from
// the prompt's input because it is not part of the input schema: it is passed
// to Execute, and the prompt reaches it either at {{history}} in a template or
// through [ai.HistoryFromContext] in a function.
type SupportTicket struct {
	Request SupportRequest `json:"request"`
	History []Turn         `json:"history,omitempty" jsonschema:"description=The conversation so far oldest first"`
}

// Triage is the structured verdict the triage prompt returns.
type Triage struct {
	Category string `json:"category" jsonschema:"enum=bug,enum=how-to,enum=billing,enum=other"`
	Urgency  string `json:"urgency" jsonschema:"enum=low,enum=medium,enum=high"`
	Summary  string `json:"summary" jsonschema:"description=One sentence restating the customer's problem"`
}

// knowledgeBase stands in for a retriever. WithDocsFn is the hook where a real
// app would run a query built from the input.
var knowledgeBase = map[string][]string{
	"billing": {
		"Invoices are issued on the first of the month and cover the previous month's usage.",
		"A second charge in the same month is usually a pending authorization, which drops off within five business days.",
	},
	"setup": {
		"Set GEMINI_API_KEY in the environment before calling genkit.Init.",
		"The dev UI runs on port 4000 and the sample servers on port 8080.",
	},
	"api": {
		"Prompt templates use handlebars syntax: {{field}} interpolates a value from the input.",
		"Content returned by a content function is sent verbatim and is never compiled as a template.",
	},
}

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the
	// recommended practice.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	DefineSupportAnswer(g)
	DefineTriage(g)

	// Optionally, start a web server to make the flows callable via HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineSupportAnswer demonstrates all four content functions working from one
// input: system text chosen by branching, a trimmed conversation, multi-part
// user content, and documents selected from a field.
func DefineSupportAnswer(g *genkit.Genkit) {
	supportPrompt := genkit.DefinePrompt(
		g, "support",
		ai.WithModelName("googleai/gemini-flash-latest"),
		// WithInputType fixes the type every content function below receives.
		// Genkit converts whatever the caller passed into it, so one function
		// covers the in-process call, the default input, and a run from the
		// Dev UI. The values here are the defaults when a caller sends none.
		ai.WithInputType(SupportRequest{Area: "api", Tier: "free"}),

		// A function fits here because the instruction is built by branching on
		// the data rather than by filling in blanks. Its result is used
		// verbatim, so nothing in it has to be escaped.
		ai.WithSystemFn(func(ctx context.Context, in SupportRequest) (string, error) {
			var b strings.Builder
			b.WriteString("You are a support agent. Answer from the reference material you were given, and say so plainly when it does not cover the question.")
			switch in.Tier {
			case "enterprise":
				b.WriteString(" This is an enterprise customer: be thorough, and offer to escalate to an engineer.")
			case "pro":
				b.WriteString(" This is a pro customer: be direct and specific.")
			default:
				b.WriteString(" Keep the answer to a few sentences and point at the docs where you can.")
			}
			return b.String(), nil
		}),

		// The conversation slot. A prompt that declares it owns it: the messages
		// passed to Execute are not spliced in behind your back, they arrive
		// here through HistoryFromContext so the function can trim, summarize,
		// or reorder them. Keeping the tail and replacing the rest with a note
		// is what holds a long ticket inside the context window.
		ai.WithMessagesFn(func(ctx context.Context, in SupportRequest) ([]*ai.Message, error) {
			history := ai.HistoryFromContext(ctx)
			if len(history) <= maxHistoryMessages {
				return history, nil
			}
			// Drop forward to a user turn so what is kept starts with a whole
			// exchange rather than an orphaned model reply.
			dropped := len(history) - maxHistoryMessages
			for dropped < len(history) && history[dropped].Role != ai.RoleUser {
				dropped++
			}
			note := ai.NewUserTextMessage(fmt.Sprintf("(%d earlier messages in this ticket were omitted for brevity.)", dropped))
			return append([]*ai.Message{note}, history[dropped:]...), nil
		}),

		// Multi-part user content: the question, plus the screenshot when the
		// customer attached one. Returning parts is how a function reaches
		// non-text content; a plain string function can only produce text.
		ai.WithPromptPartsFn(func(ctx context.Context, in SupportRequest) ([]*ai.Part, error) {
			// The question is used verbatim, so a customer asking about {{#if}}
			// is quoted to the model rather than compiled as a template.
			parts := []*ai.Part{ai.NewTextPart(in.Question)}
			if in.Screenshot != "" {
				parts = append(parts,
					ai.NewTextPart("The customer attached this screenshot:"),
					ai.NewMediaPart(in.ScreenshotType, in.Screenshot))
			}
			return parts, nil
		}),

		// Retrieval driven by the input. A real app would run a query against a
		// retriever built from these fields; what comes back becomes the
		// request's context documents.
		ai.WithDocsFn(func(ctx context.Context, in SupportRequest) ([]*ai.Document, error) {
			var docs []*ai.Document
			for _, text := range knowledgeBase[in.Area] {
				docs = append(docs, ai.DocumentFromText(text, map[string]any{"area": in.Area}))
			}
			return docs, nil
		}),
	)

	genkit.DefineStreamingFlow(g, "supportFlow",
		func(ctx context.Context, ticket SupportTicket, sendChunk core.StreamCallback[string]) (string, error) {
			stream := supportPrompt.ExecuteStream(ctx,
				ai.WithInput(ticket.Request),
				// The conversation is passed per execution rather than baked
				// into the prompt. WithMessagesFn above decides what to do with
				// it; without a prompt that claims the slot, it would simply
				// become the middle of the conversation.
				ai.WithMessages(toMessages(ticket.History)...),
			)

			for result, err := range stream {
				if err != nil {
					return "", fmt.Errorf("could not answer the ticket: %w", err)
				}
				if result.Done {
					return result.Response.Text(), nil
				}
				sendChunk(ctx, result.Chunk.Text())
			}

			return "", nil
		},
	)
}

// DefineTriage demonstrates the template side of the same split. The wording is
// fixed and only values vary, so text is the right tool: one multi-turn template
// carries a worked example and marks where the real conversation goes.
func DefineTriage(g *genkit.Genkit) {
	triagePrompt := genkit.DefineDataPrompt[SupportRequest, *Triage](
		g, "triage",
		ai.WithModelName("googleai/gemini-flash-latest"),
		// System and user text are compiled against the input, so they can
		// reference its fields.
		ai.WithSystem("You are a support triage assistant. Classify the {{area}} question from this {{tier}} customer."),
		// Each {{role}} block starts a new message, so a single string can
		// carry a worked example ahead of the real conversation. {{history}}
		// is where the messages passed to Execute land. Without the marker
		// they are inserted just before the final user turn.
		ai.WithMessagesTemplate(`{{role "user"}}My deploys started failing with a 401 right after I rotated keys.
{{role "model"}}{"category": "bug", "urgency": "high", "summary": "Deploys fail with a 401 after a key rotation."}
{{history}}`),
		// A value interpolated into a template is substituted, not compiled, so
		// a question containing {{#if}} is safe here too. What was never safe
		// was feeding computed text back through the compiler, which is why the
		// function results above are verbatim.
		ai.WithPrompt("{{question}}"),
	)

	genkit.DefineFlow(g, "triageFlow",
		func(ctx context.Context, ticket SupportTicket) (*Triage, error) {
			triage, _, err := triagePrompt.Execute(ctx, ticket.Request,
				ai.WithMessages(toMessages(ticket.History)...))
			if err != nil {
				return nil, fmt.Errorf("could not triage the ticket: %w", err)
			}
			return triage, nil
		},
	)
}

// toMessages converts the app's own turns into Genkit messages. Message text is
// used verbatim, so a customer who typed {{#if}} is quoted rather than compiled.
func toMessages(turns []Turn) []*ai.Message {
	messages := make([]*ai.Message, 0, len(turns))
	for _, t := range turns {
		if t.Role == "model" {
			messages = append(messages, ai.NewModelTextMessage(t.Text))
			continue
		}
		messages = append(messages, ai.NewUserTextMessage(t.Text))
	}
	return messages
}
