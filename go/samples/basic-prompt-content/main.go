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
// fills all four content slots (system, conversation, user, and context
// documents) from one typed input, with the template form of each alongside.
//
// The rule: template text is yours, so it is compiled; anything a function
// returns is content you already produced, so it is sent verbatim. Use a
// template when the wording is fixed and only values vary, and a function when
// the content depends on the data, such as branching on a field, trimming a
// conversation, attaching an image, or choosing documents.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to see the assembled request in a trace of every run at
// http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Answer a question with a conversation behind it:
//
//	curl -N -X POST 'http://localhost:8080/supportFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"request": {"area": "billing", "tier": "pro", "question": "Why was I charged twice this month?"}, "history": [{"role": "user", "text": "Hi, I have a question about my bill."}, {"role": "model", "text": "Happy to help. What is going on?"}]}}'
//
// Triage one into a structured verdict. This question contains handlebars
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
	"google.golang.org/genai"
)

// maxHistoryMessages is how many recent messages the support prompt keeps.
// Older ones are replaced with a one-line note.
const maxHistoryMessages = 6

// SupportRequest is the prompt's input: the app's own shape, not a map. Every
// content function below receives exactly this type, however it was called.
type SupportRequest struct {
	Area           string `json:"area" jsonschema:"enum=billing,enum=setup,enum=api,default=api" jsonschema_description:"The product area the question is about"`
	Tier           string `json:"tier" jsonschema:"enum=free,enum=pro,enum=enterprise,default=free" jsonschema_description:"The customer's support tier"`
	Question       string `json:"question" jsonschema:"default=Why was I charged twice this month?" jsonschema_description:"The customer's question in their own words"`
	Screenshot     string `json:"screenshot,omitempty" jsonschema_description:"Optional data: or https: URL of a screenshot the customer attached"`
	ScreenshotType string `json:"screenshotType,omitempty" jsonschema_description:"Media type of the screenshot for example image/png"`
}

// Turn is one exchange in the conversation the client keeps. A real app would
// read these from a session or a database.
type Turn struct {
	Role string `json:"role" jsonschema:"enum=user,enum=model" jsonschema_description:"Who said it"`
	Text string `json:"text" jsonschema_description:"What was said"`
}

// SupportTicket is what the flows receive. The conversation is separate from
// the input schema: it is passed to Execute, and the prompt reaches it at
// {{history}} in a template or through [ai.HistoryFromContext] in a function.
type SupportTicket struct {
	Request SupportRequest `json:"request"`
	History []Turn         `json:"history,omitempty" jsonschema_description:"The conversation so far oldest first"`
}

// Triage is the structured verdict the triage prompt returns.
type Triage struct {
	Category string `json:"category" jsonschema:"enum=bug,enum=how-to,enum=billing,enum=other"`
	Urgency  string `json:"urgency" jsonschema:"enum=low,enum=medium,enum=high"`
	Summary  string `json:"summary" jsonschema_description:"One sentence restating the customer's problem"`
}

// knowledgeBase stands in for a retriever.
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

// model is shared by every prompt below, so switching models or thinking levels
// for the whole sample is a one-line change.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

func main() {
	ctx := context.Background()

	// The Google AI plugin reads the API key from GEMINI_API_KEY or
	// GOOGLE_API_KEY, which is the recommended practice.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	DefineSupportAnswer(g)
	DefineTriage(g)

	// Serve every flow over HTTP.
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
		ai.WithModel(model),

		// WithInputType fixes the type every content function receives, whatever
		// the caller passed. The values here are the defaults.
		ai.WithInputType(SupportRequest{Area: "api", Tier: "free"}),

		// A function fits because the instruction branches on the data rather
		// than filling in blanks. Its result is verbatim, so nothing is escaped.
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

		// A prompt that declares the conversation slot owns it: the messages
		// passed to Execute arrive here through HistoryFromContext, to trim,
		// summarize, or reorder before they are sent.
		ai.WithMessagesFn(func(ctx context.Context, in SupportRequest) ([]*ai.Message, error) {
			history := ai.HistoryFromContext(ctx)
			if len(history) <= maxHistoryMessages {
				return history, nil
			}
			// Drop forward to a user turn, so what is kept starts with a whole
			// exchange rather than an orphaned model reply.
			dropped := len(history) - maxHistoryMessages
			for dropped < len(history) && history[dropped].Role != ai.RoleUser {
				dropped++
			}
			note := ai.NewUserTextMessage(fmt.Sprintf("(%d earlier messages in this ticket were omitted for brevity.)", dropped))
			return append([]*ai.Message{note}, history[dropped:]...), nil
		}),

		// Returning parts is how a function reaches non-text content; a string
		// function can only produce text.
		ai.WithPromptPartsFn(func(ctx context.Context, in SupportRequest) ([]*ai.Part, error) {
			// Verbatim, so a customer asking about {{#if}} is quoted.
			parts := []*ai.Part{ai.NewTextPart(in.Question)}
			if in.Screenshot != "" {
				parts = append(parts,
					ai.NewTextPart("The customer attached this screenshot:"),
					ai.NewMediaPart(in.ScreenshotType, in.Screenshot))
			}
			return parts, nil
		}),

		// Retrieval driven by the input. A real app would query a retriever
		// here; what comes back becomes the request's context documents.
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
			resp, err := supportPrompt.Execute(ctx,
				ai.WithInput(ticket.Request),

				// Passed per execution rather than baked into the prompt.
				// WithMessagesFn above decides what to do with it.
				ai.WithMessages(toMessages(ticket.History)...),

				// Forwarding every chunk is all this flow does, so WithStreaming
				// says it in one option. ExecuteStream is for a flow that has to
				// act on the chunks.
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not answer the ticket: %w", err)
			}
			return resp.Text(), nil
		},
	)
}

// DefineTriage is the template side of the same split: the wording is fixed and
// only values vary, so one multi-turn template carries a worked example and
// marks where the real conversation goes.
func DefineTriage(g *genkit.Genkit) {
	triagePrompt := genkit.DefineDataPrompt[SupportRequest, Triage](
		g, "triage",
		ai.WithModel(model),

		// System and user text are compiled against the input, so they can
		// reference its fields.
		ai.WithSystem("You are a support triage assistant. Classify the {{area}} question from this {{tier}} customer."),

		// Each {{role}} block starts a new message, so one string carries a
		// worked example ahead of the real conversation. {{history}} is where
		// the messages passed to Execute land.
		ai.WithMessagesTemplate(`{{role "user"}}My deploys started failing with a 401 right after I rotated keys.
{{role "model"}}{"category": "bug", "urgency": "high", "summary": "Deploys fail with a 401 after a key rotation."}
{{history}}`),

		// An interpolated value is substituted, not compiled, so a question
		// containing {{#if}} is safe here too.
		ai.WithPrompt("{{question}}"),
	)

	genkit.DefineFlow(g, "triageFlow",
		func(ctx context.Context, ticket SupportTicket) (Triage, error) {
			triage, _, err := triagePrompt.Execute(ctx, ticket.Request,
				ai.WithMessages(toMessages(ticket.History)...))
			if err != nil {
				return Triage{}, fmt.Errorf("could not triage the ticket: %w", err)
			}
			return triage, nil
		},
	)
}

// toMessages adapts the app's own turns to Genkit messages. Message text is
// verbatim, so a customer who typed {{#if}} is quoted.
func toMessages(turns []Turn) []*ai.Message {
	messages := make([]*ai.Message, 0, len(turns))
	for _, t := range turns {
		messages = append(messages, ai.NewTextMessage(ai.Role(t.Role), t.Text))
	}
	return messages
}
