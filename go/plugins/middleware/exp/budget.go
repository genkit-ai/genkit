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
//
// SPDX-License-Identifier: Apache-2.0

package exp

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"sync"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/invopop/jsonschema"
)

// BudgetScope is what a [Budget] counts against.
type BudgetScope string

const (
	// BudgetScopeRun counts every model call made under one Generate call:
	// the calls its TotalUsage counts (the tool loop, hooks, nested
	// generates), plus those of any subagent it runs. For a prompt-backed
	// agent, that is one turn.
	BudgetScopeRun BudgetScope = "run"
	// BudgetScopeSession counts the agent session's own usage across all its
	// turns and invocations, as the session state's Usage records it when the
	// run starts, plus every call the run makes, subagents included. It needs a
	// server-managed agent (one with a session store): client-managed state
	// comes back from the caller, who could drop its usage. Outside such an
	// agent, New fails.
	BudgetScopeSession BudgetScope = "session"
)

// Budget caps the usage a generate run, or an agent session, may spend.
//
// Limit caps each of its non-zero fields, [ai.GenerationUsage.Custom] key by
// key: InputTokens and OutputTokens, but also ThoughtsTokens, OutputImages,
// or a provider-reported cost such as Custom["cost"]. Usage counts as models
// report it, under the convention [ai.GenerationUsage] documents, so
// OutputTokens leaves out ThoughtsTokens: cap both to cap what a model
// generates. A model that reports no usage spends nothing.
//
// The limit is checked before each model call of the tool loop. Once a capped
// field reaches its limit, the run stops with [ai.ErrBudgetExceeded], which
// reports it as aborted rather than failed and keeps its partial response.
// A turn already under way finishes, so a run can overshoot by one turn.
// Because the check sits outside every WrapModel hook, a retry or fallback
// middleware never sees the stop, whatever the order in [ai.WithUse].
//
// With Interrupt set, a run-scoped budget pauses instead: the tool calls
// that would lead to the next turn are held, for the caller to decide whether
// to continue. [BudgetInterrupted] claims each held call, [BudgetExceeded] is
// what it carries, and a restart continues the run with the full limit again:
//
//	for _, part := range resp.Interrupts() {
//		if call, ok := middlewarex.BudgetInterrupted(part); ok {
//			parts = append(parts, call.Restart(nil))
//		}
//	}
//
// To stop instead, do not resume. A session-scoped budget cannot pause, since
// a later turn would start over its limit with nothing to pause on; New
// rejects the combination.
//
// A run-scoped budget counts one run across its resumes. The run's final
// message records what it spent, and a resume of that conversation continues
// from there, so answering another middleware's interrupt, such as a
// ToolApproval hold, does not renew the limit; only answering the budget's own
// hold does. A conversation the caller sends back without that record, which
// a client-managed history can do, starts the count over.
type Budget struct {
	// Limit caps each of its non-zero fields. At least one must be set.
	Limit ai.GenerationUsage `json:"limit" jsonschema_description:"Usage the run or session may spend. Each non-zero field is a cap, custom key by key."`
	// Scope is what the limit counts against. Defaults to [BudgetScopeRun].
	Scope BudgetScope `json:"scope,omitempty" jsonschema:"enum=run,enum=session" jsonschema_description:"What the limit counts against: one generate run, or the agent session. Defaults to run."`
	// Interrupt pauses a run-scoped budget for approval instead of stopping
	// the run.
	Interrupt bool `json:"interrupt,omitempty" jsonschema_description:"Pause the run for approval instead of stopping it once a limit is reached. Run scope only."`
}

// BudgetExceeded is the data a [Budget] hold carries; read it with
// [ai.InterruptAs] on the part [BudgetInterrupted] claims.
type BudgetExceeded struct {
	// Message says which limit the run reached.
	Message string `json:"message"`
	// Spent and Limit are what the run spent and its limit, for each capped
	// field of [Budget.Limit] by its JSON name ("totalTokens",
	// "custom.cost").
	Spent map[string]float64 `json:"spent"`
	Limit map[string]float64 `json:"limit"`
}

// BudgetInterrupted claims part for [Budget]: it reports whether part is a
// call a budget held and, when it is, returns the call. Restart it with nil to
// continue the run. Every Budget in a chain claims the same holds. See
// [ai.MiddlewareInterrupted].
func BudgetInterrupted(part *ai.Part) (*ai.InterruptedCall[any, any, map[string]any], bool) {
	return ai.MiddlewareInterrupted[map[string]any](Budget{}.Name(), part)
}

// budgetRecord is what a run-scoped budget writes on the run's final message,
// under its name, for a resume to continue from.
type budgetRecord struct {
	Spent *ai.GenerationUsage `json:"spent,omitempty"`
}

// Name implements [ai.Middleware].
func (b Budget) Name() string { return provider + "/budget" }

// JSONSchemaExtend describes the fields of Limit in the config schema. They
// come from a generated type, which carries no schema descriptions.
func (Budget) JSONSchemaExtend(s *jsonschema.Schema) {
	limit, ok := s.Properties.Get("limit")
	if !ok || limit.Properties == nil {
		return
	}
	for p := limit.Properties.Oldest(); p != nil; p = p.Next() {
		if p.Value.Description != "" {
			continue
		}
		if p.Key == "custom" {
			p.Value.Description = "Caps on provider-specific usage, such as cost, key by key."
			continue
		}
		p.Value.Description = "Cap on " + p.Key + ". Zero leaves it uncapped."
	}
}

// New implements [ai.Middleware]. Each Generate call gets its own count.
func (b Budget) New(ctx context.Context) (*ai.Hooks, error) {
	// JSON, which usageFields reads the limit through, cannot encode a NaN or
	// an infinity, so those are caught first.
	for k, v := range b.Limit.Custom {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, status.Errorf(status.ErrInvalidArgument, "budget: limit custom.%s must be a finite number", k)
		}
	}
	limit := usageFields(&b.Limit)
	for k, v := range limit {
		if v < 0 {
			return nil, status.Errorf(status.ErrInvalidArgument, "budget: limit %s must not be negative", k)
		}
		if v == 0 {
			delete(limit, k)
		}
	}
	if len(limit) == 0 {
		return nil, status.Errorf(status.ErrInvalidArgument, "budget: set at least one field of Limit")
	}
	s := &budgetState{limit: limit}
	hooks := &ai.Hooks{WrapGenerate: s.wrapGenerate}
	switch b.Scope {
	case "", BudgetScopeRun:
	case BudgetScopeSession:
		if b.Interrupt {
			return nil, status.Errorf(status.ErrInvalidArgument, "budget: a session budget cannot interrupt")
		}
		u, ok := aix.SessionUsageFromContext(ctx)
		if !ok {
			return nil, status.Errorf(status.ErrFailedPrecondition, "budget: session scope needs an agent with a session store")
		}
		s.session = u
	default:
		return nil, status.Errorf(status.ErrInvalidArgument, "budget: unknown scope %q", b.Scope)
	}
	if b.Interrupt {
		hooks.WrapTool = s.wrapTool
	}
	return hooks, nil
}

// budgetState is one Generate call's count. The observer and WrapTool run
// concurrently, hence the lock.
type budgetState struct {
	limit map[string]float64
	// session is the agent session's usage when the run started; nil for
	// run scope.
	session *ai.GenerationUsage

	mu sync.Mutex
	// tree counts every model call under the run, subagents included.
	tree *ai.GenerationUsage
}

// wrapGenerate counts the run's calls through an observer and checks the
// limit before each turn. Each turn's hook wraps the turns after it, so the
// first turn's sees the run's final response; only that one adds the observer
// and, for run scope, continues a resumed run's count and records the count
// on the final message.
func (s *budgetState) wrapGenerate(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
	first := params.Iteration == 0
	if first {
		ctx = tracing.WithInstrumentation(ctx, aix.Observer{ModelDone: s.count})
		if s.session == nil {
			s.resume(params)
		}
	}
	// A resume's first turn runs the restarted calls before any model call,
	// so the check waits for the turn after it; with Interrupt set, the
	// WrapTool hook holds those calls if the run is over its limit.
	resuming := first && params.Options != nil && params.Options.Resume != nil
	if msg, over := s.exceeded(); over && !resuming {
		logger.Debug(ctx, "run stopped by budget", "reason", msg)
		return nil, status.Errorf(ai.ErrBudgetExceeded, "budget: %s", msg)
	}
	resp, err := next(ctx, params)
	if first && s.session == nil && resp != nil && resp.Message != nil {
		s.mu.Lock()
		rec := budgetRecord{Spent: s.tree}
		s.mu.Unlock()
		// A copy, so the record never lands on a message the caller holds.
		msg := *resp.Message
		msg.Metadata = maps.Clone(msg.Metadata)
		if msg.Metadata == nil {
			msg.Metadata = map[string]any{}
		}
		msg.Metadata[Budget{}.Name()] = rec
		resp.Message = &msg
	}
	return resp, err
}

// resume continues the count of the run params resumes, from the record on
// its last message, unless the resume answers one of the budget's holds,
// which renews the limit. A call that is not a resume starts from zero.
func (s *budgetState) resume(params *ai.GenerateParams) {
	opts, msgs := params.Options, params.Request.Messages
	if opts == nil || opts.Resume == nil || len(msgs) == 0 {
		return
	}
	last := msgs[len(msgs)-1]
	if last.Role != ai.RoleModel {
		return
	}
	answers := slices.Concat(opts.Resume.Restart, opts.Resume.Respond)
	for _, p := range last.Content {
		if _, ok := BudgetInterrupted(p); ok && slices.ContainsFunc(answers, func(a *ai.Part) bool { return answersRequest(a, p.ToolRequest) }) {
			return
		}
	}
	rec, ok := base.ConvertTo[budgetRecord](last.Metadata[Budget{}.Name()])
	if !ok || rec.Spent == nil {
		return
	}
	s.mu.Lock()
	s.tree = ai.SumUsage(s.tree, rec.Spent)
	s.mu.Unlock()
}

// answersRequest reports whether a, a restart or a response directive,
// answers req, matched on tool name and ref as the generate loop matches it.
func answersRequest(a *ai.Part, req *ai.ToolRequest) bool {
	switch {
	case a == nil:
		return false
	case a.ToolRequest != nil:
		return a.ToolRequest.Name == req.Name && a.ToolRequest.Ref == req.Ref
	case a.ToolResponse != nil:
		return a.ToolResponse.Name == req.Name && a.ToolResponse.Ref == req.Ref
	}
	return false
}

// count adds a model call's usage to the run's count.
func (s *budgetState) count(_ context.Context, call *aix.ModelCall) {
	if u := call.Response.Usage; u != nil {
		s.mu.Lock()
		s.tree = ai.SumUsage(s.tree, u)
		s.mu.Unlock()
	}
}

// wrapTool holds the call once the run reached its limit. A restart answering
// a later stage, such as the tool's own question, passes a call this hook
// already let through.
func (s *budgetState) wrapTool(ctx context.Context, params *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
	msg, over := s.exceeded()
	if !over || tool.Released(ctx) {
		return next(ctx, params)
	}
	logger.Debug(ctx, "tool held by budget", "tool", params.Tool.Name(), "reason", msg)
	spent := s.spent()
	used := make(map[string]float64, len(s.limit))
	for k := range s.limit {
		used[k] = spent[k]
	}
	return nil, tool.Interrupt(ctx, BudgetExceeded{
		Message: "Usage budget reached: " + msg,
		Spent:   used,
		Limit:   maps.Clone(s.limit),
	})
}

// spent returns what counts against the budget, flattened by usageFields.
func (s *budgetState) spent() map[string]float64 {
	s.mu.Lock()
	spent := usageFields(s.tree)
	s.mu.Unlock()
	for k, v := range usageFields(s.session) {
		spent[k] += v
	}
	return spent
}

// exceeded reports whether a capped field reached its limit, and which.
func (s *budgetState) exceeded() (string, bool) {
	spent := s.spent()
	for _, k := range slices.Sorted(maps.Keys(s.limit)) {
		if spent[k] >= s.limit[k] {
			return fmt.Sprintf("used %v of %v %s", spent[k], s.limit[k], k), true
		}
	}
	return "", false
}

// usageFields flattens u into its numeric fields by JSON name, with Custom
// keys as "custom.<key>", so a field added to [ai.GenerationUsage] is capped
// with no change here. A nil u flattens to an empty map.
func usageFields(u *ai.GenerationUsage) map[string]float64 {
	out := map[string]float64{}
	if u == nil {
		return out
	}
	b, err := json.Marshal(u)
	if err != nil {
		return out
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return out
	}
	for k, v := range m {
		switch v := v.(type) {
		case float64:
			out[k] = v
		case map[string]any:
			for ck, cv := range v {
				if f, ok := cv.(float64); ok {
					out[k+"."+ck] = f
				}
			}
		}
	}
	return out
}
