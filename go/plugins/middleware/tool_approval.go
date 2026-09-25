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

package middleware

import (
	"context"
	"slices"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/core/logger"
)

// ToolApproval is a middleware that interrupts tool execution unless the tool
// is in [AllowedTools] or the call has been explicitly approved on resume.
//
// To approve on resume, attach a "toolApproved" flag to the resume data of the
// restart part:
//
//	restart, err := interruptPart.ToToolRestart(map[string]any{"toolApproved": true})
//
// A bare restart, resumed with no payload, is NOT treated as approval; callers
// must opt in so that unrelated resume flows (e.g. respond-only turns) cannot
// bypass approval.
//
// The hold is the middleware's own interrupt, so a tool's
// [ai.ResumableToolAction.Interrupted] declines it and the approval is
// read here, never by the tool: once approved, the tool runs as a fresh call
// and may interrupt with a question of its own, which the caller answers
// through the tool as usual. That restart passes the gate, since the call
// records that the gate let it through, unless it replaces the call's input,
// which the gate then holds for approval again. A hold that does not record
// which stage raised it, such as one stored before stages were recorded, is
// approved the same way, with "toolApproved" on the restart.
//
// Usage:
//
//	resp, err := ai.Generate(ctx, r,
//	    ai.WithModel(m),
//	    ai.WithPrompt("do something"),
//	    ai.WithTools(toolA, toolB),
//	    ai.WithUse(&middleware.ToolApproval{AllowedTools: []string{"toolA"}}),
//	)
//	// toolA runs; toolB triggers an interrupt.
//	// Resume with ai.WithResume(restart), the restart carrying {"toolApproved": true}.
type ToolApproval struct {
	// AllowedTools is the list of tool names pre-approved to run without
	// interruption. Tools not in this list trigger an interrupt. An empty
	// list interrupts all tools.
	AllowedTools []string `json:"allowedTools,omitempty" jsonschema_description:"Tool names pre-approved to run without interruption. Any tool not in this list triggers an interrupt. An empty list interrupts every tool."`
}

// toolApprovalResume is the resume payload the middleware reads on a restart:
// the caller approves the held call by restarting it with toolApproved set.
type toolApprovalResume struct {
	ToolApproved bool `json:"toolApproved"`
}

// Name implements [ai.Middleware].
func (t ToolApproval) Name() string { return provider + "/toolApproval" }

// New implements [ai.Middleware], hooking tool execution.
func (t ToolApproval) New(ctx context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{
		WrapTool: t.wrapTool,
	}, nil
}

func (t *ToolApproval) wrapTool(ctx context.Context, params *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
	name := params.Tool.Name()
	if slices.Contains(t.AllowedTools, name) {
		return next(ctx, params)
	}

	// A restart answers the stage that raised the interrupt: one answering
	// a later stage, the tool's own question after this hook released the
	// call, passes through; one answering this hook carries the approval.
	if tool.Released(ctx) {
		return next(ctx, params)
	}
	if resume, ok := tool.ResumeData[toolApprovalResume](ctx); ok && resume.ToolApproved {
		return next(ctx, params)
	}

	// No span is emitted here: the generate engine attributes a hook that
	// short-circuits the tool to the tool itself in traces.
	logger.Debug(ctx, "tool held for approval", "tool", name)
	return nil, tool.Interrupt(ctx, map[string]any{
		"message": "Tool not in approved list: " + name,
	})
}
