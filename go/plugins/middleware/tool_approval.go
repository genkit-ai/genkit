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
)

// ApprovalRequest is the interrupt payload [ToolApproval] emits when a tool
// call requires approval. Read it from the interrupted part with
// [tool.InterruptData], then resume with [Approval]:
//
//	req, ok := tool.InterruptData[middleware.ApprovalRequest](interruptPart)
type ApprovalRequest struct {
	// Tool is the name of the tool awaiting approval.
	Tool string `json:"tool"`
	// Message is a human-readable description of the request.
	Message string `json:"message"`
}

// Approval is the resume payload recognized by [ToolApproval]. Embed it in
// your own resume struct to combine approval with data for the tool:
//
//	type myResume struct {
//		middleware.Approval
//		Note string `json:"note"`
//	}
//
// The matching is by JSON shape, not Go type: any resume payload carrying a
// "toolApproved" key (e.g. map[string]any{"toolApproved": true} from a
// generic or cross-runtime caller) works the same.
type Approval struct {
	ToolApproved bool `json:"toolApproved"`
}

// ToolApproval is a middleware that interrupts tool execution unless the tool
// is in [AllowedTools] or the call has been explicitly approved on resume.
//
// To approve on resume, pass [Approval] as the resume data:
//
//	restart, err := tool.Restart(interruptPart, ai.WithResume(middleware.Approval{ToolApproved: true}))
//
// A bare resumption alone is NOT treated as approval; callers must opt in so
// that unrelated resume flows (e.g. respond-only turns) cannot bypass
// approval.
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
//	// Resume with ai.WithToolRestarts carrying {"toolApproved": true} to re-execute.
type ToolApproval struct {
	// AllowedTools is the list of tool names pre-approved to run without
	// interruption. Tools not in this list trigger an interrupt. An empty
	// list interrupts all tools.
	AllowedTools []string `json:"allowedTools,omitempty"`
}

func (t *ToolApproval) Name() string { return provider + "/toolApproval" }

func (t *ToolApproval) New(ctx context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{
		WrapTool: t.wrapTool,
	}, nil
}

func (t *ToolApproval) wrapTool(ctx context.Context, params *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
	name := params.Tool.Name()
	if slices.Contains(t.AllowedTools, name) {
		return next(ctx, params)
	}

	if resumed, ok := tool.ResumeData[Approval](ctx); ok && resumed.ToolApproved {
		return next(ctx, params)
	}

	return nil, tool.Interrupt(ApprovalRequest{
		Tool:    name,
		Message: "Tool not in approved list: " + name,
	})
}
