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

package base

// ToolFailError marks a tool error as a result for the model rather than a
// failure of the tool loop. Created by ai/exp/tool (Fail), read by
// ai/generate.go, which answers the tool request with the error message.
type ToolFailError struct {
	Err error
}

func (e *ToolFailError) Error() string { return e.Err.Error() }

func (e *ToolFailError) Unwrap() error { return e.Err }

// SoftToolErrors is the policy of one SoftToolErrors middleware
// (plugins/middleware): which tools return their own errors, and which
// unknown tool names return a not-found error, to the model. It links to the
// policy of the middleware outside it.
type SoftToolErrors struct {
	Covers func(toolName string) bool
	Outer  *SoftToolErrors
}

// Allows reports whether p or a policy outside it covers the named tool. A nil
// policy covers nothing.
func (p *SoftToolErrors) Allows(toolName string) bool {
	for ; p != nil; p = p.Outer {
		if p.Covers(toolName) {
			return true
		}
	}
	return false
}

// SoftToolErrorsKey is the context key for the call's [SoftToolErrors]. Set by
// the SoftToolErrors middleware (plugins/middleware), read by ai/generate.go.
var SoftToolErrorsKey = NewContextKey[*SoftToolErrors]()
