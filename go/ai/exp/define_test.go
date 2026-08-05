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
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
)

// Test-local define helpers: New* + Register in one call, mirroring the
// removed registry-taking ai.Define* helpers so tests stay concise.

func defineTestModel(r api.Registry, name string, opts *ai.ModelOptions, fn ai.ModelFunc) ai.Model {
	m := ai.NewModel(name, opts, fn)
	m.Register(r)
	return m
}

func defineTestTool[In, Out any](r api.Registry, name, description string, fn ai.ToolFunc[In, Out], opts ...ai.ToolOption) *ai.ToolAction[In, Out] {
	t := ai.NewTool(name, description, fn, opts...)
	t.Register(r)
	return t
}

func defineTestExpTool[In, Out any](r api.Registry, name, description string, fn ToolFunc[In, Out], opts ...ai.ToolOption) *Tool[In, Out] {
	t := NewTool(name, description, fn, opts...)
	t.Register(r)
	return t
}

func defineTestInterruptibleTool[In, Out, Res any](r api.Registry, name, description string, fn InterruptibleToolFunc[In, Out, Res], opts ...ai.ToolOption) *InterruptibleTool[In, Out, Res] {
	t := NewInterruptibleTool(name, description, fn, opts...)
	t.Register(r)
	return t
}
