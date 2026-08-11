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

package ai

import (
	"github.com/firebase/genkit/go/core/api"
)

// Test-local define helpers: New* + Register in one call, mirroring the
// removed registry-taking Define functions so tests stay concise.

func defineModel(r api.Registry, name string, opts *ModelOptions, fn ModelFunc) Model {
	m := NewModel(name, opts, fn)
	m.Register(r)
	return m
}

func defineEmbedder(r api.Registry, name string, opts *EmbedderOptions, fn EmbedderFunc) Embedder {
	e := NewEmbedder(name, opts, fn)
	e.Register(r)
	return e
}

func defineEvaluator(r api.Registry, name string, opts *EvaluatorOptions, fn EvaluatorFunc) Evaluator {
	e := NewEvaluator(name, opts, fn)
	e.Register(r)
	return e
}

func defineBatchEvaluator(r api.Registry, name string, opts *EvaluatorOptions, fn BatchEvaluatorFunc) Evaluator {
	e := NewBatchEvaluator(name, opts, fn)
	e.Register(r)
	return e
}

func defineTool[In, Out any](r api.Registry, name, description string, fn ToolFunc[In, Out], opts ...ToolOption) *ToolAction[In, Out] {
	t := NewTool(name, description, fn, opts...)
	t.Register(r)
	return t
}

func defineMultipartTool[In any](r api.Registry, name, description string, fn MultipartToolFunc[In], opts ...ToolOption) *ToolAction[In, *MultipartToolResponse] {
	t := NewMultipartTool(name, description, fn, opts...)
	t.Register(r)
	return t
}

func defineRetriever(r api.Registry, name string, opts *RetrieverOptions, fn RetrieverFunc) Retriever {
	ret := NewRetriever(name, opts, fn)
	ret.Register(r)
	return ret
}

func defineResource(r api.Registry, name string, opts *ResourceOptions, fn ResourceFunc) Resource {
	res := NewResource(name, opts, fn)
	res.Register(r)
	return res
}

func defineToolWithInputSchema[Out any](r api.Registry, name, description string, inputSchema map[string]any, fn ToolFunc[any, Out]) *ToolAction[any, Out] {
	return defineTool(r, name, description, fn, WithInputSchema(inputSchema))
}
