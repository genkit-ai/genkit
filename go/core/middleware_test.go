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

package core

import (
	"context"
	"strings"
	"testing"
)

func TestMiddlewares(t *testing.T) {
	t.Run("creates slice of middlewares", func(t *testing.T) {
		m1 := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return next
		}
		m2 := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return next
		}

		result := Middlewares(m1, m2)

		if len(result) != 2 {
			t.Errorf("len(result) = %d, want 2", len(result))
		}
	})

	t.Run("returns empty slice when no middlewares", func(t *testing.T) {
		result := Middlewares[string, string, struct{}]()

		if len(result) != 0 {
			t.Errorf("len(result) = %d, want 0", len(result))
		}
	})

	t.Run("returns single middleware slice", func(t *testing.T) {
		m := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return next
		}

		result := Middlewares(m)

		if len(result) != 1 {
			t.Errorf("len(result) = %d, want 1", len(result))
		}
	})
}

func TestChainMiddleware(t *testing.T) {
	t.Run("empty chain returns identity", func(t *testing.T) {
		handler := func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
			return "original:" + input, nil
		}

		chained := ChainMiddleware[string, string, struct{}]()(handler)
		result, err := chained(context.Background(), "test", nil)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != "original:test" {
			t.Errorf("result = %q, want %q", result, "original:test")
		}
	})

	t.Run("single middleware is applied", func(t *testing.T) {
		handler := func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
			return "handler:" + input, nil
		}

		middleware := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
				result, err := next(ctx, "m1:"+input, cb)
				return "m1:" + result, err
			}
		}

		chained := ChainMiddleware(middleware)(handler)
		result, err := chained(context.Background(), "test", nil)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Expected: m1: wraps output, m1: prepends to input
		if result != "m1:handler:m1:test" {
			t.Errorf("result = %q, want %q", result, "m1:handler:m1:test")
		}
	})

	t.Run("multiple middlewares execute in order", func(t *testing.T) {
		var executionOrder []string

		handler := func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
			executionOrder = append(executionOrder, "handler")
			return input, nil
		}

		m1 := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
				executionOrder = append(executionOrder, "m1-before")
				result, err := next(ctx, input, cb)
				executionOrder = append(executionOrder, "m1-after")
				return result, err
			}
		}

		m2 := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
				executionOrder = append(executionOrder, "m2-before")
				result, err := next(ctx, input, cb)
				executionOrder = append(executionOrder, "m2-after")
				return result, err
			}
		}

		// ChainMiddleware(m1, m2) should execute as: m1 -> m2 -> handler -> m2 -> m1
		chained := ChainMiddleware(m1, m2)(handler)
		_, err := chained(context.Background(), "test", nil)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expected := []string{"m1-before", "m2-before", "handler", "m2-after", "m1-after"}
		if len(executionOrder) != len(expected) {
			t.Errorf("execution order length = %d, want %d", len(executionOrder), len(expected))
		}
		for i, step := range expected {
			if i >= len(executionOrder) || executionOrder[i] != step {
				t.Errorf("step %d = %q, want %q", i, executionOrder[i], step)
			}
		}
	})

	t.Run("middleware can modify input", func(t *testing.T) {
		handler := func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
			return input, nil
		}

		uppercase := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
				return next(ctx, strings.ToUpper(input), cb)
			}
		}

		chained := ChainMiddleware(uppercase)(handler)
		result, err := chained(context.Background(), "hello", nil)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != "HELLO" {
			t.Errorf("result = %q, want %q", result, "HELLO")
		}
	})

	t.Run("middleware can modify output", func(t *testing.T) {
		handler := func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
			return input, nil
		}

		addSuffix := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
				result, err := next(ctx, input, cb)
				return result + "!", err
			}
		}

		chained := ChainMiddleware(addSuffix)(handler)
		result, err := chained(context.Background(), "hello", nil)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != "hello!" {
			t.Errorf("result = %q, want %q", result, "hello!")
		}
	})

	t.Run("middleware can short-circuit", func(t *testing.T) {
		handlerCalled := false
		handler := func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
			handlerCalled = true
			return input, nil
		}

		shortCircuit := func(next StreamingFunc[string, string, struct{}]) StreamingFunc[string, string, struct{}] {
			return func(ctx context.Context, input string, cb func(context.Context, struct{}) error) (string, error) {
				if input == "skip" {
					return "skipped", nil
				}
				return next(ctx, input, cb)
			}
		}

		chained := ChainMiddleware(shortCircuit)(handler)
		result, err := chained(context.Background(), "skip", nil)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if handlerCalled {
			t.Error("handler should not have been called")
		}
		if result != "skipped" {
			t.Errorf("result = %q, want %q", result, "skipped")
		}
	})
}
