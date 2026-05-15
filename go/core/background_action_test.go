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
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

func TestNewBackgroundAction(t *testing.T) {
	t.Run("creates background action with all functions", func(t *testing.T) {
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "op-1", Done: false}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: true, Output: "result"}, nil
		}
		cancelFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: true}, nil
		}

		ba := NewBackgroundAction("test/background", api.ActionTypeCustom, nil, startFn, checkFn, cancelFn)

		if ba == nil {
			t.Fatal("NewBackgroundAction returned nil")
		}
		if ba.Name() != "test/background" {
			t.Errorf("Name() = %q, want %q", ba.Name(), "test/background")
		}
		if !ba.SupportsCancel() {
			t.Error("SupportsCancel() = false, want true")
		}
	})

	t.Run("creates background action without cancel", func(t *testing.T) {
		startFn := func(ctx context.Context, input int) (*Operation[int], error) {
			return &Operation[int]{ID: "op-1", Done: false}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[int]) (*Operation[int], error) {
			return &Operation[int]{ID: op.ID, Done: true, Output: 42}, nil
		}

		ba := NewBackgroundAction("test/nocancel", api.ActionTypeCustom, nil, startFn, checkFn, nil)

		if ba == nil {
			t.Fatal("NewBackgroundAction returned nil")
		}
		if ba.SupportsCancel() {
			t.Error("SupportsCancel() = true, want false")
		}
	})

	t.Run("panics with empty name", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for empty name")
			}
		}()

		NewBackgroundAction("", api.ActionTypeCustom, nil,
			func(ctx context.Context, input string) (*Operation[string], error) {
				return nil, nil
			},
			func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
				return nil, nil
			},
			nil,
		)
	})

	t.Run("panics with nil startFn", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for nil startFn")
			}
		}()

		NewBackgroundAction[string, string]("test/nilstart", api.ActionTypeCustom, nil,
			nil,
			func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
				return nil, nil
			},
			nil,
		)
	})

	t.Run("panics with nil checkFn", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for nil checkFn")
			}
		}()

		NewBackgroundAction("test/nilcheck", api.ActionTypeCustom, nil,
			func(ctx context.Context, input string) (*Operation[string], error) {
				return nil, nil
			},
			nil,
			nil,
		)
	})
}

func TestDefineBackgroundAction(t *testing.T) {
	t.Run("creates and registers background action", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "op-1", Done: false}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: true, Output: "done"}, nil
		}

		ba := DefineBackgroundAction(r, "test/registered", api.ActionTypeCustom, nil, startFn, checkFn, nil)

		if ba == nil {
			t.Fatal("DefineBackgroundAction returned nil")
		}

		// Verify action is registered
		key := api.KeyFromName(api.ActionTypeCustom, "test/registered")
		found := r.LookupAction(key)
		if found == nil {
			t.Error("background action not found in registry")
		}
	})
}

func TestBackgroundActionStart(t *testing.T) {
	t.Run("starts operation", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "test-op", Done: false, Metadata: map[string]any{"input": input}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: op.Done, Metadata: map[string]any{}}, nil
		}

		ba := DefineBackgroundAction(r, "test/start", api.ActionTypeCustom, nil, startFn, checkFn, nil)

		op, err := ba.Start(context.Background(), "hello")
		if err != nil {
			t.Fatalf("Start error: %v", err)
		}
		if op.ID != "test-op" {
			t.Errorf("op.ID = %q, want %q", op.ID, "test-op")
		}
		if op.Done {
			t.Error("op.Done = true, want false")
		}
		// Check that Action key is set
		if op.Action == "" {
			t.Error("op.Action is empty, expected to be set")
		}
	})
}

func TestBackgroundActionCheck(t *testing.T) {
	t.Run("checks operation status", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "check-op", Done: false, Metadata: map[string]any{}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: true, Output: "completed", Metadata: map[string]any{}}, nil
		}

		ba := DefineBackgroundAction(r, "test/check", api.ActionTypeCustom, nil, startFn, checkFn, nil)

		op, err := ba.Start(context.Background(), "input")
		if err != nil {
			t.Fatalf("Start error: %v", err)
		}

		checked, err := ba.Check(context.Background(), op)
		if err != nil {
			t.Fatalf("Check error: %v", err)
		}
		if !checked.Done {
			t.Error("checked.Done = false, want true")
		}
		if checked.Output != "completed" {
			t.Errorf("checked.Output = %q, want %q", checked.Output, "completed")
		}
	})
}

func TestBackgroundActionCancel(t *testing.T) {
	t.Run("cancels operation when supported", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "cancel-op", Done: false, Metadata: map[string]any{}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: op.Done, Metadata: map[string]any{}}, nil
		}
		cancelFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: true, Metadata: map[string]any{"cancelled": true}}, nil
		}

		ba := DefineBackgroundAction(r, "test/cancel", api.ActionTypeCustom, nil, startFn, checkFn, cancelFn)

		op, err := ba.Start(context.Background(), "input")
		if err != nil {
			t.Fatalf("Start error: %v", err)
		}

		cancelled, err := ba.Cancel(context.Background(), op)
		if err != nil {
			t.Fatalf("Cancel error: %v", err)
		}
		if !cancelled.Done {
			t.Error("cancelled.Done = false, want true")
		}
	})

	t.Run("returns error when cancel not supported", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "no-cancel-op", Done: false, Metadata: map[string]any{}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: op.Done, Metadata: map[string]any{}}, nil
		}

		ba := DefineBackgroundAction(r, "test/nocancel", api.ActionTypeCustom, nil, startFn, checkFn, nil)

		op, err := ba.Start(context.Background(), "input")
		if err != nil {
			t.Fatalf("Start error: %v", err)
		}

		_, err = ba.Cancel(context.Background(), op)
		if err == nil {
			t.Error("expected error for unsupported cancel, got nil")
		}
	})
}

func TestBackgroundActionRegister(t *testing.T) {
	t.Run("registers all sub-actions", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "reg-op", Metadata: map[string]any{}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Metadata: map[string]any{}}, nil
		}
		cancelFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Metadata: map[string]any{}}, nil
		}

		ba := NewBackgroundAction("test/register", api.ActionTypeCustom, nil, startFn, checkFn, cancelFn)
		ba.Register(r)

		// Check main action
		mainKey := api.KeyFromName(api.ActionTypeCustom, "test/register")
		if r.LookupAction(mainKey) == nil {
			t.Error("main action not registered")
		}

		// Check check action
		checkKey := api.KeyFromName(api.ActionTypeCheckOperation, "test/register")
		if r.LookupAction(checkKey) == nil {
			t.Error("check action not registered")
		}

		// Check cancel action
		cancelKey := api.KeyFromName(api.ActionTypeCancelOperation, "test/register")
		if r.LookupAction(cancelKey) == nil {
			t.Error("cancel action not registered")
		}
	})

	t.Run("registers without cancel action when not provided", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "reg-op", Metadata: map[string]any{}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Metadata: map[string]any{}}, nil
		}

		ba := NewBackgroundAction("test/register-nocancel", api.ActionTypeCustom, nil, startFn, checkFn, nil)
		ba.Register(r)

		// Cancel action should not be registered
		cancelKey := api.KeyFromName(api.ActionTypeCancelOperation, "test/register-nocancel")
		if r.LookupAction(cancelKey) != nil {
			t.Error("cancel action should not be registered")
		}
	})
}

func TestLookupBackgroundAction(t *testing.T) {
	t.Run("finds registered background action", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "lookup-op", Metadata: map[string]any{}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Metadata: map[string]any{}}, nil
		}

		DefineBackgroundAction(r, "test/lookup", api.ActionTypeCustom, nil, startFn, checkFn, nil)

		key := api.KeyFromName(api.ActionTypeCustom, "test/lookup")
		found := LookupBackgroundAction[string, string](r, key)

		if found == nil {
			t.Fatal("LookupBackgroundAction returned nil")
		}
		if found.Name() != "test/lookup" {
			t.Errorf("Name() = %q, want %q", found.Name(), "test/lookup")
		}
	})

	t.Run("returns nil for non-existent action", func(t *testing.T) {
		r := registry.New()

		key := api.KeyFromName(api.ActionTypeCustom, "test/nonexistent")
		found := LookupBackgroundAction[string, string](r, key)

		if found != nil {
			t.Errorf("LookupBackgroundAction returned %v, want nil", found)
		}
	})
}

func TestCheckOperation(t *testing.T) {
	t.Run("checks operation using registry lookup", func(t *testing.T) {
		r := registry.New()
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "check-op", Done: false, Metadata: map[string]any{}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Done: true, Output: "checked", Metadata: map[string]any{}}, nil
		}

		ba := DefineBackgroundAction(r, "test/checkop", api.ActionTypeCustom, nil, startFn, checkFn, nil)

		op, err := ba.Start(context.Background(), "input")
		if err != nil {
			t.Fatalf("Start error: %v", err)
		}

		checked, err := CheckOperation[string, string](context.Background(), r, op)
		if err != nil {
			t.Fatalf("CheckOperation error: %v", err)
		}
		if !checked.Done {
			t.Error("checked.Done = false, want true")
		}
		if checked.Output != "checked" {
			t.Errorf("checked.Output = %q, want %q", checked.Output, "checked")
		}
	})

	t.Run("returns error for nil operation", func(t *testing.T) {
		r := registry.New()

		_, err := CheckOperation[string, string](context.Background(), r, nil)
		if err == nil {
			t.Error("expected error for nil operation, got nil")
		}
	})

	t.Run("returns error for operation with empty Action", func(t *testing.T) {
		r := registry.New()
		op := &Operation[string]{ID: "op-1", Metadata: map[string]any{}}

		_, err := CheckOperation[string, string](context.Background(), r, op)
		if err == nil {
			t.Error("expected error for operation with empty Action, got nil")
		}
	})

	t.Run("returns error for non-existent action", func(t *testing.T) {
		r := registry.New()
		op := &Operation[string]{
			ID:       "op-1",
			Action:   api.KeyFromName(api.ActionTypeCustom, "test/nonexistent"),
			Metadata: map[string]any{},
		}

		_, err := CheckOperation[string, string](context.Background(), r, op)
		if err == nil {
			t.Error("expected error for non-existent action, got nil")
		}
	})
}

func TestBackgroundActionWithMetadata(t *testing.T) {
	t.Run("preserves metadata", func(t *testing.T) {
		r := registry.New()
		meta := map[string]any{
			"description": "A test background action",
			"version":     "1.0",
		}
		startFn := func(ctx context.Context, input string) (*Operation[string], error) {
			return &Operation[string]{ID: "meta-op", Metadata: map[string]any{}}, nil
		}
		checkFn := func(ctx context.Context, op *Operation[string]) (*Operation[string], error) {
			return &Operation[string]{ID: op.ID, Metadata: map[string]any{}}, nil
		}

		ba := DefineBackgroundAction(r, "test/meta", api.ActionTypeCustom, meta, startFn, checkFn, nil)

		desc := ba.Desc()
		if desc.Description != "A test background action" {
			t.Errorf("Description = %q, want %q", desc.Description, "A test background action")
		}
	})
}
