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
	"context"
	"testing"
)

// TestConstructorsDoNotMutateOptions pins that a constructor owns its copy of
// the options it is handed. Plugins reuse one options value across a whole
// model table, and point every entry at one shared *ModelSupports, so a
// constructor that wrote back through either would change models it was never
// asked about.
func TestConstructorsDoNotMutateOptions(t *testing.T) {
	t.Run("model", func(t *testing.T) {
		supports := &ModelSupports{Multiturn: true, Output: []string{"text"}}
		opts := &ModelOptions{Supports: supports}
		NewModelAction("first", opts, func(context.Context, *ModelRequest, any, ModelStreamCallback) (*ModelResponse, error) {
			return nil, nil
		})
		if opts.Label != "" {
			t.Errorf("constructor wrote Label %q back into the caller's options", opts.Label)
		}
		if opts.Supports != supports {
			t.Error("constructor replaced the caller's Supports pointer")
		}
	})

	t.Run("embedder", func(t *testing.T) {
		opts := &EmbedderOptions{}
		NewEmbedderAction("first", opts, func(context.Context, *EmbedRequest, any) (*EmbedResponse, error) {
			return nil, nil
		})
		if opts.Label != "" || opts.Supports != nil {
			t.Errorf("constructor mutated the caller's options: %+v", opts)
		}
	})

	t.Run("retriever", func(t *testing.T) {
		opts := &RetrieverOptions{}
		NewRetrieverAction("first", opts, func(context.Context, *RetrieverRequest, any) (*RetrieverResponse, error) {
			return nil, nil
		})
		if opts.Label != "" || opts.Supports != nil {
			t.Errorf("constructor mutated the caller's options: %+v", opts)
		}
	})

	t.Run("background model", func(t *testing.T) {
		opts := &BackgroundModelOptions{}
		NewBackgroundModelAction("first", opts,
			func(context.Context, *ModelRequest, any) (*ModelOperation, error) { return nil, nil },
			func(context.Context, *ModelOperation) (*ModelOperation, error) { return nil, nil })
		if opts.Label != "" || opts.Supports != nil {
			t.Errorf("constructor mutated the caller's options: %+v", opts)
		}
	})

	// The capability struct a table shares must not be reachable through the
	// action either: two models built from one *ModelSupports get their own.
	t.Run("shared supports is copied", func(t *testing.T) {
		shared := &ModelSupports{Tools: true, Output: []string{"text"}}
		first := NewModelAction("first", &ModelOptions{Supports: shared}, nilModelFn)
		second := NewModelAction("second", &ModelOptions{Supports: shared}, nilModelFn)

		firstSupports := first.Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)
		secondSupports := second.Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)
		if firstSupports["tools"] != true || secondSupports["tools"] != true {
			t.Fatal("shared capabilities did not reach both models")
		}
		shared.Tools = false
		if firstSupports["tools"] != true {
			t.Error("changing the shared struct reached a model built earlier")
		}
	})
}

// TestConstructorsDefaultLabelToName pins that every primitive labels an
// unlabelled action with its name, whether or not options were supplied. The
// dev UI lists actions by label, so a blank one is a blank row.
func TestConstructorsDefaultLabelToName(t *testing.T) {
	label := func(desc map[string]any, key string) any {
		return desc[key].(map[string]any)["label"]
	}
	for _, tt := range []struct {
		name  string
		build func(string) map[string]any
		key   string
	}{
		{"model", func(n string) map[string]any {
			return NewModelAction(n, &ModelOptions{Versions: []string{"v1"}}, nilModelFn).Desc().Metadata
		}, "model"},
		{"embedder", func(n string) map[string]any {
			return NewEmbedderAction(n, &EmbedderOptions{Dimensions: 3}, func(context.Context, *EmbedRequest, any) (*EmbedResponse, error) {
				return nil, nil
			}).Desc().Metadata
		}, "info"},
		{"retriever", func(n string) map[string]any {
			return NewRetrieverAction(n, &RetrieverOptions{}, func(context.Context, *RetrieverRequest, any) (*RetrieverResponse, error) {
				return nil, nil
			}).Desc().Metadata
		}, "info"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := label(tt.build("acme/thing"), tt.key); got != "acme/thing" {
				t.Errorf("label = %v, want the action name; a non-nil options value must not lose the default", got)
			}
		})
	}
}

func nilModelFn(context.Context, *ModelRequest, any, ModelStreamCallback) (*ModelResponse, error) {
	return nil, nil
}
