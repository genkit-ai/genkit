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
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

func TestRunInFlow(t *testing.T) {
	r := registry.New()
	n := 0
	stepf := func() (int, error) {
		n++
		return n, nil
	}

	flow := DefineFlow(r, "run", func(ctx context.Context, _ any) ([]int, error) {
		g1, err := Run(ctx, "s1", stepf)
		if err != nil {
			return nil, err
		}
		g2, err := Run(ctx, "s2", stepf)
		if err != nil {
			return nil, err
		}
		return []int{g1, g2}, nil
	})
	got, err := flow.Run(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 2}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestRunFlow(t *testing.T) {
	r := registry.New()
	f := DefineFlow(r, "inc", func(ctx context.Context, i int) (int, error) {
		return i + 1, nil
	})
	got, err := f.Run(context.Background(), 2)
	if err != nil {
		t.Fatal(err)
	}
	if want := 3; got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestFlowNameFromContext(t *testing.T) {
	r := registry.New()
	flows := []*Flow[struct{}, string, struct{}, struct{}]{
		DefineFlow(r, "DefineFlow", func(ctx context.Context, _ struct{}) (string, error) {
			return FlowNameFromContext(ctx), nil
		}),
		DefineStreamingFlow(r, "DefineStreamingFlow", func(ctx context.Context, _ struct{}, s StreamCallback[struct{}]) (string, error) {
			return FlowNameFromContext(ctx), nil
		}),
	}
	for _, flow := range flows {
		t.Run(flow.Name(), func(t *testing.T) {
			got, err := flow.Run(context.Background(), struct{}{})
			if err != nil {
				t.Fatal(err)
			}
			if want := flow.Name(); got != want {
				t.Errorf("got '%s', want '%s'", got, want)
			}
		})
	}
}

func TestRunOutsideFlow(t *testing.T) {
	t.Run("returns error when called outside flow", func(t *testing.T) {
		ctx := context.Background()
		_, err := Run(ctx, "step", func() (int, error) {
			return 42, nil
		})

		if err == nil {
			t.Error("expected error when Run called outside flow, got nil")
		}
	})
}

func TestFlowStream(t *testing.T) {
	t.Run("streams values correctly", func(t *testing.T) {
		r := registry.New()
		f := DefineStreamingFlow(r, "counter", func(ctx context.Context, n int, cb StreamCallback[int]) (int, error) {
			for i := 0; i < n; i++ {
				if err := cb(ctx, i); err != nil {
					return 0, err
				}
			}
			return n, nil
		})

		var streamedValues []int
		var finalOutput int
		var finalDone bool

		for v, err := range f.Stream(context.Background(), 3) {
			if err != nil {
				t.Fatalf("Stream error: %v", err)
			}
			if v.Done {
				finalDone = true
				finalOutput = v.Output
			} else {
				streamedValues = append(streamedValues, v.Stream)
			}
		}

		wantStreamed := []int{0, 1, 2}
		if !slices.Equal(streamedValues, wantStreamed) {
			t.Errorf("streamed values = %v, want %v", streamedValues, wantStreamed)
		}
		if !finalDone {
			t.Error("expected final Done value")
		}
		if finalOutput != 3 {
			t.Errorf("final output = %d, want 3", finalOutput)
		}
	})

	t.Run("yields error on flow failure", func(t *testing.T) {
		r := registry.New()
		f := DefineStreamingFlow(r, "failing", func(ctx context.Context, input int, cb StreamCallback[int]) (int, error) {
			return 0, NewError(INTERNAL, "flow failed")
		})

		var gotErr error
		for _, err := range f.Stream(context.Background(), 1) {
			if err != nil {
				gotErr = err
			}
		}

		if gotErr == nil {
			t.Error("expected error from failing flow, got nil")
		}
	})
}

func TestFlowRegister(t *testing.T) {
	t.Run("flow can be registered with registry", func(t *testing.T) {
		r := registry.New()
		f := DefineFlow(r, "test/registerable", func(ctx context.Context, input string) (string, error) {
			return input, nil
		})

		// Flow should already be registered by DefineFlow
		if f.Name() != "test/registerable" {
			t.Errorf("Name() = %q, want %q", f.Name(), "test/registerable")
		}
	})
}

func TestFlowDesc(t *testing.T) {
	t.Run("returns flow descriptor", func(t *testing.T) {
		r := registry.New()
		f := DefineFlow(r, "test/described", func(ctx context.Context, input struct {
			Name string `json:"name"`
		}) (struct {
			Greeting string `json:"greeting"`
		}, error) {
			return struct {
				Greeting string `json:"greeting"`
			}{Greeting: "Hello " + input.Name}, nil
		})

		desc := f.Desc()

		if desc.Name != "test/described" {
			t.Errorf("Name = %q, want %q", desc.Name, "test/described")
		}
		if desc.InputSchema == nil {
			t.Error("InputSchema is nil")
		}
		if desc.OutputSchema == nil {
			t.Error("OutputSchema is nil")
		}
	})
}

func TestFlowRunJSON(t *testing.T) {
	t.Run("runs flow with JSON input and output", func(t *testing.T) {
		r := registry.New()
		f := DefineFlow(r, "test/jsonFlow", func(ctx context.Context, input int) (int, error) {
			return input * 2, nil
		})

		got, err := f.RunJSON(context.Background(), []byte("5"), nil)
		if err != nil {
			t.Fatalf("RunJSON error: %v", err)
		}

		if string(got) != "10" {
			t.Errorf("RunJSON result = %s, want %q", got, "10")
		}
	})
}

func TestFlowRunJSONWithTelemetry(t *testing.T) {
	t.Run("returns telemetry info with result", func(t *testing.T) {
		r := registry.New()
		f := DefineFlow(r, "test/telemetryFlow", func(ctx context.Context, input int) (int, error) {
			return input + 1, nil
		})

		result, err := f.RunJSONWithTelemetry(context.Background(), []byte("5"), nil)
		if err != nil {
			t.Fatalf("RunJSONWithTelemetry error: %v", err)
		}

		if result == nil {
			t.Fatal("result is nil")
		}
		if string(result.Result) != "6" {
			t.Errorf("Result = %s, want %q", result.Result, "6")
		}
		if result.TraceId == "" {
			t.Error("TraceId is empty")
		}
		if result.SpanId == "" {
			t.Error("SpanId is empty")
		}
	})
}

func TestFlowNameFromContextOutsideFlow(t *testing.T) {
	t.Run("returns empty string outside flow", func(t *testing.T) {
		ctx := context.Background()
		got := FlowNameFromContext(ctx)
		if got != "" {
			t.Errorf("FlowNameFromContext outside flow = %q, want empty string", got)
		}
	})
}

func TestBidiActionEcho(t *testing.T) {
	ctx := context.Background()

	// In=struct{} (no initial data), Out=string, OutStream=string, InStream=string
	action := NewBidiAction(
		"echo", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			var count int
			for input := range inCh {
				count++
				outCh <- fmt.Sprintf("echo: %s", input)
			}
			return fmt.Sprintf("processed %d messages", count), nil
		},
	)

	conn, err := action.StreamBidi(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	// With unbuffered channels, we must send and receive concurrently.
	go func() {
		conn.Send("hello")
		conn.Send("world")
		conn.Close()
	}()

	var chunks []string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %v", len(chunks), chunks)
	}
	if chunks[0] != "echo: hello" {
		t.Errorf("expected 'echo: hello', got %q", chunks[0])
	}
	if chunks[1] != "echo: world" {
		t.Errorf("expected 'echo: world', got %q", chunks[1])
	}

	output, err := conn.Output()
	if err != nil {
		t.Fatal(err)
	}
	if output != "processed 2 messages" {
		t.Errorf("expected 'processed 2 messages', got %q", output)
	}
}

func TestBidiActionWithConfig(t *testing.T) {
	ctx := context.Background()

	type Config struct {
		Prefix string
	}

	// In=Config (initial config), Out=string, OutStream=string, InStream=string
	action := NewBidiAction(
		"prefixed", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			for input := range inCh {
				outCh <- fmt.Sprintf("%s: %s", cfg.Prefix, input)
			}
			return "done", nil
		},
	)

	conn, err := action.StreamBidi(ctx, Config{Prefix: "INFO"})
	if err != nil {
		t.Fatal(err)
	}

	go func() {
		conn.Send("test message")
		conn.Close()
	}()

	var chunks []string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 1 || chunks[0] != "INFO: test message" {
		t.Errorf("unexpected chunks: %v", chunks)
	}
}

func TestBidiConnectionSendAfterClose(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"test", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "", nil
		},
	)

	conn, err := action.StreamBidi(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	conn.Close()
	// Wait for completion so we know the state is settled.
	<-conn.Done()

	if err := conn.Send("after close"); err == nil {
		t.Error("expected error sending after close")
	}
}

func TestBidiConnectionContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	action := NewBidiAction(
		"blocking", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		},
	)

	conn, err := action.StreamBidi(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	cancel()

	_, err = conn.Output()
	if err == nil {
		t.Error("expected error after context cancellation")
	}
}

func TestBidiFlowRegistration(t *testing.T) {
	r := registry.New()

	flow := DefineBidiFlow(
		r, "echoFlow",
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for input := range inCh {
				outCh <- input
			}
			return "done", nil
		},
	)

	if flow.Name() != "echoFlow" {
		t.Errorf("expected name 'echoFlow', got %q", flow.Name())
	}

	desc := flow.Desc()
	if desc.Type != api.ActionTypeFlow {
		t.Errorf("expected type %q, got %q", api.ActionTypeFlow, desc.Type)
	}

	// Verify bidi metadata is set.
	if bidi, ok := desc.Metadata["bidi"].(bool); !ok || !bidi {
		t.Error("expected metadata[\"bidi\"] = true")
	}

	// Verify registered in registry.
	action := r.LookupAction(desc.Key)
	if action == nil {
		t.Error("expected action to be registered")
	}
}

func TestBidiFlowEcho(t *testing.T) {
	r := registry.New()
	ctx := context.Background()

	flow := DefineBidiFlow(
		r, "echoFlow",
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			var count int
			for input := range inCh {
				count++
				outCh <- fmt.Sprintf("echo: %s", input)
			}
			return fmt.Sprintf("processed %d", count), nil
		},
	)

	conn, err := flow.StreamBidi(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	go func() {
		conn.Send("a")
		conn.Send("b")
		conn.Close()
	}()

	var chunks []string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	output, err := conn.Output()
	if err != nil {
		t.Fatal(err)
	}
	if output != "processed 2" {
		t.Errorf("expected 'processed 2', got %q", output)
	}
}

func TestBidiFlowCoreRunWorks(t *testing.T) {
	r := registry.New()
	ctx := context.Background()

	flow := DefineBidiFlow(
		r, "withSteps",
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for input := range inCh {
				// core.Run should work inside a BidiFlow.
				result, err := Run(ctx, "uppercase", func() (string, error) {
					return strings.ToUpper(input), nil
				})
				if err != nil {
					return "", err
				}
				outCh <- result
			}
			return "done", nil
		},
	)

	conn, err := flow.StreamBidi(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	go func() {
		conn.Send("hello")
		conn.Close()
	}()

	var chunks []string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 1 || chunks[0] != "HELLO" {
		t.Errorf("expected [HELLO], got %v", chunks)
	}
}

func TestBidiActionDone(t *testing.T) {
	ctx := context.Background()

	action := NewBidiAction(
		"quick", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "finished", nil
		},
	)

	conn, err := action.StreamBidi(ctx, struct{}{})
	if err != nil {
		t.Fatal(err)
	}

	conn.Close()
	<-conn.Done()

	output, err := conn.Output()
	if err != nil {
		t.Fatal(err)
	}
	if output != "finished" {
		t.Errorf("expected 'finished', got %q", output)
	}
}
