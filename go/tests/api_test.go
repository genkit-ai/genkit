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

package api_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/goccy/go-yaml"
)

type testFile struct {
	App   string
	Tests []test
}

type test struct {
	Path string
	Post any
	Body any
}

const hostPort = "http://localhost:3100"

func TestReflectionAPI(t *testing.T) {
	filenames, err := filepath.Glob(filepath.FromSlash("../../tests/specs/reflection_api.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, fn := range filenames {
		if filepath.Base(fn) == "pnpm-lock.yaml" {
			continue
		}
		data, err := os.ReadFile(fn)
		if err != nil {
			t.Fatal(err)
		}
		var file testFile
		if err := yaml.Unmarshal(data, &file); err != nil {
			t.Fatal(err)
		}
		t.Run(file.App, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			wait, err := startGenkitApp(ctx, file.App)
			if err != nil {
				t.Fatal(err)
			}
			t.Log("started app")
			defer func() {
				cancel()
				err := wait()
				t.Logf("wait returned %v", err)
			}()
			for {
				time.Sleep(50 * time.Millisecond)
				t.Log("checking...")
				if _, err := http.Get(hostPort + "/api/__health"); err == nil {
					break
				}
			}
			t.Log("app ready")
			for _, test := range file.Tests {
				runTest(t, test)
			}
		})
	}
}

func runTest(t *testing.T, test test) {
	t.Run(test.Path[1:], func(t *testing.T) {
		if t.Name() == "TestReflectionAPI/test_app/api/actions" {
			t.Skip("FIXME: skipping because Go and JS schemas are not aligned")
		}
		url := hostPort + test.Path
		var (
			res *http.Response
			err error
		)
		if test.Post != nil {
			body, err := json.Marshal(yamlToJSON(test.Post))
			if err != nil {
				t.Fatal(err)
			}
			res, err = http.Post(url, "application/json", bytes.NewReader(body))
		} else {
			res, err = http.Get(url)
		}
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			t.Fatalf("got status %s", res.Status)
		}
		var got any
		dec := json.NewDecoder(res.Body)
		dec.UseNumber()
		if err := dec.Decode(&got); err != nil {
			t.Fatal(err)
		}
		want := yamlToJSON(test.Body)
		msgs := compare(got, want)
		if len(msgs) > 0 {
			t.Logf("%s", prettyJSON(got))
			t.Fatal(strings.Join(msgs, "\n"))
		}
	})
}

func startGenkitApp(ctx context.Context, dir string) (func() error, error) {
	// If we invoke `go run`, the actual server is a child of the go command, and not
	// easy to kill. So instead we compile the app and run it directly.
	tmp := os.TempDir()
	cmd := exec.Command("go", "build", "-o", tmp, ".")
	cmd.Dir = dir
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("build: %w", err)
	}

	// `go build .` will use the working directory as the name of the executable.
	cmd = exec.CommandContext(ctx, "./"+dir)
	cmd.Dir = tmp
	cmd.Env = append(os.Environ(), "GENKIT_ENV=dev")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.WaitDelay = time.Second
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd.Wait, nil
}

func compare(got, want any) []string {
	var msgs []string
	compare1(nil, got, want, func(path []string, format string, args ...any) {
		msg := fmt.Sprintf(format, args...)
		prefix := strings.Join(path, ".")
		if prefix == "" {
			prefix = "top level"
		}
		msgs = append(msgs, fmt.Sprintf("%s: %s", prefix, msg))
	})
	return msgs
}

func compare1(path []string, got, want any, add func([]string, string, ...any)) {
	check := func() {
		if got != want {
			add(path, "\ngot  %v (%[1]T)\nwant %v (%[2]T)", got, want)
		}
	}

	switch w := want.(type) {
	case map[string]any:
		g, ok := got.(map[string]any)
		if !ok {
			add(path, "want map, got %T", got)
			return
		}
		for k, wv := range w {
			gv, ok := g[k]
			if !ok {
				add(path, "missing key %q", k)
			} else {
				compare1(append(path, k), gv, wv, add)
			}
		}
	case int:
		if n, ok := got.(json.Number); ok {
			g, err := n.Int64()
			if err != nil {
				add(path, "got number %s, want %d (%[2]T)", n, w)
			}
			if g != int64(w) {
				add(path, "got %d, want %d", g, w)
			}
		} else {
			check()
		}
	case uint64:
		if n, ok := got.(json.Number); ok {
			g, err := n.Int64()
			if err != nil {
				add(path, "got number %s, want %d (%[2]T)", n, w)
			}
			if uint64(g) != w {
				add(path, "got %d, want %d", g, w)
			}
		} else {
			check()
		}

	case float64:
		if n, ok := got.(json.Number); ok {
			g, err := n.Float64()
			if err != nil {
				add(path, "got number %s, want %f (%[2]T)", n, w)
			}
			if g != w {
				add(path, "got %f, want %f", g, w)
			}
		} else {
			check()
		}

	case []any:
		g, ok := got.([]any)
		if !ok {
			add(path, "want slice, got %T", got)
			return
		}
		if len(g) != len(w) {
			add(path, "got slice length %d, want %d", len(g), len(w))
			return
		}
		for i, ew := range w {
			compare1(append(path, strconv.Itoa(i)), g[i], ew, add)
		}
	default:
		check()
	}
}

// The yaml package unmarshals using different types than
// the json package. Convert the yaml form to the JSON form.
func yamlToJSON(y any) any {
	switch y := y.(type) {
	case map[any]any:
		j := map[string]any{}
		for k, v := range y {
			j[fmt.Sprint(k)] = yamlToJSON(v)
		}
		return j
	case []any:
		j := make([]any, len(y))
		for i, e := range y {
			j[i] = yamlToJSON(e)
		}
		return j
	default:
		return y
	}
}

func prettyJSON(x any) string {
	var sb strings.Builder
	enc := json.NewEncoder(&sb)
	enc.SetIndent("", "    ")
	if err := enc.Encode(x); err != nil {
		panic(err)
	}
	return sb.String()
}
