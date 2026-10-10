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

package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"time"
)

// startResearcher starts the researcher service as a second process: this
// program again, run with -serve on a free local port. It returns the
// service's base URL once the service accepts connections, and a function
// that stops it.
//
// This is demo plumbing, not part of the remote agent API: in a real system
// the researcher is a service deployed on its own, and the orchestrator only
// knows its URL.
func startResearcher(ctx context.Context) (url string, stop func(), err error) {
	addr, err := freeAddr()
	if err != nil {
		return "", nil, err
	}
	exe, err := os.Executable()
	if err != nil {
		return "", nil, err
	}

	cmd := exec.Command(exe, "-serve", addr)
	// Production mode keeps one Dev UI runtime (the orchestrator) when the
	// sample runs under genkit start.
	cmd.Env = append(os.Environ(), "GENKIT_ENV=prod")
	// The service runs until its stdin closes. This process holds the write
	// end, so the service also stops if this process dies without calling
	// stop.
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return "", nil, err
	}
	out := prefixed("[researcher] ")
	cmd.Stdout, cmd.Stderr = out, out
	if err := cmd.Start(); err != nil {
		return "", nil, fmt.Errorf("start researcher service: %w", err)
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait(); out.Close() }()

	stop = func() {
		stdin.Close()
		select {
		case <-exited:
		case <-time.After(5 * time.Second):
			cmd.Process.Kill()
		}
	}
	if err := waitListening(ctx, addr, exited); err != nil {
		stop()
		return "", nil, err
	}
	return "http://" + addr, stop, nil
}

// freeAddr returns a local address with a port that was free a moment ago.
func freeAddr() (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	defer l.Close()
	return l.Addr().String(), nil
}

// waitListening waits until addr accepts connections, the service exits, or
// 30 seconds pass.
func waitListening(ctx context.Context, addr string, exited <-chan error) error {
	deadline := time.After(30 * time.Second)
	for {
		if conn, err := net.DialTimeout("tcp", addr, time.Second); err == nil {
			conn.Close()
			return nil
		}
		select {
		case err := <-exited:
			return fmt.Errorf("researcher service exited before it was ready: %v", err)
		case <-deadline:
			return fmt.Errorf("researcher service did not start listening on %s", addr)
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// prefixed returns a writer that copies each line written to it to stderr,
// prefixed with prefix. Close it to stop the copy.
func prefixed(prefix string) io.WriteCloser {
	r, w := io.Pipe()
	go func() {
		lines := bufio.NewScanner(r)
		for lines.Scan() {
			fmt.Fprintln(os.Stderr, prefix+lines.Text())
		}
	}()
	return w
}
