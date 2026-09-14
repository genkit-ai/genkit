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

//go:build unix

// Skills tests that need a named pipe. syscall.Mkfifo does not exist on
// Windows, so these live behind a build constraint rather than a runtime skip:
// an undefined symbol fails the build before any skip can run.

package middleware

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// Opening a named pipe blocks until a writer appears, so the mode is checked
// before the open rather than after.
func TestSkillsSkipsFifoSkillMd(t *testing.T) {
	skillsDir := filepath.Join(t.TempDir(), "skills")
	fifoDir := filepath.Join(skillsDir, "fifo")
	if err := os.MkdirAll(fifoDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mkfifo(filepath.Join(fifoDir, "SKILL.md"), 0o644); err != nil {
		t.Skipf("cannot create a FIFO here: %v", err)
	}

	mustNotBlock(t, "scanSkills", func() error {
		if info := scanSkills(ctx, []string{skillsDir}, true); len(info) != 0 {
			t.Errorf("scanned %v, want none", sortedNames(info))
		}
		return nil
	})

	// readSkillFile is the second line of defence, for a file swapped after
	// the scan.
	if err := mustNotBlock(t, "readSkillFile", func() error {
		_, err := readSkillFile(filepath.Join(fifoDir, "SKILL.md"))
		return err
	}); err == nil {
		t.Error("readSkillFile accepted a FIFO")
	}
}

// The bundled-resource reader has the same exposure as the scan: os.Root
// confines the path but does not police the file type.
func TestSkillsResourceReadRefusesFifo(t *testing.T) {
	skillsDir := setupSkillsDir(t)
	py := filepath.Join(skillsDir, "python")
	if err := os.MkdirAll(filepath.Join(py, "scripts"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mkfifo(filepath.Join(py, "scripts", "pipe"), 0o644); err != nil {
		t.Skipf("cannot create a FIFO here: %v", err)
	}

	s := &Skills{SkillPaths: []string{skillsDir}, AllowResourceAccess: true}
	h := mustHooks(t, s)
	read := findTool(h, SkillResourceToolName)
	if read == nil {
		t.Fatal("read_skill_file was not registered")
	}
	r := newTestRegistry(t)
	read.Register(r)

	if err := mustNotBlock(t, "read_skill_file", func() error {
		_, err := read.RunRaw(ctx, map[string]any{"skillName": "python", "filePath": "scripts/pipe"})
		return err
	}); err == nil {
		t.Error("read_skill_file accepted a FIFO")
	}

	// The listing must not offer it either.
	if got := listSkillResources(ctx, py, SkillResourceToolName); strings.Contains(got, "pipe") {
		t.Errorf("the resource listing advertises a FIFO: %q", got)
	}
}

// mustNotBlock runs fn in a goroutine and fails the test if it does not return
// within the deadline. Opening a named pipe blocks forever, so a regression
// here must fail rather than hang the suite.
func mustNotBlock(t *testing.T, what string, fn func() error) error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- fn() }()
	select {
	case err := <-done:
		return err
	case <-time.After(10 * time.Second):
		t.Fatalf("%s blocked on a named pipe", what)
		return nil
	}
}
