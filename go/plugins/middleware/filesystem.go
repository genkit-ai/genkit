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

package middleware

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io/fs"
	"mime"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
)

// readMaxBytes caps a single full read or returned slice. Models can step
// past it on the read side via offset/limit, but the resulting slice still
// has to fit. fileMaxBytes is the absolute on-disk file-size ceiling: even
// a sliced read materializes the whole file via root.ReadFile, so without
// this cap a 1 GB log could OOM the process before sliceLines runs.
const (
	readMaxBytes = 256 * 1024
	fileMaxBytes = 10 * 1024 * 1024
)

// fileStateCacheMaxEntries bounds the per-call cache. Eviction is FIFO on insert.
const fileStateCacheMaxEntries = 200

// fileUnchangedStub is returned by read_file when a prior read at the same range
// is still current. The earlier tool result is still in conversation history.
const fileUnchangedStub = "File unchanged since last read. The content from the earlier read_file result in this conversation is still current — refer to that instead of re-reading."

// fileState records the on-disk state observed at the last read or write of
// a given absolute path: mtime and byte size. Used to gate edits against
// external modifications and to dedup re-reads. Comparing both mtime and
// size narrows the window where a same-mtime overwrite slips through;
// filesystem mtime resolution can be as coarse as 1 s.
type fileState struct {
	ModTime time.Time
	Size    int64
	Offset  int // 0 when the read covered the whole file
	Limit   int // 0 when the read covered the whole file
}

func newFileState(info os.FileInfo, offset, limit int) *fileState {
	return &fileState{
		ModTime: info.ModTime(),
		Size:    info.Size(),
		Offset:  offset,
		Limit:   limit,
	}
}

// fileStateCache is a per-call, bounded path→state map. The middleware's New()
// allocates one per Hooks instance so cache lifetime equals call lifetime.
type fileStateCache struct {
	mu      sync.Mutex
	max     int
	entries map[string]*fileState
	order   []string
}

func newFileStateCache(max int) *fileStateCache {
	return &fileStateCache{max: max, entries: make(map[string]*fileState)}
}

func (c *fileStateCache) get(p string) *fileState {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.entries[p]
}

func (c *fileStateCache) set(p string, s *fileState) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.entries[p]; !ok {
		if len(c.entries) >= c.max && len(c.order) > 0 {
			delete(c.entries, c.order[0])
			c.order = c.order[1:]
		}
		c.order = append(c.order, p)
	}
	c.entries[p] = s
}

// pathLockBuckets caps the lock memory footprint at compile time. A fixed
// array of mutexes is sized once at construction; no map growth.
const pathLockBuckets = 256

// pathLocks serializes read-modify-write on the same path by hashing into a
// fixed bucket of mutexes. Two different paths that hash to the same bucket
// will spuriously serialize against each other — that's harmless because
// file ops are short and 256 buckets keep the collision rate low for any
// realistic per-call working set.
type pathLocks struct {
	buckets [pathLockBuckets]sync.Mutex
}

func newPathLocks() *pathLocks { return &pathLocks{} }

func (p *pathLocks) lock(path string) func() {
	// FNV-style multiplicative hash. Distribution is good for filesystem
	// paths (which differ most in their tail) and avoids pulling in
	// hash/fnv for one call site.
	var h uint32
	for i := 0; i < len(path); i++ {
		h = h*31 + uint32(path[i])
	}
	m := &p.buckets[h%pathLockBuckets]
	m.Lock()
	return m.Unlock
}

// Filesystem is a middleware that grants the LLM scoped file access under a
// single root directory. It registers list_files and read_file, plus
// write_file and edit_file when AllowWriteAccess is true.
//
// Path safety is enforced by [os.Root] (Go 1.25+), which rejects any path
// that resolves outside the root, including via "..", absolute paths, or
// symbolic links.
//
// Usage:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(m),
//	    ai.WithPrompt("summarise docs/ and save the summary to out.md"),
//	    ai.WithUse(&middleware.Filesystem{
//	        RootDir:          "./workspace",
//	        AllowWriteAccess: true,
//	    }),
//	)
type Filesystem struct {
	// RootDir is the directory that all operations are confined to.
	RootDir string `json:"rootDirectory,omitempty"`
	// AllowWriteAccess adds write_file and edit_file.
	AllowWriteAccess bool `json:"allowWriteAccess,omitempty"`
	// ToolNamePrefix is prepended to each tool name. Use distinct prefixes
	// when attaching multiple Filesystem middlewares to one call so their
	// tool names don't collide.
	ToolNamePrefix string `json:"toolNamePrefix,omitempty"`
}

func (f *Filesystem) Name() string { return provider + "/filesystem" }

// New initializes a per-call instance: opens the [os.Root], builds the tool
// set, and allocates the message queue used to bridge tool output back to the
// model on the next turn.
func (f *Filesystem) New(ctx context.Context) (*ai.Hooks, error) {
	if strings.TrimSpace(f.RootDir) == "" {
		return nil, core.NewError(core.INVALID_ARGUMENT, "filesystem middleware: RootDir is required")
	}
	abs, err := filepath.Abs(f.RootDir)
	if err != nil {
		return nil, core.NewError(core.INTERNAL, "filesystem middleware: resolve %q: %v", f.RootDir, err)
	}
	root, err := os.OpenRoot(abs)
	if err != nil {
		return nil, core.NewError(core.FAILED_PRECONDITION, "filesystem middleware: open root %q: %v", abs, err)
	}

	var (
		mu    sync.Mutex
		queue []*ai.Message
	)
	enqueueParts := func(parts ...*ai.Part) {
		mu.Lock()
		defer mu.Unlock()
		if n := len(queue); n > 0 && queue[n-1].Role == ai.RoleUser {
			queue[n-1].Content = append(queue[n-1].Content, parts...)
			return
		}
		queue = append(queue, ai.NewUserMessage(parts...))
	}

	cache := newFileStateCache(fileStateCacheMaxEntries)
	locks := newPathLocks()
	tools := f.buildTools(root, abs, cache, locks, enqueueParts)
	toolSet := make(map[string]struct{}, len(tools))
	for _, t := range tools {
		toolSet[t.Name()] = struct{}{}
	}

	wrapGenerate := func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		mu.Lock()
		queued := queue
		queue = nil
		mu.Unlock()

		if len(queued) > 0 {
			if params.Callback != nil {
				for _, msg := range queued {
					if err := params.Callback(ctx, &ai.ModelResponseChunk{
						Role:    msg.Role,
						Index:   params.MessageIndex,
						Content: msg.Content,
					}); err != nil {
						return nil, err
					}
					params.MessageIndex++
				}
			}
			params.Request.Messages = append(params.Request.Messages, queued...)
		}

		return next(ctx, params)
	}

	wrapTool := func(ctx context.Context, params *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
		if _, ours := toolSet[params.Tool.Name()]; !ours {
			return next(ctx, params)
		}

		resp, err := next(ctx, params)
		if err == nil {
			return resp, nil
		}
		if isInterrupt, _ := ai.IsToolInterruptError(err); isInterrupt {
			return nil, err
		}

		enqueueParts(ai.NewTextPart(fmt.Sprintf("Tool %q failed: %v", params.Tool.Name(), err)))
		return &ai.MultipartToolResponse{
			Output: "Tool call failed; see user message below for details.",
		}, nil
	}

	return &ai.Hooks{
		Tools:        tools,
		WrapGenerate: wrapGenerate,
		WrapTool:     wrapTool,
	}, nil
}

// toolName returns suffix prefixed with f.ToolNamePrefix.
func (f *Filesystem) toolName(suffix string) string { return f.ToolNamePrefix + suffix }

func (f *Filesystem) buildTools(
	root *os.Root,
	rootAbs string,
	cache *fileStateCache,
	locks *pathLocks,
	enqueueParts func(parts ...*ai.Part),
) []ai.Tool {
	tools := []ai.Tool{
		f.newListFilesTool(root),
		f.newReadFileTool(root, rootAbs, cache, locks, enqueueParts),
	}
	if f.AllowWriteAccess {
		tools = append(tools,
			f.newWriteFileTool(root, rootAbs, cache, locks),
			f.newEditFileTool(root, rootAbs, cache, locks),
		)
	}
	return tools
}

// cacheKey returns the absolute path used as the fileStateCache key.
func cacheKey(rootAbs, rel string) string {
	return filepath.Join(rootAbs, filepath.FromSlash(rel))
}

// normalizeRel canonicalises an LLM-supplied path into a slash-separated
// relative path. Leading separators are stripped so "/foo" is treated as
// relative to the root; os.Root enforces the final escape check.
func normalizeRel(p string) string {
	p = strings.TrimSpace(p)
	p = filepath.ToSlash(p)
	p = strings.TrimLeft(p, "/")
	if p == "" {
		return "."
	}
	return path.Clean(p)
}

// requireFilePath validates the shared filePath argument used by read_file,
// write_file, and edit_file.
func requireFilePath(s string) error {
	if strings.TrimSpace(s) == "" {
		return errors.New("filePath is required")
	}
	return nil
}

type listFilesInput struct {
	DirPath   string `json:"dirPath,omitempty" jsonschema:"description=Directory path relative to root. Defaults to the root directory."`
	Recursive bool   `json:"recursive,omitempty" jsonschema:"description=If true, descend into subdirectories."`
}

type fileEntry struct {
	Path        string `json:"path"`
	IsDirectory bool   `json:"isDirectory"`
	SizeBytes   int64  `json:"sizeBytes,omitempty"`
}

func (f *Filesystem) newListFilesTool(root *os.Root) ai.Tool {
	return ai.NewTool(
		f.toolName("list_files"),
		"Lists files and directories in a given path. Returns a list of entries with path and type.",
		func(_ *ai.ToolContext, in listFilesInput) ([]fileEntry, error) {
			dir := normalizeRel(in.DirPath)
			fsys := root.FS()

			if !in.Recursive {
				entries, err := fs.ReadDir(fsys, dir)
				if err != nil {
					return nil, err
				}
				out := make([]fileEntry, 0, len(entries))
				for _, e := range entries {
					fe := fileEntry{Path: e.Name(), IsDirectory: e.IsDir()}
					if !e.IsDir() {
						if info, err := e.Info(); err == nil {
							fe.SizeBytes = info.Size()
						}
					}
					out = append(out, fe)
				}
				return out, nil
			}

			var out []fileEntry
			err := fs.WalkDir(fsys, dir, func(p string, d fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if p == dir {
					return nil
				}
				rel, relErr := filepath.Rel(dir, p)
				if relErr != nil {
					rel = p
				}
				fe := fileEntry{
					Path:        filepath.ToSlash(rel),
					IsDirectory: d.IsDir(),
				}
				if !d.IsDir() {
					if info, err := d.Info(); err == nil {
						fe.SizeBytes = info.Size()
					}
				}
				out = append(out, fe)
				return nil
			})
			if err != nil {
				return nil, err
			}
			return out, nil
		},
	)
}

type readFileInput struct {
	FilePath string `json:"filePath" jsonschema:"description=File path relative to root."`
	Offset   int    `json:"offset,omitempty" jsonschema:"description=1-indexed line to start reading from. 0 or 1 means start at the beginning."`
	Limit    int    `json:"limit,omitempty" jsonschema:"description=Maximum number of lines to read. 0 means read to end of file."`
}

func (f *Filesystem) newReadFileTool(
	root *os.Root,
	rootAbs string,
	cache *fileStateCache,
	locks *pathLocks,
	enqueueParts func(parts ...*ai.Part),
) ai.Tool {
	return ai.NewTool(
		f.toolName("read_file"),
		"Reads the contents of a file. The actual contents are delivered as a user message on the next turn. Use offset and limit (1-indexed lines) for large files.",
		func(_ *ai.ToolContext, in readFileInput) (string, error) {
			if err := requireFilePath(in.FilePath); err != nil {
				return "", err
			}
			rel := normalizeRel(in.FilePath)
			mimeType := mime.TypeByExtension(strings.ToLower(path.Ext(rel)))
			osPath := filepath.FromSlash(rel)
			isImage := strings.HasPrefix(mimeType, "image/")
			key := cacheKey(rootAbs, rel)

			// Serialize the stat → dedup-check → read → cache-set sequence
			// against concurrent edit_file/write_file on the same path. genkit
			// runs tool calls in parallel goroutines, so an interleaved write
			// could otherwise let the read clobber the cache with stale state.
			unlock := locks.lock(key)
			defer unlock()

			info, err := root.Stat(osPath)
			if err != nil {
				return "", err
			}
			if info.IsDir() {
				return "", fmt.Errorf("path %s is a directory; use list_files", in.FilePath)
			}

			// Dedup: text re-reads of an unchanged file at the same range get a stub
			// instead of a fresh inject. Size is compared alongside mtime because
			// mtime resolution is coarse on some filesystems.
			if !isImage {
				if prev := cache.get(key); prev != nil &&
					prev.ModTime.Equal(info.ModTime()) &&
					prev.Size == info.Size() &&
					prev.Offset == in.Offset && prev.Limit == in.Limit {
					return fileUnchangedStub, nil
				}
			}

			fullSize := info.Size()
			if fullSize > fileMaxBytes {
				return "", fmt.Errorf(
					"file %s is %d bytes (>%d); too large to read until streaming line-slicing is supported",
					in.FilePath, fullSize, fileMaxBytes,
				)
			}
			if !isImage && in.Offset == 0 && in.Limit == 0 && fullSize > readMaxBytes {
				return "", fmt.Errorf(
					"file %s is %d bytes (>%d); use offset/limit to read a slice",
					in.FilePath, fullSize, readMaxBytes,
				)
			}

			data, err := root.ReadFile(osPath)
			if err != nil {
				return "", err
			}

			content := data
			if !isImage && (in.Offset > 0 || in.Limit > 0) {
				content = sliceLines(data, in.Offset, in.Limit)
				if len(content) > readMaxBytes {
					return "", fmt.Errorf(
						"requested slice of %s is %d bytes (>%d); narrow the range",
						in.FilePath, len(content), readMaxBytes,
					)
				}
			}

			if isImage {
				encoded := "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(content)
				enqueueParts(
					ai.NewTextPart(fmt.Sprintf("\n\nread_file result %s %s", mimeType, in.FilePath)),
					ai.NewMediaPart(mimeType, encoded),
				)
				return fmt.Sprintf("File %s read successfully, see contents below.", in.FilePath), nil
			}

			totalLines := countLines(data)
			shownLines := countLines(content)
			sliced := in.Offset > 0 || in.Limit > 0

			var header, summary string
			if sliced {
				startLine := in.Offset
				if startLine < 1 {
					startLine = 1
				}
				endLine := startLine + shownLines - 1
				if endLine < startLine {
					endLine = startLine
				}
				header = fmt.Sprintf(`<read_file path=%q lines="%d-%d" totalLines="%d">`,
					in.FilePath, startLine, endLine, totalLines)
				summary = fmt.Sprintf("File %s read successfully (lines %d-%d of %d total). See contents below.",
					in.FilePath, startLine, endLine, totalLines)
			} else {
				header = fmt.Sprintf(`<read_file path=%q totalLines="%d">`, in.FilePath, totalLines)
				summary = fmt.Sprintf("File %s read successfully (%d lines). See contents below.",
					in.FilePath, totalLines)
			}
			// No "\n" before the closing tag: when content already ends with a
			// newline, the file's own terminator separates it from </read_file>;
			// when content doesn't, we shouldn't fabricate one — the model could
			// pick it up and pass it into edit_file.oldString and miss the match.
			enqueueParts(ai.NewTextPart(header + "\n" + string(content) + "</read_file>"))
			cache.set(key, newFileState(info, in.Offset, in.Limit))
			return summary, nil
		},
	)
}

// countLines returns the number of lines in data. A trailing newline does not
// add a phantom empty line; an unterminated final line still counts.
func countLines(data []byte) int {
	if len(data) == 0 {
		return 0
	}
	n := bytes.Count(data, []byte{'\n'})
	if data[len(data)-1] != '\n' {
		n++
	}
	return n
}

// sliceLines returns the byte-range covering lines [offset, offset+limit)
// of data, 1-indexed. offset <= 1 starts at the beginning; limit <= 0 reads
// to end. Returns nil when offset is past EOF. The window is sliced directly
// out of data, so any line terminators present in the original are preserved
// — preserving byte-for-byte fidelity for downstream edit_file calls.
func sliceLines(data []byte, offset, limit int) []byte {
	if offset < 1 {
		offset = 1
	}
	start := 0
	for i := 1; i < offset; i++ {
		idx := bytes.IndexByte(data[start:], '\n')
		if idx == -1 {
			return nil
		}
		start += idx + 1
	}
	if start >= len(data) {
		return nil
	}
	if limit <= 0 {
		return data[start:]
	}
	end := start
	for i := 0; i < limit && end < len(data); i++ {
		idx := bytes.IndexByte(data[end:], '\n')
		if idx == -1 {
			end = len(data)
			break
		}
		end += idx + 1
	}
	return data[start:end]
}

type writeFileInput struct {
	FilePath string `json:"filePath" jsonschema:"description=File path relative to root."`
	Content  string `json:"content" jsonschema:"description=Content to write to the file."`
}

func (f *Filesystem) newWriteFileTool(
	root *os.Root,
	rootAbs string,
	cache *fileStateCache,
	locks *pathLocks,
) ai.Tool {
	return ai.NewTool(
		f.toolName("write_file"),
		"Writes content to a file, creating it (and any missing parent directories) or overwriting it if it exists. Overwriting an existing file requires a prior read_file in this session.",
		func(_ *ai.ToolContext, in writeFileInput) (string, error) {
			if err := requireFilePath(in.FilePath); err != nil {
				return "", err
			}
			rel := normalizeRel(in.FilePath)
			osPath := filepath.FromSlash(rel)
			key := cacheKey(rootAbs, rel)

			unlock := locks.lock(key)
			defer unlock()

			info, statErr := root.Stat(osPath)
			exists := statErr == nil
			if exists && info.IsDir() {
				return "", fmt.Errorf("path %s is a directory", in.FilePath)
			}
			if exists {
				prev := cache.get(key)
				if prev == nil {
					return "", fmt.Errorf("file %s exists but has not been read yet; read_file it first before overwriting", in.FilePath)
				}
				if !info.ModTime().Equal(prev.ModTime) || info.Size() != prev.Size {
					return "", fmt.Errorf("file %s has been modified since last read; re-read it before overwriting", in.FilePath)
				}
			}

			if parent := filepath.Dir(osPath); parent != "." {
				if err := root.MkdirAll(parent, 0o755); err != nil {
					return "", fmt.Errorf("create parent dirs: %w", err)
				}
			}
			if err := root.WriteFile(osPath, []byte(in.Content), 0o644); err != nil {
				return "", err
			}
			if newInfo, err := root.Stat(osPath); err == nil {
				cache.set(key, newFileState(newInfo, 0, 0))
			}
			return fmt.Sprintf("File %s written successfully.", in.FilePath), nil
		},
	)
}

type editSpec struct {
	OldString  string `json:"oldString" jsonschema:"description=The exact text to find. Match is byte-for-byte including whitespace and indentation."`
	NewString  string `json:"newString" jsonschema:"description=The replacement text. Use an empty string to delete oldString."`
	ReplaceAll bool   `json:"replaceAll,omitempty" jsonschema:"description=If true, replace every occurrence of oldString. If false (default), oldString must match exactly once in the file."`
}

type editFileInput struct {
	FilePath string     `json:"filePath" jsonschema:"description=File path relative to root."`
	Edits    []editSpec `json:"edits" jsonschema:"description=One or more edits to apply in order. Each edit is applied to the result of the previous edit."`
}

func (f *Filesystem) newEditFileTool(
	root *os.Root,
	rootAbs string,
	cache *fileStateCache,
	locks *pathLocks,
) ai.Tool {
	return ai.NewTool(
		f.toolName("edit_file"),
		"Applies one or more structured edits to a file. Requires a prior read_file in this session. Edits are applied sequentially; later edits see the changes from earlier ones. Each edit's oldString must match the file byte-for-byte (including whitespace), and by default must be unique — set replaceAll on the edit to rename across all occurrences.",
		func(_ *ai.ToolContext, in editFileInput) (string, error) {
			if err := requireFilePath(in.FilePath); err != nil {
				return "", err
			}
			if len(in.Edits) == 0 {
				return "", errors.New("edits is required")
			}
			rel := normalizeRel(in.FilePath)
			osPath := filepath.FromSlash(rel)
			key := cacheKey(rootAbs, rel)

			unlock := locks.lock(key)
			defer unlock()

			prev := cache.get(key)
			if prev == nil {
				return "", fmt.Errorf("file %s has not been read yet; read_file it first before editing", in.FilePath)
			}

			info, err := root.Stat(osPath)
			if err != nil {
				return "", err
			}
			if !info.ModTime().Equal(prev.ModTime) || info.Size() != prev.Size {
				return "", fmt.Errorf("file %s has been modified since last read; re-read it before editing", in.FilePath)
			}

			data, err := root.ReadFile(osPath)
			if err != nil {
				return "", err
			}
			content := string(data)

			for i, e := range in.Edits {
				next, err := applyEdit(content, e, in.FilePath)
				if err != nil {
					return "", fmt.Errorf("edit %d: %w", i, err)
				}
				content = next
			}

			if err := root.WriteFile(osPath, []byte(content), 0o644); err != nil {
				return "", err
			}
			if newInfo, err := root.Stat(osPath); err == nil {
				cache.set(key, newFileState(newInfo, 0, 0))
			}
			return fmt.Sprintf("Successfully applied %d edit(s) to %s.", len(in.Edits), in.FilePath), nil
		},
	)
}

// applyEdit replaces oldString with newString in content. With replaceAll,
// every occurrence is replaced; without it, oldString must match exactly once.
func applyEdit(content string, e editSpec, filePath string) (string, error) {
	if e.OldString == "" {
		return "", errors.New("oldString is required")
	}
	if e.OldString == e.NewString {
		return "", errors.New("oldString and newString are identical")
	}
	count := strings.Count(content, e.OldString)
	if count == 0 {
		return "", fmt.Errorf("oldString not found in %s; the match must be byte-for-byte, including whitespace and indentation", filePath)
	}
	if count > 1 && !e.ReplaceAll {
		return "", fmt.Errorf("oldString matches %d locations in %s; add more surrounding context to make it unique, or set replaceAll=true", count, filePath)
	}
	if e.ReplaceAll {
		return strings.ReplaceAll(content, e.OldString, e.NewString), nil
	}
	return strings.Replace(content, e.OldString, e.NewString, 1), nil
}
