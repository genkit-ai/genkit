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
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/goccy/go-yaml"
)

// Tool names registered by [Skills]. When [Skills.ToolNamePrefix] is set, the
// registered name is the prefix followed by the constant.
//
// Use them to name the tools in [ToolApproval.AllowedTools]: that field is
// matched by exact string, so a skills tool missing from the list is held for
// approval rather than run.
const (
	// SkillToolName loads a skill's instructions by name. The name matches the
	// JS and Python runtimes so prompts and evaluations port between them.
	SkillToolName = "use_skill"

	// SkillResourceToolName reads a file bundled inside a skill directory. It
	// is registered only when [Skills.AllowResourceAccess] is set.
	SkillResourceToolName = "read_skill_file"
)

// SkillActivationMetadataKey is the metadata key stamped on every message part
// that carries a skill's instructions, whether the model loaded them with
// [SkillToolName] or [Skills.Preload] injected them. Its value is the skill
// name as a string.
//
// [Skills] reads the key back out of the conversation to recognize a skill
// that is already loaded. Context-management middleware can use it to find
// skill instructions in a transcript and exempt them from summarization.
const SkillActivationMetadataKey = "skillActivation"

const (
	// agentsSkillsPath and defaultSkillsPath are the directories scanned when
	// Skills.SkillPaths is unset. ".agents/skills" is the cross-client interoperability convention;
	// "skills" is scanned last so it keeps winning collisions.
	agentsSkillsPath  = ".agents/skills"
	defaultSkillsPath = "skills"

	// skillFileName is matched case-exactly. A case-insensitive volume would
	// otherwise let "skill.md" load on macOS and Windows but not on Linux.
	skillFileName = "SKILL.md"

	// skillsMarker marks the catalog part injected by this middleware so a
	// later tool-loop iteration refreshes it instead of appending a second
	// copy. Its value is the middleware's activation tool name, which is what
	// tells two Skills instances on one call apart: they must set distinct
	// ToolNamePrefixes anyway, since duplicate tool names fail the request.
	skillsMarker = "skills-instructions"
)

const (
	// skillMaxBytes bounds a single SKILL.md. This is a process resource
	// limit, not an enforcement of the specification's authoring guidance on
	// SKILL.md length: an oversized file is skipped, never truncated.
	skillMaxBytes = 1 << 20

	// skillResourceMaxBytes bounds one read through SkillResourceToolName.
	skillResourceMaxBytes = 1 << 20

	// Advisory bounds from the specification. Exceeding one is a diagnostic;
	// the skill still loads.
	skillNameMaxRunes        = 64
	skillDescriptionMaxRunes = 1024

	// Bounds on the bundled-resource listing appended at activation.
	skillResourceListMax  = 100
	skillResourceMaxDepth = 4
)

// skillAlreadyLoadedStub answers a repeat activation.
const skillAlreadyLoadedStub = "Skill %q is already loaded in this conversation; its instructions are still in context. Refer to the earlier result instead of loading it again."

// Skills is a middleware that makes a local library of "skills" available to
// the model, following the Agent Skills specification (https://agentskills.io).
// A skill is a directory containing a SKILL.md file whose contents become
// specialized instructions the model can load on demand.
//
// Skills implements the specification's three tiers of progressive disclosure:
//
//   - Catalog. A system prompt lists each available skill's name and
//     description, and names the tool that loads one.
//   - Instructions. A use_skill tool returns a skill's full SKILL.md, wrapped
//     in <skill_content> together with the skill's directory. A skill already
//     loaded in the conversation is not sent twice.
//   - Resources. When AllowResourceAccess is set, each activation lists the
//     files the skill bundles, and a read_skill_file tool reads them on demand,
//     confined to that one skill directory.
//
// SKILL.md should start with a YAML frontmatter block carrying name and
// description. Following the specification's guidance for clients, a skill
// that violates the format is loaded anyway and reported through the logger,
// so a library authored for another agent still works here.
//
// Security: skill paths resolve against the process working directory by
// default, and a discovered SKILL.md becomes instruction text the model
// follows. Treat a skills tree the way you treat source code, and prefer an
// absolute path over the default in a server process. Genkit applies no trust
// gate of its own.
//
// Usage:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(m),
//	    ai.WithPrompt("use the python skill to compute ..."),
//	    ai.WithUse(&middleware.Skills{SkillPaths: []string{"skills"}}),
//	)
type Skills struct {
	// SkillPaths lists directories that are scanned for skills. Each direct
	// subdirectory containing a SKILL.md file is exposed as a skill; scanning
	// is one level deep and does not follow symbolic links. Relative paths
	// resolve against the process working directory.
	//
	// When two paths hold a skill of the same name the later path wins, and
	// the shadowing is logged.
	//
	// Defaults to []string{".agents/skills", "skills"}.
	SkillPaths []string `json:"skillPaths,omitempty" jsonschema_description:"Directories scanned for skills. Each direct subdirectory containing a SKILL.md file is exposed as a skill; scanning is one level deep. If two paths hold the same skill name, the later path wins. Defaults to the \".agents/skills\" and \"skills\" directories."`

	// Preload names skills whose instructions are injected before the first
	// model turn instead of waiting for the model to call use_skill. Use it
	// when the application, rather than the model, decides that a skill
	// applies.
	//
	// Preloaded skills are left out of the catalog's list of loadable skills.
	// A name matching no discovered skill is logged and ignored: skills are
	// rescanned on every request, so a temporarily unreadable directory must
	// not fail the request.
	Preload []string `json:"preload,omitempty" jsonschema_description:"Skills whose instructions are injected before the first model turn, without waiting for the model to load them. Names matching no discovered skill are logged and ignored."`

	// AllowResourceAccess registers read_skill_file, which reads files bundled
	// inside a skill directory (references/, scripts/, assets/, and anything
	// else the skill ships), and appends a listing of those files to each
	// activation. Paths resolve against that one skill's directory and are
	// confined to it by [os.Root].
	//
	// Defaults to false: enabling it grants the model file access it does not
	// otherwise have, and adds a tool name that [ToolApproval.AllowedTools]
	// must list.
	AllowResourceAccess bool `json:"allowResourceAccess,omitempty" jsonschema_description:"Adds the read_skill_file tool and lists each skill's bundled files at activation. Reads are confined to the individual skill directory. Defaults to false."`

	// ToolNamePrefix is prepended to each registered tool name. Use distinct
	// prefixes when attaching more than one Skills middleware to a call, or to
	// avoid colliding with a caller-supplied tool of the same name, since a
	// collision fails the whole request. A prefix breaks tool-name parity with
	// the other Genkit runtimes; leave it empty unless you need it.
	ToolNamePrefix string `json:"toolNamePrefix,omitempty" jsonschema_description:"Prepended to each tool name. Use distinct prefixes when attaching multiple skills middlewares to one call so their tool names do not collide."`
}

// skillInfo records a discovered skill. Name is the directory name, which is
// both what the catalog advertises and what the tools accept. Dir is the root
// that bundled-file references resolve against.
type skillInfo struct {
	Name        string
	Dir         string
	Path        string
	Description string
}

// skillFrontmatter mirrors the YAML block expected at the top of a SKILL.md.
// The specification's other fields (license, compatibility, metadata,
// allowed-tools) are deliberately not parsed: nothing consumes them, and the
// full file delivered at activation already carries them to the model.
type skillFrontmatter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

// Name implements [ai.Middleware].
func (s Skills) Name() string { return provider + "/skills" }

// New scans the configured skill paths and returns the [ai.Hooks] that inject
// the skills catalog and expose the skill tools. Scanning happens once per
// [ai.Generate] call, so WrapGenerate and the tools agree on one skill set and
// an edited SKILL.md takes effect on the next call.
//
// A skill tool with nothing to load is not registered, as the specification
// requires. Unreadable paths, malformed frontmatter, and oversized files are
// logged and skipped rather than reported as errors.
func (s Skills) New(ctx context.Context) (*ai.Hooks, error) {
	info := scanSkills(ctx, s.paths(), len(s.SkillPaths) > 0)
	if len(info) == 0 {
		return &ai.Hooks{}, nil
	}
	logger.Debug(ctx, "skills middleware scanned", "skills", sortedNames(info))

	preload := s.resolvePreload(ctx, info)

	act := &activationSet{loaded: map[string]bool{}}
	available := availableSkillsSentence(info)

	// Registering the activation tool with nothing left to activate would offer
	// the model a tool whose every input is a dead end, which happens when
	// Preload names every discovered skill.
	loadable := loadableNames(info, preload)
	catalog := s.buildSkillsPrompt(info, loadable)

	var tools []ai.Tool
	if len(loadable) > 0 {
		tools = append(tools, s.newUseSkillTool(info, act, available))
	}
	if s.AllowResourceAccess {
		tools = append(tools, s.newReadSkillFileTool(info, available))
	}
	toolSet := map[string]bool{}
	for _, t := range tools {
		toolSet[t.Name()] = true
	}

	wrapGenerate := func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		// Inject first, then read the activation set back out of the result:
		// a preloaded skill marks itself through the metadata on the part that
		// carries it, so one whose file could not be read stays loadable
		// instead of being reported as already present.
		params.Request = s.injectSkills(ctx, params.Request, catalog, info, preload)
		act.reset(params.Request.Messages)
		return next(ctx, params)
	}

	// Recover from a failed skill tool instead of aborting the generation, the
	// way the sibling Filesystem middleware does. This has to live in the hook
	// rather than in the handler: input decoding and schema validation run
	// inside the tool action, which the hook chain wraps, so a malformed call
	// never reaches a handler at all.
	wrapTool := func(ctx context.Context, params *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
		if !toolSet[params.Tool.Name()] {
			return next(ctx, params)
		}
		resp, err := next(ctx, params)
		if err == nil {
			return resp, nil
		}
		if isInterrupt, _ := ai.IsToolInterruptError(err); isInterrupt {
			return nil, err
		}
		logger.Debug(ctx, "skills tool failed, reporting to the model",
			"tool", params.Tool.Name(), "error", err)
		// The error text carries path components from disk, so it is folded
		// like any other on-disk string before the model reads it.
		return &ai.MultipartToolResponse{
			Output: fmt.Sprintf("Tool %q failed: %s. %s",
				params.Tool.Name(), catalogText(err.Error()), available),
		}, nil
	}

	return &ai.Hooks{
		Tools:        tools,
		WrapGenerate: wrapGenerate,
		WrapTool:     wrapTool,
	}, nil
}

// paths returns the directories to scan, falling back to the defaults.
func (s *Skills) paths() []string {
	if len(s.SkillPaths) == 0 {
		return []string{agentsSkillsPath, defaultSkillsPath}
	}
	return s.SkillPaths
}

// toolName returns suffix prefixed with s.ToolNamePrefix.
func (s *Skills) toolName(suffix string) string { return s.ToolNamePrefix + suffix }

// resolvePreload maps Skills.Preload onto the discovered skills. An unknown
// name is logged and dropped: New runs on the request path, so a name that
// fails to resolve because a directory was briefly unreadable must not fail
// the request.
func (s *Skills) resolvePreload(ctx context.Context, info map[string]skillInfo) map[string]bool {
	if len(s.Preload) == 0 {
		return nil
	}
	preload := make(map[string]bool, len(s.Preload))
	for _, name := range s.Preload {
		if _, ok := info[name]; !ok {
			logger.Warn(ctx, "preloaded skill not found, ignoring",
				"skill", name, "available", sortedNames(info))
			continue
		}
		preload[name] = true
	}
	return preload
}

// newUseSkillTool builds the activation tool. It returns the full SKILL.md,
// frontmatter included, wrapped so the model and any context-management
// middleware can tell skill instructions from the rest of the conversation.
//
// WithOutputSchema is required rather than cosmetic: a multipart tool does not
// infer an output schema from a type parameter the way [ai.NewTool] does, so
// without it the tool would advertise the multipart envelope where the model
// expects a string.
func (s *Skills) newUseSkillTool(info map[string]skillInfo, act *activationSet, available string) ai.Tool {
	return ai.NewMultipartTool(
		s.toolName(SkillToolName),
		"Load a skill's instructions by name.",
		func(_ *ai.ToolContext, in useSkillInput) (*ai.MultipartToolResponse, error) {
			si, ok := lookupSkill(info, in.SkillName)
			if !ok {
				return &ai.MultipartToolResponse{
					Output: unknownSkillMessage(in.SkillName, available),
				}, nil
			}
			claimed, release := act.claim(si.Name)
			if !claimed {
				return &ai.MultipartToolResponse{
					Output: fmt.Sprintf(skillAlreadyLoadedStub, si.Name),
				}, nil
			}
			content, err := s.renderSkill(si)
			if err != nil {
				release()
				return nil, err
			}
			return &ai.MultipartToolResponse{
				Output:   content,
				Metadata: map[string]any{SkillActivationMetadataKey: si.Name},
			}, nil
		},
		ai.WithOutputSchema(map[string]any{"type": "string"}),
	)
}

// renderSkill reads a skill and returns the instructions to place in the
// conversation. Both entry points go through it, so an activation and a
// preload deliver byte-identical content.
func (s *Skills) renderSkill(si skillInfo) (string, error) {
	data, err := readSkillFile(si.Path)
	if err != nil {
		return "", err
	}
	var resources string
	if s.AllowResourceAccess {
		resources = listSkillResources(si.Dir, s.toolName(SkillResourceToolName))
	}
	return wrapSkillContent(si, string(data), resources), nil
}

// activationSet is the per-call view of which skills are already loaded into
// the conversation.
//
// claim takes the whole check-and-set under one lock. A turn runs its tool
// calls concurrently, so a split check and mark would let two calls for the
// same skill both pass and both return the body. The returned release undoes
// the reservation, so a failed read leaves the skill loadable.
type activationSet struct {
	mu     sync.Mutex
	loaded map[string]bool
}

func (a *activationSet) claim(name string) (bool, func()) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.loaded[name] {
		return false, func() {}
	}
	a.loaded[name] = true
	return true, func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		delete(a.loaded, name)
	}
}

// reset rebuilds the set from a turn's messages, so it can never report a
// skill as loaded after context management has dropped the part carrying it.
func (a *activationSet) reset(msgs []*ai.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.loaded = activatedSkills(msgs)
}

// lookupSkill resolves a model-supplied skill name. The catalog prints each
// name folded to one line, so a name whose folded form differs from its
// directory name would otherwise be advertised in a spelling the tools reject.
func lookupSkill(info map[string]skillInfo, name string) (skillInfo, bool) {
	if si, ok := info[name]; ok {
		return si, true
	}
	var (
		match skillInfo
		found bool
	)
	for _, si := range info {
		if catalogText(si.Name) != name {
			continue
		}
		if found {
			// Two skills share a folded form. Guessing between them would be
			// worse than the unknown-skill reply, which lists both.
			return skillInfo{}, false
		}
		match, found = si, true
	}
	return match, found
}

// unknownSkillMessage reports a name that resolved to no skill. available is
// the sentence from [availableSkillsSentence], built once per request.
func unknownSkillMessage(name, available string) string {
	return fmt.Sprintf("Unknown skill %q. %s", catalogText(name), available)
}

// availableSkillsSentence lists every discovered skill for a model that named
// one that does not exist. Names are quoted so whitespace an author did not
// intend is visible, and folded so a name from disk cannot forge structure in
// the text the model reads.
func availableSkillsSentence(info map[string]skillInfo) string {
	quoted := make([]string, 0, len(info))
	for _, n := range sortedNames(info) {
		quoted = append(quoted, strconv.Quote(catalogText(n)))
	}
	return fmt.Sprintf("Available skills: %s.", strings.Join(quoted, ", "))
}

// useSkillInput is the input to the activation tool.
type useSkillInput struct {
	SkillName string `json:"skillName" jsonschema_description:"The name of the skill to use, exactly as listed in <skills>."`
}

// readSkillFileInput is the input to the bundled-resource reader.
type readSkillFileInput struct {
	SkillName string `json:"skillName" jsonschema_description:"Name of the skill that bundles the file, exactly as listed in <skills>."`
	FilePath  string `json:"filePath" jsonschema_description:"Path to the file, relative to the skill directory (for example \"references/api.md\")."`
}

// newReadSkillFileTool builds the tier-3 resource reader. Each call opens an
// [os.Root] on the one skill's directory, which rejects any path resolving
// outside it, including via "..", an absolute path, or a symbolic link.
func (s *Skills) newReadSkillFileTool(info map[string]skillInfo, available string) ai.Tool {
	return ai.NewTool(
		s.toolName(SkillResourceToolName),
		"Read a file bundled inside a skill directory, such as a reference document or a script.",
		func(_ *ai.ToolContext, in readSkillFileInput) (string, error) {
			si, ok := lookupSkill(info, in.SkillName)
			if !ok {
				return "", errors.New(unknownSkillMessage(in.SkillName, available))
			}
			if err := requireFilePath(in.FilePath); err != nil {
				return "", err
			}
			rel := normalizeRel(in.FilePath)
			data, err := readSkillResource(si.Dir, rel)
			if err != nil {
				return "", fmt.Errorf("read %q from skill %q: %w", rel, in.SkillName, err)
			}
			return string(data), nil
		},
	)
}

// readSkillResource reads one bundled file, confined to dir by [os.Root],
// which rejects any path resolving outside it including via "..", an absolute
// path, or a symbolic link.
//
// It differs from [readSkillFile] in one deliberate way: Stat follows symbolic
// links, so a link to a file elsewhere in the same skill works. That is safe
// here because os.Root keeps the target inside the skill directory, whereas a
// symlinked SKILL.md would name a file outside any skill at all.
func readSkillResource(dir, rel string) ([]byte, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, err
	}
	defer root.Close()

	name := filepath.FromSlash(rel)
	st, err := root.Stat(name)
	if err != nil {
		return nil, err
	}
	if err := checkReadable(st, rel, skillResourceMaxBytes); err != nil {
		return nil, err
	}
	f, err := root.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	data := make([]byte, st.Size())
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, err
	}
	return data, nil
}

// checkReadable is the rule both skill readers apply before opening anything:
// a regular file within the byte limit. The type check has to precede the open
// rather than follow it, because opening a named pipe blocks until a writer
// appears.
func checkReadable(st os.FileInfo, name string, maxBytes int64) error {
	switch {
	case st.IsDir():
		return fmt.Errorf("%s is a directory, not a file", name)
	case !st.Mode().IsRegular():
		return fmt.Errorf("%s is not a regular file (mode %s)", name, st.Mode().Type())
	case st.Size() > maxBytes:
		return fmt.Errorf("%s is %d bytes, over the %d byte limit", name, st.Size(), maxBytes)
	}
	return nil
}

// scanSkills enumerates SKILL.md files under each path and returns a map keyed
// by the skill's directory name. Missing or unreadable paths are skipped; a
// skipped path is a warning when the caller configured it explicitly (a likely
// misconfiguration) and debug noise when it is only the unset default.
//
// Diagnostics are split by consequence. A condition that removes a skill from
// the catalog, or silently changes which instructions the model receives, is a
// warning. Advisory lint that changes nothing is debug, so that a warning
// remains worth reading.
func scanSkills(ctx context.Context, paths []string, explicit bool) map[string]skillInfo {
	result := make(map[string]skillInfo)
	skipped := func(p string, err error) {
		if explicit {
			logger.Warn(ctx, "skills path could not be read, skipping", "path", p, "error", err)
		} else {
			logger.Debug(ctx, "skills path could not be read, skipping", "path", p, "error", err)
		}
	}
	for _, p := range paths {
		abs, err := filepath.Abs(p)
		if err != nil {
			skipped(p, err)
			continue
		}
		entries, err := os.ReadDir(abs)
		if err != nil {
			skipped(abs, err)
			continue
		}
		for _, entry := range entries {
			// IsDir is false for a symbolic link, so a linked skill directory
			// is not followed.
			if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
				continue
			}
			si, ok := readSkillDir(ctx, abs, entry.Name())
			if !ok {
				continue
			}
			if prev, dup := result[si.Name]; dup {
				logger.Warn(ctx, "skill name found in more than one path, the later path wins",
					"skill", si.Name, "shadowed", prev.Path, "using", si.Path)
			}
			result[si.Name] = si
		}
	}
	return result
}

// readSkillDir loads one candidate skill directory. It reports false when the
// directory holds no SKILL.md, or holds one that cannot be used.
func readSkillDir(ctx context.Context, parent, name string) (skillInfo, bool) {
	dir := filepath.Join(parent, name)
	entries, err := os.ReadDir(dir)
	if err != nil {
		logger.Debug(ctx, "skill directory could not be read, skipping", "path", dir, "error", err)
		return skillInfo{}, false
	}

	// Match SKILL.md case-exactly. Reading the joined path would instead
	// accept "skill.md" on a case-insensitive volume, so the same library
	// would discover a different set of skills on macOS and on Linux.
	//
	// The entry must also be a regular file. A symbolic link here would read a
	// file the skill author neither owns nor can write, and a named pipe would
	// block the request that opened it.
	var found os.DirEntry
	for _, e := range entries {
		if e.Name() != skillFileName {
			continue
		}
		if !e.Type().IsRegular() {
			logger.Debug(ctx, "SKILL.md is not a regular file, skipping skill",
				"path", filepath.Join(dir, skillFileName), "mode", e.Type())
			break
		}
		found = e
		break
	}
	if found == nil {
		warnNestedSkills(ctx, dir, entries)
		return skillInfo{}, false
	}

	skillMd := filepath.Join(dir, skillFileName)
	if fi, err := found.Info(); err == nil && fi.Size() > skillMaxBytes {
		logger.Warn(ctx, "SKILL.md is over the size limit, skipping skill",
			"path", skillMd, "bytes", fi.Size(), "limit", skillMaxBytes)
		return skillInfo{}, false
	}
	data, err := readSkillFile(skillMd)
	if err != nil {
		logger.Warn(ctx, "SKILL.md could not be read, skipping skill", "path", skillMd, "error", err)
		return skillInfo{}, false
	}

	fm, yamlErr := parseFrontmatter(data)
	desc := strings.TrimSpace(fm.Description)
	switch {
	case yamlErr != nil && desc == "" && fm.Name == "":
		logger.Warn(ctx, "SKILL.md frontmatter could not be parsed; the skill is listed by name only",
			"path", skillMd, "error", yamlErr)
	case yamlErr != nil:
		logger.Debug(ctx, "SKILL.md frontmatter is not valid YAML; fields were recovered by line scan",
			"path", skillMd, "error", yamlErr)
	}
	if desc == "" {
		// The specification's client guidance suggests skipping a skill with
		// no description, since the description is the entire signal the model
		// routes on. Genkit loads it anyway, to stay compatible with the other
		// runtimes and with plain-Markdown skill files, and reports it here.
		logger.Warn(ctx, "skill has no description and will be listed by name only", "path", skillMd)
	}

	validateSkillMetadata(ctx, name, fm, desc, skillMd)
	return skillInfo{Name: name, Dir: dir, Path: skillMd, Description: desc}, true
}

// validateSkillMetadata reports specification violations that do not stop the
// skill from loading. The directory name is what the catalog advertises and
// what the tools accept, so it is checked alongside the frontmatter name.
func validateSkillMetadata(ctx context.Context, dirName string, fm skillFrontmatter, desc, skillMd string) {
	if !validSkillName(dirName) {
		logger.Debug(ctx, "skill directory name does not meet the Agent Skills naming rules "+
			"(1-64 lowercase letters, digits and single hyphens); the skill is loaded anyway",
			"skill", dirName, "path", skillMd)
	}
	if fmName := strings.TrimSpace(fm.Name); fmName != "" && fmName != dirName {
		logger.Debug(ctx, "SKILL.md name does not match its directory name; the directory name is used",
			"name", fmName, "directory", dirName, "path", skillMd)
	}
	if n := utf8.RuneCountInString(desc); n > skillDescriptionMaxRunes {
		logger.Warn(ctx, "skill description is over the length the specification allows "+
			"and will be clipped in the skills catalog; the full text still loads with the skill",
			"skill", dirName, "runes", n, "limit", skillDescriptionMaxRunes, "path", skillMd)
	}
}

// warnNestedSkills reports a directory that holds no SKILL.md but does hold a
// subdirectory that does. Scanning is one level deep, so those skills are
// invisible; this is the diagnostic that explains why.
func warnNestedSkills(ctx context.Context, dir string, entries []os.DirEntry) {
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		if _, err := os.Stat(filepath.Join(dir, e.Name(), skillFileName)); err == nil {
			logger.Debug(ctx, "directory holds nested skills, which are not scanned; "+
				"add it to SkillPaths to expose them",
				"path", dir)
			return
		}
	}
}

// readSkillFile reads a SKILL.md, refusing anything over the size limit rather
// than truncating it.
//
// Lstat, not Stat: a symbolic link here would name a file the skill author
// neither owns nor can write. The scan applies the same rule, but a file can be
// replaced between the two.
func readSkillFile(p string) ([]byte, error) {
	fi, err := os.Lstat(p)
	if err != nil {
		return nil, err
	}
	if err := checkReadable(fi, p, skillMaxBytes); err != nil {
		return nil, err
	}
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	data := make([]byte, fi.Size())
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, err
	}
	return data, nil
}

// validSkillName reports whether name meets the Agent Skills naming rules:
// 1 to 64 runes of lowercase letters, digits, and hyphens, with no leading,
// trailing, or doubled hyphen.
func validSkillName(name string) bool {
	n := utf8.RuneCountInString(name)
	if n == 0 || n > skillNameMaxRunes {
		return false
	}
	if strings.HasPrefix(name, "-") || strings.HasSuffix(name, "-") || strings.Contains(name, "--") {
		return false
	}
	for _, r := range name {
		switch {
		case r == '-', unicode.IsDigit(r):
		case unicode.IsLetter(r) && !unicode.IsUpper(r) && !unicode.IsTitle(r):
		default:
			return false
		}
	}
	return true
}

// parseFrontmatter extracts the YAML frontmatter fenced by "---" lines at the
// top of a SKILL.md.
//
// The closing fence must be a line holding only "---", so a horizontal rule or
// a run of dashes inside a block scalar does not truncate the block. When the
// YAML does not parse, name and description are recovered by scanning those
// lines directly, which is what the JS runtime does and what keeps the common
// authoring mistake of an unquoted colon in a description from losing the
// whole block. The parse error is returned either way, for the caller to log.
func parseFrontmatter(content []byte) (skillFrontmatter, error) {
	var fm skillFrontmatter
	text := strings.TrimPrefix(string(content), "\ufeff") // strip optional BOM

	rest, ok := strings.CutPrefix(text, "---")
	if !ok {
		return fm, nil
	}
	// The opening fence runs to the end of its line; trailing spaces are
	// tolerated, as they are in the other runtimes.
	nl := strings.IndexByte(rest, '\n')
	if nl < 0 || strings.TrimRight(rest[:nl], " \t\r") != "" {
		return fm, nil
	}
	rest = rest[nl+1:]

	block, ok := cutFrontmatterBlock(rest)
	if !ok {
		return fm, nil
	}
	if err := yaml.Unmarshal([]byte(block), &fm); err != nil {
		return scanFrontmatterLines(block), err
	}
	return fm, nil
}

// cutFrontmatterBlock returns everything before the closing fence: the first
// line consisting only of "---" and optional trailing whitespace.
func cutFrontmatterBlock(s string) (string, bool) {
	for offset := 0; offset < len(s); {
		line := s[offset:]
		end := len(s)
		if nl := strings.IndexByte(line, '\n'); nl >= 0 {
			line = line[:nl]
			end = offset + nl
		}
		if strings.TrimRight(line, " \t\r") == "---" {
			return s[:offset], true
		}
		if end == len(s) {
			break
		}
		offset = end + 1
	}
	return "", false
}

// scanFrontmatterLines recovers name and description from frontmatter that is
// not valid YAML, taking each value verbatim to the end of its line.
//
// A value that is only a block-scalar header ("|", ">-", and so on) is treated
// as absent: the real value is on the lines below, which this scan cannot
// reassemble, and reporting the header as the description would both mislead
// the model and hide the missing-description diagnostic.
func scanFrontmatterLines(block string) skillFrontmatter {
	var fm skillFrontmatter
	value := func(v string) string {
		v = strings.TrimSpace(v)
		if isBlockScalarHeader(v) {
			return ""
		}
		return v
	}
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimRight(line, "\r")
		if v, ok := strings.CutPrefix(line, "name:"); ok && fm.Name == "" {
			fm.Name = value(v)
			continue
		}
		if v, ok := strings.CutPrefix(line, "description:"); ok && fm.Description == "" {
			fm.Description = value(v)
		}
	}
	return fm
}

// isBlockScalarHeader reports whether s is a YAML block-scalar indicator on its
// own: "|" or ">", with an optional indentation digit and chomping indicator in
// either order.
func isBlockScalarHeader(s string) bool {
	if s == "" || (s[0] != '|' && s[0] != '>') {
		return false
	}
	for _, r := range s[1:] {
		if r != '+' && r != '-' && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// buildSkillsPrompt renders the catalog listing the skills named in names,
// which [loadableNames] has already sorted and filtered: a preloaded skill is
// left out, since its instructions are in the request and offering it would
// only invite a wasted turn.
func (s *Skills) buildSkillsPrompt(info map[string]skillInfo, names []string) string {
	if len(names) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("<skills>\n")
	b.WriteString("You have access to a library of skills that serve as specialized instructions/personas.\n")
	b.WriteString("Strongly prefer to use them when working on anything related to them.\n")
	b.WriteString("Only use them once to load the context.\n")
	fmt.Fprintf(&b, "Call the %s tool with a skill's name to load its instructions.\n", s.toolName(SkillToolName))
	b.WriteString("Here are the available skills:\n")
	for _, name := range names {
		// Clip before escaping: the specification's bound is on the author's
		// text, and clipping afterwards could cut a generated entity in half.
		desc := catalogText(clipRunes(info[name].Description, skillDescriptionMaxRunes))
		if desc == "" {
			fmt.Fprintf(&b, " - %s\n", catalogText(name))
			continue
		}
		fmt.Fprintf(&b, " - %s - %s\n", catalogText(name), desc)
	}
	b.WriteString("</skills>")
	return b.String()
}

// clipRunes truncates s to max runes, marking that it was cut. The catalog is
// injected into every request, so an over-long description is bounded here
// rather than left to inflate each one; activation still delivers the file
// whole.
func clipRunes(s string, max int) string {
	if utf8.RuneCountInString(s) <= max {
		return s
	}
	runes := []rune(s)
	return strings.TrimRight(string(runes[:max]), " ") + "..."
}

// wrapSkillContent renders an activated skill. The full file is returned,
// frontmatter included: the specification leaves stripping optional, and the
// frontmatter is the only path by which fields the harness does not parse,
// such as compatibility, reach the model at all.
func wrapSkillContent(si skillInfo, body, resources string) string {
	// The attribute values are escaped by attrText, so they are quoted
	// explicitly rather than with %q, which would escape them a second time and
	// print a Windows path with doubled separators. The prose below is not an
	// attribute, so it takes the same escaper as every other line the model
	// reads as text.
	var b strings.Builder
	fmt.Fprintf(&b, "<skill_content name=\"%s\" path=\"%s\">\n", attrText(si.Name), attrText(si.Dir))
	b.WriteString(body)
	if !strings.HasSuffix(body, "\n") {
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "\nRelative paths in this skill resolve against %s\n", catalogText(si.Dir))
	b.WriteString(resources)
	b.WriteString("</skill_content>")
	return b.String()
}

// listSkillResources enumerates the files a skill bundles, so the model knows
// they exist without any of them being read. The optional directory names in
// the specification are conventions, not a closed set, so everything the skill
// ships is listed.
func listSkillResources(dir, toolName string) string {
	root := filepath.Clean(dir)
	var (
		files     []string
		truncated bool
	)
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil //nolint:nilerr // an unreadable entry is skipped, not fatal
		}
		rel, relErr := filepath.Rel(root, p)
		if relErr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if rel == "." {
			return nil
		}
		if strings.HasPrefix(d.Name(), ".") {
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			if strings.Count(rel, "/")+1 >= skillResourceMaxDepth {
				return fs.SkipDir
			}
			return nil
		}
		if rel == skillFileName {
			return nil
		}
		// Listing a path is an invitation to read it, so anything the reader
		// would refuse is left out: a named pipe, a socket, a device node.
		if !d.Type().IsRegular() {
			return nil
		}
		if len(files) >= skillResourceListMax {
			truncated = true
			return fs.SkipAll
		}
		files = append(files, rel)
		return nil
	})
	if len(files) == 0 {
		return ""
	}
	slices.Sort(files)

	var b strings.Builder
	b.WriteString("\n<skill_resources>\n")
	fmt.Fprintf(&b, "Paths are relative to the skill directory above; read them with the %s tool.\n", toolName)
	for _, f := range files {
		fmt.Fprintf(&b, " - %s\n", catalogText(f))
	}
	if truncated {
		fmt.Fprintf(&b, " (listing truncated at %d files)\n", skillResourceListMax)
	}
	b.WriteString("</skill_resources>\n")
	return b.String()
}

// activatedSkills returns the set of skills whose instructions are present in
// msgs. It is rebuilt from the conversation on every turn, so it never reports
// a skill as loaded once context management has dropped the part carrying it.
func activatedSkills(msgs []*ai.Message) map[string]bool {
	activated := map[string]bool{}
	for _, msg := range msgs {
		if msg == nil {
			continue
		}
		for _, part := range msg.Content {
			if part == nil || part.Metadata == nil {
				continue
			}
			if name, ok := part.Metadata[SkillActivationMetadataKey].(string); ok && name != "" {
				activated[name] = true
			}
		}
	}
	return activated
}

// injectSkills returns a copy of req carrying the skills catalog and the
// instructions of any preloaded skill. The catalog is marked by skillsMarker so
// a later tool-loop iteration refreshes it in place instead of appending a
// second copy; preloaded parts are recognized by their activation metadata.
func (s *Skills) injectSkills(ctx context.Context, req *ai.ModelRequest, catalog string, info map[string]skillInfo, preload map[string]bool) *ai.ModelRequest {
	newReq := *req
	newReq.Messages = append([]*ai.Message(nil), req.Messages...)

	if catalog != "" {
		s.injectCatalogPart(&newReq, catalog)
	}
	if len(preload) == 0 {
		return &newReq
	}

	present := activatedSkills(newReq.Messages)
	var parts []*ai.Part
	for _, name := range sortedNames(info) {
		if !preload[name] || present[name] {
			continue
		}
		content, err := s.renderSkill(info[name])
		if err != nil {
			// The skill stays loadable through the activation tool, so this
			// degrades preloading rather than losing the skill.
			logger.Warn(ctx, "preloaded SKILL.md could not be read, skipping the preload",
				"skill", name, "path", info[name].Path, "error", err)
			continue
		}
		p := ai.NewTextPart(content)
		p.Metadata = map[string]any{SkillActivationMetadataKey: name}
		parts = append(parts, p)
	}
	if len(parts) > 0 {
		appendToSystemMessage(&newReq, parts...)
	}
	return &newReq
}

// injectCatalogPart places catalog in this middleware's marked part, replacing
// an existing one in place, or adding a new part when there is none.
func (s *Skills) injectCatalogPart(req *ai.ModelRequest, catalog string) {
	marker := s.toolName(SkillToolName)
	for i, msg := range req.Messages {
		if msg == nil {
			continue
		}
		for j, part := range msg.Content {
			if part == nil || !part.IsText() || !ownsMarker(part.Metadata[skillsMarker], marker) {
				continue
			}
			if part.Text == catalog {
				return
			}
			msgCopy := msg.Clone()
			msgCopy.Content[j] = s.newSkillsPart(catalog)
			req.Messages[i] = msgCopy
			return
		}
	}
	appendToSystemMessage(req, s.newSkillsPart(catalog))
}

// appendToSystemMessage adds parts to the request's system message, creating
// one at the front of the conversation when there is none.
func appendToSystemMessage(req *ai.ModelRequest, parts ...*ai.Part) {
	for i, msg := range req.Messages {
		if msg == nil || msg.Role != ai.RoleSystem {
			continue
		}
		msgCopy := msg.Clone()
		msgCopy.Content = append(msgCopy.Content, parts...)
		req.Messages[i] = msgCopy
		return
	}
	req.Messages = append([]*ai.Message{ai.NewSystemMessage(parts...)}, req.Messages...)
}

// ownsMarker reports whether a skillsMarker metadata value belongs to the
// middleware whose activation tool is named marker.
//
// A bare true is accepted from any instance. That is what this middleware wrote
// before the value carried an instance identity, and what the JS and Python
// runtimes write today, so a conversation persisted by an older Go build or
// started in another runtime has its catalog refreshed rather than gaining a
// second, stale copy. Refreshing rewrites the value, so a history self-heals on
// its first turn. Two instances would both claim such a part, but two instances
// could not have produced one: before the identity existed they collided on the
// tool name and failed the request outright.
func ownsMarker(value any, marker string) bool {
	switch v := value.(type) {
	case string:
		return v == marker
	case bool:
		return v
	default:
		return false
	}
}

// newSkillsPart builds the text part that carries the skills catalog.
func (s *Skills) newSkillsPart(text string) *ai.Part {
	p := ai.NewTextPart(text)
	p.Metadata = map[string]any{skillsMarker: s.toolName(SkillToolName)}
	return p
}

// sortedNames returns the discovered skill names in a stable order.
func sortedNames(info map[string]skillInfo) []string {
	return slices.Sorted(maps.Keys(info))
}

// loadableNames returns the skills the model may activate: everything
// discovered, less anything already injected by Preload.
func loadableNames(info map[string]skillInfo, preload map[string]bool) []string {
	names := make([]string, 0, len(info))
	for _, name := range sortedNames(info) {
		if !preload[name] {
			names = append(names, name)
		}
	}
	return names
}

// catalogText prepares author-supplied text for a catalog line. A skill file
// is instruction text the model follows, so a description must not be able to
// forge catalog structure: newlines are folded away so it stays one line, and
// a tag-like "<" is escaped so it cannot close the block it sits in. Text that
// is not markup, such as "values < 10", is left alone.
func catalogText(s string) string {
	return escapeMarkup(strings.Join(strings.Fields(s), " "))
}

// escapeMarkup escapes each "<" that begins a tag, leaving other uses intact.
func escapeMarkup(s string) string {
	if !strings.Contains(s, "<") {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] != '<' {
			b.WriteByte(s[i])
			continue
		}
		j := i + 1
		if j < len(s) && s[j] == '/' {
			j++
		}
		if j < len(s) && isASCIILetter(s[j]) {
			b.WriteString("&lt;")
			continue
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

// attrReplacer escapes text placed in a quoted attribute of the markup this
// middleware emits. A [strings.Replacer] compiles its matcher once and is safe
// for concurrent use, so it is built at package scope rather than per call.
var attrReplacer = strings.NewReplacer(
	`&`, "&amp;",
	`<`, "&lt;",
	`>`, "&gt;",
	`"`, "&quot;",
	"\n", " ",
	"\r", " ",
)

// attrText escapes text placed in a quoted attribute.
func attrText(s string) string { return attrReplacer.Replace(s) }

func isASCIILetter(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}
