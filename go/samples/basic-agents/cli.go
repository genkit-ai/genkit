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

// This file is the user-facing half of the sample: the interactive CLI
// that ties the three server-side agent definitions in main.go to a
// single experience. The CLI is deliberately small. It has three
// responsibilities:
//
//  1. Pick an agent ("thread list").
//  2. For the picked agent, pick between resuming from the latest
//     snapshot or starting fresh. If the latest snapshot is still
//     pending (a detached invocation is still processing in the
//     background), offer to wait for it, stop it, start fresh, or back
//     out.
//  3. Run a small REPL against the agent: stream the model's reply each
//     turn, accept text input, and offer /detach, /back, and /quit as
//     control commands.
//
// The detach demo is woven into step 3: typing "/detach <text>" sends
// the text as the final input and detaches, so the agent keeps
// processing in the background and the caller gets a pending snapshot
// ID. Re-picking the same agent in step 2 then surfaces the wait/resume
// flow.
//
// A turn that breaks or gets stopped is not a dead end here. Either one
// leaves a snapshot holding the conversation up to the last turn seam, so
// step 2 offers it like any other. Whether the turn it left unanswered is
// worth running again is the user's call: an empty line in step 3 runs a
// turn on the conversation as it stands, so nothing has to be retyped. The
// only rows the CLI cannot continue from are the ones the runtime refuses
// too, and for those it walks back to the snapshot before.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/status"
)

// ANSI styling for the small amount of tasteful color the CLI uses: cyan as
// the structural accent (prompt, headers, list markers, tool calls), and
// green/yellow/dim for outcomes and metadata. The user's own typed text is
// left in the terminal's default color. These are the named 16-color codes,
// not fixed RGB, so the user's terminal theme maps them to shades that suit
// its background. Kept inline and dependency-free; on a terminal without ANSI
// support these degrade to literal escape sequences, which is acceptable for
// a sample.
const (
	ansiReset  = "\033[0m"
	ansiBold   = "\033[1m"
	ansiDim    = "\033[2m"
	ansiCyan   = "\033[36m"
	ansiGreen  = "\033[32m"
	ansiYellow = "\033[33m"
)

// style wraps s in the given ANSI codes followed by a reset. With no codes it
// returns s unchanged, so it is always safe to call.
func style(s string, codes ...string) string {
	if len(codes) == 0 {
		return s
	}
	return strings.Join(codes, "") + s + ansiReset
}

// chevron is the accent-colored prompt drawn before every interactive read.
var chevron = style("> ", ansiBold, ansiCyan)

// promptLine prints prompt and reads one line. The prompt carries its own
// styling (e.g. the accent-colored chevron); the user's typed text is left in
// the terminal's default color. Routing every interactive read through it
// keeps prompting consistent across the menus and the chat REPL.
func promptLine(ctx context.Context, inputCh <-chan string, prompt string) (string, bool) {
	fmt.Print(prompt)
	return readLine(ctx, inputCh)
}

// stdoutIsTerminal reports whether stdout is an interactive terminal, computed
// once at startup. The spinner only animates when it is; piped or redirected
// output skips it so logs don't fill up with carriage-return frames.
var stdoutIsTerminal = func() bool {
	fi, err := os.Stdout.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}()

// spinner is a single-line "still working" indicator shown while waiting for
// the model to produce the first output of a generation. It animates in place
// with a carriage return and erases itself when stopped, so the reply (or tool
// call) prints on the line the spinner occupied.
type spinner struct {
	stopCh chan struct{}
	done   chan struct{}
}

// startSpinner draws label with an animated glyph and, after the first second,
// an elapsed-seconds counter, refreshing the same line ~10x/second on its own
// goroutine. When stdout is not a terminal it draws nothing. Always pair it
// with (*spinner).stop.
func startSpinner(label string) *spinner {
	s := &spinner{stopCh: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(s.done)
		if !stdoutIsTerminal {
			<-s.stopCh // nothing to draw, just wait to be released
			return
		}
		frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		start := time.Now()
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		draw := func(i int) {
			line := style(frames[i%len(frames)], ansiCyan) + " " + style(label, ansiDim)
			if secs := int(time.Since(start).Seconds()); secs >= 1 {
				line += " " + style(fmt.Sprintf("(%ds)", secs), ansiDim)
			}
			fmt.Printf("\r\033[K%s", line)
		}
		draw(0)
		for i := 1; ; i++ {
			select {
			case <-s.stopCh:
				fmt.Print("\r\033[K") // erase the spinner so content takes its line
				return
			case <-ticker.C:
				draw(i)
			}
		}
	}()
	return s
}

// stop ends the animation and erases the line, blocking until the goroutine
// has done so. The streamTurn caller guards against calling it more than once.
func (s *spinner) stop() {
	close(s.stopCh)
	<-s.done
}

// errQuit signals that the user typed /quit somewhere in the CLI; it
// bubbles up through openAgent and breaks runCLI's outer loop.
var errQuit = errors.New("quit")

// agentEntry is the CLI's view of one agent, with its session-state type
// erased. Each agent picks the state type that suits it (see barista.go for
// one that carries a structured order), so a single list cannot hold them as
// concrete values; it holds closures instead, and newEntry is the one place
// the concrete type is captured. Everything the closures call is generic over
// State, so nothing downstream loses the type.
//
// The CLI itself knows nothing about any particular agent: it lists them,
// streams their turns, and (when set) routes tool interrupts to onInterrupt.
// Agents without interruptible tools leave onInterrupt nil and the rest of the
// flow is identical. That split is what lets this same cli.go back any future
// sample, with tool interrupts or without.
type agentEntry struct {
	name        string
	description string
	// summarize renders the one-line status shown under the agent in the list.
	summarize func(ctx context.Context, sessionID string) string
	// open runs one visit to the agent, returning the session ID it ran under.
	open func(ctx context.Context, inputCh <-chan string, lastSessionID string) (string, error)
}

// agentHooks is what the streaming path needs about an agent beyond its
// connection: what to call it in messages, and how to resolve interrupts.
type agentHooks struct {
	name        string
	onInterrupt InterruptHandler
}

// newEntry builds a list entry for an agent of any session-state type.
func newEntry[State any](a *aix.Agent[State], onInterrupt InterruptHandler) agentEntry {
	hooks := agentHooks{name: a.Name(), onInterrupt: onInterrupt}
	return agentEntry{
		name:        a.Name(),
		description: a.Desc().Description,
		summarize: func(ctx context.Context, sessionID string) string {
			return summarizeLatest(ctx, a, sessionID)
		},
		open: func(ctx context.Context, inputCh <-chan string, lastSessionID string) (string, error) {
			return openAgent(ctx, inputCh, a, hooks, lastSessionID)
		},
	}
}

// InterruptHandler resolves a single tool interrupt into a resume part. It
// receives the interrupted tool-request part (read its typed payload with
// tool.InterruptAs) and a Prompter for asking the user questions through
// the CLI's input stream. It returns one of:
//
//   - a restart part (tool.Resume) to re-run the tool with resume data,
//   - a response part (tool.Respond) to answer the tool directly, or
//   - nil to leave this interrupt unresolved.
//
// The CLI sorts the returned part into the right half of aix.ToolResume,
// so handlers never touch the wire shape.
type InterruptHandler func(p *Prompter, interrupt *ai.Part) (*ai.Part, error)

// runCLI is the entry point for the interactive client. It alternates
// between two screens forever: the agent list and a per-agent chat.
// Returning from a chat brings the user back to the agent list. /quit
// (anywhere) and Ctrl-C both unwind back here and exit cleanly.
func runCLI(ctx context.Context, agents []agentEntry) error {
	fmt.Println(style("Genkit Basic Agents", ansiBold, ansiCyan))
	fmt.Println(style("===================", ansiDim))
	fmt.Println()
	fmt.Println("Pick an agent below, choose to resume the last conversation")
	fmt.Println("or start a new one, and chat. Inside a chat:")
	fmt.Println("  (text)             send a message and stream the reply")
	fmt.Println("  (empty line)       run a turn on the conversation as it")
	fmt.Println("                     stands, without adding a message")
	fmt.Println("  /detach (text...)  send the text (optional) as the final")
	fmt.Println("                     input, then detach. The agent finishes")
	fmt.Println("                     in the background and writes a pending")
	fmt.Println("                     snapshot. Re-pick the agent later to")
	fmt.Println("                     wait for it, or stop it and resume from")
	fmt.Println("                     the turns it finished.")
	fmt.Println("  /back              return to the agent list")
	fmt.Println("  /quit              exit the program")
	fmt.Println()
	fmt.Println("Some agents pause mid-turn to ask for input (a tool")
	fmt.Println("interrupt). When that happens, the CLI prompts you inline and")
	fmt.Println("resumes the turn with your answer.")
	fmt.Println()
	fmt.Println("A turn that breaks, or one you stop, still leaves a snapshot")
	fmt.Println("you can continue: re-pick the agent, resume from it, and press")
	fmt.Println("Enter to run the turn it left unanswered again.")

	// lastSession remembers, per agent, the session ID of the most recent
	// conversation this process ran with it. That is all a client needs to
	// resume: re-entering an agent resolves the session's latest snapshot
	// from the store (see SnapshotReader.GetLatestSnapshot). It lives in
	// memory, so a fresh run starts clean rather than rediscovering prior
	// conversations from the store.
	lastSession := map[string]string{}

	inputCh := readLines(ctx)
	for {
		choice, ok := pickAgent(ctx, inputCh, agents, lastSession)
		if !ok {
			return nil
		}
		entry := agents[choice]
		sessionID, err := entry.open(ctx, inputCh, lastSession[entry.name])
		if err != nil {
			if errors.Is(err, errQuit) {
				return nil
			}
			return err
		}
		lastSession[entry.name] = sessionID
	}
}

// pickAgent renders the agent list and reads the user's choice. The
// list is re-rendered between selections so the user can see updated
// pending/terminal status after returning from a chat.
func pickAgent(ctx context.Context, inputCh <-chan string, agents []agentEntry, lastSession map[string]string) (int, bool) {
	for {
		fmt.Println()
		fmt.Println(style("Agents:", ansiBold))
		for i, entry := range agents {
			fmt.Printf("  %s %s %s\n",
				style(fmt.Sprintf("%d.", i+1), ansiCyan),
				style(entry.name, ansiBold),
				style("— "+entry.description, ansiDim))
			if summary := entry.summarize(ctx, lastSession[entry.name]); summary != "" {
				fmt.Printf("       %s\n", summary)
			}
		}
		fmt.Printf("  %s %s\n", style("q.", ansiCyan), style("quit", ansiDim))
		fmt.Println()

		line, ok := promptLine(ctx, inputCh, chevron)
		if !ok {
			return -1, false
		}
		line = strings.TrimSpace(line)
		if line == "q" || line == "quit" || line == "exit" {
			return -1, false
		}
		idx, err := strconv.Atoi(line)
		if err != nil || idx < 1 || idx > len(agents) {
			fmt.Println("Invalid choice. Type a number from the list or 'q' to quit.")
			continue
		}
		return idx - 1, true
	}
}

// openAgent is the per-agent screen: it surfaces the latest snapshot
// (asking the user how to handle a still-pending detached invocation),
// asks whether to resume or start fresh, and then hands off to the
// chat REPL.
//
// The pending and non-pending paths return the same (resume, ok) shape,
// so the rest of the flow is uniform: ok=false means the user backed
// out, otherwise hand the chosen snapshot (or nil for fresh) to
// runChat.
func openAgent[State any](ctx context.Context, inputCh <-chan string, a *aix.Agent[State], hooks agentHooks, lastSessionID string) (string, error) {
	// Resolve where the last conversation left off. With no tracked
	// session (a first visit this run) there is nothing to resume;
	// otherwise the store resolves the session's latest snapshot, which
	// also surfaces a still-pending detached invocation.
	//
	// Reads go through the agent, not its store: the agent normalizes what a
	// client should see, which is what turns a detached worker that stopped
	// beating into an expired row instead of one stuck pending forever. It
	// reports a missing snapshot as an error where the raw store returns nil,
	// so an unknown session reads as "nothing to resume".
	var tip *aix.SessionSnapshot[State]
	if lastSessionID != "" {
		var err error
		tip, err = a.GetLatestSnapshot(ctx, lastSessionID)
		if err != nil && !errors.Is(err, aix.ErrSnapshotNotFound) {
			fmt.Fprintf(os.Stderr, "Read snapshot for %q: %v\n", a.Name(), err)
			return lastSessionID, nil
		}
	}

	var (
		resume *aix.SessionSnapshot[State]
		ok     bool
	)
	if tip != nil && tip.Status == aix.SnapshotStatusPending {
		// Background invocation still in flight. handlePending makes the
		// final decision itself (wait & resume, stop & resume, new, or back),
		// so we don't fall through to pickSession: the user already chose,
		// and asking again would just be noise. An expired row is not offered
		// the wait, since its worker is presumed dead; it goes to pickSession,
		// which walks back to the last snapshot that can be continued.
		resume, ok = handlePending(ctx, inputCh, a, tip)
	} else {
		resume, ok = pickSession(ctx, inputCh, a, tip)
	}
	if !ok {
		return lastSessionID, nil
	}
	return runChat(ctx, inputCh, a, hooks, resume, lastSessionID)
}

// handlePending offers the three reasonable responses when a previous
// invocation of this agent is still running in the background:
//
//  1. wait for it to finalize and resume from it directly,
//  2. stop it now and resume from the turns it finished,
//  3. ignore it and start a fresh conversation,
//  4. go back to the agent list.
//
// Returns the snapshot to resume from (options 1 and 2) or nil (option 3, or
// 1 and 2 when nothing in the chain can be continued). ok=false means the
// user chose 4 or the context was canceled.
//
// Crucially, options that imply "use this conversation" return the
// snapshot directly so the caller can skip the resume / new prompt:
// the user already committed to the choice by waiting, and re-asking
// would be redundant.
func handlePending[State any](ctx context.Context, inputCh <-chan string, a *aix.Agent[State], pending *aix.SessionSnapshot[State]) (*aix.SessionSnapshot[State], bool) {
	for {
		fmt.Printf("\nThe last %s session is still running in the background (%s).\n",
			style(a.Name(), ansiBold), style(shortID(pending.SnapshotID), ansiDim))
		fmt.Printf("  %s Wait for it to finalize\n", style("1.", ansiCyan))
		fmt.Printf("  %s Stop it now and resume from the turns it finished\n", style("2.", ansiCyan))
		fmt.Printf("  %s Start a new conversation\n", style("3.", ansiCyan))
		fmt.Printf("  %s Back to agent list\n", style("4.", ansiCyan))

		line, ok := promptLine(ctx, inputCh, chevron)
		if !ok {
			return nil, false
		}
		choice := strings.TrimSpace(line)
		switch choice {
		case "1", "2":
			if choice == "2" {
				// Abort flips the pending row, and the runtime observes the
				// flip and cancels the background work. The flip alone writes
				// no state: the invocation that owns the row stamps the turns
				// it finished onto it as it unwinds, which is what the wait
				// below is for. Stopping and waiting are therefore the same
				// flow, differing only in who ends the work.
				if _, err := a.Abort(ctx, pending.SnapshotID); err != nil {
					fmt.Fprintf(os.Stderr, "Stop error: %v\n", err)
					return nil, false
				}
				fmt.Println(style("Stopping it...", ansiYellow))
			} else {
				fmt.Println(style("Waiting for it to finalize...", ansiYellow))
			}
			final, err := waitForFinalize(ctx, a, pending.SnapshotID)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Wait error: %v\n", err)
				return nil, false
			}
			if final == nil {
				fmt.Println(style("Snapshot disappeared while waiting. Starting a new conversation.", ansiYellow))
				return nil, true
			}
			tone := ansiGreen
			if final.Status != aix.SnapshotStatusCompleted {
				tone = ansiYellow
			}
			fmt.Println(style(fmt.Sprintf("Settled (%s).", final.Status), tone))
			// A stopped run holds the turns that finished, so it resumes like
			// a completed one. Only a row its invocation never finished
			// writing has nothing to continue from, and resumePoint falls
			// back to the snapshot before it.
			resume := resumePoint(ctx, a, final)
			if resume == nil {
				fmt.Println(style("Nothing here can be continued. Starting a new conversation.", ansiYellow))
				return nil, true
			}
			if resume.SnapshotID != final.SnapshotID {
				fmt.Println(style("That snapshot holds no state; falling back to the one before it.", ansiYellow))
			}
			return resume, true
		case "3":
			// Ignore the pending snapshot and start fresh. The background
			// invocation keeps running; this CLI just stops tracking it.
			return nil, true
		case "4":
			return nil, false
		default:
			fmt.Println("Invalid choice. Type 1, 2, 3, or 4.")
		}
	}
}

// resumable reports whether a snapshot can be handed to WithSnapshotID. It
// mirrors the runtime's own gate. A failed row qualifies: it holds what its
// turn committed before the failure, so resuming picks up there. So does an
// aborted one, which holds the turns that finished before the stop.
//
// Two shapes do not. A pending row is still being written to by the
// invocation that owns it (an expired one is a pending row whose worker
// stopped beating). And an aborted row with no state was flipped by an abort
// whose invocation never got to stamp the conversation onto it, so it holds
// nothing to continue from.
func resumable[State any](snap *aix.SessionSnapshot[State]) bool {
	switch snap.Status {
	case aix.SnapshotStatusCompleted, aix.SnapshotStatusFailed:
		return true
	case aix.SnapshotStatusAborted:
		return snap.State != nil
	}
	return false
}

// resumePoint walks back from tip along ParentID to the newest snapshot that
// can be continued, or nil when the whole chain is dead ends. That is what
// turns an orphaned tip (a detached worker that died, or an abort nothing
// finalized) into a conversation the user keeps rather than loses.
func resumePoint[State any](ctx context.Context, a *aix.Agent[State], tip *aix.SessionSnapshot[State]) *aix.SessionSnapshot[State] {
	for snap := tip; snap != nil && snap.SnapshotID != ""; {
		if resumable(snap) {
			return snap
		}
		if snap.ParentID == "" {
			return nil
		}
		parent, err := a.GetSnapshot(ctx, snap.ParentID)
		if err != nil {
			return nil
		}
		snap = parent
	}
	return nil
}

// endedEarly reports whether a finish reason means the run did not get to
// finish on its own terms. Both terminals propagate out of the agent's turn
// loop, so the invocation is over either way and the connection is closing:
// the REPL has to stop reading rather than send into it. Which of the two it
// is changes the wording the CLI uses, never the handling.
func endedEarly(r aix.AgentFinishReason) bool {
	return r == aix.AgentFinishReasonFailed || r == aix.AgentFinishReasonAborted
}

// pickSession decides which snapshot (if any) to resume from. It only
// offers two paths so the demo stays focused: resume from the newest
// snapshot that can be continued (returns the snapshot pointer), or start
// fresh (returns nil).
func pickSession[State any](ctx context.Context, inputCh <-chan string, a *aix.Agent[State], tip *aix.SessionSnapshot[State]) (*aix.SessionSnapshot[State], bool) {
	var latest *aix.SessionSnapshot[State]
	if tip != nil {
		latest = resumePoint(ctx, a, tip)
	}
	if latest == nil {
		fmt.Printf("\nStarting a new conversation with %s.\n", style(a.Name(), ansiBold))
		return nil, true
	}
	if latest.SnapshotID != tip.SnapshotID {
		// The tip is orphaned: a detached worker that stopped beating, or an
		// abort that flipped the row before anything wrote state onto it.
		// Neither loses the conversation, since its parent still holds it.
		fmt.Printf("\n%s\n", style(fmt.Sprintf("The last snapshot is %s and holds nothing to continue; using the one before it.", tip.Status), ansiYellow))
	}

	msgs := 0
	if latest.State != nil {
		msgs = len(latest.State.Messages)
	}
	fmt.Printf("\nLast %s session: %s\n",
		style(a.Name(), ansiBold),
		style(fmt.Sprintf("%s (%s, %d msgs)", shortID(latest.SnapshotID), latest.UpdatedAt.Format(time.RFC822), msgs), ansiDim))
	switch latest.Status {
	case aix.SnapshotStatusFailed:
		why := "its last turn failed"
		if latest.Error != nil && latest.Error.Message != "" {
			why = latest.Error.Message
		}
		fmt.Println(style("This session stopped on an error: "+why, ansiYellow))
	case aix.SnapshotStatusAborted:
		fmt.Println(style("This session was stopped early.", ansiYellow))
	}
	fmt.Println("Resume from it? " + style("[Y/n] (n = start a new conversation)", ansiDim))

	line, ok := promptLine(ctx, inputCh, chevron)
	if !ok {
		return nil, false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "", "y", "yes":
		return latest, true
	default:
		return nil, true
	}
}

// resumeOption picks how to resume the chosen snapshot. Resuming by
// session ID is the canonical "continue this conversation" flow, so the
// session is validated against the store first: if it does not resolve
// to a resumable snapshot (a detached invocation still pending, legacy
// rows with no session ID, or a store error), fall back to the exact
// snapshot the user was just shown, which is also how a walk back past an
// orphaned tip stays on the snapshot pickSession settled on. Validating up
// front keeps the chat from opening on a connection whose invocation already
// failed, which would surface the error only after the user types a message.
func resumeOption[State any](ctx context.Context, a *aix.Agent[State], resume *aix.SessionSnapshot[State]) aix.InvocationOption[State] {
	if resume.SessionID != "" {
		tip, err := a.GetLatestSnapshot(ctx, resume.SessionID)
		if err == nil && tip != nil && resumable(tip) {
			return aix.WithSessionID[State](resume.SessionID)
		}
		fmt.Println(style("(this conversation's session can't be resumed as a whole; continuing from the selected snapshot)", ansiDim))
	}
	return aix.WithSnapshotID[State](resume.SnapshotID)
}

// runChat opens the bidi connection (optionally resuming from a
// snapshot) and runs the per-turn REPL. When resuming, the prior
// conversation is replayed first so the user sees the context they're
// picking up, then the REPL takes over. /detach is the one interesting
// branch: it sends the optional trailing text as the final input and
// detaches the connection, leaving a pending snapshot for the user to
// observe. It returns the session ID the chat ran under (falling back to
// prevSessionID) so the caller can offer to resume it later.
func runChat[State any](ctx context.Context, inputCh <-chan string, a *aix.Agent[State], hooks agentHooks, resume *aix.SessionSnapshot[State], prevSessionID string) (string, error) {
	prompter := &Prompter{ctx: ctx, inputCh: inputCh}
	fmt.Printf("\n%s\n", style("=== Chatting with "+a.Name()+" ===", ansiBold, ansiCyan))
	if resume != nil {
		fmt.Println(style("Resumed from "+shortID(resume.SnapshotID), ansiDim))
	}
	fmt.Println(style("Commands: /detach [text], /back, /quit", ansiDim))

	resumed := resume != nil && resume.State != nil && len(resume.State.Messages) > 0
	if resumed {
		fmt.Println()
		fmt.Println(style("(picking up where you left off)", ansiDim))
		printHistory(resume.State.Messages)
	}
	fmt.Println()

	var opts []aix.InvocationOption[State]
	if resume != nil {
		opts = append(opts, resumeOption(ctx, a, resume))
	}
	conn, err := a.Connect(ctx, opts...)
	if err != nil {
		// A snapshot the store accepted a moment ago can still be refused
		// here, so hand the user back to the agent list rather than take the
		// process down with the connection.
		fmt.Fprintf(os.Stderr, "Open agent %q: %v\n", a.Name(), err)
		return prevSessionID, nil
	}

	var (
		detached bool
		quit     bool
		runEnded bool
		// conversation is whether there is anything for an empty line to run
		// a turn on. A stray Enter before the first message has nothing, and
		// the agent rejects an empty input on an empty session.
		conversation = resumed
	)

	// Whether the turn a stopped run left unanswered is worth running again
	// is the user's call, not the CLI's, so it is a keystroke rather than an
	// inference. Saying so here is the only thing the status decides.
	if resume != nil && (resume.Status == aix.SnapshotStatusFailed || resume.Status == aix.SnapshotStatusAborted) {
		fmt.Println(style("Its last turn stopped. Press Enter on an empty line to run", ansiYellow))
		fmt.Println(style("it again on the messages it kept, or just type a new one.", ansiYellow))
		fmt.Println()
	}

repl:
	for {
		if runEnded {
			break
		}
		line, ok := promptLine(ctx, inputCh, chevron)
		if !ok {
			break
		}
		text := strings.TrimSpace(line)
		if text == "" {
			if !conversation {
				continue
			}
			// An input with no payload runs a turn on the conversation as it
			// stands. That is how the turn a stopped run left unanswered is
			// picked back up: its message is already in the session, so
			// sending it again would only duplicate it.
			if err := conn.Send(&aix.AgentInput{}); err != nil {
				fmt.Fprintf(os.Stderr, "Send error: %v\n", err)
				break
			}
		} else {
			switch {
			case text == "/back":
				break repl
			case text == "/quit" || text == "/exit":
				quit = true
				break repl
			case text == "/detach" || strings.HasPrefix(text, "/detach "):
				trailing := strings.TrimSpace(strings.TrimPrefix(text, "/detach"))
				// Send (optional message) + detach in a single wire input so
				// the trailing text becomes the last buffered message. The
				// agent will process it in the background after the
				// connection closes.
				input := &aix.AgentInput{Detach: true}
				if trailing != "" {
					input.Message = ai.NewUserTextMessage(trailing)
				}
				if err := conn.Send(input); err != nil {
					fmt.Fprintf(os.Stderr, "Detach error: %v\n", err)
					break repl
				}
				detached = true
				break repl
			case strings.HasPrefix(text, "/"):
				fmt.Println("Unknown command. Try /detach, /back, or /quit.")
				continue
			}

			if err := conn.SendText(text); err != nil {
				fmt.Fprintf(os.Stderr, "Send error: %v\n", err)
				break
			}
		}
		conversation = true
		fmt.Println()
		end, wrote, ok := streamReply(conn, hooks, prompter)
		if !ok {
			// The stream errored or closed before the turn settled; the
			// error was already reported. Output below reports the outcome.
			break repl
		}
		// finishReason rides the turn-end signal, so the client learns how
		// the turn ended without scanning the message content. The dim
		// turn/snapshot footer sits one blank line below the reply: when the
		// turn printed something the markers' leading newlines make that
		// blank, but when it printed nothing (e.g. an immediate failure) the
		// erased spinner already left a blank line, so the first marker reuses
		// it rather than stacking more blanks.
		if end.FinishReason != "" || end.SnapshotID != "" {
			sep := "  "
			if wrote {
				fmt.Println() // close the reply's line
				sep = "\n  "  // and put a blank line above the footer
			}
			if r := end.FinishReason; r != "" {
				fmt.Printf("%s%s", sep, style(fmt.Sprintf("[turn: %s]", r), ansiDim))
				sep = "\n  "
			}
			if end.SnapshotID != "" {
				fmt.Printf("%s%s", sep, style(fmt.Sprintf("[snapshot %s]", shortID(end.SnapshotID)), ansiDim))
			}
		}
		fmt.Println()
		fmt.Println()
		if endedEarly(end.FinishReason) {
			// A turn that broke or was stopped ends the invocation; Output
			// below reports the error and the snapshot to pick up from. The
			// footer above already spaced things off, so flag it to skip the
			// pre-outcome blank.
			runEnded = true
			break repl
		}
	}

	out, outErr := conn.Output()
	if outErr != nil && !errors.Is(outErr, context.Canceled) {
		fmt.Fprintf(os.Stderr, "Output error: %v\n", outErr)
	}

	// Set the closing status apart from the last input line (e.g. /back or
	// /detach) with a blank line. A failed turn already printed its footer
	// with trailing spacing just above, so skip the extra blank there.
	if out != nil && !runEnded {
		fmt.Println()
	}
	switch {
	case detached && out != nil && out.SnapshotID != "":
		fmt.Printf("%s Pending snapshot: %s.\n",
			style("Detached.", ansiYellow), style(shortID(out.SnapshotID), ansiDim))
		fmt.Println(style("The agent keeps processing in the background. Pick this", ansiDim))
		fmt.Println(style("agent again from the list to wait for it to finalize and", ansiDim))
		fmt.Println(style("resume from the cumulative final state.", ansiDim))
	case out != nil && endedEarly(out.FinishReason):
		// Both terminals resolve with the error and a snapshot to resume
		// from; which one it is picks the wording, not the handling.
		what := "Agent failed"
		if out.FinishReason == aix.AgentFinishReasonAborted {
			what = "Agent stopped"
		}
		if out.Error != nil {
			fmt.Fprintf(os.Stderr, "%s\n", style(fmt.Sprintf("%s (%s): %s", what, out.Error.Status, out.Error.Message), ansiYellow))
			// The output's error carries the status the failure was
			// classified with, so the client can offer the right next step
			// without parsing message text.
			switch out.Error.Status {
			case status.Unavailable, status.ResourceExhausted:
				fmt.Println(style("The model looks overloaded. Resume from the snapshot below in a moment and try again.", ansiDim))
			case status.InvalidArgument:
				fmt.Println(style("The model or a tool rejected the request. Rephrase and try again.", ansiDim))
			case status.Cancelled, status.DeadlineExceeded:
				fmt.Println(style("The turn was cut short. Resume from the snapshot below to continue.", ansiDim))
			case status.Aborted:
				// A limit the run was given, such as ai.WithMaxTurns. Resuming
				// re-runs the same turn into the same limit, so say so rather
				// than invite a retry that burns tokens to fail identically.
				fmt.Println(style("The run hit a limit it was given. Resuming re-runs the same turn, so it needs a bigger limit or a simpler ask.", ansiDim))
			}
		}
		if out.SnapshotID != "" {
			// The stopped turn's own row when it committed anything, the turn
			// before it otherwise. Either way it is where to pick up.
			fmt.Printf("%s\n", style(fmt.Sprintf("Resume point: %s. Pick this agent again to continue from it.", shortID(out.SnapshotID)), ansiDim))
		}
	case out != nil && out.SnapshotID != "":
		fmt.Printf("%s Final snapshot: %s.\n",
			style(fmt.Sprintf("Done (%s).", out.FinishReason), ansiGreen),
			style(shortID(out.SnapshotID), ansiDim))
	}

	// Hand back the session this chat ran under so the caller can offer to
	// resume it. An invocation that never produced output (e.g. the stream
	// errored before any turn) keeps the prior session.
	sessionID := prevSessionID
	if out != nil && out.SessionID != "" {
		sessionID = out.SessionID
	}
	if quit {
		return sessionID, errQuit
	}
	return sessionID, nil
}

// streamReply consumes one user turn end to end: it streams the reply and,
// whenever the turn pauses on tool interrupts, routes them through the agent's
// handler and resumes, until the turn settles or the stream ends. It returns
// the settling TurnEnd (ok=true) or reports a stream error (ok=false).
//
// Interrupt rounds reuse the connection: per AgentConnection.Receive, a
// caller may break on TurnEnd, send the next input (here a resume), and
// range Receive again for the continuation. A resumed turn can interrupt
// again, so this loops rather than handling just one round.
//
// It also reports whether any visible content (text or a tool call) was
// printed, so the caller can space the turn footer correctly when a turn
// produces nothing (e.g. an immediate failure).
func streamReply[State any](conn *aix.AgentConnection[State], hooks agentHooks, p *Prompter) (end *aix.TurnEnd, wrote bool, ok bool) {
	for {
		interrupts, e, w, sok := streamTurn(conn)
		wrote = wrote || w
		if !sok {
			return nil, wrote, false
		}
		if len(interrupts) == 0 {
			return e, wrote, true
		}
		resume := resolveInterrupts(hooks, p, interrupts)
		if resume == nil {
			// Nothing could be resolved (no handler, or the handler declined
			// every interrupt). Surface the interrupted turn as-is rather
			// than sending an empty resume the agent can't act on.
			return e, wrote, true
		}
		if err := conn.SendResume(resume); err != nil {
			fmt.Fprintf(os.Stderr, "\nResume error: %v\n", err)
			return nil, wrote, false
		}
		// Blank line so the resumed reply (and its spinner) sits apart from the
		// interrupt prompt, matching the spacing of a fresh turn.
		fmt.Println()
		// The resumed turn streams on the next Receive; loop to consume it.
	}
}

// streamTurn streams a single turn's chunks: it shows a spinner until the
// first visible output arrives, prints model text as it streams, renders tool
// calls inline for feedback, and collects any tool interrupts. It returns when
// the turn ends (end != nil, ok=true) or the stream errors/closes (ok=false).
//
// Tool requests reach the client because the agent forwards the model's raw
// streaming chunks verbatim, so the function calls the model emits show up
// here alongside its text. Interrupts (a tool that paused for input) are kept
// for the agent's handler; every other tool request is printed so the user
// sees what the agent is doing between bursts of text.
//
// The spinner is stopped (and erased) the moment any visible content is about
// to print, so the reply or tool line takes over the line it occupied.
func streamTurn[State any](conn *aix.AgentConnection[State]) (interrupts []*ai.Part, end *aix.TurnEnd, wrote bool, ok bool) {
	sp := startSpinner("Thinking")
	stopSpinner := func() {
		if sp != nil {
			sp.stop()
			sp = nil
		}
	}
	defer stopSpinner()

	for chunk, err := range conn.Receive() {
		if err != nil {
			stopSpinner()
			fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
			return nil, nil, wrote, false
		}
		if mc := chunk.ModelChunk; mc != nil {
			if t := mc.Text(); t != "" {
				stopSpinner()
				fmt.Print(t)
				wrote = true
			}
			for _, p := range mc.Content {
				switch {
				case p.IsInterrupt():
					interrupts = append(interrupts, p)
				case p.IsToolRequest() && !p.ToolRequest.Partial:
					stopSpinner()
					printToolRequest(p.ToolRequest)
					wrote = true
				}
			}
		}
		if chunk.TurnEnd != nil {
			stopSpinner()
			return interrupts, chunk.TurnEnd, wrote, true
		}
	}
	// Stream closed without a turn end (e.g. ctx canceled / connection gone).
	stopSpinner()
	return nil, nil, wrote, false
}

// printToolRequest renders a single tool call as a compact, colored line so
// the user gets feedback about what the agent is doing as a turn streams.
// The tool name is highlighted and its JSON input dimmed. A blank line on
// each side sets the call apart from the messages before and after it (the
// agent's "about to do X" note and its follow-up reply).
func printToolRequest(tr *ai.ToolRequest) {
	if tr == nil {
		return
	}
	fmt.Printf("\n\n  %s%s⚙ %s%s %s%s%s\n\n",
		ansiBold, ansiCyan, tr.Name, ansiReset,
		ansiDim, formatToolInput(tr.Input), ansiReset)
}

// formatToolInput renders a tool's input as compact single-line JSON,
// truncated if very long so the feedback line stays readable. It falls back
// to %v if the input can't be marshaled.
func formatToolInput(input any) string {
	if input == nil {
		return "{}"
	}
	b, err := json.Marshal(input)
	if err != nil {
		return fmt.Sprintf("%v", input)
	}
	const max = 300
	s := string(b)
	if r := []rune(s); len(r) > max {
		s = string(r[:max]) + "…"
	}
	return s
}

// resolveInterrupts turns a batch of tool interrupts into one resume payload by
// asking the agent's handler about each. The handler builds the resume part;
// this sorts each into the matching half of aix.ToolResume by part kind, so
// handlers never deal with the wire shape. Returns nil if none resolved.
func resolveInterrupts(hooks agentHooks, p *Prompter, interrupts []*ai.Part) *aix.ToolResume {
	if hooks.onInterrupt == nil {
		// An interruptible tool fired on an agent the CLI wasn't told how to
		// resume. That's a wiring bug in the sample, not user input, so say
		// so plainly instead of hanging on a prompt that never comes.
		fmt.Printf("\n[%s paused on %d tool interrupt(s), but no interrupt handler is registered for it]\n",
			hooks.name, len(interrupts))
		return nil
	}
	// No separator needed here: the interrupting tool's request was already
	// streamed and rendered by printToolRequest, whose trailing blank line
	// sets the upcoming prompts apart from the reply.
	resume := &aix.ToolResume{}
	for _, interrupt := range interrupts {
		part, err := hooks.onInterrupt(p, interrupt)
		switch {
		case err != nil:
			fmt.Fprintf(os.Stderr, "Interrupt handler error: %v\n", err)
		case part == nil:
			// Handler chose not to resolve this one; leave it out.
		case part.ToolResponse != nil:
			resume.Respond = append(resume.Respond, part)
		case part.ToolRequest != nil:
			resume.Restart = append(resume.Restart, part)
		}
	}
	if len(resume.Restart) == 0 && len(resume.Respond) == 0 {
		return nil
	}
	return resume
}

// summarizeLatest is the one-line summary printed under each agent in the
// list: the tip of the conversation last run with it this session. Empty
// when none has run yet, or the session has no resumable snapshot, so a
// fresh list shows no clutter.
//
// It returns the line already styled: dim metadata, with a detach that is
// still running (or whose worker died) highlighted in yellow. Each segment
// carries its own reset so the yellow status doesn't bleed into the
// surrounding dim text. Neither row has finalized messages yet, so the count
// is omitted for both.
func summarizeLatest[State any](ctx context.Context, a *aix.Agent[State], sessionID string) string {
	if sessionID == "" {
		return ""
	}
	latest, err := a.GetLatestSnapshot(ctx, sessionID)
	if err != nil || latest == nil {
		return ""
	}
	when := latest.UpdatedAt.Format(time.RFC822)
	switch latest.Status {
	case aix.SnapshotStatusPending, aix.SnapshotStatusExpired:
		return style("last: "+shortID(latest.SnapshotID)+" (", ansiDim) +
			style(string(latest.Status), ansiYellow) +
			style(", "+when+")", ansiDim)
	}
	msgs := 0
	if latest.State != nil {
		msgs = len(latest.State.Messages)
	}
	return style(fmt.Sprintf("last: %s (%s, %d msgs, %s)",
		shortID(latest.SnapshotID), latest.Status, msgs, when), ansiDim)
}

// waitForFinalize blocks until a snapshot leaves pending and returns the final
// one, or nil if it disappeared while we waited.
//
// WaitForSnapshot does the waiting, so the CLI does not care how the store it
// was given observes the change: the agent subscribes where the store can push
// and re-reads otherwise, and either way a worker that stopped heartbeating
// ends the wait as expired rather than leaving it hanging.
func waitForFinalize[State any](ctx context.Context, a *aix.Agent[State], snapshotID string) (*aix.SessionSnapshot[State], error) {
	final, err := a.WaitForSnapshot(ctx, snapshotID)
	if errors.Is(err, aix.ErrSnapshotNotFound) {
		return nil, nil
	}
	return final, err
}

// printHistory replays prior turns in the format the live REPL uses, so a
// resumed chat reads continuously. It renders user and model text plus the tool
// calls the model made, as the same lines streamTurn shows live, so a
// delegation or transfer is visible on resume exactly as it happened.
//
// Tool result messages (role=tool) are skipped, matching the live stream; the
// agent's per-turn loop still has the full history under the hood.
func printHistory(msgs []*ai.Message) {
	prevToolReq := false
	for _, m := range msgs {
		switch m.Role {
		case ai.RoleUser:
			text := strings.TrimSpace(m.Text())
			if text == "" {
				continue
			}
			fmt.Println()
			fmt.Printf("%s%s\n", chevron, text)
			prevToolReq = false
		case ai.RoleModel:
			text := strings.TrimSpace(m.Text())
			var toolReqs []*ai.ToolRequest
			for _, part := range m.Content {
				if part.IsToolRequest() {
					toolReqs = append(toolReqs, part.ToolRequest)
				}
			}
			if text == "" && len(toolReqs) == 0 {
				continue
			}
			// Leading blank unless the previous line was a tool call, whose
			// printToolRequest already left a trailing blank line.
			if !prevToolReq {
				fmt.Println()
			}
			// Print text with no trailing newline; the tool call's leading
			// blank (or the close below) ends the line.
			if text != "" {
				fmt.Print(text)
			}
			for _, tr := range toolReqs {
				printToolRequest(tr)
			}
			if len(toolReqs) == 0 {
				fmt.Println()
			}
			prevToolReq = len(toolReqs) > 0
		}
	}
}

// shortID trims a UUID-shaped snapshot ID to its first segment so the
// CLI stays readable. The full ID is still available on disk in
// .genkit/snapshots/<agent>/.
func shortID(id string) string {
	if i := strings.Index(id, "-"); i > 0 {
		return id[:i]
	}
	if len(id) > 8 {
		return id[:8]
	}
	return id
}

// Prompter lets an InterruptHandler ask the user questions through the same
// input stream and cancellation semantics the rest of the CLI uses, so a
// handler never touches os.Stdin directly. Every read honors Ctrl-C/EOF: a
// false (Ask/Confirm) or -1 (Choose) return means input closed mid-question.
type Prompter struct {
	ctx     context.Context
	inputCh <-chan string
}

// Printf prints informational text to the user. Use it for context the
// handler wants to show before asking (e.g. why the tool paused).
func (p *Prompter) Printf(format string, args ...any) {
	fmt.Printf(format, args...)
}

// Ask prints prompt and returns the user's trimmed reply. ok=false if the
// input stream closed or the context was canceled.
func (p *Prompter) Ask(prompt string) (reply string, ok bool) {
	line, ok := promptLine(p.ctx, p.inputCh, prompt)
	if !ok {
		return "", false
	}
	return strings.TrimSpace(line), true
}

// Confirm asks a yes/no question, defaulting to yes on empty input and
// returning false on anything else (including a closed input stream).
func (p *Prompter) Confirm(question string) bool {
	reply, ok := p.Ask(question + style(" [Y/n] ", ansiDim))
	if !ok {
		return false
	}
	switch strings.ToLower(reply) {
	case "", "y", "yes":
		return true
	default:
		return false
	}
}

// Choose prints a numbered menu and returns the selected option's index
// (0-based), re-prompting until the input is valid. It returns -1 if the
// input stream closed.
func (p *Prompter) Choose(title string, options ...string) int {
	for {
		if title != "" {
			fmt.Println(title)
		}
		for i, opt := range options {
			fmt.Printf("  %s %s\n", style(fmt.Sprintf("%d.", i+1), ansiCyan), opt)
		}
		fmt.Println()
		reply, ok := p.Ask(chevron)
		if !ok {
			return -1
		}
		if n, err := strconv.Atoi(reply); err == nil && n >= 1 && n <= len(options) {
			return n - 1
		}
		fmt.Printf("Enter a number between 1 and %d.\n", len(options))
	}
}

// readLine reads one line from inputCh, returning false if the channel
// closes (EOF) or ctx is canceled (Ctrl-C). All CLI prompts go through
// this so cancellation is honored uniformly.
func readLine(ctx context.Context, inputCh <-chan string) (string, bool) {
	select {
	case <-ctx.Done():
		fmt.Println()
		return "", false
	case line, ok := <-inputCh:
		if !ok {
			fmt.Println()
			return "", false
		}
		return line, true
	}
}

// readLines reads lines from stdin on a background goroutine and yields
// them via the returned channel. The channel is closed on EOF, read
// error, or ctx cancellation. The goroutine cannot interrupt a blocked
// stdin read; on ctx cancellation it exits as soon as a line completes
// (or, in practice, when the process terminates).
func readLines(ctx context.Context) <-chan string {
	ch := make(chan string)
	go func() {
		defer close(ch)
		reader := bufio.NewReader(os.Stdin)
		for {
			line, err := reader.ReadString('\n')
			if line != "" {
				select {
				case ch <- line:
				case <-ctx.Done():
					return
				}
			}
			if err != nil {
				return
			}
		}
	}()
	return ch
}
