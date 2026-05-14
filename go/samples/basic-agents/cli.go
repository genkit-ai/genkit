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

// This file is the user-facing half of the sample: the interactive CLI
// that ties the three server-side agent definitions in main.go to a
// single experience. The CLI is deliberately small. It has three
// responsibilities:
//
//  1. Pick an agent ("thread list").
//  2. For the picked agent, pick between resuming from the latest
//     snapshot or starting fresh. If the latest snapshot is still
//     pending (a detached invocation is still processing in the
//     background), offer to wait, start fresh, or back out.
//  3. Run a small REPL against the agent: stream the model's reply each
//     turn, accept text input, and offer /detach, /back, and /quit as
//     control commands.
//
// The detach demo is woven into step 3: typing "/detach <text>" sends
// the text as the final input and detaches, so the agent keeps
// processing in the background and the caller gets a pending snapshot
// ID. Re-picking the same agent in step 2 then surfaces the wait/resume
// flow.

package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
)

// sampleAgent pairs an agent with the store it persists to and a
// one-line description for the CLI list view. The embedded
// *aix.Agent[any, any] makes Name(), StreamBidi() etc. callable
// directly on a sampleAgent value, so the CLI does not need a
// separate field-threading layer.
//
// Store is tracked alongside the agent (rather than fished out of it)
// because we use FileSessionStore-specific helpers like
// LatestSnapshot and OnSnapshotStatusChange; carrying the concrete
// type avoids a type assertion at every call site.
type sampleAgent struct {
	*aix.Agent[any, any]
	Store       *localstore.FileSessionStore[any]
	Description string
}

// errQuit signals that the user typed /quit somewhere in the CLI; it
// bubbles up through openAgent and breaks runCLI's outer loop.
var errQuit = errors.New("quit")

// runCLI is the entry point for the interactive client. It alternates
// between two screens forever: the agent list and a per-agent chat.
// Returning from a chat brings the user back to the agent list. /quit
// (anywhere) and Ctrl-C both unwind back here and exit cleanly.
func runCLI(ctx context.Context, agents []sampleAgent) error {
	fmt.Println("Genkit Basic Agents")
	fmt.Println("===================")
	fmt.Println()
	fmt.Println("Pick an agent below, choose to resume the last conversation")
	fmt.Println("or start a new one, and chat. Inside a chat:")
	fmt.Println("  (text)             send a message and stream the reply")
	fmt.Println("  /detach (text...)  send the text (optional) as the final")
	fmt.Println("                     input, then detach. The agent finishes")
	fmt.Println("                     in the background and writes a pending")
	fmt.Println("                     snapshot. Re-pick the agent later to")
	fmt.Println("                     wait for it and resume from the final")
	fmt.Println("                     state.")
	fmt.Println("  /back              return to the agent list")
	fmt.Println("  /quit              exit the program")

	inputCh := readLines(ctx)
	for {
		choice, ok := pickAgent(ctx, inputCh, agents)
		if !ok {
			return nil
		}
		if err := openAgent(ctx, inputCh, agents[choice]); err != nil {
			if errors.Is(err, errQuit) {
				return nil
			}
			return err
		}
	}
}

// pickAgent renders the agent list and reads the user's choice. The
// list is re-rendered between selections so the user can see updated
// pending/terminal status after returning from a chat.
func pickAgent(ctx context.Context, inputCh <-chan string, agents []sampleAgent) (int, bool) {
	for {
		fmt.Println()
		fmt.Println("Agents:")
		for i, a := range agents {
			fmt.Printf("  %d. %s — %s\n", i+1, a.Name(), a.Description)
			if summary := summarizeLatest(ctx, a); summary != "" {
				fmt.Printf("       last: %s\n", summary)
			}
		}
		fmt.Println("  q. quit")
		fmt.Println()
		fmt.Print("> ")

		line, ok := readLine(ctx, inputCh)
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
func openAgent(ctx context.Context, inputCh <-chan string, a sampleAgent) error {
	latest, err := a.Store.LatestSnapshot(ctx)
	if err != nil {
		return fmt.Errorf("read snapshots for %q: %w", a.Name(), err)
	}

	var (
		resume *aix.SessionSnapshot[any]
		ok     bool
	)
	if latest != nil && latest.Status == aix.SnapshotStatusPending {
		// Background invocation still in flight. handlePending makes
		// the final decision itself (wait & resume, new, or back), so
		// we don't fall through to pickSession — the user already
		// chose; asking again would just be noise.
		resume, ok = handlePending(ctx, inputCh, a, latest)
	} else {
		resume, ok = pickSession(ctx, inputCh, a, latest)
	}
	if !ok {
		return nil
	}
	return runChat(ctx, inputCh, a, resume)
}

// handlePending offers the three reasonable responses when a previous
// invocation of this agent is still running in the background:
//
//  1. wait for it to finalize and resume from it directly,
//  2. ignore it and start a fresh conversation,
//  3. go back to the agent list.
//
// Returns the snapshot to resume from (option 1, succeeded) or nil
// (option 2, or option 1 when the snapshot terminated non-succeeded).
// ok=false means the user chose 3 or the context was canceled.
//
// Crucially, options that imply "use this conversation" return the
// snapshot directly so the caller can skip the resume / new prompt:
// the user already committed to the choice by waiting, and re-asking
// would be redundant.
func handlePending(ctx context.Context, inputCh <-chan string, a sampleAgent, pending *aix.SessionSnapshot[any]) (*aix.SessionSnapshot[any], bool) {
	for {
		fmt.Printf("\nThe last %s session is still running in the background (%s).\n", a.Name(), shortID(pending.SnapshotID))
		fmt.Println("  1. Wait for it to finalize")
		fmt.Println("  2. Start a new conversation")
		fmt.Println("  3. Back to agent list")
		fmt.Print("> ")

		line, ok := readLine(ctx, inputCh)
		if !ok {
			return nil, false
		}
		switch strings.TrimSpace(line) {
		case "1":
			fmt.Println("Waiting for it to finalize...")
			final, err := waitForFinalize(ctx, a.Store, pending.SnapshotID)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Wait error: %v\n", err)
				return nil, false
			}
			if final == nil {
				fmt.Println("Snapshot disappeared while waiting. Starting a new conversation.")
				return nil, true
			}
			fmt.Printf("Done (%s).\n", final.Status)
			if final.Status != aix.SnapshotStatusSucceeded {
				// failed / aborted snapshots aren't resumable; the
				// agent runtime would reject WithSnapshotID on them.
				// Fall through to a fresh chat instead.
				fmt.Println("Cannot resume this snapshot. Starting a new conversation.")
				return nil, true
			}
			return final, true
		case "2":
			// Ignore the pending snapshot; start a fresh chat. The
			// background invocation keeps running and writes its
			// terminal status — this CLI just stops tracking it.
			return nil, true
		case "3":
			return nil, false
		default:
			fmt.Println("Invalid choice. Type 1, 2, or 3.")
		}
	}
}

// pickSession decides which snapshot (if any) to resume from. It only
// offers two paths so the demo stays focused: resume from the most
// recent terminal snapshot (returns the snapshot pointer), or start
// fresh (returns nil).
func pickSession(ctx context.Context, inputCh <-chan string, a sampleAgent, latest *aix.SessionSnapshot[any]) (*aix.SessionSnapshot[any], bool) {
	if latest == nil || latest.Status != aix.SnapshotStatusSucceeded {
		fmt.Printf("\nStarting a new conversation with %s.\n", a.Name())
		return nil, true
	}

	msgs := 0
	if latest.State != nil {
		msgs = len(latest.State.Messages)
	}
	fmt.Printf("\nLast %s session: %s (%s, %d msgs).\n",
		a.Name(), shortID(latest.SnapshotID), latest.UpdatedAt.Format(time.RFC822), msgs)
	fmt.Println("Resume from it? [Y/n] (n = start a new conversation)")
	fmt.Print("> ")

	line, ok := readLine(ctx, inputCh)
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

// runChat opens the bidi connection (optionally resuming from a
// snapshot) and runs the per-turn REPL. When resuming, the prior
// conversation is replayed first so the user sees the context they're
// picking up, then the REPL takes over. /detach is the one interesting
// branch — it sends the optional trailing text as the final input and
// detaches the connection, returning the pending snapshot ID for the
// user to observe.
func runChat(ctx context.Context, inputCh <-chan string, a sampleAgent, resume *aix.SessionSnapshot[any]) error {
	fmt.Printf("\n=== Chatting with %s ===\n", a.Name())
	if resume != nil {
		fmt.Printf("Resumed from %s\n", shortID(resume.SnapshotID))
	}
	fmt.Println("Commands: /detach [text], /back, /quit")

	if resume != nil && resume.State != nil && len(resume.State.Messages) > 0 {
		fmt.Println()
		fmt.Println("(picking up where you left off)")
		printHistory(resume.State.Messages)
	}
	fmt.Println()

	var opts []aix.InvocationOption[any]
	if resume != nil {
		opts = append(opts, aix.WithSnapshotID[any](resume.SnapshotID))
	}
	conn, err := a.StreamBidi(ctx, opts...)
	if err != nil {
		return fmt.Errorf("open agent %q: %w", a.Name(), err)
	}

	var (
		detached bool
		quit     bool
	)

repl:
	for {
		fmt.Print("> ")
		line, ok := readLine(ctx, inputCh)
		if !ok {
			break
		}
		text := strings.TrimSpace(line)
		if text == "" {
			continue
		}

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
		fmt.Println()
		for chunk, err := range conn.Receive() {
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				break
			}
			if chunk.ModelChunk != nil {
				fmt.Print(chunk.ModelChunk.Text())
			}
			if chunk.TurnEnd != nil {
				if chunk.TurnEnd.SnapshotID != "" {
					fmt.Printf("\n  [snapshot %s]", shortID(chunk.TurnEnd.SnapshotID))
				}
				fmt.Println()
				fmt.Println()
				break
			}
		}
	}

	out, outErr := conn.Output()
	if outErr != nil && !errors.Is(outErr, context.Canceled) {
		fmt.Fprintf(os.Stderr, "Output error: %v\n", outErr)
	}

	switch {
	case detached && out != nil && out.SnapshotID != "":
		fmt.Printf("Detached. Pending snapshot: %s.\n", shortID(out.SnapshotID))
		fmt.Println("The agent keeps processing in the background. Pick this")
		fmt.Println("agent again from the list to wait for it to finalize and")
		fmt.Println("resume from the cumulative final state.")
	case out != nil && out.SnapshotID != "":
		fmt.Printf("Done. Final snapshot: %s.\n", shortID(out.SnapshotID))
	}

	if quit {
		return errQuit
	}
	return nil
}

// summarizeLatest is the one-line summary printed under each agent in
// the list. Empty if there is no snapshot yet, so a freshly-installed
// sample doesn't show clutter.
func summarizeLatest(ctx context.Context, a sampleAgent) string {
	latest, err := a.Store.LatestSnapshot(ctx)
	if err != nil || latest == nil {
		return ""
	}
	msgs := 0
	if latest.State != nil {
		msgs = len(latest.State.Messages)
	}
	return fmt.Sprintf("%s (%s, %d msgs, %s)",
		shortID(latest.SnapshotID), latest.Status, msgs, latest.UpdatedAt.Format(time.RFC822))
}

// waitForFinalize subscribes to a snapshot's status and blocks until it
// transitions out of pending. The returned snapshot is the final one (or
// nil if it disappeared). OnSnapshotStatusChange yields the current
// status first, so a snapshot that finalized between the directory scan
// and the subscription is observed immediately.
func waitForFinalize(ctx context.Context, store *localstore.FileSessionStore[any], snapshotID string) (*aix.SessionSnapshot[any], error) {
	subCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	statusCh := store.OnSnapshotStatusChange(subCtx, snapshotID)
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case status, ok := <-statusCh:
			if !ok {
				// Subscription closed (e.g. snapshot deleted under us).
				return store.GetSnapshot(ctx, snapshotID)
			}
			if status == aix.SnapshotStatusPending {
				continue
			}
			return store.GetSnapshot(ctx, snapshotID)
		}
	}
}

// printHistory replays prior turns in the same format the live REPL
// uses, so a resumed chat reads continuously rather than dumping the
// user into an empty prompt. Non-user/model roles (e.g. tool messages)
// and empty content are skipped — they would only be noise here, and
// the agent's per-turn loop still has the full history under the hood.
func printHistory(msgs []*ai.Message) {
	for _, m := range msgs {
		text := strings.TrimSpace(m.Text())
		if text == "" {
			continue
		}
		switch m.Role {
		case ai.RoleUser:
			fmt.Println()
			fmt.Printf("> %s\n", text)
		case ai.RoleModel:
			fmt.Println()
			fmt.Println(text)
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
