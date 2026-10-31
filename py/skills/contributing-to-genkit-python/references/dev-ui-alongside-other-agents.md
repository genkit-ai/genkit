# Working alongside other agents

Assume you're not alone on this machine. Other agents may be editing their own copies of the repo and running their own Dev UIs at the same time, and nothing stops you from quietly debugging one of theirs.

Work in your own git worktree, created off the branch you were asked to change, and run everything from inside it. Use that worktree's environment (`uv run` from inside the worktree), not a shared one, or you'll be testing whatever another checkout installed.

When you use `genkit start`, make sure the Dev UI you're looking at is talking to your app:

- Start it from your worktree's project directory. The Dev UI only sees apps that registered under the directory it was started from (`.genkit/runtimes`), so separate worktrees get separate Dev UIs. Within one directory it can see several apps at once, each with its own runtime ID, and anything that doesn't name a runtime goes to the one started most recently. If a second app starts in your directory, your runs quietly go to it.
- Let it pick free ports, and read them from its output. Don't assume 4000 for the UI or 3100 for reflection. If you pin a port with `-p` or `GENKIT_REFLECTION_PORT`, expect it might be taken; a busy reflection port makes `Genkit()` fail at startup.
- Don't carry over `GENKIT_*` variables from another shell. A leftover `GENKIT_TELEMETRY_SERVER` sends your traces to another agent's Dev UI, and yours stays empty while things look like they're working.
- Before trusting a result, check you're in the right place: the flow you just added is in the list, and the trace you just ran shows up. If either is missing, you're connected to the wrong app.

Clean up only what you started. Stop your own process by its PID. Never run `pkill genkit`, kill whatever's on a port, or clear `.genkit/` in a directory you didn't create; that's someone else's session.
