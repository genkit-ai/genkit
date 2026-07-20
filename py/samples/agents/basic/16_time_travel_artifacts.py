#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Git for agent state: rewinding restores the artifact, not just the messages.

A builder agent keeps a landing page in a `landing.md` artifact. The main line
drifts into a stiff enterprise direction we don't love, so we rewind to the
checkpoint right after the headline. Loading that snapshot restores the WHOLE
state — the artifact reverts with the conversation — so the playful timeline we
build next grows from the original headline, and the enterprise page is untouched.

Every turn is a snapshot you can rewind to. Swap InMemorySessionStore
for FileSessionStore to keep the tree on disk. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI
from genkit_middleware import Artifacts, Middleware

from genkit import Genkit
from genkit.agent import InMemorySessionStore

ai = Genkit(plugins=[GoogleAI(), Middleware()])

writer = ai.define_agent(
    name='writer',
    model='googleai/gemini-flash-latest',
    system=(
        'You build a landing page in a single artifact named "landing.md". On every '
        'request rewrite the whole file, keep it under 14 lines, and reply with one '
        'short sentence about what you changed.'
    ),
    use=[Artifacts()],
    store=InMemorySessionStore(),  # every turn is a snapshot you can rewind to
)


def page(chat) -> str:
    """The landing.md the agent is maintaining in this timeline."""
    for art in chat.artifacts:
        if art.name == 'landing.md':
            return ''.join(getattr(getattr(p, 'root', p), 'text', '') for p in art.parts).strip()
    return ''


async def main() -> None:
    chat = writer.chat()
    await chat.send('Start a landing page for "Quill", an AI note-taking app: punchy headline + subhead.').response
    checkpoint = chat.snapshot_id  # bookmark this exact moment
    assert checkpoint  # populated once the turn is store-backed
    headline_page = page(chat)  # landing.md as it stands at the checkpoint
    assert headline_page

    # The main line drifts corporate, rewriting landing.md twice...
    await chat.send('Add enterprise feature bullets: SOC 2, SSO, audit logs.').response
    await chat.send('Add an enterprise pricing table with "Contact Sales".').response
    # → landing.md now carries an enterprise section; it's moved past the checkpoint
    assert page(chat) != headline_page

    # Don't love that direction? Rewind to the checkpoint. Loading the snapshot
    # restores the whole state — the landing.md artifact reverts with the messages.
    alt = await writer.load_chat(snapshot_id=checkpoint)
    # → the enterprise edits are gone; alt's landing.md is back to the headline version
    assert page(alt) == headline_page

    # Build a different, playful timeline from that same headline. The enterprise
    # page is untouched — both landing.md timelines coexist off one checkpoint.
    await alt.send('Add playful, indie feature bullets with emoji.').response
    await alt.send('Add a warm "why we built this" founder note instead of pricing.').response
    assert page(alt) != page(chat)


if __name__ == '__main__':
    ai.run_main(main())
