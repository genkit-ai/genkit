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

"""Artifacts middleware for Genkit agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from genkit._ai._model import Message
from genkit._ai._tools import tool
from genkit._core._model import GenerateActionOptions, ModelResponse
from genkit._core._typing import Artifact, Part, Role, TextPart
from genkit.middleware import BaseMiddleware, GenerateHookParams, GenerateMiddlewareContext

ARTIFACTS_LISTING_MARKER = 'artifacts-middleware-listing'


class ArtifactsConfig(BaseModel):
    """Options for session artifact tools and prompt injection."""

    readonly: bool = Field(
        default=False,
        description=('When true, only read_artifact is provided — the model cannot create or update artifacts.'),
    )


class ReadArtifactInput(BaseModel):
    name: str = Field(description='The name of the artifact to read.')


class ReadArtifactOutput(BaseModel):
    name: str = Field(description='The artifact name.')
    content: str = Field(description='The text content of the artifact.')
    found: bool = Field(description='Whether the artifact was found in the session.')


class WriteArtifactInput(BaseModel):
    name: str = Field(description='A unique name for the artifact (e.g. a filename like "report.md").')
    content: str = Field(description='The full text content of the artifact.')


class WriteArtifactOutput(BaseModel):
    status: str = Field(description='Confirmation that the artifact was created or updated.')


def extract_artifact_text(artifact: Artifact) -> str:
    parts: list[str] = []
    for part in artifact.parts:
        root = part.root
        if isinstance(root, TextPart) and root.text:
            parts.append(root.text)
    return '\n'.join(parts)


def artifact_source(artifact: Artifact) -> str | None:
    meta = artifact.metadata
    if isinstance(meta, dict):
        source = meta.get('source')
        return str(source) if source is not None else None
    return None


def build_artifact_listing(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return '<artifacts>\nNo artifacts are currently available in the session.\n</artifacts>'

    lines = [
        '<artifacts>',
        'The following artifacts are available in the session. Use the read_artifact tool to view their content.',
    ]
    for art in artifacts:
        text = extract_artifact_text(art)
        size_hint = f' ({len(text)} chars)' if text else ''
        source = artifact_source(art)
        source_hint = f' [from: {source}]' if source else ''
        label = art.name or '(unnamed)'
        lines.append(f'  - {label}{size_hint}{source_hint}')
    lines.append('</artifacts>')
    return '\n'.join(lines)


def inject_artifact_listing_messages(messages: list[Message], listing: str) -> list[Message]:
    """Strip prior listing parts and append a fresh listing to the system message."""
    out = list(messages)

    for i, msg in enumerate(out):
        filtered: list[Part] = []
        for part in msg.content or []:
            root = part.root
            meta = root.metadata if isinstance(root, TextPart) else None
            if isinstance(meta, dict) and meta.get(ARTIFACTS_LISTING_MARKER):
                continue
            filtered.append(part)
        if len(filtered) != len(msg.content or []):
            out[i] = Message(role=msg.role, content=filtered)

    listing_part = Part(
        root=TextPart(text=listing, metadata={ARTIFACTS_LISTING_MARKER: True}),
    )

    system_idx: int | None = None
    for i, msg in enumerate(out):
        if msg.role == Role.SYSTEM:
            system_idx = i
            break

    if system_idx is not None:
        msg = out[system_idx]
        out[system_idx] = Message(
            role=Role.SYSTEM,
            content=[*msg.content, listing_part],
        )
    else:
        out.insert(0, Message(role=Role.SYSTEM, content=[listing_part]))

    return out


def inject_artifact_listing(options: GenerateActionOptions, listing: str) -> GenerateActionOptions:
    new_options = options.model_copy()
    new_options.messages = inject_artifact_listing_messages(list(options.messages), listing)
    return new_options


class Artifacts(BaseMiddleware[ArtifactsConfig]):
    """Session artifact tools plus an injected artifact listing in the system prompt."""

    def tools(self, ctx: GenerateMiddlewareContext) -> list[Any]:
        tools: list[Any] = []

        async def read_artifact(input: ReadArtifactInput) -> ReadArtifactOutput:
            session = ctx.ai.current_session()
            if session is None:
                return ReadArtifactOutput(
                    name=input.name,
                    content=(
                        'Artifacts-based tools are not available, as there is no active agent '
                        'session detected. Artifacts middleware only work when passed to an agent.'
                    ),
                    found=False,
                )

            artifacts = await session.get_artifacts()
            match = next((a for a in artifacts if a.name == input.name), None)
            if match is None:
                return ReadArtifactOutput(
                    name=input.name,
                    content=f'Artifact "{input.name}" not found.',
                    found=False,
                )

            return ReadArtifactOutput(
                name=input.name,
                content=extract_artifact_text(match),
                found=True,
            )

        tools.append(
            tool(
                read_artifact,
                name='read_artifact',
                description=(
                    'Reads the content of a named artifact from the session. '
                    'Use this to inspect artifacts produced by sub-agents or '
                    'previously created artifacts.'
                ),
            ).action()
        )

        if not self.config.readonly:

            async def write_artifact(input: WriteArtifactInput) -> WriteArtifactOutput:
                session = ctx.ai.current_session()
                if session is None:
                    return WriteArtifactOutput(status='Error: no active session.')

                await session.add_artifacts(Artifact(name=input.name, parts=[Part(TextPart(text=input.content))]))
                return WriteArtifactOutput(status=f'Artifact "{input.name}" saved successfully.')

            tools.append(
                tool(
                    write_artifact,
                    name='write_artifact',
                    description=(
                        'Creates or updates a named artifact in the session. '
                        'If an artifact with the same name already exists, it will be '
                        'replaced. Use this to produce files, reports, code, or other '
                        'deliverables.'
                    ),
                ).action()
            )

        return tools

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        session = ctx.ai.current_session()
        artifacts = await session.get_artifacts() if session is not None else []
        listing = build_artifact_listing(artifacts)
        params.options = inject_artifact_listing(params.options, listing)
        return await next_fn(params, ctx)
