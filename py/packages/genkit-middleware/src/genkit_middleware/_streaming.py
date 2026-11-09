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

"""Track whether a model attempt already streamed chunks to the caller."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from genkit import ModelResponseChunk
from genkit.middleware import MiddlewareContext


class StreamWatch:
    """Counts chunks one model attempt sent downstream."""

    __slots__ = ('chunks',)

    def __init__(self) -> None:
        self.chunks = 0

    @property
    def committed(self) -> bool:
        """True once the caller has seen part of this attempt's answer."""
        return self.chunks > 0


@contextlib.contextmanager
def watch_stream(ctx: MiddlewareContext) -> Iterator[StreamWatch]:
    """Count chunks sent through ``ctx`` while the block runs.

    Retrying or failing over after chunks reached the caller would splice a
    second answer onto a partial first one, so callers check ``committed``.
    """
    watch = StreamWatch()
    inner = ctx.on_chunk
    if inner is None:
        yield watch
        return

    def counting(chunk: ModelResponseChunk) -> None:
        watch.chunks += 1
        inner(chunk)

    previous = ctx.replace_on_chunk(counting)
    try:
        yield watch
    finally:
        ctx.replace_on_chunk(previous)
