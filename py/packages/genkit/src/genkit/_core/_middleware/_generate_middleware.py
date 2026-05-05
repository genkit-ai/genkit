# Copyright 2025 Google LLC
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

"""Middleware descriptor helpers and registry hooks.

Stock middleware descriptors (retry, fallback, …) are registered in a follow-up change;
this module re-exports ``new_middleware`` and exposes empty built-in registration for
``Genkit``.
"""

from __future__ import annotations

from genkit._core._registry import Registry

from ._base import (
    MiddlewareDesc,
    new_middleware,
)


def builtin_middleware_descs() -> list[MiddlewareDesc]:
    """Stock middleware descriptors; empty until the built-in middleware PR lands."""
    return []


def register_builtin_middleware(registry: Registry) -> None:
    """Register built-in middleware descriptors on the given registry."""
    for d in builtin_middleware_descs():
        registry.register_value('middleware', d.name, d)
