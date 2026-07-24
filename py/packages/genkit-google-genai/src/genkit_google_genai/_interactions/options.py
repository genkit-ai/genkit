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

"""Small option types for Interactions-backed Google AI models."""

from __future__ import annotations

from typing import Literal, TypedDict

ResponseModality = Literal['text', 'image', 'audio']


class ClientOptions(TypedDict, total=False):
    """HTTP settings reconstructed across background poll calls.

    Includes api_key so check/cancel can reuse a per-request override from start,
    matching the JS Operation.metadata.clientOptions shape.
    """

    api_key: str
    api_version: str
    base_url: str
    custom_headers: dict[str, str]
    timeout: float
