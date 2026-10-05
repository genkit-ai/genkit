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

"""What a model author can declare in Supports."""

from typing import Any

import pytest
from pydantic import ValidationError

from genkit.model import Supports


def test_supports_output_accepts_text_json_and_media() -> None:
    """`Supports(output=['text', 'json', 'media'])` validates."""
    supports = Supports(output=['text', 'json', 'media'])

    assert supports.output == ['text', 'json', 'media']


def test_supports_output_json_mode_raises_validation_error() -> None:
    """`Supports(output=['json_mode'])` fails at construction."""
    output: Any = ['json_mode']

    with pytest.raises(ValidationError, match="'text', 'json' or 'media'"):
        Supports(output=output)


def test_supports_output_typo_raises_validation_error() -> None:
    """`Supports(output=['jsno'])` fails at construction."""
    output: Any = ['jsno']

    with pytest.raises(ValidationError, match="'text', 'json' or 'media'"):
        Supports(output=output)


def test_supports_output_unset_stays_none() -> None:
    """Control: omitting `output` is still allowed."""
    supports = Supports(multiturn=True)

    assert supports.output is None


def test_supports_context_field_raises_validation_error() -> None:
    """`Supports(context=True)` fails at construction."""
    fields: dict[str, Any] = {'context': True}

    with pytest.raises(ValidationError, match='context'):
        Supports(**fields)
