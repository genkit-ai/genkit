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

"""`OllamaConfig` rejects typos; `extra` is merged into `options` as-is."""

import pytest
from genkit_ollama._models import OllamaConfig, OllamaModel
from pydantic import ValidationError


def test_unknown_key_raises() -> None:
    """`{'repeatPenalty': 1.1}` is no longer a silent passthrough; it raises."""
    with pytest.raises(ValidationError, match='repeatPenalty'):
        OllamaConfig.model_validate({'repeatPenalty': 1.1})


def test_extra_merges_into_options_verbatim() -> None:
    """`extra={'repeat_penalty': 1.1}` lands in `options` with its key unchanged."""
    options = OllamaModel.build_request_options(OllamaConfig(num_ctx=4096, extra={'repeat_penalty': 1.1}))

    assert options['repeat_penalty'] == 1.1
    assert options['num_ctx'] == 4096
    assert 'extra' not in options


def test_extra_colliding_key_wins() -> None:
    """`extra` is merged last, so it overrides the declared field."""
    options = OllamaModel.build_request_options(OllamaConfig(num_ctx=4096, extra={'num_ctx': 8192}))

    assert options['num_ctx'] == 8192


def test_dumped_dict_extra_merges_too() -> None:
    """A config the framework already dumped to a dict reads `extra` the same way."""
    options = OllamaModel.build_request_options({'temperature': 0.2, 'extra': {'mirostat': 2}})

    assert options['mirostat'] == 2
    assert 'extra' not in options
