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

"""Pin OutputConfig's Python name vs wire key."""

import pytest
from pydantic import ValidationError

from genkit._core._schema import to_json_schema
from genkit._core._typing import OutputConfig

_SAMPLE = {'type': 'object', 'properties': {'name': {'type': 'string'}}}


def test_python_name_dumps_wire_schema() -> None:
    dumped = OutputConfig(json_schema=_SAMPLE).model_dump()
    assert dumped == {'schema': _SAMPLE}


def test_accepts_wire_schema_key() -> None:
    cfg = OutputConfig.model_validate({'schema': _SAMPLE})
    assert cfg.json_schema == _SAMPLE
    assert cfg.model_dump() == {'schema': _SAMPLE}


def test_accepts_python_name_on_validate() -> None:
    cfg = OutputConfig.model_validate({'json_schema': _SAMPLE})
    assert cfg.json_schema == _SAMPLE
    assert cfg.model_dump() == {'schema': _SAMPLE}


def test_advertised_schema_uses_wire_key() -> None:
    props = to_json_schema(OutputConfig)['properties']
    assert 'schema' in props
    assert 'json_schema' not in props
    assert 'jsonSchema' not in props
    assert 'schema_' not in props


def test_rejects_old_python_name() -> None:
    with pytest.raises(ValidationError, match='schema_'):
        OutputConfig.model_validate({'schema_': _SAMPLE})
