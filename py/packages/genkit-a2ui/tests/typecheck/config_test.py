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

"""Typed construction of SurfacesConfig.

pyright, pyrefly and ty must accept these snake_case kwargs with no
suppressions. The runtime asserts pin the wire dump.
"""

from genkit_a2ui import Surfaces, SurfacesConfig


def test_surfaces_config_snake_case_kwargs() -> None:
    """Field names type-check as kwargs; the dump keeps the validate and surfaceId wire names."""
    config = SurfacesConfig(instructions='none', validation='strict', surface_id='order-card', catalog='menu')

    assert config.model_dump(by_alias=True, exclude_none=True, mode='json') == {
        'instructions': 'none',
        'validate': 'strict',
        'surfaceId': 'order-card',
        'catalog': 'menu',
        'version': config.version,
    }


def test_surfaces_config_accepts_wire_names() -> None:
    """The validate and surfaceId JSON keys build the same config as the field names."""
    typed = SurfacesConfig(validation='off', surface_id='order-card')

    assert SurfacesConfig.model_validate({'validate': 'off', 'surfaceId': 'order-card'}) == typed
    assert Surfaces(validation='off', surface_id='order-card').config == typed
    assert Surfaces(config=typed).config.validation == 'off'
    assert Surfaces(validate='strict').config.validation == 'strict'
