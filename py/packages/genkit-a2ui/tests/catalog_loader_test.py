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

"""Pins: load a catalog, generate against it, and list it for the Developer UI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from genkit_a2ui import (
    A2UI_CATALOG_VALUE_TYPE,
    A2uiCatalog,
    A2uiCatalogComponent,
    A2uiParseError,
    Surfaces,
    SurfacesConfig,
    load_catalog,
    load_catalog_file,
    register_basic_catalog,
)
from helpers import (
    BASIC_CATALOG_ID,
    assert_finished_message,
    create_surface_ids,
    envelopes,
    fence_block,
    model_ok,
    request_system_text,
    setup,
    weather_fence,
)
from pydantic import ValidationError

BANNER_CATALOG = A2uiCatalog(
    id='https://example.com/catalogs/banner.json',
    components=(A2uiCatalogComponent(name='Banner', description='A banner.', props='title: string.'),),
)


def banner_fence(*, catalog_id: str = BANNER_CATALOG.id) -> str:
    return fence_block([
        {'createSurface': {'surfaceId': 'SURFACE_ID', 'catalogId': catalog_id}},
        {
            'updateComponents': {
                'surfaceId': 'SURFACE_ID',
                'components': [{'id': 'root', 'component': 'Banner', 'title': 'hi'}],
            }
        },
    ])


@pytest.mark.asyncio
async def test_generate_uses_the_bundled_catalog_when_nothing_is_registered() -> None:
    ai, pm = setup()
    pm.responses = [model_ok(weather_fence())]

    response = await ai.generate(model='programmableModel', prompt='weather', use=[Surfaces()])
    message = assert_finished_message(response)
    assert create_surface_ids(message.content)
    assert '- Text:' in request_system_text(pm)
    assert '- Banner:' not in request_system_text(pm)


@pytest.mark.asyncio
async def test_generate_uses_a_loaded_catalog() -> None:
    ai, pm = setup()
    load_catalog(ai, BANNER_CATALOG)
    pm.responses = [model_ok(banner_fence())]

    response = await ai.generate(
        model='programmableModel',
        prompt='banner',
        use=[Surfaces(catalog=BANNER_CATALOG.id)],
    )
    message = assert_finished_message(response)
    assert any(env.get('createSurface', {}).get('catalogId') == BANNER_CATALOG.id for env in envelopes(message.content))
    assert '- Banner:' in request_system_text(pm)
    assert '- Text:' not in request_system_text(pm)


@pytest.mark.asyncio
async def test_generate_default_stays_basic_when_a_custom_catalog_is_registered() -> None:
    ai, pm = setup()
    load_catalog(ai, BANNER_CATALOG)
    pm.responses = [model_ok(weather_fence())]

    response = await ai.generate(model='programmableModel', prompt='weather', use=[Surfaces()])
    message = assert_finished_message(response)
    assert create_surface_ids(message.content)
    assert '- Text:' in request_system_text(pm)
    assert '- Banner:' not in request_system_text(pm)


@pytest.mark.asyncio
async def test_generate_unknown_catalog_fails_before_the_model() -> None:
    ai, pm = setup()
    pm.responses = [model_ok(weather_fence())]

    with pytest.raises(ValueError, match='no catalog registered'):
        await ai.generate(
            model='programmableModel',
            prompt='banner',
            use=[Surfaces(catalog=BANNER_CATALOG.id)],
        )
    assert pm.last_request is None


@pytest.mark.asyncio
async def test_generate_catalog_basic_falls_back_without_register() -> None:
    ai, pm = setup()
    pm.responses = [model_ok(weather_fence())]

    response = await ai.generate(model='programmableModel', prompt='weather', use=[Surfaces(catalog='basic')])
    message = assert_finished_message(response)
    assert create_surface_ids(message.content)


@pytest.mark.asyncio
async def test_generate_strict_rejects_a_component_the_loaded_catalog_lacks() -> None:
    ai, pm = setup()
    load_catalog(ai, BANNER_CATALOG)
    pm.responses = [model_ok(weather_fence())]

    with pytest.raises(A2uiParseError, match='not in catalog'):
        await ai.generate(
            model='programmableModel',
            prompt='weather',
            use=[Surfaces(catalog=BANNER_CATALOG.id, validate='strict')],
        )


@pytest.mark.asyncio
async def test_generate_warn_drops_a_component_the_loaded_catalog_lacks() -> None:
    ai, pm = setup()
    load_catalog(ai, BANNER_CATALOG)
    pm.responses = [model_ok(weather_fence())]

    response = await ai.generate(
        model='programmableModel',
        prompt='weather',
        use=[Surfaces(catalog=BANNER_CATALOG.id)],
    )
    message = assert_finished_message(response)
    # createSurface can still land; the unknown-component update does not.
    assert not any('updateComponents' in env for env in envelopes(message.content))


def test_load_catalog_appears_in_the_registry_the_dev_ui_lists() -> None:
    ai, _ = setup()
    load_catalog(ai, BANNER_CATALOG)
    listed = ai.registry.list_values(A2UI_CATALOG_VALUE_TYPE)
    assert BANNER_CATALOG.id in listed
    assert listed[BANNER_CATALOG.id] == BANNER_CATALOG.as_value()


def test_register_basic_catalog_appears_in_the_registry_the_dev_ui_lists() -> None:
    ai, _ = setup()
    register_basic_catalog(ai)
    listed = ai.registry.list_values(A2UI_CATALOG_VALUE_TYPE)
    assert BASIC_CATALOG_ID in listed
    basic = listed[BASIC_CATALOG_ID]
    assert isinstance(basic, dict)
    components = basic.get('components')
    assert isinstance(components, list)
    assert 'Text' in {item['name'] for item in components if isinstance(item, dict)}


def test_load_catalog_file_registers_and_returns_the_catalog(tmp_path: Path) -> None:
    ai, _ = setup()
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(BANNER_CATALOG.as_value()), encoding='utf-8')
    loaded = load_catalog_file(ai, str(path))
    assert loaded == BANNER_CATALOG
    assert ai.registry.lookup_value(A2UI_CATALOG_VALUE_TYPE, BANNER_CATALOG.id) == BANNER_CATALOG.as_value()


def test_load_catalog_same_id_keeps_the_first() -> None:
    ai, _ = setup()
    load_catalog(ai, BANNER_CATALOG)
    other = A2uiCatalog(
        id=BANNER_CATALOG.id,
        components=(A2uiCatalogComponent(name='Other', description='x', props='y'),),
    )
    kept = load_catalog(ai, other)
    assert kept == BANNER_CATALOG
    stored = A2uiCatalog.from_value(ai.registry.lookup_value(A2UI_CATALOG_VALUE_TYPE, BANNER_CATALOG.id))
    assert stored == BANNER_CATALOG


def test_load_catalog_same_catalog_twice_is_ok() -> None:
    ai, _ = setup()
    assert load_catalog(ai, BANNER_CATALOG) == BANNER_CATALOG
    assert load_catalog(ai, BANNER_CATALOG) == BANNER_CATALOG


def test_load_catalog_raises_when_id_already_holds_something_else() -> None:
    ai, _ = setup()
    ai.registry.register_value(A2UI_CATALOG_VALUE_TYPE, BANNER_CATALOG.id, 'not-a-catalog')
    with pytest.raises(ValueError, match='is not a catalog'):
        load_catalog(ai, BANNER_CATALOG)


def test_load_catalog_file_names_the_path_when_the_file_is_not_utf8(tmp_path: Path) -> None:
    ai, _ = setup()
    path = tmp_path / 'catalog.json'
    path.write_bytes(b'\xff\xfe not utf-8')
    with pytest.raises(ValueError, match=str(path)) as exc_info:
        load_catalog_file(ai, str(path))
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_load_catalog_file_names_the_path_when_a_component_has_no_name(tmp_path: Path) -> None:
    path = tmp_path / 'catalog.json'
    path.write_text(
        json.dumps({'id': BANNER_CATALOG.id, 'components': [{'description': 'no name'}]}),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match=str(path)):
        load_catalog_file(setup()[0], str(path))


def test_a2ui_catalog_is_a_registry_id_string() -> None:
    assert Surfaces(catalog=BANNER_CATALOG.id).config.catalog == BANNER_CATALOG.id


def test_a2ui_catalog_object_is_not_a_knob() -> None:
    with pytest.raises(ValidationError):
        Surfaces(catalog=BANNER_CATALOG)


def test_a2ui_config_schema_lists_catalog_as_a_string() -> None:
    """The Developer UI picks a registered catalog by the same id generate uses."""
    props = SurfacesConfig.model_json_schema().get('properties', {})
    schema = props['catalog']
    types = {schema.get('type')} | {opt.get('type') for opt in schema.get('anyOf', [])}
    assert 'string' in types
