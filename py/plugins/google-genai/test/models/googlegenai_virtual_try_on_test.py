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

"""Tests for the Virtual Try-On model implementation."""

import base64

import pytest
from google.genai.errors import ClientError
from pytest_mock import MockerFixture

from genkit import (
    ActionRunContext,
    FinishReason,
    GenkitError,
    Media,
    MediaPart,
    Message,
    ModelRequest,
    Part,
    Role,
)
from genkit.plugins.google_genai.models.virtual_try_on import (
    PART_METADATA_TYPE_PERSON_IMAGE,
    PART_METADATA_TYPE_PRODUCT_IMAGE,
    VirtualTryOnConfig,
    VirtualTryOnModel,
    VirtualTryOnVersion,
    _extract_media_by_type,
    _to_virtual_try_on_request,
    is_virtual_try_on_model,
)


def _person_part(url: str) -> Part:
    return Part(
        root=MediaPart(
            media=Media(url=url, content_type='image/png'),
            metadata={'type': PART_METADATA_TYPE_PERSON_IMAGE},
        )
    )


def _product_part(url: str) -> Part:
    return Part(
        root=MediaPart(
            media=Media(url=url, content_type='image/png'),
            metadata={'type': PART_METADATA_TYPE_PRODUCT_IMAGE},
        )
    )


def _request_with(parts: list[Part]) -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=parts)])


def test_is_virtual_try_on_model() -> None:
    """is_virtual_try_on_model recognises virtual-try-on-* names."""
    assert is_virtual_try_on_model('virtual-try-on-001')
    assert is_virtual_try_on_model('virtual-try-on-future')
    assert not is_virtual_try_on_model('imagen-4.0-generate-001')
    assert not is_virtual_try_on_model('gemini-2.5-flash')


def test_extract_media_by_type_gcs() -> None:
    """gs:// URIs are passed through as gcsUri."""
    req = _request_with([_person_part('gs://bucket/person.png')])
    got = _extract_media_by_type(req, PART_METADATA_TYPE_PERSON_IMAGE)
    assert got == [{'image': {'gcsUri': 'gs://bucket/person.png'}}]


def test_extract_media_by_type_data_base64() -> None:
    """data:*;base64,* URIs are extracted as bytesBase64Encoded."""
    payload = base64.b64encode(b'\x89PNG').decode('ascii')
    url = f'data:image/png;base64,{payload}'
    req = _request_with([_product_part(url)])
    got = _extract_media_by_type(req, PART_METADATA_TYPE_PRODUCT_IMAGE)
    assert got == [{'image': {'bytesBase64Encoded': payload}}]


def test_extract_media_by_type_ignores_other_types() -> None:
    """Parts whose metadata.type does not match are skipped."""
    req = _request_with([_person_part('gs://b/p.png')])
    got = _extract_media_by_type(req, PART_METADATA_TYPE_PRODUCT_IMAGE)
    assert got == []


def test_to_virtual_try_on_request_requires_person() -> None:
    """Missing personImage raises a descriptive ValueError."""
    req = _request_with([_product_part('gs://b/shirt.png')])
    with pytest.raises(ValueError, match='personImage'):
        _to_virtual_try_on_request(req, None)


def test_to_virtual_try_on_request_requires_product() -> None:
    """Missing productImage raises a descriptive ValueError."""
    req = _request_with([_person_part('gs://b/person.png')])
    with pytest.raises(ValueError, match='productImage'):
        _to_virtual_try_on_request(req, None)


def test_to_virtual_try_on_request_shape() -> None:
    """Resulting body has one instance with personImage and all productImages."""
    req = _request_with([
        _person_part('gs://b/person.png'),
        _product_part('gs://b/shirt.png'),
        _product_part('gs://b/hat.png'),
    ])
    cfg = VirtualTryOnConfig(sample_count=2, person_generation='allow_adult')
    body = _to_virtual_try_on_request(req, cfg)
    assert body == {
        'instances': [
            {
                'personImage': {'image': {'gcsUri': 'gs://b/person.png'}},
                'productImages': [
                    {'image': {'gcsUri': 'gs://b/shirt.png'}},
                    {'image': {'gcsUri': 'gs://b/hat.png'}},
                ],
            }
        ],
        'parameters': {
            'sampleCount': 2,
            'personGeneration': 'allow_adult',
        },
    }


@pytest.mark.asyncio
async def test_generate_blocked_when_no_predictions(mocker: MockerFixture) -> None:
    """Empty predictions should surface as a FinishReasonBlocked response, not raise."""
    client = mocker.MagicMock()
    client._api_client.vertexai = True

    class _FakeResp:
        body = '{"predictions": []}'

    client._api_client.async_request = mocker.AsyncMock(return_value=_FakeResp())

    req = _request_with([
        _person_part('gs://b/person.png'),
        _product_part('gs://b/shirt.png'),
    ])
    model = VirtualTryOnModel(VirtualTryOnVersion.VIRTUAL_TRY_ON_001, client)
    resp = await model.generate(req, ActionRunContext())
    assert resp.finish_reason == FinishReason.BLOCKED
    assert resp.message is not None
    assert resp.message.content == []


@pytest.mark.asyncio
async def test_generate_maps_client_error_to_genkit_error(mocker: MockerFixture) -> None:
    """HTTP errors from the direct predict call should use Genkit status names."""
    client = mocker.MagicMock()
    client._api_client.vertexai = True
    client._api_client.async_request = mocker.AsyncMock(
        side_effect=ClientError(429, {'error': {'message': 'quota exceeded'}})
    )

    req = _request_with([
        _person_part('gs://b/person.png'),
        _product_part('gs://b/shirt.png'),
    ])
    model = VirtualTryOnModel(VirtualTryOnVersion.VIRTUAL_TRY_ON_001, client)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(req, ActionRunContext())

    assert exc_info.value.status == 'RESOURCE_EXHAUSTED'
    assert exc_info.value.original_message == 'quota exceeded'


@pytest.mark.asyncio
async def test_generate_emits_media_parts(mocker: MockerFixture) -> None:
    """A non-empty predict response is converted into MediaPart data URLs."""
    client = mocker.MagicMock()
    client._api_client.vertexai = True

    image_b64 = base64.b64encode(b'\x89PNG\r\n\x1a\n').decode('ascii')

    class _FakeResp:
        body = '{"predictions": [{"bytesBase64Encoded": "' + image_b64 + '", "mimeType": "image/png"}]}'

    client._api_client.async_request = mocker.AsyncMock(return_value=_FakeResp())

    req = _request_with([
        _person_part('gs://b/person.png'),
        _product_part('gs://b/shirt.png'),
    ])
    model = VirtualTryOnModel(VirtualTryOnVersion.VIRTUAL_TRY_ON_001, client)
    resp = await model.generate(req, ActionRunContext())

    assert resp.message is not None
    assert len(resp.message.content) == 1
    part = resp.message.content[0].root
    assert isinstance(part, MediaPart)
    assert part.media.content_type == 'image/png'
    assert part.media.url == f'data:image/png;base64,{image_b64}'
