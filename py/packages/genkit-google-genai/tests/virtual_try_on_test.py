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

"""Tests for the Virtual Try-On model: request shape, response shape, failures."""

import base64
import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai.models.virtual_try_on import VirtualTryOnConfig, VirtualTryOnModel, VirtualTryOnOutputOptions
from google.auth.credentials import AnonymousCredentials
from google.genai import models as genai_models, types as genai_types
from google.genai._api_client import BaseApiClient
from google.genai._common import recursive_dict_update
from google.genai.errors import APIError
from pydantic import ValidationError

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
    TextPart,
)

PNG_BYTES = b'\x89PNG\r\n\x1a\n-person'
PNG_B64 = base64.b64encode(PNG_BYTES).decode('ascii')
PERSON_URL = f'data:image/png;base64,{PNG_B64}'
PRODUCT_URL = 'gs://bucket/shirt.png'


def _media_part(url: str, part_type: str | None) -> Part:
    metadata = {'type': part_type} if part_type else None
    return Part(MediaPart(media=Media(url=url, content_type='image/png'), metadata=metadata))


def _request(
    *,
    parts: list[Part] | None = None,
    config: object | None = None,
    history: list[Message] | None = None,
) -> ModelRequest:
    content = (
        parts
        if parts is not None
        else [_media_part(PERSON_URL, 'personImage'), _media_part(PRODUCT_URL, 'productImage')]
    )
    messages = list(history or [])
    messages.append(Message(role=Role.USER, content=content))
    return ModelRequest(messages=messages, config=config)  # type: ignore[arg-type]


def _client(*, response: genai_types.RecontextImageResponse | None = None, vertexai: bool = True) -> MagicMock:
    client = MagicMock()
    client.vertexai = vertexai
    client.aio.models.recontext_image = AsyncMock(
        return_value=response if response is not None else genai_types.RecontextImageResponse()
    )
    return client


def _sdk_response(predictions: list[dict[str, Any]] | None) -> genai_types.RecontextImageResponse:
    """Build a response the way the SDK builds one from a Vertex predict body."""
    body = {} if predictions is None else {'predictions': predictions}
    return genai_types.RecontextImageResponse._from_response(  # noqa: SLF001
        response=genai_models._RecontextImageResponse_from_vertex(body),  # noqa: SLF001
        kwargs={},
    )


def _wire_body(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Render the :predict body the SDK would send for these call kwargs."""
    api_client = BaseApiClient(
        vertexai=True, project='test-project', location='us-central1', credentials=AnonymousCredentials()
    )
    config = call_kwargs.get('config')
    body = genai_models._RecontextImageParameters_to_vertex(  # noqa: SLF001
        api_client,
        genai_types._RecontextImageParameters(  # noqa: SLF001
            model=call_kwargs['model'],
            source=call_kwargs['source'],
            config=config,
        ),
    )
    extra_body = config.http_options.extra_body if config is not None and config.http_options else None
    if extra_body:
        recursive_dict_update(body, extra_body)
    body.pop('_url', None)
    return body


async def _generate(model: VirtualTryOnModel, request: ModelRequest) -> Any:  # noqa: ANN401
    return await model.generate(request, ActionRunContext())


class TestRequestShape:
    """What the driver hands the SDK, and what that becomes on the wire."""

    @pytest.mark.asyncio
    async def test_person_and_products_reach_the_sdk(self) -> None:
        """A data url is sent inline; a gs:// path is sent by reference."""
        client = _client()
        await _generate(
            VirtualTryOnModel('virtual-try-on-001', client),
            _request(
                parts=[
                    _media_part(PERSON_URL, 'personImage'),
                    _media_part(PRODUCT_URL, 'productImage'),
                    _media_part('gs://bucket/hat.png', 'productImage'),
                ]
            ),
        )

        source = client.aio.models.recontext_image.await_args.kwargs['source']
        assert source.person_image.image_bytes == PNG_BYTES
        assert source.person_image.mime_type is None
        assert [p.product_image.gcs_uri for p in source.product_images] == [
            'gs://bucket/shirt.png',
            'gs://bucket/hat.png',
        ]

    @pytest.mark.asyncio
    async def test_bare_base64_is_decoded(self) -> None:
        """An unprefixed base64 payload is treated as inline image data."""
        client = _client()
        await _generate(
            VirtualTryOnModel('virtual-try-on-001', client),
            _request(parts=[_media_part(PNG_B64, 'personImage'), _media_part(PRODUCT_URL, 'productImage')]),
        )

        source = client.aio.models.recontext_image.await_args.kwargs['source']
        assert source.person_image.image_bytes == PNG_BYTES

    @pytest.mark.asyncio
    async def test_line_wrapped_base64_is_decoded(self) -> None:
        """Whitespace inside the payload is not image data."""
        client = _client()
        image = PNG_BYTES * 8
        wrapped = base64.encodebytes(image).decode('ascii')
        assert '\n' in wrapped.strip()
        await _generate(
            VirtualTryOnModel('virtual-try-on-001', client),
            _request(
                parts=[
                    _media_part(f'data:image/png;base64,{wrapped}', 'personImage'),
                    _media_part(PRODUCT_URL, 'productImage'),
                ]
            ),
        )

        source = client.aio.models.recontext_image.await_args.kwargs['source']
        assert source.person_image.image_bytes == image

    @pytest.mark.asyncio
    async def test_url_safe_base64_is_decoded(self) -> None:
        """The URL-safe alphabet is accepted alongside the standard one."""
        client = _client()
        image = b'\xfb\xff\xbf' * 4
        encoded = base64.urlsafe_b64encode(image).decode('ascii')
        assert '-' in encoded and '_' in encoded
        await _generate(
            VirtualTryOnModel('virtual-try-on-001', client),
            _request(
                parts=[
                    _media_part(f'data:image/png;base64,{encoded}', 'personImage'),
                    _media_part(PRODUCT_URL, 'productImage'),
                ]
            ),
        )

        source = client.aio.models.recontext_image.await_args.kwargs['source']
        assert source.person_image.image_bytes == image

    @pytest.mark.asyncio
    async def test_unpadded_base64_is_decoded(self) -> None:
        """A payload whose encoder dropped the trailing '=' still decodes."""
        client = _client()
        image = PNG_BYTES[:-1]
        encoded = base64.b64encode(image).decode('ascii')
        assert encoded.endswith('=')
        await _generate(
            VirtualTryOnModel('virtual-try-on-001', client),
            _request(
                parts=[
                    _media_part(f'data:image/png;base64,{encoded.rstrip("=")}', 'personImage'),
                    _media_part(PRODUCT_URL, 'productImage'),
                ]
            ),
        )

        source = client.aio.models.recontext_image.await_args.kwargs['source']
        assert source.person_image.image_bytes == image

    @pytest.mark.asyncio
    async def test_only_the_last_message_is_read(self) -> None:
        """Earlier turns are history; the images come from the request being served."""
        client = _client()
        history = [Message(role=Role.USER, content=[_media_part('gs://old/person.png', 'personImage')])]
        await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request(history=history))

        source = client.aio.models.recontext_image.await_args.kwargs['source']
        assert source.person_image.gcs_uri is None
        assert source.person_image.image_bytes == PNG_BYTES

    @pytest.mark.asyncio
    async def test_untagged_and_other_tags_are_ignored(self) -> None:
        """Only personImage/productImage parts are inputs."""
        client = _client()
        await _generate(
            VirtualTryOnModel('virtual-try-on-001', client),
            _request(
                parts=[
                    Part(TextPart(text='dress her in this')),
                    _media_part('gs://bucket/ignored.png', None),
                    _media_part('gs://bucket/mask.png', 'mask'),
                    _media_part(PERSON_URL, 'personImage'),
                    _media_part(PRODUCT_URL, 'productImage'),
                ]
            ),
        )

        source = client.aio.models.recontext_image.await_args.kwargs['source']
        assert len(source.product_images) == 1

    @pytest.mark.asyncio
    async def test_golden_wire_body(self) -> None:
        """The full config renders the predict body Vertex expects."""
        client = _client()
        config = VirtualTryOnConfig.model_validate({
            'sampleCount': 2,
            'seed': 7,
            'baseSteps': 32,
            'storageUri': 'gs://out/',
            'personGeneration': 'allow_adult',
            'safetySetting': 'block_few',
            'outputOptions': {'mimeType': 'image/jpeg', 'compressionQuality': 80},
        })
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request(config=config))
            body = _wire_body(client.aio.models.recontext_image.await_args.kwargs)

        assert body == {
            'instances': [
                {
                    'personImage': {'image': {'bytesBase64Encoded': PNG_B64}},
                    'productImages': [{'image': {'gcsUri': PRODUCT_URL}}],
                }
            ],
            'parameters': {
                'sampleCount': 2,
                'seed': 7,
                'baseSteps': 32,
                'storageUri': 'gs://out/',
                'personGeneration': 'allow_adult',
                'safetySetting': 'block_few',
                'outputOptions': {'mimeType': 'image/jpeg', 'compressionQuality': 80},
            },
        }

    @pytest.mark.asyncio
    async def test_extra_config_keys_reach_the_wire(self) -> None:
        """A key the schema does not declare rides to parameters rather than being dropped."""
        client = _client()
        config = VirtualTryOnConfig.model_validate({'sampleCount': 1, 'addWatermark': True})
        await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request(config=config))

        body = _wire_body(client.aio.models.recontext_image.await_args.kwargs)
        assert body['parameters'] == {'sampleCount': 1, 'addWatermark': True}

    @pytest.mark.asyncio
    async def test_extra_output_option_lands_beside_the_declared_ones(self) -> None:
        """An extra nested under outputOptions merges with the declared keys."""
        client = _client()
        config = VirtualTryOnConfig.model_validate({'outputOptions': {'mimeType': 'image/jpeg', 'lossless': False}})
        await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request(config=config))

        body = _wire_body(client.aio.models.recontext_image.await_args.kwargs)
        assert body['parameters']['outputOptions'] == {'mimeType': 'image/jpeg', 'lossless': False}

    @pytest.mark.asyncio
    async def test_no_config_sends_no_config(self) -> None:
        """A request with no config leaves the parameters block off."""
        client = _client()
        await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request())

        assert client.aio.models.recontext_image.await_args.kwargs['config'] is None


class TestResponseShape:
    """What the driver returns for each prediction shape."""

    @pytest.mark.asyncio
    async def test_inline_image_becomes_one_data_url(self) -> None:
        """Bytes are base64-encoded exactly once."""
        client = _client(response=_sdk_response([{'bytesBase64Encoded': PNG_B64, 'mimeType': 'image/png'}]))
        response = await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request())

        assert response.finish_reason == FinishReason.STOP
        media = response.message.content[0].root.media
        assert media.url == f'data:image/png;base64,{PNG_B64}'
        assert media.content_type == 'image/png'

    @pytest.mark.asyncio
    async def test_two_predictions_become_two_media_parts(self) -> None:
        """Every generated image is a media part on the one model message."""
        client = _client(
            response=_sdk_response([
                {'bytesBase64Encoded': PNG_B64, 'mimeType': 'image/png'},
                {'gcsUri': 'gs://out/2.png', 'mimeType': 'image/png'},
            ])
        )
        response = await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request())

        assert response.finish_reason == FinishReason.STOP
        assert response.finish_message is None
        assert [part.root.media.url for part in response.message.content] == [
            f'data:image/png;base64,{PNG_B64}',
            'gs://out/2.png',
        ]
        assert response.usage.input_images == 2
        assert response.usage.output_images == 2
        assert response.usage.custom == {'generations': 2}

    @pytest.mark.asyncio
    async def test_missing_mime_type_defaults_to_png(self) -> None:
        """A prediction with no mimeType still yields a usable data url."""
        client = _client(response=_sdk_response([{'bytesBase64Encoded': PNG_B64}]))
        response = await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request())

        assert response.message.content[0].root.media.content_type == 'image/png'

    @pytest.mark.asyncio
    async def test_filtered_prediction_alongside_a_real_one(self) -> None:
        """An empty filtered image is skipped, not turned into an empty media part."""
        client = _client(
            response=_sdk_response([
                {'raiFilteredReason': 'blocked by safety'},
                {'bytesBase64Encoded': PNG_B64, 'mimeType': 'image/png'},
            ])
        )
        response = await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request())

        assert response.finish_reason == FinishReason.STOP
        assert len(response.message.content) == 1
        assert response.finish_message == 'blocked by safety'
        assert response.usage.custom == {'generations': 1}

    @pytest.mark.asyncio
    async def test_all_filtered_is_blocked(self) -> None:
        """When every sample is filtered, say so instead of returning an empty success."""
        client = _client(response=_sdk_response([{'raiFilteredReason': 'blocked by safety'}]))
        response = await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request())

        assert response.finish_reason == FinishReason.BLOCKED
        assert response.message.content == []
        assert response.finish_message == 'blocked by safety'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('predictions', [[], None])
    async def test_no_predictions_is_blocked(self, predictions: list[dict[str, Any]] | None) -> None:
        """An empty or absent predictions list is a block, not a success."""
        client = _client(response=_sdk_response(predictions))
        response = await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request())

        assert response.finish_reason == FinishReason.BLOCKED
        assert 'content filters' in (response.finish_message or '')

    @pytest.mark.asyncio
    async def test_streaming_context_returns_the_whole_response(self) -> None:
        """There is nothing to stream; the caller still gets the complete response."""
        client = _client(response=_sdk_response([{'bytesBase64Encoded': PNG_B64, 'mimeType': 'image/png'}]))
        chunks: list[object] = []
        ctx = ActionRunContext(streaming_callback=chunks.append)

        response = await VirtualTryOnModel('virtual-try-on-001', client).generate(_request(), ctx)

        assert response.finish_reason == FinishReason.STOP
        assert chunks == []


class TestRequestClient:
    """context.secrets and client knobs go on a request-scoped client, never on the wire."""

    @pytest.mark.asyncio
    async def test_empty_context_uses_plugin_client(self) -> None:
        """No secret and no knobs means no new client."""
        client = _client()

        with patch('genkit_google_genai.models.virtual_try_on.genai.Client') as ctor:
            await _generate(VirtualTryOnModel('virtual-try-on-001', client), _request())

        ctor.assert_not_called()
        client.aio.models.recontext_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_secrets_api_key_builds_an_express_client(self) -> None:
        """A tenant key replaces the plugin credentials, project, location and host."""
        plugin = _client()
        override = _client()
        model = VirtualTryOnModel(
            'virtual-try-on-001',
            plugin,
            client_kwargs={
                'vertexai': True,
                'project': 'plugin-project',
                'location': 'us-central1',
                'http_options': genai_types.HttpOptions(base_url='https://plugin.example'),
            },
        )

        with patch('genkit_google_genai.models.virtual_try_on.genai.Client', return_value=override) as ctor:
            await model.generate(_request(), ActionRunContext(context={'secrets': {'api_key': 'sk-tenant'}}))

        kwargs = ctor.call_args.kwargs
        assert kwargs['api_key'] == 'sk-tenant'
        assert kwargs['vertexai'] is True
        assert kwargs['credentials'] is None
        assert 'project' not in kwargs
        assert 'location' not in kwargs
        assert kwargs['http_options'].base_url is None
        plugin.aio.models.recontext_image.assert_not_called()
        override.aio.models.recontext_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_client_knobs_route_the_client_and_stay_off_the_wire(self) -> None:
        """location, baseUrl and apiVersion pick the client; the predict body never sees them."""
        plugin = _client()
        override = _client()
        model = VirtualTryOnModel(
            'virtual-try-on-001',
            plugin,
            client_kwargs={'vertexai': True, 'project': 'plugin-project', 'location': 'us-central1'},
        )
        config = VirtualTryOnConfig.model_validate({
            'sampleCount': 1,
            'location': 'europe-west4',
            'baseUrl': 'https://request.example',
            'apiVersion': 'v1beta1',
        })

        with patch('genkit_google_genai.models.virtual_try_on.genai.Client', return_value=override) as ctor:
            await _generate(model, _request(config=config))

        kwargs = ctor.call_args.kwargs
        assert kwargs['location'] == 'europe-west4'
        assert kwargs['project'] == 'plugin-project'
        assert kwargs['http_options'].base_url == 'https://request.example'
        assert kwargs['http_options'].api_version == 'v1beta1'
        body = _wire_body(override.aio.models.recontext_image.await_args.kwargs)
        assert body['parameters'] == {'sampleCount': 1}

    @pytest.mark.asyncio
    async def test_multi_regional_location_picks_its_host(self) -> None:
        """A multi-region location is served from its own host, not the regional pattern."""
        override = _client()
        model = VirtualTryOnModel('virtual-try-on-001', _client(), client_kwargs={'vertexai': True, 'project': 'p'})
        config = VirtualTryOnConfig.model_validate({'location': 'eu'})

        with patch('genkit_google_genai.models.virtual_try_on.genai.Client', return_value=override) as ctor:
            await _generate(model, _request(config=config))

        kwargs = ctor.call_args.kwargs
        assert kwargs['location'] == 'eu'
        assert kwargs['http_options'].base_url == 'https://aiplatform.eu.rep.googleapis.com'

    @pytest.mark.asyncio
    async def test_location_override_keeps_a_pinned_base_url(self) -> None:
        """A plugin constructed with an explicit host keeps it when only the location changes."""
        override = _client()
        model = VirtualTryOnModel(
            'virtual-try-on-001',
            _client(),
            client_kwargs={
                'vertexai': True,
                'project': 'plugin-project',
                'http_options': genai_types.HttpOptions(base_url='https://egress-proxy.example'),
            },
            base_url_pinned=True,
        )
        config = VirtualTryOnConfig.model_validate({'location': 'europe-west4'})

        with patch('genkit_google_genai.models.virtual_try_on.genai.Client', return_value=override) as ctor:
            await _generate(model, _request(config=config))

        kwargs = ctor.call_args.kwargs
        assert kwargs['location'] == 'europe-west4'
        assert kwargs['http_options'].base_url == 'https://egress-proxy.example'

    def test_compression_quality_is_bounded(self) -> None:
        """The JPEG quality range is 0-100; anything else fails before the request is built."""
        for quality in (-1, 101):
            with pytest.raises(ValidationError):
                VirtualTryOnOutputOptions.model_validate({'compressionQuality': quality})
        assert VirtualTryOnOutputOptions.model_validate({'compressionQuality': 100}).compression_quality == 100

    def test_client_knobs_are_typed(self) -> None:
        """The knobs are declared fields, so a wrong type fails validation before it reaches the SDK."""
        with pytest.raises(ValidationError):
            VirtualTryOnConfig.model_validate({'location': 123})
        assert VirtualTryOnConfig.model_validate({'baseUrl': 'https://request.example'}).base_url == (
            'https://request.example'
        )

    @pytest.mark.asyncio
    async def test_only_client_knobs_sends_no_config(self) -> None:
        """A config made of nothing but client knobs leaves the parameters block off."""
        override = _client()
        model = VirtualTryOnModel('virtual-try-on-001', _client(), client_kwargs={'vertexai': True})
        config = VirtualTryOnConfig.model_validate({'baseUrl': 'https://request.example'})

        with patch('genkit_google_genai.models.virtual_try_on.genai.Client', return_value=override):
            await _generate(model, _request(config=config))

        assert override.aio.models.recontext_image.await_args.kwargs['config'] is None

    @pytest.mark.asyncio
    async def test_key_outside_secrets_is_refused(self) -> None:
        """A key on the top-level context does not fall through to the plugin client."""
        model = VirtualTryOnModel('virtual-try-on-001', _client())

        with pytest.raises(GenkitError) as exc_info:
            await model.generate(_request(), ActionRunContext(context={'api_key': 'sk-tenant'}))
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'context.secrets' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_client_construction_failure_is_invalid_argument(self) -> None:
        """Bad override kwargs surface as a caller error, not an SDK traceback."""
        model = VirtualTryOnModel('virtual-try-on-001', _client())

        with (
            patch('genkit_google_genai.models.virtual_try_on.genai.Client', side_effect=ValueError('bad kwargs')),
            pytest.raises(GenkitError) as exc_info,
        ):
            await model.generate(_request(), ActionRunContext(context={'secrets': {'api_key': 'sk-tenant'}}))
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'bad kwargs' in str(exc_info.value)


class TestFailureModes:
    """Everything that must fail with a named Genkit status."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('missing', ['personImage', 'productImage'])
    async def test_missing_media_names_the_tag(self, missing: str) -> None:
        """The error says which metadata tag the caller left out."""
        kept = 'productImage' if missing == 'personImage' else 'personImage'
        model = VirtualTryOnModel('virtual-try-on-001', _client())

        with pytest.raises(GenkitError) as exc_info:
            await _generate(model, _request(parts=[_media_part(PRODUCT_URL, kept)]))
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert missing in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('url', ['https://example.com/x.png', 'http://example.com/x.png'])
    async def test_http_urls_are_refused(self, url: str) -> None:
        """http(s) images are not fetched; the caller is pointed at Cloud Storage."""
        model = VirtualTryOnModel('virtual-try-on-001', _client())

        with pytest.raises(GenkitError) as exc_info:
            await _generate(
                model, _request(parts=[_media_part(url, 'personImage'), _media_part(PRODUCT_URL, 'productImage')])
            )
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'Cloud Storage' in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'url', ['data:image/png;base64,!!!not-base64!!!', 'data:image/png;base64,', 'data:nocomma']
    )
    async def test_undecodable_payloads_are_invalid_argument(self, url: str) -> None:
        """A broken data url is a caller error, not a binascii traceback."""
        model = VirtualTryOnModel('virtual-try-on-001', _client())

        with pytest.raises(GenkitError) as exc_info:
            await _generate(
                model, _request(parts=[_media_part(url, 'personImage'), _media_part(PRODUCT_URL, 'productImage')])
            )
        assert exc_info.value.status == 'INVALID_ARGUMENT'

    @pytest.mark.asyncio
    async def test_non_vertex_client_is_failed_precondition(self) -> None:
        """The Gemini API backend does not serve this model."""
        model = VirtualTryOnModel('virtual-try-on-001', _client(vertexai=False))

        with pytest.raises(GenkitError) as exc_info:
            await _generate(model, _request())
        assert exc_info.value.status == 'FAILED_PRECONDITION'

    @pytest.mark.asyncio
    async def test_dict_config_is_refused(self) -> None:
        """Action hands the driver the family instance; a raw dict means it did not."""
        model = VirtualTryOnModel('virtual-try-on-001', _client())

        with pytest.raises(GenkitError) as exc_info:
            await _generate(model, _request(config={'sampleCount': 1}))
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'family schema instance' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_key_on_config_is_refused(self) -> None:
        """A tenant key belongs on context.secrets."""
        model = VirtualTryOnModel('virtual-try-on-001', _client())
        config = VirtualTryOnConfig.model_validate({'apiKey': 'tenant-key'})

        with pytest.raises(GenkitError) as exc_info:
            await _generate(model, _request(config=config))
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'context.secrets' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_sdk_field_raises_invalid_argument(self) -> None:
        """SDK type errors become a named INVALID_ARGUMENT."""
        model = VirtualTryOnModel('virtual-try-on-001', _client())
        config = VirtualTryOnConfig.model_construct(sample_count='nope')

        with pytest.raises(GenkitError) as exc_info:
            await _generate(model, _request(config=config))
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'number_of_images' in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('code', 'status'),
        [
            (400, 'INVALID_ARGUMENT'),
            (401, 'UNAUTHENTICATED'),
            (403, 'PERMISSION_DENIED'),
            (404, 'NOT_FOUND'),
            (429, 'RESOURCE_EXHAUSTED'),
            (503, 'UNAVAILABLE'),
        ],
    )
    async def test_api_errors_keep_their_status(self, code: int, status: str) -> None:
        """An HTTP failure keeps the status a caller can retry on."""
        client = _client()
        client.aio.models.recontext_image = AsyncMock(side_effect=APIError(code, {'error': {'message': 'boom'}}))
        model = VirtualTryOnModel('virtual-try-on-001', client)

        with pytest.raises(GenkitError) as exc_info:
            await _generate(model, _request())
        assert exc_info.value.status == status
