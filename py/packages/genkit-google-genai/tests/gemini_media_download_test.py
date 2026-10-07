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

"""What Gemini receives for each kind of media URL passed to `ai.generate`."""

import base64
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from genkit_google_genai import GoogleAI
from genkit_google_genai._google import GenaiModels
from google import genai

from genkit import FinishReason, Genkit, Message, Part, Role

MB = 1024 * 1024
CAP = 20 * MB
PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100


@pytest.fixture
def gemini() -> Iterator[tuple[Genkit, MagicMock]]:
    """A Genkit app on GoogleAI whose Gemini API calls are recorded, not sent."""
    with (
        patch('genkit_google_genai._google.genai.client.Client') as client_cls,
        patch('genkit_google_genai._google._list_genai_models', return_value=GenaiModels()),
    ):
        sdk = client_cls.return_value
        sdk.vertexai = False
        sdk.aio.models.generate_content = AsyncMock(
            return_value=genai.types.GenerateContentResponse(
                candidates=[
                    genai.types.Candidate(
                        content=genai.types.Content(role='model', parts=[genai.types.Part(text='ok')]),
                        finish_reason=genai.types.FinishReason.STOP,
                    )
                ]
            )
        )
        yield Genkit(plugins=[GoogleAI(api_key='test-key')]), sdk


def _serve(handler: Callable[[httpx.Request], httpx.Response]) -> Any:  # noqa: ANN401
    """Route the plugin's media downloads to `handler` instead of the network."""

    def make_client(*, cache_key: str, **kwargs: Any) -> httpx.AsyncClient:  # noqa: ANN401
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    return patch('genkit_google_genai._models._utils.get_cached_client', side_effect=make_client)


def _no_download() -> Any:  # noqa: ANN401
    return patch('genkit_google_genai._models._utils.get_cached_client', side_effect=AssertionError('downloaded'))


async def _chunks(total: int, consumed: list[int] | None = None) -> AsyncIterator[bytes]:
    sent = 0
    while sent < total:
        size = min(MB, total - sent)
        sent += size
        if consumed is not None:
            consumed.append(size)
        yield b'\x00' * size


async def _describe(ai: Genkit, url: str, content_type: str = 'image/png') -> Any:  # noqa: ANN401
    return await ai.generate(
        model='googleai/gemini-2.5-flash',
        messages=[
            Message(
                role=Role.USER,
                content=[Part.from_text('describe'), Part.from_media(url=url, content_type=content_type)],
            )
        ],
    )


def _sent_media(sdk: MagicMock) -> genai.types.Part:
    contents = sdk.aio.models.generate_content.call_args.kwargs['contents']
    return contents[-1].parts[-1]


async def _over_cap(ai: Genkit, sdk: MagicMock, url: str, content_type: str = 'video/mp4') -> str:
    """Generate on an over-cap URL fails with INVALID_ARGUMENT and never calls Gemini; returns the message."""
    response = await _describe(ai, url, content_type)
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.finish_message is not None
    assert 'larger than 20MB' in response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
    sdk.aio.models.generate_content.assert_not_called()
    return response.finish_message


@pytest.mark.asyncio
async def test_generate_gemini_http_image_under_cap_is_inlined(gemini: tuple[Genkit, MagicMock]) -> None:
    """A 1MB image URL is downloaded and sent as inline data."""
    ai, sdk = gemini
    body = PNG + b'\x00' * MB

    with _serve(lambda request: httpx.Response(200, headers={'content-type': 'image/png'}, content=body)):
        response = await _describe(ai, 'https://example.com/cat.png')

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'ok'
    sent = _sent_media(sdk)
    assert sent.inline_data is not None
    assert sent.inline_data.data == body
    assert sent.inline_data.mime_type == 'image/png'


@pytest.mark.asyncio
async def test_generate_gemini_http_media_exactly_20mb_is_inlined(gemini: tuple[Genkit, MagicMock]) -> None:
    """A body of exactly 20MB is sent whole; the cap only rejects bytes past it."""
    ai, sdk = gemini

    with _serve(lambda request: httpx.Response(200, headers={'content-type': 'video/mp4'}, content=b'\x00' * CAP)):
        response = await _describe(ai, 'https://example.com/clip.mp4', 'video/mp4')

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'ok'
    sent = _sent_media(sdk)
    assert sent.inline_data is not None
    assert sent.inline_data.data is not None
    assert len(sent.inline_data.data) == CAP


@pytest.mark.asyncio
async def test_generate_gemini_http_media_over_cap_raises_invalid_argument(gemini: tuple[Genkit, MagicMock]) -> None:
    """A 21MB body raises INVALID_ARGUMENT before the model is called, instead of sending a truncated file."""
    ai, sdk = gemini

    with _serve(lambda request: httpx.Response(200, content=b'\x00' * (21 * MB))):
        message = await _over_cap(ai, sdk, 'https://example.com/huge.mp4')

    assert 'https://example.com/huge.mp4' in message


@pytest.mark.asyncio
async def test_generate_gemini_http_media_over_cap_error_omits_query_string(gemini: tuple[Genkit, MagicMock]) -> None:
    """The over-cap error names the host and path but not a signed URL's query string."""
    ai, sdk = gemini
    url = 'https://storage.example.com/bucket/huge.mp4?X-Goog-Signature=s3cr3t&X-Goog-Credential=me'

    with _serve(lambda request: httpx.Response(200, content=b'\x00' * (21 * MB))):
        message = await _over_cap(ai, sdk, url)

    assert 'https://storage.example.com/bucket/huge.mp4' in message
    assert 's3cr3t' not in message
    assert 'X-Goog-Credential' not in message


@pytest.mark.asyncio
async def test_generate_gemini_http_media_with_large_content_length_raises_before_body(
    gemini: tuple[Genkit, MagicMock],
) -> None:
    """A `Content-Length` over 20MB fails without reading the body."""
    ai, sdk = gemini
    consumed: list[int] = []

    with _serve(
        lambda request: httpx.Response(
            200, headers={'content-length': str(21 * MB)}, content=_chunks(21 * MB, consumed)
        )
    ):
        await _over_cap(ai, sdk, 'https://example.com/huge.mp4')

    assert consumed == []


@pytest.mark.asyncio
async def test_generate_gemini_http_media_without_content_length_over_cap_raises(
    gemini: tuple[Genkit, MagicMock],
) -> None:
    """A chunked 21MB body with no `Content-Length` raises once it passes 20MB."""
    ai, sdk = gemini

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, content=_chunks(21 * MB))
        assert 'content-length' not in response.headers
        return response

    with _serve(handler):
        await _over_cap(ai, sdk, 'https://example.com/huge.mp4')


@pytest.mark.asyncio
async def test_generate_gemini_http_media_with_lying_content_length_stops_at_cap(
    gemini: tuple[Genkit, MagicMock],
) -> None:
    """A small `Content-Length` over a huge body still stops reading just past 20MB."""
    ai, sdk = gemini
    consumed: list[int] = []

    with _serve(
        lambda request: httpx.Response(200, headers={'content-length': '1024'}, content=_chunks(100 * MB, consumed))
    ):
        await _over_cap(ai, sdk, 'https://example.com/huge.mp4')

    assert CAP < sum(consumed) <= CAP + MB


@pytest.mark.asyncio
async def test_generate_gemini_http_image_follows_redirect(gemini: tuple[Genkit, MagicMock]) -> None:
    """A 302 to the image is followed and the image is inlined."""
    ai, sdk = gemini

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/old.png':
            return httpx.Response(302, headers={'location': 'https://cdn.example.com/cat.png'})
        return httpx.Response(200, headers={'content-type': 'image/png'}, content=PNG)

    with _serve(handler):
        response = await _describe(ai, 'https://example.com/old.png')

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'ok'
    sent = _sent_media(sdk)
    assert sent.inline_data is not None
    assert sent.inline_data.data == PNG


@pytest.mark.asyncio
async def test_generate_gemini_http_image_on_loopback_is_downloaded(gemini: tuple[Genkit, MagicMock]) -> None:
    """A `127.0.0.1` URL is still fetched; vetting user-supplied URLs is up to the app."""
    ai, sdk = gemini
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, headers={'content-type': 'image/png'}, content=PNG)

    with _serve(handler):
        response = await _describe(ai, 'http://127.0.0.1:8080/cat.png')

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'ok'
    assert hosts == ['127.0.0.1']
    sent = _sent_media(sdk)
    assert sent.inline_data is not None
    assert sent.inline_data.data == PNG


@pytest.mark.asyncio
async def test_generate_gemini_large_data_uri_is_sent_without_download(gemini: tuple[Genkit, MagicMock]) -> None:
    """A 25MB `data:` URL is sent whole as inline data and never makes a request."""
    ai, sdk = gemini
    body = b'\x00' * (25 * MB)
    url = 'data:video/mp4;base64,' + base64.b64encode(body).decode()

    with _no_download():
        response = await _describe(ai, url, 'video/mp4')

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'ok'
    sent = _sent_media(sdk)
    assert sent.inline_data is not None
    assert sent.inline_data.data == body


@pytest.mark.asyncio
async def test_generate_gemini_gs_uri_is_sent_as_file_uri(gemini: tuple[Genkit, MagicMock]) -> None:
    """A `gs://` URI is passed to Gemini as a file URI, not downloaded."""
    ai, sdk = gemini

    with _no_download():
        response = await _describe(ai, 'gs://bucket/cat.png')

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'ok'
    sent = _sent_media(sdk)
    assert sent.inline_data is None
    assert sent.file_data is not None
    assert sent.file_data.file_uri == 'gs://bucket/cat.png'


@pytest.mark.asyncio
async def test_generate_gemini_youtube_url_is_sent_as_file_uri(gemini: tuple[Genkit, MagicMock]) -> None:
    """A YouTube URL is passed to Gemini as a file URI, not downloaded."""
    ai, sdk = gemini
    url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

    with _no_download():
        response = await _describe(ai, url, 'video/mp4')

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'ok'
    sent = _sent_media(sdk)
    assert sent.inline_data is None
    assert sent.file_data is not None
    assert sent.file_data.file_uri == url


@pytest.mark.asyncio
async def test_generate_gemini_local_file_path_media_is_not_opened(
    gemini: tuple[Genkit, MagicMock], tmp_path: Path
) -> None:
    """A path to a real file on the server is passed as a file URI; its bytes are never read or sent."""
    ai, sdk = gemini
    secret = tmp_path / 'cat.png'
    secret.write_bytes(b'server-side secret')

    for url in (str(secret), secret.as_uri()):
        with _no_download():
            response = await _describe(ai, url)

        assert response.finish_reason == FinishReason.STOP
        assert response.text == 'ok'
        sent = _sent_media(sdk)
        assert sent.inline_data is None
        assert sent.file_data is not None
        assert sent.file_data.file_uri == url
