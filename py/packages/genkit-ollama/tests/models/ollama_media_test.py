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

"""What an Ollama vision model receives for each kind of media string passed to `ai.generate`."""

import base64
import gzip
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import ollama as ollama_api
import pytest
from genkit_ollama import Ollama
from genkit_ollama._models import _image_fetch_client

from genkit import FinishReason, Genkit, Message, Part, Role

MB = 1024 * 1024
PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
_GZIP_25MB_ZEROS = gzip.compress(b'\x00' * (25 * MB))


@pytest.fixture
def ollama() -> tuple[Genkit, MagicMock]:
    """A Genkit app on a local `llava` whose Ollama chat calls are recorded, not sent."""
    plugin = Ollama()
    client = MagicMock()
    client.show = AsyncMock(
        return_value=ollama_api.ShowResponse.model_validate({
            'capabilities': ['tools', 'vision'],
            'model_info': {},
        })
    )
    client.chat = AsyncMock(
        return_value=ollama_api.ChatResponse(
            message=ollama_api.Message(role='assistant', content='ok'), done=True, done_reason='stop'
        )
    )
    plugin.client = lambda: client
    return Genkit(plugins=[plugin]), client


@contextmanager
def _serve(handler: Callable[[httpx.Request], httpx.Response]) -> Iterator[None]:
    """Route the plugin's image downloads to `handler` while keeping `_image_fetch_client`'s headers and settings."""
    client = _image_fetch_client()
    with patch.object(client, '_transport', httpx.MockTransport(handler)):
        yield


async def _describe(ai: Genkit, url: str) -> Any:  # noqa: ANN401
    return await ai.generate(
        model='ollama/llava',
        messages=[
            Message(
                role=Role.USER,
                content=[Part.from_text('describe'), Part.from_media(url=url, content_type='image/png')],
            )
        ],
    )


def _sent_image(client: MagicMock) -> bytes:
    """The bytes Ollama gets for the one image in the request."""
    messages = client.chat.call_args.kwargs['messages']
    images = messages[-1].images
    assert len(images) == 1
    return base64.b64decode(images[0].model_dump())


async def _sent(ai: Genkit, client: MagicMock, url: str) -> bytes:
    """Generate on `url` succeeds; returns the image bytes Ollama received."""
    response = await _describe(ai, url)
    assert response.error is None
    assert response.text == 'ok'
    assert response.message is not None
    assert response.messages[-1] == response.message
    return _sent_image(client)


async def _rejected(ai: Genkit, client: MagicMock, url: str) -> str:
    """Generate on `url` fails with INVALID_ARGUMENT and never calls Ollama; returns the message."""
    response = await _describe(ai, url)
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.finish_message is not None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
    client.chat.assert_not_called()
    return response.finish_message


def _never_read_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: Path) -> bytes:
        raise AssertionError(f'read {self}')

    monkeypatch.setattr(Path, 'read_bytes', refuse)


@pytest.mark.asyncio
async def test_generate_ollama_local_file_path_media_raises_invalid_argument(
    ollama: tuple[Genkit, MagicMock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path to a real file on the server raises INVALID_ARGUMENT, and the file is never opened."""
    ai, client = ollama
    secret = tmp_path / 'cat.png'
    secret.write_bytes(PNG)
    _never_read_files(monkeypatch)

    message = await _rejected(ai, client, str(secret))

    assert 'data: URL' in message
    assert 'http(s) URL' in message
    assert 'base64' in message


@pytest.mark.asyncio
async def test_generate_ollama_file_url_media_raises_invalid_argument(
    ollama: tuple[Genkit, MagicMock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `file://` URL raises INVALID_ARGUMENT, and the file is never opened."""
    ai, client = ollama
    secret = tmp_path / 'cat.png'
    secret.write_bytes(PNG)
    _never_read_files(monkeypatch)

    await _rejected(ai, client, secret.as_uri())


@pytest.mark.asyncio
async def test_generate_ollama_path_that_is_valid_base64_is_sent_as_decoded_bytes(
    ollama: tuple[Genkit, MagicMock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A string that is both valid base64 and an existing file name is decoded as base64; the file is not read."""
    ai, client = ollama
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'abcd').write_bytes(b'server-side secret')

    sent = await _sent(ai, client, 'abcd')

    assert sent == base64.b64decode('abcd')


@pytest.mark.asyncio
async def test_generate_ollama_data_uri_image_is_sent(ollama: tuple[Genkit, MagicMock]) -> None:
    """A `data:image/png;base64,...` part reaches Ollama as that image."""
    ai, client = ollama

    sent = await _sent(ai, client, 'data:image/png;base64,' + base64.b64encode(PNG).decode())

    assert sent == PNG


@pytest.mark.asyncio
async def test_generate_ollama_raw_base64_media_is_sent(ollama: tuple[Genkit, MagicMock]) -> None:
    """A bare base64 string with no `data:` prefix still reaches Ollama as that image."""
    ai, client = ollama

    sent = await _sent(ai, client, base64.b64encode(PNG).decode())

    assert sent == PNG


@pytest.mark.asyncio
async def test_generate_ollama_invalid_base64_media_raises_invalid_argument(ollama: tuple[Genkit, MagicMock]) -> None:
    """A string that is not a data: URL, an http(s) URL, or valid base64 raises INVALID_ARGUMENT."""
    ai, client = ollama

    message = await _rejected(ai, client, 'not base64!')

    assert 'base64' in message


@pytest.mark.asyncio
async def test_generate_ollama_malformed_data_uri_raises_invalid_argument(ollama: tuple[Genkit, MagicMock]) -> None:
    """A `data:` URL with no comma, or with a payload that isn't base64, raises INVALID_ARGUMENT."""
    ai, client = ollama

    for url in ('data:image/png;base64', 'data:image/png;base64,not base64!'):
        message = await _rejected(ai, client, url)
        assert 'data: URL' in message


@pytest.mark.asyncio
async def test_generate_ollama_http_image_is_downloaded_and_sent(ollama: tuple[Genkit, MagicMock]) -> None:
    """An http(s) image URL is downloaded with `User-Agent`, following redirects, and its bytes reach Ollama."""
    ai, client = ollama
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        if request.url.path == '/old.png':
            return httpx.Response(302, headers={'location': 'https://cdn.example.com/cat.png'})
        return httpx.Response(200, headers={'content-type': 'image/png'}, content=PNG)

    with _serve(handler):
        sent = await _sent(ai, client, 'https://example.com/old.png')

    assert sent == PNG
    assert len(seen_headers) == 2
    assert 'genkit' in seen_headers[0]['user-agent'].lower()
    assert seen_headers[0]['accept-encoding'] == 'identity'


@pytest.mark.asyncio
async def test_generate_ollama_multiline_base64_in_data_uri_and_bare_string_is_sent(
    ollama: tuple[Genkit, MagicMock],
) -> None:
    """ASCII whitespace in a `data:` URI or bare base64 string is stripped before decode."""
    ai, client = ollama
    encoded = base64.b64encode(PNG).decode()
    wrapped = f'  {encoded[:16]}\r\n\t{encoded[16:32]}\n{encoded[32:]}  '

    assert await _sent(ai, client, f'data:image/png;base64,{wrapped}') == PNG
    assert await _sent(ai, client, wrapped) == PNG


async def _chunks(total: int, consumed: list[int] | None = None) -> AsyncIterator[bytes]:
    for _ in range(total // MB):
        if consumed is not None:
            consumed.append(MB)
        yield b'\x00' * MB


@pytest.mark.asyncio
async def test_generate_ollama_http_media_over_cap_raises_invalid_argument(ollama: tuple[Genkit, MagicMock]) -> None:
    """A 21MB download raises INVALID_ARGUMENT naming host and path, without the query string."""
    ai, client = ollama

    with _serve(lambda request: httpx.Response(200, content=_chunks(21 * MB))):
        message = await _rejected(ai, client, 'https://example.com/huge.png?sig=s3cr3t')

    assert 'larger than 20MB' in message
    assert 'https://example.com/huge.png' in message
    assert 's3cr3t' not in message


@pytest.mark.asyncio
async def test_generate_ollama_http_media_with_large_content_length_raises_before_body(
    ollama: tuple[Genkit, MagicMock],
) -> None:
    """A `Content-Length` over 20MB fails without reading any body chunks."""
    ai, client = ollama
    consumed: list[int] = []

    with _serve(
        lambda request: httpx.Response(
            200, headers={'content-length': str(21 * MB)}, content=_chunks(21 * MB, consumed)
        )
    ):
        message = await _rejected(ai, client, 'https://example.com/huge.png?sig=s3cr3t')

    assert 'larger than 20MB' in message
    assert consumed == []


@pytest.mark.asyncio
async def test_generate_ollama_http_media_rejects_gzip_content_encoding_without_decompressing(
    ollama: tuple[Genkit, MagicMock],
) -> None:
    """A gzipped media response is rejected before decompressing."""
    ai, client = ollama

    with _serve(
        lambda request: httpx.Response(
            200,
            headers={'content-type': 'image/png', 'content-encoding': 'gzip'},
            content=_GZIP_25MB_ZEROS,
        )
    ):
        message = await _rejected(ai, client, 'https://example.com/bomb.png?sig=s3cr3t')

    assert 'https://example.com/bomb.png' in message
    assert 's3cr3t' not in message


@pytest.mark.asyncio
async def test_generate_ollama_http_media_status_errors_classify_and_redact_url(
    ollama: tuple[Genkit, MagicMock],
) -> None:
    """4xx (other than 408/429) raises INVALID_ARGUMENT; 5xx/408/429 stay retryable INTERNAL; both redact the URL."""
    from genkit_ollama._models import _download_media

    ai, client = ollama
    url = 'https://user:pw@storage.example.com/bucket/cat.png?X-Goog-Signature=s3cr3t'

    with _serve(lambda request: httpx.Response(404)):
        msg_404 = await _rejected(ai, client, url)
        with pytest.raises(Exception) as exc_404:
            await _download_media(url)

    for text in (msg_404, str(exc_404.value)):
        assert 'https://storage.example.com/bucket/cat.png' in text
        assert 's3cr3t' not in text
        assert 'pw' not in text

    for status in (408, 429, 503):
        with _serve(lambda request, s=status: httpx.Response(s)):
            res = await _describe(ai, url)
            with pytest.raises(httpx.HTTPStatusError) as exc_retryable:
                await _download_media(url)
        assert res.finish_reason == FinishReason.FAILED
        assert res.error is not None
        assert res.error.status == 'INTERNAL'
        assert res.finish_message is not None
        assert 'https://storage.example.com/bucket/cat.png' in str(exc_retryable.value)
        for text in (res.finish_message, str(exc_retryable.value)):
            assert 's3cr3t' not in text
            assert 'pw' not in text
    client.chat.assert_not_called()
