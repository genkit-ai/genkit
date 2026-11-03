# Copyright 2025 Google LLC
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

"""Action context definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass(init=False)
class RequestData:
    """What a context_provider sees for one HTTP request.

    ``method``, ``headers`` and ``input`` (the parsed ``{"data": ...}``) are the
    same on every adapter, so a provider that reads only those works in Flask,
    Django and FastAPI. ``headers`` keys are lowercase so ``Authorization`` and
    ``authorization`` look the same on every served flow. Repeated names become
    one comma-joined string, so ``headers.get('authorization')`` sees every
    value that arrived. Pass ``headers`` as a mapping or as the raw
    ``(name, value)`` pairs off the wire.

    ``request`` is the framework's own request object (Django ``HttpRequest``,
    Flask ``Request``, Starlette ``Request``) for what the shared fields don't
    carry: cookies, sessions, the logged-in user. A provider that reads it works
    on that framework only.

    Example:
        ```python
        from genkit import FlowContext, PublicError
        from genkit.web import RequestData
        from genkit_django import genkit_django_handler


        # 1. Django session auth; auser() because the view is async (Django 5.0+,
        #    on 4.2 use sync_to_async(lambda: request.request.user)())
        async def django_user(request: RequestData) -> dict[str, object]:
            user = await request.request.auser()
            if not user.is_authenticated:
                raise PublicError('UNAUTHENTICATED', 'Sign in first')
            return {'user_id': user.pk}


        # 2. The flow reads it from ctx.context
        @genkit_django_handler(ai, context_provider=django_user)
        @ai.flow()
        async def reorder(order_id: str, ctx: FlowContext) -> str:
            return f'reordered {order_id} for {ctx.context["user_id"]}'
            # => reordered ORD-9921 for 42
        ```
    """

    request: Any = None
    method: str = ''
    headers: dict[str, str] = field(default_factory=dict)
    input: Any = None

    def __init__(
        self,
        request: Any = None,  # noqa: ANN401
        method: str = '',
        headers: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
        input: Any = None,  # noqa: ANN401
    ) -> None:
        """Build request data; header names are lowercased and repeats comma-joined."""
        self.request = request
        self.method = method
        self.input = input
        pairs: Iterable[tuple[str, str]] = (
            cast(Mapping[str, str], headers).items() if isinstance(headers, Mapping) else headers or ()
        )
        # The same header can arrive twice (two Authorization lines, each hop
        # in an X-Forwarded-For chain). A context_provider looks the name up
        # once, so keep every part instead of dropping all but the last.
        self.headers = {}
        for name, value in pairs:
            key = name.lower()
            existing = self.headers.get(key)
            self.headers[key] = value if existing is None else f'{existing}, {value}'


ContextProvider = Callable[[RequestData], dict[str, Any] | Awaitable[dict[str, Any]]]
"""Middleware can read request data and add information to the context that will be passed to the
Action. If middleware throws an error, that error will fail the request and the Action will not
be called.

Expected cases should return a PublicError, which allows the request handler to
know what data is safe to return to end users.
Middleware can provide validation in addition to parsing. For example, an auth middleware can have
policies for validating auth in addition to passing auth context to the Action.
"""
