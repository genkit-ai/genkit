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

r"""For serving flows over HTTP or writing a web framework adapter.

``read_body`` turns the parsed JSON body into the flow input: callers send
``{"data": ...}``, or ``{}`` for no input. ``wants_stream`` says whether to
answer with server-sent events. When a flow fails, send ``error_body`` as the
JSON body and ``error_status`` as the HTTP status. A PublicError's text
reaches the caller, and bad input names the failing fields; anything else is
a generic ``Internal Error`` so server details stay in your logs. A
``ContextProvider`` reads the ``RequestData`` for one request and returns what
goes into the flow's context.

``to_sse`` frames a streaming handle (``flow.stream()``,
``ai.generate_stream()``, ``chat.send_stream()``) as the SSE lines every
adapter sends: a ``message`` frame per chunk, then ``result``, or an
``error`` frame instead. It's also how a hand-written route, say a FastAPI one
with typed params and ``Depends(...)``, streams without writing frames itself.
Non-streaming results are often Pydantic models, so serialize them with
``pydantic_core.to_json``: compact JSON by alias, the same bytes the adapters
send.

Example:
    ```python
    from pydantic_core import to_json
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse

    from genkit.web import RequestData, error_body, error_status, read_body, to_sse, wants_stream


    def to_wire(payload: object) -> bytes:
        return to_json(payload, by_alias=True, exclude_none=True)


    # 1. Read auth off the request; the result becomes ctx.context in the flow
    async def auth(request_data: RequestData) -> dict[str, object]:
        return {'token': request_data.headers.get('authorization')}


    # 2. One route: parse, build context, then stream or run once
    async def serve(request: Request) -> Response:
        try:
            body = await request.json()
            flow_input = read_body(body)
            request_data = RequestData(
                request=request,
                method=request.method,
                headers=request.headers.items(),
                input=body.get('data'),
            )
            ctx = await auth(request_data)
            if wants_stream(accept=request.headers.get('accept'), stream=request.query_params.get('stream')):
                stream = order_flow.stream(input=flow_input, context=ctx)
                return StreamingResponse(to_sse(stream), media_type='text/event-stream')
            response = await order_flow.run(input=flow_input, context=ctx)
            return Response(to_wire({'result': response.response}), media_type='application/json')
        except Exception as e:
            return Response(to_wire(error_body(e)), status_code=error_status(e), media_type='application/json')


    # POST /order {"data": {"dish": "pad thai"}}
    # => {"result":{"text":"1x pad thai"}}
    # POST /order?stream=true {"data": {"dish": "pad thai"}}
    # => data: {"message":"Plating pad thai"}
    #    data: {"result":{"text":"1x pad thai"}}
    # POST /order {"input": "pad thai"}
    # => 400 {"message":"Flow request must be wrapped in {\"data\": ...}","status":"INVALID_ARGUMENT"}
    ```
"""

from genkit._core._context import ContextProvider, RequestData
from genkit._core._error import error_body, error_status
from genkit._core._web import read_body, wants_stream
from genkit.web._sse import StreamHandle, to_sse

__all__ = [
    'ContextProvider',
    'RequestData',
    'StreamHandle',
    'error_body',
    'error_status',
    'read_body',
    'to_sse',
    'wants_stream',
]
