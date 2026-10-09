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
JSON body and ``error_status`` as the HTTP status. Only a PublicError's text
reaches the caller; anything else is a generic ``Internal Error`` so server
details stay in your logs.

Example:
    ```python
    from genkit.web import error_body, error_status, read_body, wants_stream


    async def events(flow_input):
        try:
            stream = my_flow.stream(input=flow_input)
            async for chunk in stream.stream:
                yield f'data: {json.dumps({"message": chunk})}\n\n'
            yield f'data: {json.dumps({"result": await stream.response})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"error": error_body(e)})}\n\n'


    async def serve(request):
        try:
            flow_input = read_body(await request.json())
            if wants_stream(accept=request.headers.get('accept'), stream=request.query_params.get('stream')):
                return StreamingResponse(events(flow_input), media_type='text/event-stream')
            response = await my_flow.run(input=flow_input)
            return JSONResponse({'result': response.response})
        except Exception as e:
            return JSONResponse(error_body(e), status_code=error_status(e))
    ```
"""

from genkit._core._error import error_body, error_status
from genkit._core._web import read_body, wants_stream

__all__ = ['error_body', 'error_status', 'read_body', 'wants_stream']
