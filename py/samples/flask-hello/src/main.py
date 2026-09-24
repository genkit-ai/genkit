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

"""Hang a flow on the Flask app you already have."""

from flask import Flask
from genkit_flask import genkit_flask_handler
from genkit_google_genai import GoogleAI

from genkit import ActionRunContext, Genkit

ai = Genkit(
    plugins=[GoogleAI()],
    model=GoogleAI.gemini_model('gemini-flash-latest'),
)
app = Flask(__name__)


@app.post('/say_hi')
@genkit_flask_handler(ai)
@ai.flow()
async def say_hi(name: str, ctx: ActionRunContext) -> str:
    # generate_stream + send_chunk is how the Flask handler streams tokens.
    stream = ai.generate_stream(prompt=f'tell a short joke about {name}')
    async for chunk in stream.stream:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    return (await stream.response).text


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
