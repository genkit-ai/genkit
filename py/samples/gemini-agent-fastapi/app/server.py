# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""HTTP server for the Gemini agent — assembles the app, nothing more.

Runtime/env bootstrap lives in ``app.core.config``; the agent lives in ``app.ai``;
routes live in ``app.api``. This file just wires them together.
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.sessions import router as sessions_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title='Gemini Agent on Cloud Run',
        description='Streaming Gemini agent API with auth, sessions, and token accounting.',
        version='0.1.0',
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    app.include_router(auth_router, prefix='/api/auth', tags=['auth'])
    app.include_router(sessions_router, prefix='/api/sessions', tags=['sessions'])
    app.include_router(chat_router, prefix='/api', tags=['chat'])

    @app.get('/health')
    async def health() -> dict[str, str]:
        return {'status': 'ok'}

    return app


app = create_app()


if __name__ == '__main__':
    settings = get_settings()
    uvicorn.run(
        'app.server:app',
        host='0.0.0.0',  # noqa: S104
        port=settings.port,
        reload=os.environ.get('GENKIT_ENV') == 'dev',
    )
