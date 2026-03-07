from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from exoqml.api.routes import router as api_router
from exoqml.config import get_settings
from exoqml.db import init_db

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.api_prefix)
init_db()


@app.on_event("startup")
def startup() -> None:
    Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)
    init_db()
