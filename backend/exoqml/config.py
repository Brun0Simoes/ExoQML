from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="EXOQML_",
    )

    app_name: str = "ExoQML API"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./exoqml.db"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
    max_points: int = 2048
    enable_qml: bool = False
    allow_synthetic_fallback: bool = True
    model_path: str = ""
    qml_model_path: str = ""
    device: str = "auto"
    cache_dir: str = "./data/cache"
    log_level: str = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        allowed = {"auto", "cpu", "cuda"}
        value = value.lower()
        if value not in allowed:
            raise ValueError(f"device must be one of: {sorted(allowed)}")
        return value

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        value = value.upper()
        if value not in allowed:
            raise ValueError(f"log_level must be one of: {sorted(allowed)}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
