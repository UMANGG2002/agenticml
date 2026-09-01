"""Shared FastAPI dependencies (DB session, settings, etc.)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "postgresql+psycopg2://ame:ame@localhost:5432/ame"
    artifact_storage_path: str = "./data/artifacts"


@lru_cache
def get_settings() -> Settings:
    return Settings()
