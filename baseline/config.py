from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import AnyHttpUrl

BASE_DIR = Path(__file__).parent.parent


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    host: str
    port: int
    collection_name: str


class DrafterAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    drafter_url: AnyHttpUrl | None = None
    drafter_key: str


class GeneratorAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    generator_url: AnyHttpUrl | None = None
    generator_key: str


class Settings(BaseSettings):
    qdrant: QdrantConfig = QdrantConfig()
    drafter_api: DrafterAPIConfig = DrafterAPIConfig()


settings = Settings()
