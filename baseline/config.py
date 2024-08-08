from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import AnyHttpUrl

BASE_DIR = Path(__file__).parent.parent


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    host: str
    port: int
    collection_name: str


class LLMAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")
    url: AnyHttpUrl | None = None
    key: str


class Settings(BaseSettings):
    qdrant: QdrantConfig = QdrantConfig()
    llm_api: LLMAPIConfig = LLMAPIConfig()


settings = Settings()
